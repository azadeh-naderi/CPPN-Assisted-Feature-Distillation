import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import Subset

from src.cppn.compile import genome_to_pattern
from src.cppn.evolve import load_neat_config
from src.cppn.serialize import load_genome, load_top_k_genomes
from src.cppn.trainable import train_trainable_cppn
from src.data.datasets import load_dataset, split_dataset
from src.distill.modes import MODES
from src.distill.trainer import DistillTrainer
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.logging import RunLogger, get_logger
from src.utils.seed import set_seed

log = get_logger("train_student")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--teacher-ckpt", default=None)
    parser.add_argument("--genome-path", default=None)
    parser.add_argument("--ensemble-genomes", default=None)
    parser.add_argument("--random-cppn-variant", default="coord", choices=["legacy", "coord"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, eval_ds = load_dataset(cfg["dataset_name"], cfg.get("data_root", "./data"))
    if args.smoke and "smoke_subset_size" in cfg:
        n = min(cfg["smoke_subset_size"], len(train_ds))
        train_ds, eval_ds = Subset(train_ds, range(n)), Subset(eval_ds, range(n))

    train_loader, val_loader, test_loader = split_dataset(
        train_ds, eval_ds, cfg["batch_size"], cfg["train_size_percent"], cfg["val_size_percent"], seed
    )

    student = build_model(cfg["model_name"], cfg["input_channels"], cfg["num_classes"], pretrained=False)

    teacher = None
    if args.mode != "student_only":
        if not args.teacher_ckpt:
            raise ValueError("--teacher-ckpt is required unless --mode student_only")
        teacher = build_model(
            cfg["model_name"], cfg["input_channels"], cfg["num_classes"], cfg.get("pretrained", False)
        )
        teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
        teacher = teacher.to(device)

    cppn_cfg = cfg.get("cppn", {})
    trained_cfg = cfg.get("trained_cppn", {})

    patterns = None
    neat_config = None
    if args.mode == "kd_evolved_cppn":
        if args.ensemble_genomes:
            top_k = load_top_k_genomes(args.ensemble_genomes)
            neat_config = load_neat_config(cppn_cfg["neat_config"])
            patterns = [
                genome_to_pattern(g, neat_config.genome_config, cfg["image_size"], cfg["input_channels"], device)
                for _fitness, g in top_k
            ]
        else:
            if not args.genome_path:
                raise ValueError("--genome-path (or --ensemble-genomes) required for mode kd_evolved_cppn")
            genome, neat_config = load_genome(args.genome_path)
            patterns = [
                genome_to_pattern(
                    genome, neat_config.genome_config, cfg["image_size"], cfg["input_channels"], device
                )
            ]
    elif args.mode == "kd_trained_cppn":
        num_steps = trained_cfg.get("num_steps", 500)
        if args.smoke:
            num_steps = min(num_steps, 20)
        pattern = train_trainable_cppn(
            teacher=teacher,
            probe_dataset=eval_ds,
            image_size=cfg["image_size"],
            channels=cfg["input_channels"],
            view_op=cppn_cfg.get("view_op", "multiplicative"),
            view_scale=cppn_cfg.get("view_scale", 0.5),
            num_steps=num_steps,
            probe_batch_size=trained_cfg.get("probe_batch_size", 256),
            lr=trained_cfg.get("lr", 0.01),
            temperature=trained_cfg.get("temperature", 4),
            diversity_weight=trained_cfg.get("diversity_weight", 1.0),
            kl_weight=trained_cfg.get("kl_weight", 1.0),
            seed=seed,
            device=device,
        )
        patterns = [pattern]
    elif args.mode == "kd_random_cppn" and args.random_cppn_variant == "coord":
        neat_config = load_neat_config(cppn_cfg["neat_config"])

    student_cfg = cfg["student"]
    num_epochs = student_cfg["num_epochs"] if not args.smoke else min(student_cfg["num_epochs"], 2)

    trainer = DistillTrainer(
        student=student,
        teacher=teacher,
        mode=args.mode,
        dataset_name=cfg["dataset_name"],
        device=device,
        patterns=patterns,
        random_cppn_variant=args.random_cppn_variant,
        random_cppn_seed=cppn_cfg.get("random_seed", seed),
        neat_config=neat_config,
        image_size=cfg["image_size"],
        channels=cfg["input_channels"],
        view_op=cppn_cfg.get("view_op", "multiplicative"),
        view_scale=cppn_cfg.get("view_scale", 0.5),
        temperature=student_cfg["temperature"],
        alpha=student_cfg["alpha"],
        lr=student_cfg["lr"],
        momentum=student_cfg["momentum"],
    )

    run_id = f"{cfg['dataset_name']}_{cfg['model_name']}_{args.mode}_{seed}_{int(time.time())}"
    run_dir = Path("results/students") / run_id
    run_logger = RunLogger(run_dir)
    run_logger.log_config(cfg)

    trainer.fit(train_loader, val_loader, num_epochs, run_logger)
    test_acc = trainer.evaluate(test_loader)
    ckpt_path = run_logger.save_artifact("checkpoint.pt", student.state_dict())
    run_logger.log_summary(
        mode=args.mode,
        test_accuracy=test_acc,
        checkpoint_path=str(ckpt_path),
        teacher_ckpt=args.teacher_ckpt,
        smoke=args.smoke,
    )
    log.info("[%s] student test accuracy: %.2f", args.mode, test_acc)
    print(f"{args.mode}\t{test_acc:.4f}\t{run_dir}")


if __name__ == "__main__":
    main()

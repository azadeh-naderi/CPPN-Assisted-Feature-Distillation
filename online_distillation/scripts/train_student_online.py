import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import Subset

from online_distillation.src.online_trainer import ONLINE_MODES, OnlineDistillTrainer
from src.cppn.compile import genome_to_pattern
from src.cppn.evolve import create_random_genome, load_neat_config
from src.data.datasets import load_dataset, split_dataset
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.logging import RunLogger, get_logger
from src.utils.seed import set_seed

log = get_logger("train_student_online")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=ONLINE_MODES)
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

    cppn_cfg = cfg["cppn"]
    neat_config = load_neat_config(cppn_cfg["neat_config"])
    genome = create_random_genome(neat_config, cppn_cfg.get("random_seed", seed))
    pattern = genome_to_pattern(
        genome, neat_config.genome_config, cfg["image_size"], cfg["input_channels"], device
    )

    student_cfg = cfg["student"]
    num_epochs = student_cfg["num_epochs"] if not args.smoke else min(student_cfg["num_epochs"], 2)

    trainer = OnlineDistillTrainer(
        student=student,
        mode=args.mode,
        dataset_name=cfg["dataset_name"],
        device=device,
        pattern=pattern,
        view_op=cppn_cfg.get("view_op", "multiplicative"),
        view_scale=cppn_cfg.get("view_scale", 0.5),
        temperature=student_cfg["temperature"],
        alpha=student_cfg["alpha"],
        lr=student_cfg["lr"],
        momentum=student_cfg["momentum"],
        scheduler=student_cfg.get("scheduler", False),
        step_size=student_cfg.get("step_size", 30),
        gamma=student_cfg.get("gamma", 0.1),
    )

    run_id = f"{cfg['dataset_name']}_{cfg['model_name']}_{args.mode}_{seed}_{int(time.time())}"
    run_dir = Path("results/online_distillation") / run_id
    run_logger = RunLogger(run_dir)
    run_logger.log_config(cfg)

    trainer.fit(train_loader, val_loader, num_epochs, run_logger)
    test_acc = trainer.evaluate(test_loader)
    ckpt_path = run_logger.save_artifact("checkpoint.pt", student.state_dict())
    run_logger.log_summary(
        mode=args.mode, test_accuracy=test_acc, checkpoint_path=str(ckpt_path), smoke=args.smoke
    )
    log.info("[%s] student test accuracy: %.2f", args.mode, test_acc)
    print(f"{args.mode}\t{test_acc:.4f}\t{run_dir}")


if __name__ == "__main__":
    main()

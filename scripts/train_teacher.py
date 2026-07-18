import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import Subset

from src.data.datasets import load_dataset, split_dataset
from src.distill.trainer import evaluate_model, train_teacher
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.logging import RunLogger, get_logger
from src.utils.seed import set_seed

log = get_logger("train_teacher")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
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

    model = build_model(cfg["model_name"], cfg["input_channels"], cfg["num_classes"], cfg.get("pretrained", False))

    run_id = f"{cfg['dataset_name']}_{cfg['model_name']}_teacher_{seed}_{int(time.time())}"
    run_dir = Path("results/teachers") / run_id
    run_logger = RunLogger(run_dir)
    run_logger.log_config(cfg)

    teacher_cfg = cfg["teacher"]
    num_epochs = teacher_cfg["num_epochs"] if not args.smoke else min(teacher_cfg["num_epochs"], 2)
    teacher = train_teacher(
        model,
        cfg["dataset_name"],
        train_loader,
        val_loader,
        num_epochs,
        teacher_cfg["lr"],
        teacher_cfg["momentum"],
        device,
        run_logger,
    )

    test_acc = evaluate_model(teacher, cfg["dataset_name"], test_loader, device)
    ckpt_path = run_logger.save_artifact("checkpoint.pt", teacher.state_dict())
    run_logger.log_summary(
        test_accuracy=test_acc, checkpoint_path=str(ckpt_path), config_path=args.config, smoke=args.smoke
    )
    log.info("teacher test accuracy: %.2f", test_acc)
    print(run_dir)


if __name__ == "__main__":
    main()

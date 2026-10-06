"""Train one teacher or student under the benchmark protocol.

    # teacher
    python method_paper/scripts/train.py --config method_paper/configs/cifar100.yaml \
        --role teacher --arch resnet56 --seed 0
    # student_only
    python method_paper/scripts/train.py --config method_paper/configs/cifar100.yaml \
        --role student --arch resnet20 --mode ce --seed 0
    # kd (needs the teacher's run to have finished)
    python method_paper/scripts/train.py --config method_paper/configs/cifar100.yaml \
        --role student --arch resnet20 --mode kd --teacher resnet56 --seed 0

Run directories are deterministic (no timestamps), so a student finds its
teacher's checkpoint by name, and a finished run (summary.json present) is
skipped unless --overwrite is passed.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from method_paper.src.data import NUM_CLASSES, get_loaders
from method_paper.src.models.registry import SMALL_LR_MODELS, build_model
from method_paper.src.trainer import evaluate, fit
from src.utils.config import load_config
from src.utils.logging import RunLogger, get_logger
from src.utils.seed import set_seed

log = get_logger("method_paper.train")

DEFAULT_RESULTS_ROOT = Path("results/method_paper")


def run_dir_for(
    results_root: Path, dataset: str, role: str, arch: str, mode: str, seed: int, teacher: str | None = None
) -> Path:
    if role == "teacher":
        return results_root / dataset / "teachers" / f"{arch}_seed{seed}"
    run_mode = "student_only" if mode == "ce" else mode
    return results_root / dataset / "students" / f"{teacher or 'none'}__{arch}__{run_mode}__seed{seed}"


def run_training(
    cfg: dict,
    role: str,
    arch: str,
    seed: int,
    mode: str = "ce",
    teacher: str | None = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    fake_data: bool = False,
    max_batches: int | None = None,
    epochs: int | None = None,
    overwrite: bool = False,
    device: torch.device | None = None,
) -> Path:
    if role not in ("teacher", "student"):
        raise ValueError(f"role must be 'teacher' or 'student', got {role!r}")
    if role == "teacher" and mode != "ce":
        raise ValueError("teachers are trained with mode='ce'")
    if mode == "kd" and not teacher:
        raise ValueError("mode='kd' requires --teacher")

    dataset = cfg["dataset"]
    results_root = Path(results_root)
    run_dir = run_dir_for(results_root, dataset, role, arch, mode, seed, teacher)
    if (run_dir / "summary.json").exists() and not overwrite:
        log.info("skipping %s: already finished (pass --overwrite to rerun)", run_dir)
        return run_dir
    (run_dir / "training_log.csv").unlink(missing_ok=True)  # don't append onto a stale log

    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, _val_loader, test_loader = get_loaders(
        dataset, cfg.get("data_root", "./data"), cfg["training"]["batch_size"],
        cfg.get("num_workers", 4), cfg.get("val_size", 0), seed, fake_data,
    )
    num_classes = NUM_CLASSES[dataset]
    model = build_model(arch, num_classes)

    teacher_model = None
    teacher_top1 = None
    if mode == "kd":
        teacher_dir = run_dir_for(results_root, dataset, "teacher", teacher, "ce", cfg["teacher_seed"])
        ckpt = teacher_dir / "checkpoint.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"teacher checkpoint not found: {ckpt} -- train the teacher first")
        teacher_model = build_model(teacher, num_classes)
        teacher_model.load_state_dict(torch.load(ckpt, map_location=device))
        teacher_model = teacher_model.to(device)
        teacher_top1 = evaluate(teacher_model, test_loader, dataset, device)["top1"]
        log.info("loaded teacher %s from %s (test top1 %.2f)", teacher, ckpt, teacher_top1)

    tcfg = cfg["training"]
    lr = tcfg["small_model_lr"] if arch in SMALL_LR_MODELS else tcfg["lr"]
    num_epochs = epochs if epochs is not None else tcfg["epochs"]

    run_logger = RunLogger(run_dir)
    run_logger.log_config(
        {**cfg, "role": role, "arch": arch, "mode": mode, "seed": seed, "teacher": teacher,
         "effective_lr": lr, "effective_epochs": num_epochs, "fake_data": fake_data, "max_batches": max_batches}
    )

    summary = fit(
        model, train_loader, test_loader, dataset, device,
        epochs=num_epochs, lr=lr, lr_milestones=tcfg["lr_milestones"], lr_decay=tcfg["lr_decay"],
        momentum=tcfg["momentum"], weight_decay=tcfg["weight_decay"], mode=mode, teacher=teacher_model,
        temperature=cfg["kd"]["temperature"], alpha=cfg["kd"]["alpha"], gamma=cfg["kd"]["gamma"],
        run_logger=run_logger, max_batches=max_batches,
    )

    run_logger.save_artifact("checkpoint.pt", model.state_dict())
    run_logger.log_summary(
        **summary, dataset=dataset, role=role, arch=arch, mode=mode, seed=seed, teacher=teacher,
        teacher_test_top1=teacher_top1, lr=lr, epochs=num_epochs, fake_data=fake_data,
    )
    log.info("%s done: final test top1 %.2f (best %.2f @ epoch %d)", run_dir.name,
             summary["final_test_top1"], summary["best_test_top1"], summary["best_epoch"])
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True, choices=["teacher", "student"])
    parser.add_argument("--arch", required=True)
    parser.add_argument("--mode", default="ce", choices=["ce", "kd"])
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--fake-data", action="store_true", help="random data, no download (wiring check)")
    parser.add_argument("--max-batches", type=int, default=None, help="cap batches per epoch (wiring check)")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs (wiring check)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_training(
        load_config(args.config), args.role, args.arch, args.seed, args.mode, args.teacher,
        Path(args.results_root), args.fake_data, args.max_batches, args.epochs, args.overwrite,
    )


if __name__ == "__main__":
    main()

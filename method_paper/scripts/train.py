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
    # any Phase 1 baseline, e.g. evolved CPPN views
    python method_paper/scripts/train.py --config method_paper/configs/cifar100.yaml \
        --role student --arch resnet20 --mode kd_evolved_cppn --teacher resnet56 --seed 0

Run directories are deterministic (no timestamps), so a student finds its
teacher's checkpoint by name, and a finished run (summary.json present) is
skipped unless --overwrite is passed.

CPPN-view modes run their view search first (seeded by the student seed,
against the frozen teacher, on unaugmented training images) and save the
patterns and search logs under <run_dir>/views/.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from method_paper.src.data import NUM_CLASSES, get_loaders, get_probe_dataset, train_transform
from method_paper.src.models.registry import SMALL_LR_MODELS, build_model
from method_paper.src.trainer import evaluate, fit
from src.utils.config import load_config
from src.utils.logging import RunLogger, get_logger
from src.utils.seed import set_seed

log = get_logger("method_paper.train")

DEFAULT_RESULTS_ROOT = Path("results/method_paper")

# run mode -> (trainer mode, loader augmentation, CPPN view source)
RUN_MODES = {
    "ce": ("ce", None, None),
    "kd": ("kd", None, None),
    "kd_randaugment": ("kd", "randaugment", None),
    "kd_cutmix": ("kd_cutmix", None, None),
    "kd_random_cppn": ("kd_view", None, "random"),
    "kd_trained_cppn": ("kd_view", None, "trained"),
    "kd_evolved_cppn": ("kd_view", None, "evolved"),
    "kd_random_search": ("kd_view", None, "random_search"),
}


def _resolve(path: str) -> str:
    """Config paths (e.g. the NEAT config) are relative to the repo root."""
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)


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
    if mode not in RUN_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Choose one of {list(RUN_MODES)}.")
    if role == "teacher" and mode != "ce":
        raise ValueError("teachers are trained with mode='ce'")
    if mode != "ce" and not teacher:
        raise ValueError(f"mode={mode!r} requires --teacher")
    trainer_mode, augment, view_source = RUN_MODES[mode]

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

    train_tf = None
    if augment == "randaugment":
        ra = cfg["augment"]["randaugment"]
        train_tf = train_transform("randaugment", ra["num_ops"], ra["magnitude"])
    train_loader, _val_loader, test_loader = get_loaders(
        dataset, cfg.get("data_root", "./data"), cfg["training"]["batch_size"],
        cfg.get("num_workers", 4), cfg.get("val_size", 0), seed, fake_data, train_tf,
    )
    num_classes = NUM_CLASSES[dataset]
    model = build_model(arch, num_classes)

    teacher_model = None
    teacher_top1 = None
    if mode != "ce":
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

    mode_kwargs = {}
    if view_source is not None:
        vcfg = {**cfg["views"], "neat_config": _resolve(cfg["views"]["neat_config"])}
        probe_ds = get_probe_dataset(dataset, cfg.get("data_root", "./data"), fake_data)
        from method_paper.src.views import build_views  # imports neat; only needed here

        patterns = build_views(view_source, teacher_model, dataset, probe_ds, vcfg, seed, device, run_dir / "views")
        mode_kwargs = dict(
            patterns=patterns, view_op=vcfg["view_op"], view_scale=vcfg["view_scale"],
            view_weight=vcfg["view_weight"], view_sampling=vcfg["view_sampling"],
        )
    elif trainer_mode == "kd_cutmix":
        cm = cfg["augment"]["cutmix"]
        mode_kwargs = dict(cutmix_alpha=cm["alpha"], cutmix_prob=cm["prob"])

    summary = fit(
        model, train_loader, test_loader, dataset, device,
        epochs=num_epochs, lr=lr, lr_milestones=tcfg["lr_milestones"], lr_decay=tcfg["lr_decay"],
        momentum=tcfg["momentum"], weight_decay=tcfg["weight_decay"], mode=trainer_mode, teacher=teacher_model,
        temperature=cfg["kd"]["temperature"], alpha=cfg["kd"]["alpha"], gamma=cfg["kd"]["gamma"],
        run_logger=run_logger, max_batches=max_batches, **mode_kwargs,
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
    parser.add_argument("--mode", default="ce", choices=list(RUN_MODES))
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

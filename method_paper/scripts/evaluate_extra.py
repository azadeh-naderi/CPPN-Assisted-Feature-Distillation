"""Phase 3 post-hoc evaluation of finished student runs: calibration (ECE,
NLL) and agreement with the teacher on the official test split, and
accuracy/ECE on CIFAR-100-C. Writes <run_dir>/extra_metrics.json, which
summarize.py picks up. No training; one GPU job covers every run.

    # all finished students (skips runs that already have extra_metrics.json)
    python method_paper/scripts/evaluate_extra.py --config method_paper/configs/cifar100.yaml
    # only some runs
    python method_paper/scripts/evaluate_extra.py --config ... --runs 'resnet56__resnet20__*'

CIFAR-100-C must be downloaded beforehand (compute nodes have no internet);
without it, only the clean-test metrics are computed. See method_paper/README.md.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from method_paper.scripts.train import DEFAULT_RESULTS_ROOT, run_dir_for
from method_paper.src.data import NUM_CLASSES, get_loaders
from method_paper.src.metrics import (
    agreement_metrics,
    available_corruptions,
    cifar_c_metrics,
    classification_metrics,
    predict,
)
from method_paper.src.models.registry import build_model
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("method_paper.evaluate_extra")


def load_model(arch: str, num_classes: int, ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(arch, num_classes)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


def evaluate_run(run_dir: Path, cfg: dict, results_root: Path, test_loader, device: torch.device,
                 cifar_c_dir: Path | None, teacher_cache: dict) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    dataset = cfg["dataset"]
    num_classes = NUM_CLASSES[dataset]
    student = load_model(summary["arch"], num_classes, run_dir / "checkpoint.pt", device)
    logits, labels = predict(student, test_loader, dataset, device)
    metrics = {"test": classification_metrics(logits, labels)}

    teacher = summary.get("teacher")
    if teacher:
        if teacher not in teacher_cache:
            teacher_dir = run_dir_for(results_root, dataset, "teacher", teacher, "ce", cfg["teacher_seed"])
            teacher_model = load_model(teacher, num_classes, teacher_dir / "checkpoint.pt", device)
            teacher_cache[teacher] = predict(teacher_model, test_loader, dataset, device)[0]
        metrics["test"].update(agreement_metrics(logits, teacher_cache[teacher]))

    if cifar_c_dir is not None:
        metrics["cifar_c"] = cifar_c_metrics(student, cifar_c_dir, dataset, device)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--runs", default="*", help="glob over student run directory names")
    parser.add_argument("--cifar-c-dir", default="data/CIFAR-100-C")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fake-data", action="store_true", help="random test data (wiring check)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_root = Path(args.results_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train, _val, test_loader = get_loaders(cfg["dataset"], cfg.get("data_root", "./data"), 500,
                                            cfg.get("num_workers", 4), fake_data=args.fake_data)

    cifar_c_dir = Path(args.cifar_c_dir)
    if available_corruptions(cifar_c_dir):
        log.info("CIFAR-100-C: %d corruptions in %s", len(available_corruptions(cifar_c_dir)), cifar_c_dir)
    else:
        log.warning("no CIFAR-100-C files in %s -- computing clean-test metrics only", cifar_c_dir)
        cifar_c_dir = None

    students_dir = results_root / cfg["dataset"] / "students"
    run_dirs = sorted(d for d in students_dir.glob(args.runs) if (d / "summary.json").exists())
    teacher_cache: dict = {}
    for run_dir in run_dirs:
        out = run_dir / "extra_metrics.json"
        if out.exists() and not args.overwrite:
            continue
        metrics = evaluate_run(run_dir, cfg, results_root, test_loader, device, cifar_c_dir, teacher_cache)
        out.write_text(json.dumps(metrics, indent=2))
        c = metrics.get("cifar_c", {}).get("mean_top1")
        log.info("%s: top1 %.2f, ECE %.2f%s", run_dir.name, metrics["test"]["top1"], metrics["test"]["ece"],
                 f", CIFAR-100-C {c:.2f}" if c is not None else "")
    log.info("evaluated %d of %d runs", sum(1 for d in run_dirs if (d / "extra_metrics.json").exists()),
             len(run_dirs))


if __name__ == "__main__":
    main()

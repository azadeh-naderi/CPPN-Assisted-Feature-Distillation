"""Plots mean +/- std validation-accuracy learning curves across seeds for
one or more modes, reading per-epoch training_log.csv files archived under
results_archive/ (git-tracked, see README.md).

Where multiple runs share the same mode/seed directory-naming prefix (e.g.
kd_evolved_cppn across 13 CIFAR-10 fitness-tuning attempts, or
hard_label_augmentation/self_consistency_random_cppn across the
online_distillation folder's 3 attempts), disambiguates by matching each
candidate's summary.json test_accuracy against a caller-supplied "known"
value from experiments/EXPERIMENT_LOG.md / online_distillation/EXPERIMENT_LOG.md
-- picking the wrong attempt's run silently would plot a stale/invalidated
result. Modes without ambiguity (e.g. student_only/kd/kd_random_cppn, whose
training recipe never changed across CPPN-fitness attempts) just use the
most recent run per seed.

Usage:
    python scripts/plot_learning_curves.py --dataset cifar_10 --model resnet18 --num-seeds 10 \
        --mode student_only --results-dir results_archive/students \
        --mode kd --results-dir results_archive/students \
        --mode kd_random_cppn --results-dir results_archive/students \
        --mode kd_evolved_cppn --results-dir results_archive/students --known-accuracies 82.80,74.54,82.28,79.76,65.82,79.04,81.44,79.42,29.40,73.20 \
        --mode hard_label_augmentation --results-dir results_archive/online_distillation --known-accuracies 82.76,82.26,83.42,79.02,83.10,82.12,84.20,83.94,82.86,84.04 \
        --mode self_consistency_random_cppn --results-dir results_archive/online_distillation --known-accuracies 81.06,83.06,82.70,83.12,81.64,82.86,83.74,83.34,82.70,82.92 \
        --output experiments/figures/cifar10_resnet18_learning_curves.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = [
    "#555555", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf", "#8c564b",
]


def find_dirs(base: Path, dataset: str, model: str, mode: str, seed: int) -> list[Path]:
    prefix = f"{dataset}_{model}_{mode}_{seed}_"
    return [d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)]


def pick_dir(base: Path, dataset: str, model: str, mode: str, seed: int, known_acc: float | None) -> Path:
    candidates = find_dirs(base, dataset, model, mode, seed)
    if not candidates:
        raise RuntimeError(f"No dirs found for {mode} seed={seed} under {base}")
    if known_acc is None:
        return max(candidates, key=lambda d: int(d.name.rsplit("_", 1)[-1]))
    best, best_diff = None, None
    for d in candidates:
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        acc = json.loads(summary_path.read_text())["test_accuracy"]
        diff = abs(acc - known_acc)
        if best_diff is None or diff < best_diff:
            best, best_diff = d, diff
    if best is None or best_diff > 0.01:
        raise RuntimeError(f"No dir matched known_accuracy={known_acc} for {mode} seed={seed} (best_diff={best_diff})")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar_10")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument(
        "--mode", action="append", dest="modes", required=True,
        help="repeat once per mode to plot, in order paired with --results-dir/--known-accuracies",
    )
    parser.add_argument(
        "--results-dir", action="append", dest="results_dirs", required=True,
        help="results_archive subdir for the preceding --mode (e.g. results_archive/students)",
    )
    parser.add_argument(
        "--known-accuracies", action="append", dest="known_accuracies", default=[],
        help="comma-separated per-seed test_accuracy for the preceding --mode, to disambiguate "
        "multiple attempts sharing the same directory naming; omit (or pass '') for modes with no ambiguity",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.modes) != len(args.results_dirs):
        raise ValueError("--mode and --results-dir must be given the same number of times, in matching order")
    known_by_mode = args.known_accuracies + [""] * (len(args.modes) - len(args.known_accuracies))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, (mode, results_dir, known_str) in enumerate(zip(args.modes, args.results_dirs, known_by_mode)):
        base = Path(results_dir)
        known = [float(x) for x in known_str.split(",")] if known_str else [None] * args.num_seeds
        seed_curves = []
        for seed in range(args.num_seeds):
            d = pick_dir(base, args.dataset, args.model, mode, seed, known[seed])
            df = pd.read_csv(d / "training_log.csv")
            seed_curves.append(df["val_accuracy"].to_numpy())
        min_len = min(len(c) for c in seed_curves)
        arr = np.stack([c[:min_len] for c in seed_curves])
        epochs = np.arange(arr.shape[1])
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        color = COLORS[i % len(COLORS)]
        ax.plot(epochs, mean, label=mode, color=color, linewidth=1.8)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.12)
        print(f"{mode}: {arr.shape[0]} seeds x {arr.shape[1]} epochs, final mean={mean[-1]:.2f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"{args.dataset} / {args.model} — learning curves (mean ± std over {args.num_seeds} seeds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

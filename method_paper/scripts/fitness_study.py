"""Fitness-vs-accuracy study (see method_paper/src/fitness_study.py).

    # 1. build the genome bank (GPU, ~30 min): slurm/fitness_study_bank.sbatch
    python method_paper/scripts/fitness_study.py bank --config method_paper/configs/cifar100.yaml
    # 2. short student runs: slurm/fitness_study_train.sbatch (launch.py stage fitness_study)
    # 3. correlate fitness (and each term) with the student's accuracy change
    python method_paper/scripts/fitness_study.py analyze --config method_paper/configs/cifar100.yaml

`analyze` writes per_genome.csv, correlations.csv and fitness_study.png next
to the bank and prints the correlation table. The accuracy change is each
genome's mean final test top-1 over seeds minus the short-schedule kd mean.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from method_paper.scripts.train import DEFAULT_RESULTS_ROOT, _resolve, run_dir_for
from src.utils.config import load_config

# Predictors correlated with the accuracy change: the fitness itself, its
# terms, and descriptors the fitness doesn't use directly.
PREDICTORS = ("fitness", "diversity", "agreement", "top1_agreement", "rel_teacher_acc", "teacher_view_acc",
              "pattern_mean", "pattern_std", "channel_divergence", "num_connections")


def cmd_bank(cfg: dict, results_root: Path, fake_data: bool) -> None:
    import torch

    from method_paper.src.data import NUM_CLASSES, get_probe_dataset
    from method_paper.src.fitness_study import bank_dir, build_bank
    from method_paper.src.models.registry import build_model

    fs, dataset = cfg["fitness_study"], cfg["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_ckpt = run_dir_for(results_root, dataset, "teacher", fs["teacher"], "ce", cfg["teacher_seed"]) / "checkpoint.pt"
    teacher = build_model(fs["teacher"], NUM_CLASSES[dataset])
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    vcfg = {**cfg["views"], "neat_config": _resolve(cfg["views"]["neat_config"])}
    build_bank(teacher, dataset, get_probe_dataset(dataset, cfg.get("data_root", "./data"), fake_data), vcfg,
               fs["num_genomes"], fs["generations"], fs["rescore_probe_size"], fs["bank_seed"], device,
               bank_dir(results_root, dataset))


def collect_runs(students_dir: Path) -> pd.DataFrame:
    rows = []
    for path in students_dir.glob("*/summary.json"):
        s = json.loads(path.read_text())
        tag = s.get("tag") or ""
        if tag.startswith("fs") and not s.get("val_size"):
            rows.append({"tag": tag, "seed": s["seed"], "top1": s["final_test_top1"]})
    return pd.DataFrame(rows, columns=["tag", "seed", "top1"])


def spearman_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """Spearman rho, its p-value, and a percentile bootstrap 95% CI over genomes."""
    from scipy.stats import spearmanr

    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) > 0 and np.ptp(y[idx]) > 0:
            boots.append(spearmanr(x[idx], y[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(rho), float(p), float(lo), float(hi)


def analyze(bank: pd.DataFrame, runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    kd = runs[runs.tag == "fs_kd"].top1
    if kd.empty:
        raise ValueError("no short-schedule kd reference runs (tag fs_kd) found")
    ref = {"kd_mean": kd.mean(), "kd_std": kd.std(ddof=1) if len(kd) > 1 else 0.0, "kd_n": len(kd)}

    per = runs[runs.tag != "fs_kd"].assign(index=lambda d: d.tag.str[2:].astype(int))
    per = per.groupby("index").top1.agg(["mean", "std", "count"]).rename(
        columns={"mean": "top1", "std": "top1_std", "count": "n_seeds"})
    per = bank.set_index("index").join(per, how="inner").reset_index()
    per["delta_vs_kd"] = per.top1 - ref["kd_mean"]

    corr_rows = []
    for subset, frame in (("all", per), ("evolved", per[per.source == "evolved"]),
                          ("random", per[per.source == "random"])):
        if len(frame) < 4:
            continue
        for name in PREDICTORS:
            x, y = frame[name].to_numpy(float), frame.delta_vs_kd.to_numpy(float)
            if np.ptp(x) == 0:
                continue
            rho, p, lo, hi = spearman_with_ci(x, y)
            corr_rows.append({"subset": subset, "predictor": name, "n": len(frame), "spearman_rho": rho, "p": p,
                              "ci_low": lo, "ci_high": hi})
    return per, pd.DataFrame(corr_rows), ref


def plot(per: pd.DataFrame, ref: dict, path: Path) -> None:
    panels = (("fitness", "teacher-only fitness (rescored)"), ("rel_teacher_acc", "teacher accuracy on view / raw"),
              ("pattern_mean", "pattern mean (brightness kept)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True)
    for ax, (col, label) in zip(axes, panels):
        ax.axhspan(-ref["kd_std"], ref["kd_std"], color="0.9", zorder=0, label="kd ± 1 std")
        ax.axhline(0, color="0.5", lw=0.8, zorder=1)
        for source, marker in (("evolved", "o"), ("random", "^")):
            d = per[per.source == source]
            ax.errorbar(d[col], d.delta_vs_kd, yerr=d.top1_std.fillna(0), fmt=marker, ms=5, capsize=2, lw=0.8,
                        label=source, zorder=2)
        ax.set_xlabel(label)
    axes[0].set_ylabel("student top-1 change vs kd (pt)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"short-schedule students, one view each (kd = {ref['kd_mean']:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_analyze(cfg: dict, results_root: Path) -> None:
    from method_paper.src.fitness_study import bank_dir

    out = bank_dir(results_root, cfg["dataset"]).parent
    bank = pd.read_csv(bank_dir(results_root, cfg["dataset"]) / "bank.csv")
    per, corr, ref = analyze(bank, collect_runs(results_root / cfg["dataset"] / "students"))
    per.to_csv(out / "per_genome.csv", index=False)
    corr.to_csv(out / "correlations.csv", index=False)
    plot(per, ref, out / "fitness_study.png")

    print(f"short-schedule kd: {ref['kd_mean']:.2f} ± {ref['kd_std']:.2f} (n={ref['kd_n']}); "
          f"{len(per)} of {len(bank)} bank genomes have runs")
    print(f"views better than kd: {(per.delta_vs_kd > 0).sum()}, worse: {(per.delta_vs_kd < 0).sum()}; "
          f"mean change {per.delta_vs_kd.mean():+.2f}")
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(corr.sort_values(["subset", "p"]).to_string(index=False))
    print(f"\nwrote {out / 'per_genome.csv'}, {out / 'correlations.csv'}, {out / 'fitness_study.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["bank", "analyze"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--fake-data", action="store_true", help="bank: random probe images (wiring check)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.command == "bank":
        cmd_bank(cfg, Path(args.results_root), args.fake_data)
    else:
        cmd_analyze(cfg, Path(args.results_root))


if __name__ == "__main__":
    main()

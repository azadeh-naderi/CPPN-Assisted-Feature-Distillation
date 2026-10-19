"""Builds the analysis paper's figures from the tracked results archive
(results_archive/method_paper/, written by method_paper/scripts/archive_results.sh),
so they can be regenerated from the repo alone.

    python paper/make_figures.py                       # -> paper/figures/*.pdf, *.png
    python paper/make_figures.py --archive <dir> --out <dir>

Figures whose runs aren't archived yet are skipped with a note.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV = ("resnet56", "resnet20")  # development pair

# Colors: dataviz reference palette, light mode. Reference rows (no views)
# are neutral gray; the three method groups take categorical slots 1-3 in
# order (checked: worst CVD dE 9.2, worst normal-vision dE 24.0). Every
# mark is also labeled, so color never carries identity alone.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
REFERENCE = "#898781"
GROUP_COLORS = {"reference": REFERENCE, "offline, unscored": "#2a78d6",
                "offline, fitness-selected": "#eb6834", "online": "#1baf7a"}

MODES = [  # (mode, label, group) in display order
    ("ce", "student only", "reference"),
    ("kd", "KD", "reference"),
    ("kd_randaugment", "KD + RandAugment", "reference"),
    ("kd_cutmix", "KD + CutMix", "reference"),
    ("kd_random_cppn", "one random CPPN view", "offline, unscored"),
    ("kd_trained_cppn", "one trained CPPN view", "offline, unscored"),
    ("kd_random_search", "random search, top-5 views", "offline, fitness-selected"),
    ("kd_evolved_cppn", "NEAT, top-5 views", "offline, fitness-selected"),
    ("kd_online_teacher", "online, teacher-only fitness", "online"),
    ("kd_online_gradalign", "online, gradient alignment", "online"),
    ("kd_online_disagree", "online, disagreement", "online"),
]
ABLATIONS = [  # (tag, label) for kd_online_disagree
    (None, "method (one view/batch, N=20)"),
    ("every10", "re-evolve every 10 epochs"),
    ("every40", "re-evolve every 40 epochs"),
    ("nojitter", "no shift/flip"),
    ("rsearch", "random search instead of NEAT"),
    ("average", "all 5 views every batch"),
    ("randviews", "unscored random views"),
    ("noise_filtered", "noise masks, fitness-selected"),
    ("noise", "noise masks, unscored"),
]


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": "white",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    })


def load_runs(archive: Path) -> pd.DataFrame:
    rows = []
    for path in (archive / "cifar100" / "students").glob("*/summary.json"):
        s = json.loads(path.read_text())
        extra_path = path.parent / "extra_metrics.json"
        extra = json.loads(extra_path.read_text()) if extra_path.exists() else {}
        rows.append({
            "teacher": s.get("teacher") or "none", "student": s["arch"], "mode": s["mode"], "seed": s["seed"],
            "tag": s.get("tag"), "val_size": s.get("val_size") or 0, "top1": s["final_test_top1"],
            "ece": extra.get("test", {}).get("ece"), "cifar_c": extra.get("cifar_c", {}).get("mean_top1"),
        })
    return pd.DataFrame(rows)


def benchmark(runs: pd.DataFrame, teacher: str, student: str, mode: str, tag=None) -> pd.Series:
    """Final top-1 per seed for one untuned full-data configuration."""
    sel = (runs.student == student) & (runs["mode"] == mode) & (runs.val_size == 0)
    sel &= (runs.teacher == ("none" if mode == "ce" else teacher))
    sel &= runs.tag.isna() if tag is None else (runs.tag == tag)
    return runs[sel].sort_values("seed").top1


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def dot_rows(ax, rows, ref_value=None, ref_label=None):
    """Horizontal dot plot: per row, seed dots (hollow) and mean +- std (filled)."""
    for y, (label, values, color) in enumerate(rows):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(values, y + np.asarray(jitter), s=14, facecolors="none", edgecolors=color, linewidths=0.8,
                   zorder=3)
        mean, std = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
        ax.errorbar(mean, y, xerr=std, fmt="o", ms=5, color=color, mec="white", mew=0.8, elinewidth=1.4,
                    capsize=0, zorder=4)
        ax.text(1.005, y, f"{mean:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5,
                color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if ref_value is not None:
        ax.axvline(ref_value, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(ref_value, -0.9, ref_label, ha="center", va="bottom", fontsize=6.5, color=INK2)


def fig_main_results(runs, out):
    rows = [(label, benchmark(runs, *DEV[:1], DEV[1], mode).to_numpy(), GROUP_COLORS[group])
            for mode, label, group in MODES]
    kd = benchmark(runs, DEV[0], DEV[1], "kd").mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    dot_rows(ax, rows, kd, f"KD {kd:.2f}")
    ax.set_xlabel("final test top-1 (%), CIFAR-100, resnet56 → resnet20, 3 seeds")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", frameon=False)
    save(fig, out, "fig1_main_results")


def fig_ablations(runs, out):
    rows = []
    for tag, label in ABLATIONS:
        values = benchmark(runs, DEV[0], DEV[1], "kd_online_disagree", tag).to_numpy()
        rows.append((label, values, GROUP_COLORS["online"]))  # all ablations are online variants
    if sum(len(v) > 0 for _l, v, _c in rows) < 2:
        print("skip fig2_ablations: ablation runs not archived")
        return
    kd = benchmark(runs, DEV[0], DEV[1], "kd").mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    dot_rows(ax, rows, kd, f"KD {kd:.2f}")
    ax.set_xlabel("final test top-1 (%), resnet56 → resnet20, 3 seeds")
    save(fig, out, "fig2_ablations")


def fig_fitness_study(archive, out):
    path = archive / "cifar100" / "fitness_study" / "per_genome.csv"
    if not path.exists():
        print("skip fig3_fitness_study: per_genome.csv not archived")
        return
    p = pd.read_csv(path)
    gate_open = p.agreement > 0.5  # the fitness's own gate threshold (tau_low)
    markers = {"evolved": ("o", "#2a78d6"), "random": ("^", "#eb6834")}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axes[0]
    for source, (marker, color) in markers.items():
        for is_open, face in ((True, color), (False, "none")):
            d = p[(p.source == source) & (gate_open == is_open)]
            ax.scatter(d.fitness, d.delta_vs_kd, marker=marker, s=20, facecolors=face, edgecolors=color,
                       linewidths=0.8, label=f"{source}, gate {'open' if is_open else 'closed'}", zorder=3)
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel("teacher-only fitness")
    ax.set_ylabel("student top-1 change vs KD (pt)")
    ax.set_title("(a) all 40 views", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.06))

    for ax, col, xlabel, title in ((axes[1], "fitness", "teacher-only fitness", "(b) gate open: fitness"),
                                   (axes[2], "diversity", "diversity (fitness objective)", "(c) gate open: diversity")):
        d_open = p[gate_open]
        for source, (marker, color) in markers.items():
            d = d_open[d_open.source == source]
            ax.scatter(d[col], d.delta_vs_kd, marker=marker, s=20, color=color, edgecolors="white",
                       linewidths=0.5, zorder=3)
        ev = d_open[d_open.source == "evolved"]
        rho, pval = spearmanr(ev[col], ev.delta_vs_kd)
        ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title}\nevolved ρ = {rho:+.2f}, p = {pval:.3f}, n = {len(ev)}", loc="left")
    fig.tight_layout()
    save(fig, out, "fig3_fitness_study")


def fig_view_gallery(archive, out):
    bank = archive / "cifar100" / "fitness_study"
    if not (bank / "per_genome.csv").exists() or not (bank / "bank" / "patterns.pt").exists():
        print("skip fig4_view_gallery: bank not archived")
        return
    p = pd.read_csv(bank / "per_genome.csv")
    patterns = torch.load(bank / "bank" / "patterns.pt")
    groups = [("highest fitness", p.nlargest(5, "fitness")), ("most harmful", p.nsmallest(5, "delta_vs_kd")),
              ("most helpful", p.nlargest(5, "delta_vs_kd"))]
    fig, axes = plt.subplots(len(groups), 5, figsize=(5.2, 3.6))
    for r, (title, sel) in enumerate(groups):
        for c, (_i, row) in enumerate(sel.iterrows()):
            ax = axes[r, c]
            ax.imshow(patterns[int(row["index"])].numpy(), vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{row.source[:3]} f={row.fitness:.2f}\nΔ={row.delta_vs_kd:+.1f}", fontsize=6, color=INK2)
        axes[r, 0].set_ylabel(title, fontsize=7, color=INK)
    fig.tight_layout()
    save(fig, out, "fig4_view_gallery")


def fig_robustness(runs, out):
    pts = []
    for mode, label, group in MODES:
        sel = (runs.student == DEV[1]) & (runs["mode"] == mode) & (runs.val_size == 0) & runs.tag.isna()
        sel &= runs.teacher == ("none" if mode == "ce" else DEV[0])
        d = runs[sel].dropna(subset=["cifar_c"])
        if len(d):
            pts.append((label, group, d.top1.mean(), d.top1.std(ddof=1), d.cifar_c.mean(), d.cifar_c.std(ddof=1)))
    if len(pts) < 3:
        print("skip fig5_robustness: extra metrics not archived")
        return
    # Selective direct labels, each nudged clear of its neighbors.
    offsets = {"student only": (4, -9), "KD": (-12, 5), "KD + RandAugment": (5, 3),
               "NEAT, top-5 views": (5, 3), "online, disagreement": (5, 3), "random search, top-5 views": (-20, 6)}
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for label, group, x, xs, y, ys in pts:
        ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="o", ms=5, color=GROUP_COLORS[group], mec="white", mew=0.6,
                    elinewidth=0.8, capsize=0, zorder=3)
        if label in offsets:
            ax.annotate(label, (x, y), xytext=offsets[label], textcoords="offset points", fontsize=6, color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6)
    ax.set_xlabel("clean test top-1 (%)")
    ax.set_ylabel("CIFAR-100-C top-1 (%)")
    ax.set_title("mean over 15 corruptions × 5 severities; ± 1 std, 3 seeds", loc="left", fontsize=7, color=INK2)
    save(fig, out, "fig5_robustness")


def fig_generality(runs, out):
    pairs = [("resnet56", "resnet20"), ("wrn_40_2", "wrn_16_2"), ("vgg13", "mobilenetv2")]
    modes = [("kd_random_cppn", "one random CPPN view", "offline, unscored"),
             ("kd_evolved_cppn", "NEAT, top-5 views", "offline, fitness-selected"),
             ("kd_online_teacher", "online, teacher-only fitness", "online")]
    deltas = defaultdict(dict)
    for teacher, student in pairs:
        kd = benchmark(runs, teacher, student, "kd")
        for mode, _label, _group in modes:
            v = benchmark(runs, teacher, student, mode)
            if len(v) and len(kd):
                deltas[(teacher, student)][mode] = (v - kd.mean()).to_numpy()
    if sum(len(d) == len(modes) for d in deltas.values()) < 2:
        print("skip fig6_generality: generality runs not archived yet")
        return
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    width = 0.25
    for k, (mode, label, group) in enumerate(modes):
        for j, pair in enumerate(pairs):
            v = deltas.get(pair, {}).get(mode)
            if v is None:
                continue
            x = j + (k - 1) * width
            ax.scatter(np.full(len(v), x), v, s=12, facecolors="none", edgecolors=GROUP_COLORS[group], lw=0.8)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="o", ms=5, color=GROUP_COLORS[group], mec="white",
                        mew=0.6, elinewidth=1.2, label=label if j == 0 else None)
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(range(len(pairs)), [f"{t} → {s}" for t, s in pairs])
    ax.set_ylabel("top-1 change vs KD (pt)")
    ax.legend(frameon=False, loc="lower left")
    save(fig, out, "fig6_generality")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=str(REPO_ROOT / "results_archive" / "method_paper"))
    parser.add_argument("--out", default=str(REPO_ROOT / "paper" / "figures"))
    args = parser.parse_args()
    archive, out = Path(args.archive), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    runs = load_runs(archive)
    print(f"{len(runs)} student runs in {archive}")
    fig_main_results(runs, out)
    fig_ablations(runs, out)
    fig_fitness_study(archive, out)
    fig_view_gallery(archive, out)
    fig_robustness(runs, out)
    fig_generality(runs, out)


if __name__ == "__main__":
    main()

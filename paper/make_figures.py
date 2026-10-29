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


def _robustness_points(runs, teacher, student):
    pts = []
    for mode, label, group in MODES:
        sel = (runs.student == student) & (runs["mode"] == mode) & (runs.val_size == 0) & runs.tag.isna()
        sel &= runs.teacher == ("none" if mode == "ce" else teacher)
        d = runs[sel].dropna(subset=["cifar_c"])
        if len(d):
            pts.append((label, group, d.top1.mean(), d.top1.std(ddof=1), d.cifar_c.mean(), d.cifar_c.std(ddof=1)))
    return pts


def fig_robustness(runs, out):
    """Clean vs CIFAR-100-C accuracy, one panel per pair that has metrics for
    at least one view method (the development pair has every mode)."""
    pairs = [("resnet56", "resnet20"), ("wrn_40_2", "wrn_16_2"), ("vgg13", "mobilenetv2")]
    panels = [(pair, pts) for pair in pairs if len(pts := _robustness_points(runs, *pair)) >= 3]
    if not panels:
        print("skip fig5_robustness: extra metrics not archived")
        return
    # Selective direct labels, each nudged clear of its neighbors. The
    # development pair has every mode, so it labels a subset; the other pairs
    # have only the generality modes and label all of them.
    dev_offsets = {"student only": (4, -9), "KD": (-12, 5), "KD + RandAugment": (5, 3),
                   "NEAT, top-5 views": (5, 3), "online, disagreement": (5, 3),
                   "random search, top-5 views": (-20, 6)}
    pair_offsets = {
        DEV: dev_offsets,
        ("wrn_40_2", "wrn_16_2"): {"student only": (4, -9), "KD": (-4, 7), "NEAT, top-5 views": (5, 3),
                                   "one random CPPN view": (5, -9), "online, teacher-only fitness": (5, 3)},
        ("vgg13", "mobilenetv2"): {"student only": (4, -9), "KD": (5, 3), "NEAT, top-5 views": (5, 3),
                                   "one random CPPN view": (-75, 6), "online, teacher-only fitness": (5, -9)},
    }
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.0), squeeze=False)
    for ax, ((teacher, student), pts) in zip(axes[0], panels):
        offsets = pair_offsets.get((teacher, student), {})
        for label, group, x, xs, y, ys in pts:
            ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="o", ms=5, color=GROUP_COLORS[group], mec="white", mew=0.6,
                        elinewidth=0.8, capsize=0, zorder=3)
            if label in offsets:
                ax.annotate(label, (x, y), xytext=offsets[label], textcoords="offset points", fontsize=6,
                            color=INK2)
        ax.set_xlabel("clean test top-1 (%)")
        ax.set_title(f"{teacher} → {student}", loc="left")
    axes[0][0].set_ylabel("CIFAR-100-C top-1 (%)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.suptitle("CIFAR-100-C: mean over 15 corruptions × 5 severities; ± 1 std, 3 seeds", x=0.01, ha="left",
                 fontsize=7, color=INK2)
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    save(fig, out, "fig5_robustness")
    if len(panels) < len(pairs):
        print(f"  fig5: {len(pairs) - len(panels)} pair(s) without view-method metrics yet")


def fig_refresh_instability(archive, out):
    """Test accuracy around late view refreshes: unscored random views vs the
    fitness-filtered method, 3 seeds each, plus the views behind the final drop."""
    base = archive / "cifar100" / "students"
    configs = [("randviews", "unscored random views", GROUP_COLORS["offline, unscored"]),
               (None, "fitness-filtered (method)", GROUP_COLORS["online"])]
    curves = {}
    for tag, label, color in configs:
        for seed in (0, 1, 2):
            name = f"{DEV[0]}__{DEV[1]}__kd_online_disagree__seed{seed}" + (f"__{tag}" if tag else "")
            path = base / name / "training_log.csv"
            if path.exists():
                curves[(label, seed)] = (pd.read_csv(path), color)
    if len(curves) < 6:
        print("skip fig7_refresh_instability: training logs not archived")
        return
    final_round = base / f"{DEV[0]}__{DEV[1]}__kd_online_disagree__seed2__randviews" / "views" / "round11_epoch220"
    patterns = [torch.load(f) for f in sorted(final_round.glob("pattern_*.pt"))]

    fig = plt.figure(figsize=(5.4, 3.6))
    grid = fig.add_gridspec(2, 5, height_ratios=[3.2, 1.0], hspace=0.55)
    ax = fig.add_subplot(grid[0, :])
    for (label, seed), (log, color) in curves.items():
        d = log[log.epoch >= 140]
        ax.plot(d.epoch, d.test_top1, color=color, lw=1.0 if label.startswith("unscored") else 1.4,
                ls="-" if label.startswith("unscored") else (0, (4, 1.5)), label=label if seed == 0 else None)
    for epoch in range(140, 240, 20):
        ax.axvline(epoch, color=GRID, lw=0.8, zorder=0)
    ax.text(200, ax.get_ylim()[1], " view refresh", fontsize=6, color=MUTED, va="top")
    ax.set_xlabel("epoch (LR decays at 150, 180, 210)")
    ax.set_ylabel("test top-1 (%)")
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    ax.set_title("resnet56 → resnet20, 3 seeds each; vertical lines = view refreshes", loc="left")
    for i, p in enumerate(patterns[:5]):
        pax = fig.add_subplot(grid[1, i])
        pax.imshow(p.numpy(), vmin=0, vmax=1)
        pax.set_xticks([]), pax.set_yticks([])
        pax.set_title(f"mean {p.mean():.2f}", fontsize=6, color=INK2)
        for spine in pax.spines.values():
            spine.set_visible(False)
    fig.text(0.5, 0.0, "views drawn at the epoch-220 refresh, unscored seed 2 (in use until the end)",
             ha="center", va="top", fontsize=6.5, color=INK)
    save(fig, out, "fig7_refresh_instability")


PUBLISHED_TEACHER = {"resnet56": 72.34, "wrn_40_2": 75.61, "resnet32x4": 79.42, "vgg13": 74.64, "resnet110": 74.31}
PUBLISHED_STUDENT = {"resnet20": 69.06, "wrn_16_2": 73.26, "resnet8x4": 72.50, "mobilenetv2": 64.60,
                     "resnet32": 71.14, "vgg8": 70.36}
PUBLISHED_KD = {("resnet56", "resnet20"): 70.66, ("wrn_40_2", "wrn_16_2"): 74.92, ("resnet32x4", "resnet8x4"): 73.33,
                ("vgg13", "mobilenetv2"): 67.37, ("resnet110", "resnet32"): 73.08, ("vgg13", "vgg8"): 72.98}
BENCHMARK_PAIRS = list(PUBLISHED_KD)


def table_benchmark(archive, runs, out):
    """Table 1: our teachers / student_only / KD vs Tian et al. (2020).
    Final-epoch test top-1, mean ± std over 3 student seeds (teachers: 1 run)."""
    teachers = {}
    for path in (archive / "cifar100" / "teachers").glob("*/summary.json"):
        s = json.loads(path.read_text())
        teachers[s["arch"]] = s["final_test_top1"]
    rows = []
    for teacher, student in BENCHMARK_PAIRS:
        so = benchmark(runs, teacher, student, "ce")
        kd = benchmark(runs, teacher, student, "kd")
        if teacher not in teachers or so.empty or kd.empty:
            continue
        rows.append((teacher, student, teachers[teacher], PUBLISHED_TEACHER[teacher], so.mean(), so.std(ddof=1),
                     PUBLISHED_STUDENT[student], kd.mean(), kd.std(ddof=1), PUBLISHED_KD[(teacher, student)]))
    tables = out.parent / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    md = ["| teacher → student | teacher (ours / pub.) | student only (ours / pub.) | KD (ours / pub.) |",
          "|---|---|---|---|"]
    tex = [r"\begin{tabular}{llccc}", r"\toprule",
           r"Teacher & Student & Teacher & Student only & KD \\", r"\midrule"]
    for t, s, tt, tp, som, sos, sop, kdm, kds, kdp in rows:
        md.append(f"| {t} → {s} | {tt:.2f} / {tp:.2f} | {som:.2f} ± {sos:.2f} / {sop:.2f} | "
                  f"{kdm:.2f} ± {kds:.2f} / {kdp:.2f} |")
        esc = lambda x: x.replace("_", r"\_")  # noqa: E731
        tex.append(f"{esc(t)} & {esc(s)} & {tt:.2f} ({tp:.2f}) & {som:.2f}$\\pm${sos:.2f} ({sop:.2f}) & "
                   f"{kdm:.2f}$\\pm${kds:.2f} ({kdp:.2f}) \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (tables / "table1_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (tables / "table1_benchmark.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"wrote tables/table1_benchmark.md/.tex ({len(rows)} pairs)")


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
    fig_refresh_instability(archive, out)
    table_benchmark(archive, runs, out)


if __name__ == "__main__":
    main()

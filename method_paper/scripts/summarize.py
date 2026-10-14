"""Collects every finished run's summary.json and prints teacher and student
tables next to the published benchmark numbers (method_paper/EXPERIMENT_LOG.md).

    python method_paper/scripts/summarize.py                  # cifar100
    python method_paper/scripts/summarize.py --csv out.csv    # also save the student table

Wall time per run is the time between config_snapshot.json (written at the
start of training, after the teacher is loaded) and summary.json (written at
the end); for CPPN-view modes it includes view search.

Standard library only (no torch import), so it runs instantly and works
without activating the training environment.
"""

from __future__ import annotations  # also runs under the system python3.9

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_RESULTS_ROOT = Path("results/method_paper")  # same as scripts/train.py

# Tian et al., ICLR 2020 (CRD), CIFAR-100 top-1.
PUBLISHED_TEACHER = {"resnet56": 72.34, "wrn_40_2": 75.61, "resnet32x4": 79.42, "vgg13": 74.64, "resnet110": 74.31}
PUBLISHED_STUDENT = {"resnet20": 69.06, "wrn_16_2": 73.26, "resnet8x4": 72.50, "mobilenetv2": 64.60,
                     "resnet32": 71.14, "vgg8": 70.36}
PUBLISHED_KD = {("resnet56", "resnet20"): 70.66, ("wrn_40_2", "wrn_16_2"): 74.92,
                ("resnet32x4", "resnet8x4"): 73.33, ("vgg13", "mobilenetv2"): 67.37,
                ("resnet110", "resnet20"): 70.67, ("resnet110", "resnet32"): 73.08, ("vgg13", "vgg8"): 72.98}


def load_runs(stage_dir: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(stage_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        config_path = summary_path.parent / "config_snapshot.json"
        if config_path.exists():
            seconds = summary_path.stat().st_mtime - config_path.stat().st_mtime
            summary["hours"] = seconds / 3600
            summary["min_per_epoch"] = seconds / 60 / max(summary.get("epochs") or 1, 1)
        runs.append(summary)
    return runs


def _fmt(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def _delta(value: float | None, published: float | None) -> str:
    return f"{value - published:+.2f}" if value is not None and published is not None else "-"


def student_rows(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        mode = "student_only" if r["mode"] == "ce" else r["mode"]
        groups[(r.get("teacher") or "none", r["arch"], mode)].append(r)

    rows = []
    for (teacher, arch, mode), group in sorted(groups.items()):
        finals = [r["final_test_top1"] for r in group]
        bests = [r["best_test_top1"] for r in group]
        if mode == "student_only":
            published = PUBLISHED_STUDENT.get(arch)
        elif mode == "kd":
            published = PUBLISHED_KD.get((teacher, arch))
        else:
            published = None
        rows.append({
            "teacher": teacher, "student": arch, "mode": mode, "n": len(group),
            "final": _fmt(finals), "best": _fmt(bests),
            "published": f"{published:.2f}" if published is not None else "-",
            "best_minus_published": _delta(statistics.mean(bests), published),
            "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"])),
            "min_per_epoch": f"{statistics.mean(r.get('min_per_epoch', 0) for r in group):.2f}",
        })
    return rows


def print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("  (no finished runs)")
        return
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print("  " + "  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  " + "  ".join(str(r[c]).ljust(widths[c]) for c in columns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--csv", default=None, help="also write the student table here")
    args = parser.parse_args()

    root = Path(args.results_root) / args.dataset

    print("Teachers (final = last epoch, best = max over epochs)")
    teacher_rows = [
        {"teacher": t["arch"], "final": f"{t['final_test_top1']:.2f}", "best": f"{t['best_test_top1']:.2f}",
         "published": f"{PUBLISHED_TEACHER[t['arch']]:.2f}" if t["arch"] in PUBLISHED_TEACHER else "-",
         "best_minus_published": _delta(t["best_test_top1"], PUBLISHED_TEACHER.get(t["arch"])),
         "hours": f"{t.get('hours', 0):.1f}", "min_per_epoch": f"{t.get('min_per_epoch', 0):.2f}"}
        for t in load_runs(root / "teachers")
    ]
    print_table(teacher_rows, ["teacher", "final", "best", "published", "best_minus_published", "hours", "min_per_epoch"])

    print("\nStudents (mean ± std over seeds)")
    rows = student_rows(load_runs(root / "students"))
    print_table(rows, ["teacher", "student", "mode", "n", "final", "best", "published", "best_minus_published",
                       "min_per_epoch"])

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["teacher"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

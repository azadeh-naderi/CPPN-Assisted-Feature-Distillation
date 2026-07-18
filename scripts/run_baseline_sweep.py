import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subprocess

from src.distill.modes import MODES
from src.utils.logging import get_logger

log = get_logger("run_baseline_sweep")
REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> str:
    """Runs a CLI script, returns its last stdout line (each script's
    run_dir or "mode\\taccuracy\\trun_dir" line)."""
    log.info("+ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.stdout:
        log.info(result.stdout[-2000:])
    if result.returncode != 0:
        log.error(result.stderr[-4000:])
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result.stdout.strip().splitlines()[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--modes", default="all", help="comma-separated modes, or 'all'")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    modes = MODES if args.modes == "all" else args.modes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    python = sys.executable
    smoke_flag = ["--smoke"] if args.smoke else []

    sweep_id = f"sweep_{int(time.time())}"
    sweep_dir = REPO_ROOT / "results" / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in seeds:
        teacher_run_dir = run(
            [python, "scripts/train_teacher.py", "--config", args.config, "--seed", str(seed), *smoke_flag]
        )
        teacher_ckpt = str(REPO_ROOT / teacher_run_dir / "checkpoint.pt")

        genome_path = None
        if "kd_evolved_cppn" in modes:
            cppn_run_dir = run(
                [
                    python,
                    "scripts/evolve_cppn.py",
                    "--config",
                    args.config,
                    "--teacher-ckpt",
                    teacher_ckpt,
                    "--seed",
                    str(seed),
                    *smoke_flag,
                ]
            )
            genome_path = str(REPO_ROOT / cppn_run_dir / "best_genome.pkl")

        for mode in modes:
            cmd = [
                python,
                "scripts/train_student.py",
                "--config",
                args.config,
                "--mode",
                mode,
                "--seed",
                str(seed),
                *smoke_flag,
            ]
            if mode != "student_only":
                cmd += ["--teacher-ckpt", teacher_ckpt]
            if mode == "kd_evolved_cppn":
                cmd += ["--genome-path", genome_path]

            last_line = run(cmd)
            mode_out, test_acc, run_dir = last_line.split("\t")
            rows.append(
                {
                    "seed": seed,
                    "mode": mode_out,
                    "test_accuracy": float(test_acc),
                    "run_dir": run_dir,
                    "teacher_ckpt": teacher_ckpt,
                }
            )

    summary_path = sweep_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "mode", "test_accuracy", "run_dir", "teacher_ckpt"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Sweep summary written to %s", summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()

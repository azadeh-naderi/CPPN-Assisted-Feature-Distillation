import argparse
import csv
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from online_distillation.src.online_trainer import ONLINE_MODES
from src.utils.logging import get_logger

log = get_logger("run_online_sweep")


def run(cmd: list[str]) -> str:
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

    modes = ONLINE_MODES if args.modes == "all" else args.modes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    python = sys.executable
    smoke_flag = ["--smoke"] if args.smoke else []

    sweep_id = f"online_sweep_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    sweep_dir = REPO_ROOT / "results" / "online_distillation" / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in seeds:
        for mode in modes:
            cmd = [
                python,
                "online_distillation/scripts/train_student_online.py",
                "--config",
                args.config,
                "--mode",
                mode,
                "--seed",
                str(seed),
                *smoke_flag,
            ]
            last_line = run(cmd)
            mode_out, test_acc, run_dir = last_line.split("\t")
            rows.append({"seed": seed, "mode": mode_out, "test_accuracy": float(test_acc), "run_dir": run_dir})

    summary_path = sweep_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "mode", "test_accuracy", "run_dir"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Sweep summary written to %s", summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()

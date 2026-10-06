"""Maps a SLURM array index to one training job, so a single sbatch script
can run a whole stage.

    # see the job list (and its size, for --array)
    python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage teachers --list
    # run job 3 of the student stage
    python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage students --index 3

Stages:
- teachers: one CE run per unique teacher architecture (seed teacher_seed)
- students: student_only for each unique student architecture x seed, then
  kd for each pair x seed
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from method_paper.scripts.train import DEFAULT_RESULTS_ROOT, run_training
from src.utils.config import load_config

STAGES = ("teachers", "students")


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def build_jobs(cfg: dict, stage: str) -> list[dict]:
    pairs = cfg["pairs"]
    if stage == "teachers":
        return [
            {"role": "teacher", "arch": t, "mode": "ce", "seed": cfg["teacher_seed"], "teacher": None}
            for t in _unique([p["teacher"] for p in pairs])
        ]
    if stage == "students":
        jobs = [
            {"role": "student", "arch": s, "mode": "ce", "seed": seed, "teacher": None}
            for s in _unique([p["student"] for p in pairs])
            for seed in cfg["student_seeds"]
        ]
        jobs += [
            {"role": "student", "arch": p["student"], "mode": "kd", "seed": seed, "teacher": p["teacher"]}
            for p in pairs
            for seed in cfg["student_seeds"]
        ]
        return jobs
    raise ValueError(f"Unknown stage {stage!r}. Choose one of {STAGES}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--list", action="store_true")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    jobs = build_jobs(cfg, args.stage)

    if args.list:
        for i, job in enumerate(jobs):
            teacher = f" <- {job['teacher']}" if job["teacher"] else ""
            print(f"{i:3d}  {job['role']:7s} {job['arch']:12s} {job['mode']:2s} seed{job['seed']}{teacher}")
        print(f"{len(jobs)} jobs -> sbatch --array=0-{len(jobs) - 1}")
        return

    if not 0 <= args.index < len(jobs):
        raise IndexError(f"--index {args.index} out of range for {len(jobs)} {args.stage} jobs")
    job = jobs[args.index]
    run_training(
        cfg, job["role"], job["arch"], job["seed"], job["mode"], job["teacher"],
        Path(args.results_root), args.fake_data, args.max_batches, args.epochs, args.overwrite,
    )


if __name__ == "__main__":
    main()

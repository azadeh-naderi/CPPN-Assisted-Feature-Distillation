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
- phase1: every Phase 1 baseline mode x phase1 pair x seed (needs the
  Phase 0 teachers)
- phase2: every Phase 2 online-view mode x phase2 pair x seed
- tune: the `tune` section's modes x pairs x seeds -- the kd reference plus
  the method, for hyperparameter tuning on a validation split
  (slurm/tune.sbatch)
- phase3_main: every benchmark pair x (phase3.main_baselines + phase3.method)
  x seed
- phase3_ablations: phase3.method with each phase3.ablations override set,
  on phase3.ablation_pair, x seed (each ablation is a tagged run)
- fitness_study: short-schedule kd reference (tag fs_kd) x kd_seeds, then
  one short kd_bank_view run per bank genome (tag fsNN) x seeds

Jobs may carry their own config overrides and run tag (Phase 3).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from method_paper.scripts.train import DEFAULT_RESULTS_ROOT, add_override_args, prepare_config, run_training
from src.utils.config import apply_overrides

STAGES = ("teachers", "students", "phase1", "phase2", "tune", "phase3_main", "phase3_ablations", "fitness_study")


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
    if stage in ("phase1", "phase2", "tune"):
        return [
            {"role": "student", "arch": p["student"], "mode": mode, "seed": seed, "teacher": p["teacher"]}
            for p in cfg[stage]["pairs"]
            for mode in cfg[stage]["modes"]
            for seed in cfg["student_seeds"]
        ]
    if stage in ("phase3_main", "phase3_ablations"):
        p3 = cfg["phase3"]
        method_set = list(p3.get("method_set") or [])
        method_tag = p3["method_tag"] if method_set else None
        if stage == "phase3_main":
            runs = [(mode, [], None) for mode in p3["main_baselines"]] + [(p3["method"], method_set, method_tag)]
            pairs = cfg["pairs"]
        else:
            runs = [(p3["method"], method_set + list(a["set"]), a["tag"]) for a in p3["ablations"]]
            pairs = [p3["ablation_pair"]]
        return [
            {"role": "student", "arch": p["student"], "mode": mode, "seed": seed, "teacher": p["teacher"],
             "overrides": overrides, "tag": tag}
            for p in pairs
            for mode, overrides, tag in runs
            for seed in cfg["student_seeds"]
        ]
    if stage == "fitness_study":
        fs = cfg["fitness_study"]
        short = [f"training.epochs={fs['epochs']}", f"training.lr_milestones={list(fs['lr_milestones'])}"]
        common = {"role": "student", "arch": fs["student"], "teacher": fs["teacher"]}
        jobs = [{**common, "mode": "kd", "seed": seed, "overrides": short, "tag": "fs_kd"} for seed in fs["kd_seeds"]]
        jobs += [
            {**common, "mode": "kd_bank_view", "seed": seed, "overrides": [*short, f"fitness_study.index={i}"],
             "tag": f"fs{i:02d}"}
            for i in range(fs["num_genomes"])
            for seed in fs["seeds"]
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
    add_override_args(parser)
    args = parser.parse_args()

    cfg = prepare_config(args.config, args.overrides, args.tag)
    jobs = build_jobs(cfg, args.stage)

    if args.list:
        for i, job in enumerate(jobs):
            teacher = f" <- {job['teacher']}" if job["teacher"] else ""
            tag = f" [{job['tag']}]" if job.get("tag") else ""
            print(f"{i:3d}  {job['role']:7s} {job['arch']:12s} {job['mode']:20s} seed{job['seed']}{teacher}{tag}")
        print(f"{len(jobs)} jobs -> sbatch --array=0-{len(jobs) - 1}")
        return

    if not 0 <= args.index < len(jobs):
        raise IndexError(f"--index {args.index} out of range for {len(jobs)} {args.stage} jobs")
    job = jobs[args.index]
    # Phase 3 jobs carry their own overrides and tag, applied on top of any --set / --tag.
    job_cfg = apply_overrides(cfg, job.get("overrides") or [])
    tag = "_".join(t for t in (args.tag, job.get("tag")) if t) or None
    run_training(
        job_cfg, job["role"], job["arch"], job["seed"], job["mode"], job["teacher"],
        Path(args.results_root), args.fake_data, args.max_batches, args.epochs, args.overwrite, tag=tag,
    )


if __name__ == "__main__":
    main()

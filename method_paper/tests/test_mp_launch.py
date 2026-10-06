import json
from pathlib import Path

import pytest

from method_paper.scripts.launch import build_jobs
from method_paper.scripts.train import run_dir_for, run_training
from src.utils.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cifar100.yaml"


def _cfg():
    return load_config(CONFIG_PATH)


def test_teacher_stage_has_one_job_per_unique_teacher():
    jobs = build_jobs(_cfg(), "teachers")
    assert [j["arch"] for j in jobs] == ["resnet56", "wrn_40_2", "resnet32x4", "vgg13"]
    assert all(j["mode"] == "ce" and j["role"] == "teacher" for j in jobs)


def test_student_stage_covers_student_only_and_kd_for_every_seed():
    cfg = _cfg()
    jobs = build_jobs(cfg, "students")
    n_seeds = len(cfg["student_seeds"])
    n_pairs = len(cfg["pairs"])
    assert len(jobs) == 2 * n_pairs * n_seeds == 24  # matches phase0_students.sbatch --array=0-23
    kd = [j for j in jobs if j["mode"] == "kd"]
    assert {(j["teacher"], j["arch"]) for j in kd} == {(p["teacher"], p["student"]) for p in cfg["pairs"]}


def test_run_dirs_are_deterministic_and_distinct():
    root = Path("r")
    teacher = run_dir_for(root, "cifar100", "teacher", "resnet56", "ce", 0)
    only = run_dir_for(root, "cifar100", "student", "resnet20", "ce", 0)
    kd = run_dir_for(root, "cifar100", "student", "resnet20", "kd", 0, "resnet56")
    assert teacher == root / "cifar100" / "teachers" / "resnet56_seed0"
    assert only.name == "none__resnet20__student_only__seed0"
    assert kd.name == "resnet56__resnet20__kd__seed0"


def test_end_to_end_teacher_then_kd_student_on_fake_data(tmp_path):
    cfg = {**_cfg(), "num_workers": 0}
    common = dict(results_root=tmp_path, fake_data=True, max_batches=2, epochs=2)

    teacher_dir = run_training(cfg, "teacher", "resnet20", cfg["teacher_seed"], **common)
    assert (teacher_dir / "checkpoint.pt").exists()

    student_dir = run_training(cfg, "student", "resnet8", 0, mode="kd", teacher="resnet20", **common)
    summary = json.loads((student_dir / "summary.json").read_text())
    assert summary["mode"] == "kd" and summary["teacher"] == "resnet20"
    assert summary["teacher_test_top1"] is not None
    assert (student_dir / "training_log.csv").read_text().count("\n") == 3  # header + 2 epochs


def test_finished_run_is_skipped_unless_overwrite(tmp_path):
    cfg = {**_cfg(), "num_workers": 0}
    common = dict(results_root=tmp_path, fake_data=True, max_batches=1, epochs=1)
    run_dir = run_training(cfg, "student", "resnet8", 0, **common)
    stamp = (run_dir / "summary.json").stat().st_mtime_ns
    run_training(cfg, "student", "resnet8", 0, **common)
    assert (run_dir / "summary.json").stat().st_mtime_ns == stamp


def test_kd_without_trained_teacher_fails_clearly(tmp_path):
    cfg = {**_cfg(), "num_workers": 0}
    with pytest.raises(FileNotFoundError):
        run_training(cfg, "student", "resnet8", 0, mode="kd", teacher="resnet56",
                     results_root=tmp_path, fake_data=True, max_batches=1, epochs=1)

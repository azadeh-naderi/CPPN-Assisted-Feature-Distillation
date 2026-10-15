import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from method_paper.scripts.summarize import student_rows
from method_paper.scripts.train import prepare_config, run_dir_for, run_training
from method_paper.src.data import get_loaders, get_probe_dataset
from method_paper.src.trainer import fit
from src.utils.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cifar100.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 10)

    def forward(self, x, return_features=False):
        logits = self.fc(x.flatten(1))
        return (logits, x.flatten(1)) if return_features else logits


class Recorder:
    def __init__(self):
        self.rows = []

    def log_epoch(self, epoch, **metrics):
        self.rows.append(metrics)


def _data():
    torch.manual_seed(0)
    return [(torch.rand(8, 3, 8, 8), torch.randint(0, 10, (8,)))]


def _fit(val_loader):
    rec = Recorder()
    summary = fit(TinyNet(), _data(), _data(), "cifar10", torch.device("cpu"), epochs=2, lr=0.1, lr_milestones=[5],
                  lr_decay=0.1, momentum=0.9, weight_decay=0.0, mode="ce", run_logger=rec, val_loader=val_loader)
    return summary, rec.rows


def test_fit_reports_validation_when_given():
    summary, rows = _fit(_data())
    assert all("val_top1" in r and "val_top5" in r for r in rows)
    assert summary["final_val_top1"] == rows[-1]["val_top1"]
    assert summary["best_val_top1"] == max(r["val_top1"] for r in rows)


def test_fit_without_validation_keeps_the_phase0_log_format():
    summary, rows = _fit(None)
    assert all("val_top1" not in r for r in rows)
    assert "final_val_top1" not in summary


def test_probe_pool_excludes_validation_images():
    _train, val_loader, _test = get_loaders("cifar100", "./data", 16, 0, val_size=50, seed=3, fake_data=True)
    probe = get_probe_dataset("cifar100", "./data", fake_data=True, val_size=50, seed=3)
    val_idx = set(val_loader.dataset.indices)
    assert len(val_idx) == 50
    assert val_idx.isdisjoint(probe.indices) and len(probe.indices) == 256 - 50
    # val_size=0 keeps the whole training set, as before.
    assert len(get_probe_dataset("cifar100", "./data", fake_data=True)) == 256


def test_run_dirs_get_suffixes_only_when_needed():
    root = Path("r")
    base = run_dir_for(root, "cifar100", "student", "resnet20", "kd", 0, "resnet56")
    assert base.name == "resnet56__resnet20__kd__seed0"  # unchanged for Phase 0/1 runs
    tuned = run_dir_for(root, "cifar100", "student", "resnet20", "kd", 0, "resnet56", val_size=5000, tag="g05")
    assert tuned.name == "resnet56__resnet20__kd__seed0__val5000__g05"


def test_overrides_other_than_val_size_need_a_tag():
    with pytest.raises(ValueError):
        prepare_config(str(CONFIG_PATH), ["online.gate.tau_low=0.5"], None)
    assert prepare_config(str(CONFIG_PATH), ["val_size=5000"], None)["val_size"] == 5000
    cfg = prepare_config(str(CONFIG_PATH), ["val_size=5000", "online.gate.tau_low=0.5"], "g05")
    assert cfg["online"]["gate"]["tau_low"] == 0.5


def test_validation_run_end_to_end_uses_the_full_data_teacher(tmp_path):
    cfg = {**load_config(CONFIG_PATH), "num_workers": 0}
    common = dict(results_root=tmp_path, fake_data=True, max_batches=1, epochs=1)
    teacher_dir = run_training(cfg, "teacher", "resnet8", cfg["teacher_seed"], **common)

    cfg_val = {**cfg, "val_size": 32}
    cfg_val["views"] = {**cfg["views"], "neat_config": str(REPO_ROOT / "configs" / "neat" / "cppn_neat_smoke.cfg"),
                        "probe_batch_size": 8}
    cfg_val["online"] = {**cfg["online"], "initial_generations": 1, "top_k": 2}
    run_dir = run_training(cfg_val, "student", "resnet8", 0, mode="kd_online_disagree", teacher="resnet8",
                           tag="smoke", **common)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert run_dir.name.endswith("__val32__smoke")
    assert summary["val_size"] == 32 and summary["tag"] == "smoke"
    assert summary["final_val_top1"] is not None
    assert teacher_dir.name == "resnet8_seed0"  # the student loaded this, not a __val teacher


def test_tune_stage_is_kd_reference_then_method():
    from method_paper.scripts.launch import build_jobs

    cfg = load_config(CONFIG_PATH)
    modes = [j["mode"] for j in build_jobs(cfg, "tune")]
    assert modes == ["kd"] * 3 + ["kd_online_disagree"] * 3  # matches tune.sbatch's 0-2 / 3-5 docs


def test_summary_groups_tuning_runs_separately():
    def run(seed, **extra):
        return {"mode": "kd_online_disagree", "teacher": "resnet56", "arch": "resnet20", "seed": seed,
                "final_test_top1": 71.0, "best_test_top1": 71.2, **extra}

    rows = student_rows([run(0), run(1), run(0, val_size=5000, tag="g05", final_val_top1=70.0)])
    assert [(r["mode"], r["holdout"], r["n"]) for r in rows] == [
        ("kd_online_disagree", "-", 2), ("kd_online_disagree[g05]", 5000, 1)]
    assert rows[1]["val_final"] == "70.00" and rows[0]["val_final"] == "-"

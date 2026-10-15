import json
from pathlib import Path

import numpy as np
import pytest
import torch

from method_paper.scripts.launch import build_jobs
from method_paper.scripts.summarize import extra_rows
from method_paper.scripts.train import RUN_MODES, run_training
from method_paper.src.data import get_probe_dataset
from method_paper.src.metrics import (
    CIFAR_C_CORRUPTIONS,
    agreement_metrics,
    cifar_c_metrics,
    expected_calibration_error,
)
from method_paper.src.models.registry import build_model
from method_paper.src.online_views import OnlineViewEvolver
from src.utils.config import apply_overrides, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cifar100.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _smoke_cfg():
    cfg = load_config(CONFIG_PATH)
    cfg["views"]["neat_config"] = str(REPO_ROOT / "configs" / "neat" / "cppn_neat_smoke.cfg")
    cfg["views"]["probe_batch_size"] = 8
    cfg["views"]["random_search"] = {"max_mutations": 3}
    cfg["online"] = {**cfg["online"], "evolve_every": 1, "initial_generations": 1, "generations_per_round": 2,
                     "top_k": 2}
    return {**cfg, "num_workers": 0}


# --- metrics ----------------------------------------------------------------

def test_ece_is_zero_when_confidence_matches_accuracy_and_large_when_overconfident():
    labels = torch.zeros(100, dtype=torch.long)
    # Confidence exactly 0.5 on class 0 (two classes tied) with 50% accuracy -> ECE 0.
    calibrated = torch.zeros(100, 2)
    calibrated_labels = torch.tensor([0, 1] * 50)
    assert expected_calibration_error(calibrated, calibrated_labels) == pytest.approx(0.0, abs=1e-6)
    # Near-certain predictions of the wrong class -> ECE ~100%.
    overconfident = torch.tensor([[-20.0, 20.0]]).repeat(100, 1)
    assert expected_calibration_error(overconfident, labels) == pytest.approx(100.0, abs=1e-3)


def test_agreement_metrics():
    t = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    out = agreement_metrics(t.clone(), t)
    assert out["teacher_agreement"] == 100.0 and out["teacher_kl"] == pytest.approx(0.0, abs=1e-6)


def _fake_cifar_c(root: Path, n_per_severity=4):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 5 * n_per_severity
    for c in CIFAR_C_CORRUPTIONS:
        np.save(root / f"{c}.npy", rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8))
    np.save(root / "labels.npy", rng.integers(0, 100, n))


def test_cifar_c_metrics_cover_every_corruption_and_severity(tmp_path, monkeypatch):
    import method_paper.src.metrics as metrics

    monkeypatch.setattr(metrics, "IMAGES_PER_SEVERITY", 4)
    _fake_cifar_c(tmp_path)
    out = cifar_c_metrics(build_model("resnet8", 100), tmp_path, "cifar100", torch.device("cpu"))
    assert set(out["per_corruption"]) == set(CIFAR_C_CORRUPTIONS)
    assert all(set(v) == {"1", "2", "3", "4", "5"} for v in out["per_corruption"].values())
    assert 0.0 <= out["mean_top1"] <= 100.0 and "mean_ece" in out


# --- online controls --------------------------------------------------------

@pytest.mark.parametrize("override,expected_rows", [
    (["online.search=random"], 2 * 6),   # generations_per_round (2) x pop_size (6), scored
    (["online.search=none"], 0),         # unscored
    (["online.family=noise"], 0),        # no genomes at all
])
def test_online_controls_produce_patterns(tmp_path, override, expected_rows):
    cfg = apply_overrides(_smoke_cfg(), override)
    evolver = OnlineViewEvolver(
        build_model("resnet8", 100).eval(), "cifar100", get_probe_dataset("cifar100", "./data", fake_data=True),
        cfg["views"]["neat_config"], "disagreement", cfg["online"], cfg["views"]["evolution"], "multiplicative",
        0.5, 8, 4.0, 0, torch.device("cpu"), tmp_path, max_mutations=3,
    )
    patterns = evolver.evolve(build_model("resnet8", 100), 2, epoch=0)
    assert 1 <= len(patterns) <= 2
    assert all(p.shape == (32, 32, 3) and 0.0 <= p.min() and p.max() <= 1.0 for p in patterns)
    assert len(evolver.rows) == expected_rows
    if override == ["online.family=noise"]:
        assert patterns[0].std() > 0.01  # actual spatial variation, not a constant


def test_bad_online_control_rejected(tmp_path):
    cfg = apply_overrides(_smoke_cfg(), ["online.search=grid"])
    with pytest.raises(ValueError):
        OnlineViewEvolver(build_model("resnet8", 100), "cifar100", None, cfg["views"]["neat_config"],
                          "disagreement", cfg["online"], cfg["views"]["evolution"], "multiplicative", 0.5, 8,
                          4.0, 0, torch.device("cpu"), tmp_path)


# --- stages -----------------------------------------------------------------

def test_phase3_main_covers_every_pair_with_baselines_and_method():
    cfg = load_config(CONFIG_PATH)
    jobs = build_jobs(cfg, "phase3_main")
    modes = cfg["phase3"]["main_baselines"] + [cfg["phase3"]["method"]]
    assert len(jobs) == len(cfg["pairs"]) * len(modes) * len(cfg["student_seeds"]) == 36  # phase3_main.sbatch
    assert {(j["teacher"], j["arch"]) for j in jobs} == {(p["teacher"], p["student"]) for p in cfg["pairs"]}
    assert all(j["tag"] is None and j["overrides"] == [] for j in jobs)  # untuned method: same dirs as Phase 2


def test_tuned_method_gets_its_own_tag_in_main_and_ablations():
    cfg = load_config(CONFIG_PATH)
    cfg["phase3"]["method_set"] = ["online.gate.tau_low=0.5"]
    method = cfg["phase3"]["method"]
    main = [j for j in build_jobs(cfg, "phase3_main") if j["mode"] == method]
    assert all(j["tag"] == cfg["phase3"]["method_tag"] and j["overrides"] == cfg["phase3"]["method_set"]
               for j in main)
    ablations = build_jobs(cfg, "phase3_ablations")
    assert all(j["overrides"][0] == "online.gate.tau_low=0.5" for j in ablations)


def test_phase3_ablations_are_tagged_method_runs():
    cfg = load_config(CONFIG_PATH)
    jobs = build_jobs(cfg, "phase3_ablations")
    assert len(jobs) == len(cfg["phase3"]["ablations"]) * len(cfg["student_seeds"]) == 21  # phase3_ablations.sbatch
    assert {j["mode"] for j in jobs} == {cfg["phase3"]["method"]}
    assert len({j["tag"] for j in jobs}) == len(cfg["phase3"]["ablations"])
    assert {j["mode"] for j in jobs} <= set(RUN_MODES)


# --- end to end -------------------------------------------------------------

def test_evaluate_extra_end_to_end(tmp_path, monkeypatch):
    import method_paper.scripts.evaluate_extra as ev
    import method_paper.src.metrics as metrics

    cfg = _smoke_cfg()
    common = dict(results_root=tmp_path, fake_data=True, max_batches=1, epochs=1)
    run_training(cfg, "teacher", "resnet8", cfg["teacher_seed"], **common)
    run_dir = run_training(cfg, "student", "resnet8", 0, mode="kd", teacher="resnet8", **common)

    monkeypatch.setattr(metrics, "IMAGES_PER_SEVERITY", 4)
    _fake_cifar_c(tmp_path / "cifar_c")
    monkeypatch.setattr("sys.argv", ["evaluate_extra.py", "--config", str(CONFIG_PATH), "--results-root",
                                     str(tmp_path), "--cifar-c-dir", str(tmp_path / "cifar_c"), "--fake-data"])
    monkeypatch.setattr(ev, "load_config", lambda _p: cfg)
    ev.main()

    extra = json.loads((run_dir / "extra_metrics.json").read_text())
    assert {"top1", "nll", "ece", "teacher_agreement", "teacher_kl"} <= set(extra["test"])
    assert "mean_top1" in extra["cifar_c"]

    summary = json.loads((run_dir / "summary.json").read_text())
    rows = extra_rows([{**summary, "extra": extra}])
    assert rows[0]["mode"] == "kd" and rows[0]["cifar_c_top1"] != "-"

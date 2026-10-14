import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from method_paper.scripts.launch import build_jobs
from method_paper.scripts.train import RUN_MODES, run_training
from method_paper.src.data import get_probe_dataset
from method_paper.src.models.registry import build_model
from method_paper.src.online_views import (
    FITNESSES,
    OnlineViewEvolver,
    ce_dlogits,
    kd_dlogits,
    last_layer_grad,
    per_sample_kd,
)
from method_paper.src.trainer import fit, jitter_pattern
from src.distill.losses import kd_loss
from src.utils.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cifar100.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _smoke_cfg():
    """Full config with view search and online evolution shrunk to CPU size."""
    cfg = load_config(CONFIG_PATH)
    cfg["views"]["neat_config"] = str(REPO_ROOT / "configs" / "neat" / "cppn_neat_smoke.cfg")
    cfg["views"]["probe_batch_size"] = 8
    cfg["online"] = {**cfg["online"], "evolve_every": 1, "initial_generations": 1, "generations_per_round": 1,
                     "top_k": 2}
    return {**cfg, "num_workers": 0}


# --- fitness math -----------------------------------------------------------

def test_per_sample_kd_averages_to_kd_loss():
    torch.manual_seed(0)
    s, t = torch.randn(6, 10), torch.randn(6, 10)
    assert per_sample_kd(s, t, 4.0).mean().item() == pytest.approx(kd_loss(s, t, 4.0).item(), rel=1e-5)


@pytest.mark.parametrize("loss_name", ["kd", "ce"])
def test_closed_form_last_layer_grad_matches_autograd(loss_name):
    torch.manual_seed(0)
    features = torch.randn(6, 8)
    fc = nn.Linear(8, 10)
    logits = fc(features)
    teacher_logits = torch.randn(6, 10)
    labels = torch.randint(0, 10, (6,))

    if loss_name == "kd":
        loss = kd_loss(logits, teacher_logits, 4.0)
        dlogits = kd_dlogits(logits.detach(), teacher_logits, 4.0)
    else:
        loss = F.cross_entropy(logits, labels)
        dlogits = ce_dlogits(logits.detach(), labels)
    loss.backward()

    expected = torch.cat([fc.weight.grad.flatten(), fc.bias.grad])
    assert torch.allclose(last_layer_grad(dlogits, features), expected, atol=1e-6)


# --- pattern jitter and the before_epoch hook --------------------------------

def test_jitter_is_a_noop_without_shift_or_flip_and_consumes_no_rng():
    # Phase 1's kd_view runs use shift 0 / no flip; they must be unaffected.
    pattern = torch.rand(32, 32, 3)
    state = torch.get_rng_state()
    assert torch.equal(jitter_pattern(pattern), pattern)
    assert torch.equal(torch.get_rng_state(), state)


def test_jitter_only_rearranges_pixels():
    torch.manual_seed(0)
    pattern = torch.rand(32, 32, 3)
    for _ in range(10):
        out = jitter_pattern(pattern, max_shift=4, flip=True)
        assert out.shape == pattern.shape
        assert torch.equal(out.flatten().sort().values, pattern.flatten().sort().values)


def test_before_epoch_can_swap_patterns_each_epoch():
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, 10)

        def forward(self, x, return_features=False):
            logits = self.fc(x.flatten(1))
            return (logits, x.flatten(1)) if return_features else logits

    data = [(torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
    calls = []

    def before_epoch(epoch, student):
        calls.append(epoch)
        return {"patterns": [torch.full((32, 32, 3), 0.1 * (epoch + 1))]}

    fit(TinyNet(), data, data, "cifar10", torch.device("cpu"), epochs=3, lr=0.1, lr_milestones=[10],
        lr_decay=0.1, momentum=0.9, weight_decay=0.0, mode="kd_view", teacher=TinyNet(),
        before_epoch=before_epoch, view_sampling="one")
    assert calls == [0, 1, 2]


# --- online evolver ---------------------------------------------------------

@pytest.mark.parametrize("fitness", FITNESSES)
def test_online_evolver_warm_starts_across_rounds(tmp_path, fitness):
    cfg = _smoke_cfg()
    teacher, student = build_model("resnet8", 100).eval(), build_model("resnet8", 100)
    evolver = OnlineViewEvolver(
        teacher, "cifar100", get_probe_dataset("cifar100", "./data", fake_data=True),
        cfg["views"]["neat_config"], fitness, cfg["online"], cfg["views"]["evolution"], "multiplicative", 0.5,
        8, 4.0, 0, torch.device("cpu"), tmp_path,
    )
    first = evolver.evolve(student, 2, epoch=0)
    assert evolver.population.generation == 2
    second = evolver.evolve(student, 1, epoch=20)
    assert evolver.population.generation == 3  # continued, not restarted
    assert student.training  # restored after scoring in eval mode

    for patterns in (first, second):
        assert 1 <= len(patterns) <= 2
        assert all(p.shape == (32, 32, 3) and 0.0 <= p.min() and p.max() <= 1.0 for p in patterns)
    log = pd.read_csv(tmp_path / "online_evolution_log.csv")
    assert set(log["round"]) == {0, 1} and set(log["epoch"]) == {0, 20}
    assert (tmp_path / "round00_epoch000" / "pattern_0.png").exists()
    assert (tmp_path / "round01_epoch020" / "pattern_0.png").exists()


def test_unknown_fitness_rejected(tmp_path):
    cfg = _smoke_cfg()
    with pytest.raises(ValueError):
        OnlineViewEvolver(build_model("resnet8", 100), "cifar100", None, cfg["views"]["neat_config"], "accuracy",
                          cfg["online"], cfg["views"]["evolution"], "multiplicative", 0.5, 8, 4.0, 0,
                          torch.device("cpu"), tmp_path)


# --- jobs and end-to-end ----------------------------------------------------

def test_phase2_stage_covers_every_mode_pair_and_seed():
    cfg = load_config(CONFIG_PATH)
    jobs = build_jobs(cfg, "phase2")
    assert len(jobs) == len(cfg["phase2"]["pairs"]) * len(cfg["phase2"]["modes"]) * len(cfg["student_seeds"]) == 9
    assert {j["mode"] for j in jobs} <= set(RUN_MODES)  # matches phase2_online.sbatch --array=0-8


@pytest.mark.parametrize("mode", ["kd_online_disagree", "kd_online_gradalign"])
def test_online_modes_end_to_end_on_fake_data(tmp_path, mode):
    cfg = _smoke_cfg()
    common = dict(results_root=tmp_path, fake_data=True, max_batches=1)
    run_training(cfg, "teacher", "resnet8", cfg["teacher_seed"], epochs=1, **common)
    run_dir = run_training(cfg, "student", "resnet8", 0, mode=mode, teacher="resnet8", epochs=2, **common)
    assert json.loads((run_dir / "summary.json").read_text())["mode"] == mode
    rounds = sorted(p.name for p in (run_dir / "views").glob("round*"))
    assert rounds == ["round00_epoch000", "round01_epoch001"]  # evolve_every=1 in the smoke config

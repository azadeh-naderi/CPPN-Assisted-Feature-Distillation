import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from method_paper.scripts.launch import build_jobs
from method_paper.scripts.train import RUN_MODES, run_training
from method_paper.src.augment import cutmix
from method_paper.src.data import normalize, train_transform
from method_paper.src.trainer import train_one_epoch
from method_paper.src.views import NormalizedModel, build_views
from src.utils.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cifar100.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _smoke_cfg():
    """Full config with view search shrunk to CPU size."""
    cfg = load_config(CONFIG_PATH)
    views = cfg["views"]
    views["neat_config"] = str(REPO_ROOT / "configs" / "neat" / "cppn_neat_smoke.cfg")
    views["probe_batch_size"] = 8
    views["evolution"] = {**views["evolution"], "num_generations": 2, "top_k": 2}
    views["random_search"] = {"max_mutations": 3}
    views["trained"] = {**views["trained"], "num_steps": 2}
    return {**cfg, "num_workers": 0}


class TinyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x, return_features=False):
        logits = self.fc(x.flatten(1))
        return (logits, x.flatten(1)) if return_features else logits


def _loader(n=16, batch=8):
    torch.manual_seed(0)
    images = torch.rand(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    return [(images[i : i + batch], labels[i : i + batch]) for i in range(0, n, batch)]


# --- augmentation -----------------------------------------------------------

def test_cutmix_lam_matches_pasted_area():
    torch.manual_seed(0)
    labels = torch.arange(4)
    images = labels.float().view(4, 1, 1, 1).expand(4, 3, 32, 32).clone()  # image i is constant i
    for _ in range(20):
        mixed, shuffled, lam = cutmix(images, labels, alpha=1.0)
        assert mixed.shape == images.shape and 0.0 <= lam <= 1.0
        assert sorted(shuffled.tolist()) == [0, 1, 2, 3]
        for i in range(4):
            own = (mixed[i, 0] == i).float().mean().item()
            partner = (mixed[i, 0] == shuffled[i].item()).float().mean().item()
            if shuffled[i] == i:
                assert own == 1.0
            else:
                assert own == pytest.approx(lam) and own + partner == pytest.approx(1.0)


def test_randaugment_transform_is_added_only_when_requested():
    names = lambda tf: [type(t).__name__ for t in tf.transforms]  # noqa: E731
    assert "RandAugment" not in names(train_transform())
    assert "RandAugment" in names(train_transform("randaugment"))
    with pytest.raises(ValueError):
        train_transform("autoaugment")


# --- trainer modes ----------------------------------------------------------

@pytest.mark.parametrize("sampling", ["all", "one"])
def test_kd_view_trains_and_leaves_teacher_alone(sampling):
    student, teacher = TinyNet(), TinyNet().eval()
    before = [p.clone() for p in teacher.parameters()]
    opt = torch.optim.SGD(student.parameters(), lr=0.1)
    patterns = [torch.rand(32, 32, 3), torch.rand(32, 32, 3)]
    out = train_one_epoch(student, _loader(), opt, "cifar10", torch.device("cpu"), "kd_view",
                          teacher=teacher, patterns=patterns, view_sampling=sampling)
    assert out["loss"] > 0
    assert all(torch.equal(b, p) for b, p in zip(before, teacher.parameters()))


def test_kd_view_with_weight_zero_equals_plain_kd():
    torch.manual_seed(0)
    student, teacher = TinyNet(), TinyNet().eval()
    common = dict(teacher=teacher, max_batches=1)
    loader = _loader()
    s1 = TinyNet()
    s1.load_state_dict(student.state_dict())
    kd = train_one_epoch(student, loader, torch.optim.SGD(student.parameters(), lr=0.0),
                         "cifar10", torch.device("cpu"), "kd", **common)
    view = train_one_epoch(s1, loader, torch.optim.SGD(s1.parameters(), lr=0.0), "cifar10",
                           torch.device("cpu"), "kd_view", patterns=[torch.rand(32, 32, 3)],
                           view_weight=0.0, **common)
    assert view["loss"] == pytest.approx(kd["loss"])


def test_kd_view_needs_patterns_and_cutmix_needs_teacher():
    student = TinyNet()
    opt = torch.optim.SGD(student.parameters(), lr=0.1)
    with pytest.raises(ValueError):
        train_one_epoch(student, _loader(), opt, "cifar10", torch.device("cpu"), "kd_view", teacher=TinyNet())
    with pytest.raises(ValueError):
        train_one_epoch(student, _loader(), opt, "cifar10", torch.device("cpu"), "kd_cutmix")


def test_kd_cutmix_runs():
    student = TinyNet()
    opt = torch.optim.SGD(student.parameters(), lr=0.1)
    out = train_one_epoch(student, _loader(), opt, "cifar10", torch.device("cpu"), "kd_cutmix",
                          teacher=TinyNet().eval(), cutmix_prob=1.0)
    assert out["loss"] > 0


# --- view search ------------------------------------------------------------

def test_normalized_model_normalizes_before_forward():
    class Spy(nn.Module):
        def forward(self, x, return_features=False):
            self.seen = x
            return x.flatten(1)[:, :10]

    spy = Spy()
    images = torch.rand(2, 3, 32, 32)
    NormalizedModel(spy, "cifar100")(images)
    assert torch.allclose(spy.seen, normalize(images, "cifar100"))


@pytest.mark.parametrize("source,expected", [("random", 1), ("trained", 1), ("evolved", 2), ("random_search", 2)])
def test_build_views_returns_valid_patterns(tmp_path, source, expected):
    from method_paper.src.data import get_probe_dataset
    from method_paper.src.models.registry import build_model

    cfg = _smoke_cfg()
    teacher = build_model("resnet8", 100).eval()
    probe = get_probe_dataset("cifar100", "./data", fake_data=True)
    patterns = build_views(source, teacher, "cifar100", probe, cfg["views"], 0, torch.device("cpu"), tmp_path)
    assert len(patterns) == expected
    for p in patterns:
        assert p.shape == (32, 32, 3) and 0.0 <= p.min() and p.max() <= 1.0
    assert len(list(tmp_path.glob("pattern_*.png"))) == expected


def test_random_search_uses_the_evolution_budget(tmp_path):
    import pandas as pd

    from method_paper.src.data import get_probe_dataset
    from method_paper.src.models.registry import build_model
    from src.cppn.evolve import load_neat_config

    cfg = _smoke_cfg()
    build_views("random_search", build_model("resnet8", 100).eval(), "cifar100",
                get_probe_dataset("cifar100", "./data", fake_data=True), cfg["views"], 0,
                torch.device("cpu"), tmp_path)
    log = pd.read_csv(tmp_path / "random_search_log.csv")
    pop_size = load_neat_config(cfg["views"]["neat_config"]).pop_size
    assert len(log) == pop_size * cfg["views"]["evolution"]["num_generations"]
    assert log["num_mutations"].between(0, cfg["views"]["random_search"]["max_mutations"]).all()


# --- jobs and end-to-end ----------------------------------------------------

def test_phase1_stage_covers_every_mode_pair_and_seed():
    cfg = load_config(CONFIG_PATH)
    jobs = build_jobs(cfg, "phase1")
    n = len(cfg["phase1"]["pairs"]) * len(cfg["phase1"]["modes"]) * len(cfg["student_seeds"])
    assert len(jobs) == n == 18  # matches phase1_baselines.sbatch --array=0-17
    assert {j["mode"] for j in jobs} <= set(RUN_MODES)
    assert all(j["teacher"] for j in jobs)


@pytest.mark.parametrize("mode", ["kd_randaugment", "kd_cutmix", "kd_random_cppn", "kd_evolved_cppn"])
def test_phase1_modes_end_to_end_on_fake_data(tmp_path, mode):
    cfg = _smoke_cfg()
    common = dict(results_root=tmp_path, fake_data=True, max_batches=1, epochs=1)
    run_training(cfg, "teacher", "resnet8", cfg["teacher_seed"], **common)
    run_dir = run_training(cfg, "student", "resnet8", 0, mode=mode, teacher="resnet8", **common)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["mode"] == mode and run_dir.name == f"resnet8__resnet8__{mode}__seed0"
    has_views = RUN_MODES[mode][2] is not None
    assert (run_dir / "views").exists() == has_views

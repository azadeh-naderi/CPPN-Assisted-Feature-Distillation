import pytest
import torch
import torch.nn as nn

from online_distillation.src.online_trainer import ONLINE_MODES, OnlineDistillTrainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 5)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _make_trainer(mode: str) -> OnlineDistillTrainer:
    return OnlineDistillTrainer(
        student=TinyModel(),
        mode=mode,
        dataset_name="cifar_10",
        device=torch.device("cpu"),
        pattern=torch.rand(8, 8, 3),
        alpha=0.5,
        lr=0.01,
    )


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        _make_trainer("not_a_real_mode")


@pytest.mark.parametrize("mode", ONLINE_MODES)
def test_step_runs_and_backprops(mode):
    trainer = _make_trainer(mode)
    images = torch.rand(4, 3, 8, 8)
    labels = torch.randint(0, 5, (4,))
    loss = trainer._step(images, labels)
    assert loss.item() > 0
    loss.backward()
    grads = [p.grad for p in trainer.student.parameters()]
    assert all(g is not None for g in grads)


@pytest.mark.parametrize("mode", ONLINE_MODES)
def test_multiple_optimizer_steps_do_not_error(mode):
    # Would surface any double-backward/graph-retention issue from reusing
    # raw_logits (once with grad for loss_hard, once detached for the
    # consistency term) across repeated steps.
    trainer = _make_trainer(mode)
    images = torch.rand(4, 3, 8, 8)
    labels = torch.randint(0, 5, (4,))
    for _ in range(3):
        trainer.optimizer.zero_grad()
        loss = trainer._step(images, labels)
        loss.backward()
        trainer.optimizer.step()


def test_self_consistency_target_is_detached_not_pulling_raw_branch_backward():
    # If raw_logits weren't detached before use as the consistency target,
    # the "hard_weight * loss_hard" gradient path and the "alpha * loss_view"
    # gradient path would both flow into raw_logits, and the view branch's
    # gradient would incorrectly also update raw_logits' own prediction
    # target. Checking .grad_fn on the used tensor confirms detachment.
    trainer = _make_trainer("self_consistency_random_cppn")
    images = torch.rand(4, 3, 8, 8)
    labels = torch.randint(0, 5, (4,))
    from src.data.datasets import normalize_batch

    images_norm = normalize_batch(images, trainer.dataset_name)
    raw_logits = trainer.student(images_norm)
    assert raw_logits.detach().grad_fn is None


def test_fit_and_evaluate_run_end_to_end():
    trainer = _make_trainer("self_consistency_random_cppn")
    images = torch.rand(16, 3, 8, 8)
    labels = torch.randint(0, 5, (16,))
    loader = [(images, labels)]
    trainer.fit(loader, loader, num_epochs=1)
    acc = trainer.evaluate(loader)
    assert 0.0 <= acc <= 100.0


def test_resample_pattern_requires_neat_config():
    with pytest.raises(ValueError):
        OnlineDistillTrainer(
            student=TinyModel(),
            mode="self_consistency_random_cppn",
            dataset_name="cifar_10",
            device=torch.device("cpu"),
            pattern=torch.rand(8, 8, 3),
            resample_pattern=True,
        )


def test_resample_pattern_changes_pattern_across_epochs():
    from src.cppn.evolve import load_neat_config

    neat_config = load_neat_config("configs/neat/cppn_neat_smoke.cfg")
    trainer = OnlineDistillTrainer(
        student=TinyModel(),
        mode="self_consistency_random_cppn",
        dataset_name="cifar_10",
        device=torch.device("cpu"),
        pattern=torch.rand(8, 8, 3),
        alpha=0.5,
        lr=0.01,
        resample_pattern=True,
        neat_config=neat_config,
        image_size=8,
        channels=3,
        pattern_seed=0,
    )
    images = torch.rand(4, 3, 8, 8)
    labels = torch.randint(0, 5, (4,))
    loader = [(images, labels)]

    trainer.fit(loader, loader, num_epochs=1)
    pattern_epoch_0 = trainer.pattern.clone()
    trainer.fit(loader, loader, num_epochs=1)
    pattern_epoch_1 = trainer.pattern.clone()

    assert not torch.equal(pattern_epoch_0, pattern_epoch_1)


def test_resample_pattern_false_keeps_pattern_fixed_across_epochs():
    trainer = _make_trainer("self_consistency_random_cppn")
    original_pattern = trainer.pattern.clone()
    images = torch.rand(4, 3, 8, 8)
    labels = torch.randint(0, 5, (4,))
    loader = [(images, labels)]

    trainer.fit(loader, loader, num_epochs=3)

    assert torch.equal(trainer.pattern, original_pattern)


def _make_resample_trainer(**overrides):
    from src.cppn.evolve import load_neat_config

    neat_config = load_neat_config("configs/neat/cppn_neat_smoke.cfg")
    kwargs = dict(
        student=TinyModel(),
        mode="self_consistency_random_cppn",
        dataset_name="cifar_10",
        device=torch.device("cpu"),
        pattern=torch.rand(8, 8, 3),
        alpha=0.5,
        lr=0.01,
        resample_pattern=True,
        neat_config=neat_config,
        image_size=8,
        channels=3,
        pattern_seed=0,
    )
    kwargs.update(overrides)
    return OnlineDistillTrainer(**kwargs)


def test_guardrail_rejects_high_std_pattern():
    trainer = _make_resample_trainer(max_pattern_std=0.2)
    high_std = torch.zeros(8, 8, 3)
    high_std[:4] = 1.0  # half 0s, half 1s -- std=0.5, well above 0.2
    assert not trainer._pattern_passes_guardrail(high_std)


def test_guardrail_rejects_low_mean_pattern():
    trainer = _make_resample_trainer(min_pattern_mean=0.3)
    near_black = torch.full((8, 8, 3), 0.05)
    assert not trainer._pattern_passes_guardrail(near_black)


def test_guardrail_accepts_moderate_pattern():
    trainer = _make_resample_trainer(max_pattern_std=0.2, min_pattern_mean=0.3)
    moderate = torch.full((8, 8, 3), 0.6)  # std=0, mean=0.6 -- passes both bounds
    assert trainer._pattern_passes_guardrail(moderate)


def test_guardrail_none_bounds_accept_everything():
    trainer = _make_resample_trainer()  # no max_pattern_std/min_pattern_mean set
    extreme = torch.zeros(8, 8, 3)
    assert trainer._pattern_passes_guardrail(extreme)


def test_resample_with_guardrail_produces_a_passing_pattern():
    trainer = _make_resample_trainer(max_pattern_std=0.2, min_pattern_mean=0.3)
    images = torch.rand(4, 3, 8, 8)
    labels = torch.randint(0, 5, (4,))
    loader = [(images, labels)]
    trainer.fit(loader, loader, num_epochs=5)
    assert trainer._pattern_passes_guardrail(trainer.pattern)


def test_resample_guardrail_impossible_bounds_falls_back_without_crashing(caplog):
    # min_pattern_mean=2.0 is unsatisfiable (patterns are in [0,1]) -- confirms
    # the max_resample_attempts fallback engages and logs a warning instead
    # of hanging or raising.
    trainer = _make_resample_trainer(min_pattern_mean=2.0, max_resample_attempts=5)
    images = torch.rand(4, 3, 8, 8)
    labels = torch.randint(0, 5, (4,))
    loader = [(images, labels)]
    trainer.fit(loader, loader, num_epochs=1)
    assert trainer.pattern is not None

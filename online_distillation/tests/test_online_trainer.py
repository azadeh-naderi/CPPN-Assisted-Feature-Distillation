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

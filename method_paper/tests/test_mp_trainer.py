import pytest
import torch
import torch.nn as nn

from method_paper.src.trainer import evaluate, fit, train_one_epoch


class TinyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, x, return_features=False):
        logits = self.fc(x.flatten(1))
        return (logits, x.flatten(1)) if return_features else logits


def _loader(n=32, num_classes=10, batch=8):
    torch.manual_seed(0)
    images = torch.rand(n, 3, 8, 8)
    labels = torch.randint(0, num_classes, (n,))
    return [(images[i : i + batch], labels[i : i + batch]) for i in range(0, n, batch)]


def test_evaluate_returns_percentages():
    result = evaluate(TinyNet(), _loader(), "cifar10", torch.device("cpu"))
    assert 0.0 <= result["top1"] <= result["top5"] <= 100.0


def test_kd_requires_teacher():
    model = TinyNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError):
        train_one_epoch(model, _loader(), opt, "cifar10", torch.device("cpu"), mode="kd")


def test_unknown_mode_rejected():
    model = TinyNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError):
        train_one_epoch(model, _loader(), opt, "cifar10", torch.device("cpu"), mode="mixup")


def test_kd_does_not_update_teacher():
    student, teacher = TinyNet(), TinyNet()
    before = [p.clone() for p in teacher.parameters()]
    fit(student, _loader(), _loader(), "cifar10", torch.device("cpu"), epochs=2, lr=0.1,
        lr_milestones=[1], lr_decay=0.1, momentum=0.9, weight_decay=5e-4, mode="kd", teacher=teacher)
    assert all(torch.equal(b, p) for b, p in zip(before, teacher.parameters()))


def test_fit_applies_lr_milestones_and_logs_each_epoch():
    class Recorder:
        rows = []

        def log_epoch(self, epoch, **metrics):
            self.rows.append({"epoch": epoch, **metrics})

    rec = Recorder()
    rec.rows = []
    summary = fit(TinyNet(), _loader(), _loader(), "cifar10", torch.device("cpu"), epochs=4, lr=0.1,
                  lr_milestones=[2, 3], lr_decay=0.1, momentum=0.9, weight_decay=5e-4, mode="ce", run_logger=rec)
    assert [round(r["lr"], 6) for r in rec.rows] == [0.1, 0.1, 0.01, 0.001]
    assert summary["final_test_top1"] == rec.rows[-1]["test_top1"]
    assert summary["best_test_top1"] == max(r["test_top1"] for r in rec.rows)


def test_weight_decay_is_applied():
    # The main pipeline's optimizer has no weight decay; make sure this one does.
    model = TinyNet()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    zero_lr_data = [(torch.zeros(4, 3, 8, 8), torch.zeros(4, dtype=torch.long))]
    norm_before = sum(p.norm() for p in model.parameters()).item()
    fit(model, zero_lr_data, zero_lr_data, "cifar10", torch.device("cpu"), epochs=3, lr=0.1,
        lr_milestones=[100], lr_decay=0.1, momentum=0.0, weight_decay=0.5, mode="ce")
    assert sum(p.norm() for p in model.parameters()).item() < norm_before

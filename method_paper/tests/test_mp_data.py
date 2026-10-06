import torch
from torchvision import datasets

from method_paper.src import data
from method_paper.src.data import STATS, get_loaders, normalize


class _RecordingCIFAR100:
    """Stand-in for torchvision's CIFAR100 that records which split was asked
    for, so the official-test-split requirement is checked without a download."""

    calls: list[bool] = []

    def __init__(self, root, train, download, transform):
        _RecordingCIFAR100.calls.append(train)
        self._inner = datasets.FakeData(size=100, image_size=(3, 32, 32), num_classes=100, transform=transform)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, i):
        return self._inner[i]


def test_test_loader_uses_official_test_split(monkeypatch):
    _RecordingCIFAR100.calls = []
    monkeypatch.setitem(data._DATASET_CLS, "cifar100", _RecordingCIFAR100)
    get_loaders("cifar100", "./data", batch_size=8, num_workers=0)
    assert False in _RecordingCIFAR100.calls  # train=False requested for the test set
    assert True in _RecordingCIFAR100.calls


def test_val_split_is_held_out_from_train(monkeypatch):
    monkeypatch.setitem(data._DATASET_CLS, "cifar100", _RecordingCIFAR100)
    train_loader, val_loader, _ = get_loaders("cifar100", "./data", batch_size=8, num_workers=0, val_size=20)
    assert len(val_loader.dataset) == 20
    assert len(train_loader.dataset) == 80
    assert set(val_loader.dataset.indices).isdisjoint(train_loader.dataset.indices)


def test_no_val_split_by_default(monkeypatch):
    monkeypatch.setitem(data._DATASET_CLS, "cifar100", _RecordingCIFAR100)
    train_loader, val_loader, _ = get_loaders("cifar100", "./data", batch_size=8, num_workers=0)
    assert val_loader is None
    assert len(train_loader.dataset) == 100


def test_normalize_uses_cifar100_stats():
    mean, std = STATS["cifar100"]
    x = torch.tensor(mean).view(1, 3, 1, 1).expand(2, 3, 4, 4)
    assert torch.allclose(normalize(x, "cifar100"), torch.zeros(2, 3, 4, 4), atol=1e-6)
    # CIFAR-100's stats, not CIFAR-10's (the main pipeline reuses CIFAR-10's).
    assert STATS["cifar100"] != STATS["cifar10"]


def test_fake_data_loaders_yield_raw01_images():
    train_loader, _, test_loader = get_loaders("cifar100", "./data", batch_size=16, num_workers=0, fake_data=True)
    images, labels = next(iter(test_loader))
    assert images.shape == (16, 3, 32, 32)
    assert images.min() >= 0.0 and images.max() <= 1.0
    assert labels.max() < 100

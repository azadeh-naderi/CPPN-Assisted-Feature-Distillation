"""Dataset loading/splitting.

The random-split logic here mirrors the original prototype scripts
(`legacy/test_cppn_lenet.py`, `legacy/test_cppn_resnet.py`) verbatim: same
percentages, same DataLoader construction. The one deliberate change is that
transforms are now split into a raw-[0,1] pipeline (used everywhere the CPPN
pathway touches an image) and a separate `Normalize` step applied only right
before a tensor enters a model — the CIFAR/ResNet legacy script fed
already-normalized (mean-subtracted, possibly negative) pixels into
`-log(pixel + eps)`, silently producing NaNs for part of the batch. Train/val/
test also now get independently-appropriate transforms (augmentation only on
the train split) via a shared index permutation, since the legacy scripts
applied one dataset-wide transform before splitting, leaking random-crop/flip
augmentation into validation and test accuracy.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetMeta:
    torch_dataset_cls: type
    input_channels: int
    num_classes: int
    image_size: int
    normalize_mean: tuple[float, ...] | None
    normalize_std: tuple[float, ...] | None


DATASET_META: dict[str, DatasetMeta] = {
    "mnist": DatasetMeta(datasets.MNIST, 1, 10, 28, None, None),
    "fashionmnist": DatasetMeta(datasets.FashionMNIST, 1, 10, 28, None, None),
    "cifar_10": DatasetMeta(
        datasets.CIFAR10, 3, 10, 32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    ),
    "cifar_100": DatasetMeta(
        datasets.CIFAR100, 3, 100, 32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    ),
}


def get_dataset_meta(dataset_name: str) -> DatasetMeta:
    key = dataset_name.lower()
    if key not in DATASET_META:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose one of {list(DATASET_META)}.")
    return DATASET_META[key]


def get_transform(dataset_name: str, train: bool) -> transforms.Compose:
    """Raw-[0,1] transform (no Normalize) — augmented for CIFAR train split only."""
    meta = get_dataset_meta(dataset_name)
    if meta.normalize_mean is None:
        # MNIST/FashionMNIST: no standard augmentation in the legacy scripts.
        return transforms.Compose([transforms.ToTensor()])
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(meta.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose([transforms.ToTensor()])


def normalize_batch(images_raw01: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """Applies Normalize to a raw-[0,1] batch. Call this immediately before a
    tensor enters a model, after any CPPN view has already been applied."""
    meta = get_dataset_meta(dataset_name)
    if meta.normalize_mean is None:
        return images_raw01
    mean = torch.tensor(meta.normalize_mean, device=images_raw01.device).view(1, -1, 1, 1)
    std = torch.tensor(meta.normalize_std, device=images_raw01.device).view(1, -1, 1, 1)
    return (images_raw01 - mean) / std


def load_dataset(dataset_name: str, data_root: str = "./data") -> tuple[Dataset, Dataset]:
    """Returns (train_transform_dataset, eval_transform_dataset) — two views of
    the same underlying train split, differing only in whether augmentation is
    applied. Use the same indices from `split_dataset` on both."""
    meta = get_dataset_meta(dataset_name)
    train_ds = meta.torch_dataset_cls(
        root=data_root, train=True, download=True, transform=get_transform(dataset_name, train=True)
    )
    eval_ds = meta.torch_dataset_cls(
        root=data_root, train=True, download=True, transform=get_transform(dataset_name, train=False)
    )
    return train_ds, eval_ds


def split_dataset(
    train_transform_ds: Dataset,
    eval_transform_ds: Dataset,
    batch_size: int,
    train_size_percent: float,
    val_size_percent: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    n = len(train_transform_ds)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()

    train_size = int(train_size_percent * n)
    val_size = int(val_size_percent * n)
    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size :]

    train_loader = DataLoader(
        Subset(train_transform_ds, train_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(Subset(eval_transform_ds, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(eval_transform_ds, test_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_probe_batch(
    dataset: Dataset, probe_size: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """A single fresh random batch of `probe_size` raw-[0,1] images/labels,
    used as the fixed-per-generation, resampled-across-generations probe set
    for CPPN fitness evaluation. Caller controls resampling via `seed`."""
    generator = torch.Generator().manual_seed(seed)
    n = len(dataset)
    idx = torch.randperm(n, generator=generator)[:probe_size].tolist()
    loader = DataLoader(Subset(dataset, idx), batch_size=probe_size, shuffle=False)
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device)

"""CIFAR data loading for the benchmark protocol.

Differences from src/data/datasets.py (the main pipeline), all deliberate:
- Evaluation uses the official 10k test split (train=False). The main
  pipeline only ever loads train=True and reports accuracy on a 10% slice of
  the training set, which isn't comparable to published numbers.
- Training uses the full 50k training set by default. `val_size > 0` holds
  out a seeded validation subset (evaluated without augmentation) for any
  hyperparameter tuning, so tuning never touches the test split.
- CIFAR-100 uses its own normalization statistics; the main pipeline reuses
  CIFAR-10's.

Loaders yield raw [0, 1] images; call `normalize` on-device right before a
tensor enters a model. That keeps the raw images available for CPPN views
(src/cppn/apply.py expects [0, 1] inputs) in later phases.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
_DATASET_CLS = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


AUGMENTS = (None, "randaugment")


def train_transform(augment: str | None = None, randaugment_ops: int = 2, randaugment_magnitude: int = 9):
    """Standard crop + flip. `augment="randaugment"` adds RandAugment
    (Cubuk et al., 2020) after them, for the KD + strong-augmentation
    baseline -- applied per image in the loader, so teacher and student see
    the same augmented image."""
    if augment not in AUGMENTS:
        raise ValueError(f"Unknown augment {augment!r}. Choose one of {AUGMENTS}.")
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if augment == "randaugment":
        steps.append(transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_magnitude))
    return transforms.Compose([*steps, transforms.ToTensor()])


def test_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def normalize(images_raw01: torch.Tensor, dataset: str) -> torch.Tensor:
    mean, std = STATS[dataset]
    mean_t = torch.tensor(mean, device=images_raw01.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=images_raw01.device).view(1, -1, 1, 1)
    return (images_raw01 - mean_t) / std_t


def _build_datasets(
    dataset: str, data_root: str, fake_data: bool, train_tf: transforms.Compose | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    """Returns (train_augmented, train_unaugmented, test)."""
    train_tf = train_tf or train_transform()
    if fake_data:
        # Random images/labels, for CPU wiring checks only -- no download.
        num_classes = NUM_CLASSES[dataset]
        make = lambda size, tf, offset: datasets.FakeData(  # noqa: E731
            size=size, image_size=(3, 32, 32), num_classes=num_classes, transform=tf, random_offset=offset
        )
        return make(256, train_tf, 0), make(256, test_transform(), 0), make(128, test_transform(), 1000)
    cls = _DATASET_CLS[dataset]
    train_aug = cls(root=data_root, train=True, download=True, transform=train_tf)
    train_plain = cls(root=data_root, train=True, download=True, transform=test_transform())
    test = cls(root=data_root, train=False, download=True, transform=test_transform())
    return train_aug, train_plain, test


def get_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    val_size: int = 0,
    seed: int = 0,
    fake_data: bool = False,
    train_tf: transforms.Compose | None = None,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """Returns (train_loader, val_loader_or_None, test_loader). `train_tf`
    overrides the default training transform (see `train_transform`)."""
    if dataset not in _DATASET_CLS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose one of {list(_DATASET_CLS)}.")
    train_aug, train_plain, test = _build_datasets(dataset, data_root, fake_data, train_tf)

    val_ds = None
    train_ds: Dataset = train_aug
    if val_size > 0:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train_aug), generator=generator).tolist()
        val_ds = Subset(train_plain, perm[:val_size])
        train_ds = Subset(train_aug, perm[val_size:])

    pin = torch.cuda.is_available()
    loader = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin
    )
    return loader(train_ds, True), (loader(val_ds, False) if val_ds is not None else None), loader(test, False)


def get_probe_dataset(dataset: str, data_root: str, fake_data: bool = False) -> Dataset:
    """Unaugmented training set (raw [0, 1] images), the pool that view
    search draws probe batches from. Never the test split."""
    return _build_datasets(dataset, data_root, fake_data)[1]

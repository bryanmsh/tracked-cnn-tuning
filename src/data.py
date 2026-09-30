"""
Data loading and preprocessing pipeline for CIFAR-10.
Supports deterministic train/val splits and augmentation.
"""

from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_transforms(augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and test transforms.
    Train transform applies RandomCrop and RandomHorizontalFlip if augment=True.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, eval_transform


class TransformSubset(Dataset):
    """Wraps a Subset with a custom transform."""

    def __init__(self, subset: Subset, transform: transforms.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        x, y = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)


def get_cifar10_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
    augment: bool = True,
    download: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads and prepares CIFAR-10 data loaders.

    Args:
        data_dir: Directory to cache CIFAR-10 raw files.
        batch_size: Mini-batch size.
        val_split: Fraction of training set reserved for validation (e.g. 0.1 = 5,000 images).
        seed: Random seed for deterministic train/val split.
        num_workers: DataLoader worker count.
        augment: Whether to apply data augmentation to train set.
        download: Whether to download if dataset is missing.
        pin_memory: Pin memory for GPU transfer speed.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_transform, eval_transform = get_transforms(augment=augment)

    # Base dataset without transforms so we can apply different transforms to train vs val
    base_trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=None
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=eval_transform
    )

    total_train = len(base_trainset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_train), [train_size, val_size], generator=generator
    )

    train_subset = TransformSubset(
        Subset(base_trainset, train_indices.indices),
        transform=train_transform,
    )
    val_subset = TransformSubset(
        Subset(base_trainset, val_indices.indices),
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

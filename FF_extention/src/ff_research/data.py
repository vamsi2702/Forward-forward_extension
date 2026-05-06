from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_mnist_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    jitter: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_transforms = [transforms.ToTensor()]
    if jitter:
        train_transforms = [
            transforms.RandomAffine(degrees=0.0, translate=(2.0 / 28.0, 2.0 / 28.0)),
            transforms.ToTensor(),
        ]

    train_ds = datasets.MNIST(
        root=data_root,
        train=True,
        transform=transforms.Compose(train_transforms),
        download=True,
    )
    test_ds = datasets.MNIST(
        root=data_root,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    if train_subset is not None:
        train_ds = Subset(train_ds, list(range(min(train_subset, len(train_ds)))))
    if test_subset is not None:
        test_ds = Subset(test_ds, list(range(min(test_subset, len(test_ds)))))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def overlay_label_channels(flat_images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    x = flat_images.clone()
    x[:, :10] = 0.0
    x[torch.arange(x.size(0), device=x.device), labels] = 1.0
    return x


def make_wrong_labels(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    random_offset = torch.randint(1, num_classes, size=labels.shape, device=labels.device)
    return (labels + random_offset) % num_classes

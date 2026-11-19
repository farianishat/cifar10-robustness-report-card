"""CIFAR-10 dataset loading, reproducible subsetting, and normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import CIFAR10_MEAN, CIFAR10_STD


def _stats_tensors(
    device: torch.device,
    dtype: torch.dtype,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean_t, std_t


def normalize(
    images: torch.Tensor,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> torch.Tensor:
    """Normalize a batch (N, C, H, W) or single image (C, H, W) from [0, 1] space."""
    squeeze = images.ndim == 3
    x = images.unsqueeze(0) if squeeze else images
    mean_t, std_t = _stats_tensors(x.device, x.dtype, mean, std)
    out = (x - mean_t) / std_t
    return out.squeeze(0) if squeeze else out


def denormalize(
    images: torch.Tensor,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> torch.Tensor:
    """Convert normalized tensors back to [0, 1] pixel space."""
    squeeze = images.ndim == 3
    x = images.unsqueeze(0) if squeeze else images
    mean_t, std_t = _stats_tensors(x.device, x.dtype, mean, std)
    out = x * std_t + mean_t
    return out.squeeze(0) if squeeze else out


def get_cifar10_dataloaders(
    data_dir: Path | str,
    batch_size: int = 128,
    train_subset_size: Optional[int] = None,
    test_subset_size: Optional[int] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Download CIFAR-10 if needed and return train, validation, and test DataLoaders.

    Args:
        data_dir: Directory where CIFAR-10 will be stored or downloaded.
        batch_size: Mini-batch size for all loaders.
        train_subset_size: Optional total cap on (train + val) samples for fast CPU runs.
        test_subset_size: Optional cap on test samples for fast CPU runs.
        val_ratio: Fraction of training pool reserved for validation.
        seed: Random seed for reproducible index sampling and shuffling.
        num_workers: Number of DataLoader worker processes (0 is safest on CPU/notebooks).

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
    """
    if not (0.0 < val_ratio < 0.5):
        raise ValueError(f"val_ratio must be in (0, 0.5), got {val_ratio}.")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    try:
        full_train_aug = datasets.CIFAR10(
            root=str(data_path), train=True, download=True, transform=train_transform
        )
        full_train_eval = datasets.CIFAR10(
            root=str(data_path), train=True, download=False, transform=eval_transform
        )
        full_test = datasets.CIFAR10(
            root=str(data_path), train=False, download=True, transform=eval_transform
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download or load CIFAR-10 in '{data_path}'. "
            "Check your internet connection or dataset permissions."
        ) from exc

    generator = torch.Generator().manual_seed(seed)

    # Reproducible train/val split with optional subsetting
    total_train_pool = len(full_train_aug)
    if train_subset_size is not None:
        if train_subset_size <= 0 or train_subset_size > total_train_pool:
            raise ValueError(
                f"train_subset_size must be in [1, {total_train_pool}], got {train_subset_size}."
            )
        pool_indices = torch.randperm(total_train_pool, generator=generator)[:train_subset_size]
    else:
        pool_indices = torch.randperm(total_train_pool, generator=generator)

    val_size = max(1, int(len(pool_indices) * val_ratio))
    train_size = len(pool_indices) - val_size
    if train_size < 1:
        raise ValueError("Training subset is too small to split into train and validation sets.")

    train_indices = pool_indices[:train_size].tolist()
    val_indices = pool_indices[train_size:].tolist()

    train_dataset = Subset(full_train_aug, train_indices)
    val_dataset = Subset(full_train_eval, val_indices)

    # Reproducible test subsetting
    total_test = len(full_test)
    if test_subset_size is not None:
        if test_subset_size <= 0 or test_subset_size > total_test:
            raise ValueError(
                f"test_subset_size must be in [1, {total_test}], got {test_subset_size}."
            )
        test_indices = torch.randperm(total_test, generator=generator)[:test_subset_size].tolist()
        test_dataset = Subset(full_test, test_indices)
    else:
        test_dataset = full_test

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
    
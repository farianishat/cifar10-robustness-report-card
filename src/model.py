"""Lightweight CNN architecture for CIFAR-10 (~305k trainable parameters)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """Compact 3-block CNN designed for CPU training on 32x32 RGB CIFAR-10 images.

    Total trainable parameters: 304,794.
    """

    def __init__(self, num_classes: int = 10, feature_dim: int = 80, dropout: float = 0.3) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, feature_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer feature embeddings of shape (N, feature_dim)."""
        h = self.features(x)
        h = self.flatten(h)
        h = self.relu(self.fc1(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape (N, num_classes)."""
        h = self.extract_features(x)
        h = self.dropout(h)
        return self.fc2(h)


def count_trainable_parameters(model: nn.Module, verbose: bool = True) -> int:
    """Calculate and optionally print the number of trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"{model.__class__.__name__} trainable parameters: {total:,}")
    return total
    
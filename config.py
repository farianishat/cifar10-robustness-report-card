"""Centralized configuration, reproducibility utilities, and CPU-friendly presets."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Standard CIFAR-10 channel statistics (computed on training split)
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

EPSILONS: List[float] = [
    0.0,
    1 / 255,
    2 / 255,
    4 / 255,
    8 / 255,
]


@dataclass
class ExperimentConfig:
    """Configuration for CPU-only CIFAR-10 robustness experiments."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    # Compute & reproducibility
    device: str = "cpu"
    seed: int = 42
    num_workers: int = 0

    # Dataset subsets (None uses full split)
    train_subset_size: Optional[int] = 10000
    val_ratio: float = 0.1
    test_subset_size: Optional[int] = 2000

    # Shared training hyperparameters
    batch_size: int = 128
    optimizer: str = "adam"  # "adam" or "sgd"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Standard training epochs
    standard_epochs: int = 12

    # Adversarial training hyperparameters (PGD)
    adv_epochs: int = 8
    adv_train_epsilon: float = 4 / 255
    adv_train_alpha: float = 2 / 255
    adv_train_steps: int = 3
    adv_train_random_start: bool = True

    # Evaluation attack hyperparameters
    epsilons: List[float] = field(default_factory=lambda: list(EPSILONS))
    eval_pgd_steps: int = 10
    eval_pgd_random_start: bool = True
    eval_vis_epsilon: float = 8 / 255

    # Visualization parameters
    tsne_samples: int = 500
    figure_dpi: int = 300

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.data_dir = self.project_root / "data"
        self.figures_dir = self.project_root / "figures"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.results_dir = self.project_root / "results"
        # Enforce CPU default even if CUDA is present unless explicitly overridden
        if self.device != "cpu" and not torch.cuda.is_available():
            self.device = "cpu"

    def ensure_directories(self) -> None:
        """Create required project directories if they do not exist."""
        for directory in (
            self.data_dir,
            self.figures_dir,
            self.checkpoints_dir,
            self.results_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_config(mode: str = "default", project_root: Optional[Path] = None) -> ExperimentConfig:
    """Return a preset configuration.

    Args:
        mode: Either ``"sanity_check"`` (fast verification on tiny subsets)
            or ``"default"`` (CPU-friendly portfolio experiment).
        project_root: Optional override for repository root directory.
    """
    kwargs = {"project_root": project_root} if project_root is not None else {}

    if mode == "sanity_check":
        return ExperimentConfig(
            train_subset_size=512,
            val_ratio=0.2,
            test_subset_size=256,
            batch_size=64,
            standard_epochs=2,
            adv_epochs=2,
            adv_train_steps=2,
            eval_pgd_steps=3,
            tsne_samples=120,
            **kwargs,
        )
    if mode == "default":
        return ExperimentConfig(**kwargs)
    raise ValueError(f"Unknown config mode '{mode}'. Choose 'default' or 'sanity_check'.")


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)
    
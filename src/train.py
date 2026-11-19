"""Standard and PGD adversarial training pipeline with checkpointing and history logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import set_seed
from src.attacks import pgd_attack


def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Use 'adam' or 'sgd'.")


@torch.no_grad()
def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute average cross-entropy loss and classification accuracy on clean data."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adam",
    adversarial_training: bool = False,
    pgd_epsilon: float = 4 / 255,
    pgd_alpha: float = 2 / 255,
    pgd_steps: int = 3,
    pgd_random_start: bool = True,
    device: str = "cpu",
    seed: int = 42,
    checkpoint_path: Optional[Path | str] = None,
    history_path: Optional[Path | str] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train a CNN using either standard clean training or PGD adversarial training.

    Args:
        model: PyTorch model to train (modified in-place).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (evaluated on clean validation samples).
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 weight decay coefficient.
        optimizer_name: ``"adam"`` or ``"sgd"``.
        adversarial_training: If True, train on PGD adversarial examples.
        pgd_epsilon: L-infinity perturbation radius for adversarial training.
        pgd_alpha: PGD step size for adversarial training.
        pgd_steps: Number of PGD iterations per batch during adversarial training.
        pgd_random_start: Whether to use random initialization in PGD.
        device: Target compute device (default ``"cpu"``).
        seed: Random seed for reproducibility.
        checkpoint_path: Optional path to save the best model state_dict.
        history_path: Optional path to save training metrics JSON.
        verbose: Print per-epoch progress if True.

    Returns:
        Dictionary containing ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``.
    """
    set_seed(seed)
    torch_device = torch.device(device)
    model.to(torch_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, optimizer_name, learning_rate, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state_dict = None
    mode_label = "Adversarial (PGD)" if adversarial_training else "Standard"

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(torch_device)
            labels = labels.to(torch_device)

            if adversarial_training:
                inputs = pgd_attack(
                    model=model,
                    images=images,
                    labels=labels,
                    epsilon=pgd_epsilon,
                    alpha=pgd_alpha,
                    num_steps=pgd_steps,
                    random_start=pgd_random_start,
                )
            else:
                inputs = images

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

        scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = _evaluate_loader(model, val_loader, criterion, torch_device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            elapsed = time.perf_counter() - start_time
            print(
                f"[{mode_label}] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"{elapsed:.1f}s"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "adversarial_training": adversarial_training,
                "history": history,
            },
            ckpt,
        )

    if history_path is not None:
        hist = Path(history_path)
        hist.parent.mkdir(parents=True, exist_ok=True)
        with hist.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: str = "cpu",
) -> Dict:
    """Load saved weights into ``model`` with graceful error handling."""
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt}'. Train the model first before loading."
        )
    payload = torch.load(ckpt, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return payload
    
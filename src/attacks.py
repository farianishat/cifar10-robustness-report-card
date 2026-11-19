"""L-infinity FGSM and PGD adversarial attacks operating consistently in [0, 1] pixel space."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CIFAR10_MEAN, CIFAR10_STD
from src.data import denormalize, normalize


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> torch.Tensor:
    """Generate L-infinity FGSM adversarial examples from normalized inputs.

    Perturbations are computed and clamped in [0, 1] pixel space, then re-normalized
    so epsilon is strictly measured in pixel intensity units.

    Args:
        model: Target PyTorch classifier.
        images: Normalized input batch of shape (N, 3, H, W).
        labels: Ground-truth integer labels of shape (N,).
        epsilon: Maximum L-infinity perturbation in [0, 1] pixel space.
        mean: Channel-wise normalization mean.
        std: Channel-wise normalization standard deviation.

    Returns:
        Normalized adversarial batch (detached; original input tensor is unmodified).
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
    if epsilon == 0.0:
        return images.clone().detach()

    was_training = model.training
    model.eval()
    try:
        x_orig_pixel = denormalize(images.detach(), mean, std).clamp(0.0, 1.0)
        x_adv_pixel = x_orig_pixel.clone().detach().requires_grad_(True)

        logits = model(normalize(x_adv_pixel, mean, std))
        loss = F.cross_entropy(logits, labels)

        # Use autograd.grad so model parameter .grad buffers are never modified
        grad = torch.autograd.grad(loss, x_adv_pixel, retain_graph=False, create_graph=False)[0]

        x_adv_pixel = x_orig_pixel + epsilon * grad.sign()
        x_adv_pixel = torch.clamp(x_adv_pixel, min=0.0, max=1.0).detach()
        return normalize(x_adv_pixel, mean, std)
    finally:
        if was_training:
            model.train()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool = True,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> torch.Tensor:
    """Generate iterative L-infinity PGD adversarial examples from normalized inputs.

    All updates, L-infinity projections, and [0, 1] pixel clamps occur in original
    pixel space before re-normalizing the output tensor.

    Args:
        model: Target PyTorch classifier.
        images: Normalized input batch of shape (N, 3, H, W).
        labels: Ground-truth integer labels of shape (N,).
        epsilon: Maximum L-infinity perturbation in [0, 1] pixel space.
        alpha: Step size per iteration in [0, 1] pixel space.
        num_steps: Number of PGD iterations.
        random_start: Whether to initialize uniformly inside [-epsilon, +epsilon].
        mean: Channel-wise normalization mean.
        std: Channel-wise normalization standard deviation.

    Returns:
        Normalized adversarial batch (detached; original input tensor is unmodified).
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}.")
    if epsilon == 0.0 or num_steps == 0:
        return images.clone().detach()

    was_training = model.training
    model.eval()
    try:
        x_orig_pixel = denormalize(images.detach(), mean, std).clamp(0.0, 1.0)

        if random_start:
            noise = torch.empty_like(x_orig_pixel).uniform_(-epsilon, epsilon)
            x_adv_pixel = torch.clamp(x_orig_pixel + noise, min=0.0, max=1.0).detach()
        else:
            x_adv_pixel = x_orig_pixel.clone().detach()

        for _ in range(num_steps):
            x_adv_pixel.requires_grad_(True)
            logits = model(normalize(x_adv_pixel, mean, std))
            loss = F.cross_entropy(logits, labels)

            grad = torch.autograd.grad(
                loss, x_adv_pixel, retain_graph=False, create_graph=False
            )[0]

            with torch.no_grad():
                x_adv_pixel = x_adv_pixel + alpha * grad.sign()
                delta = torch.clamp(x_adv_pixel - x_orig_pixel, min=-epsilon, max=epsilon)
                x_adv_pixel = torch.clamp(x_orig_pixel + delta, min=0.0, max=1.0)
            x_adv_pixel = x_adv_pixel.detach()

        return normalize(x_adv_pixel, mean, std)
    finally:
        if was_training:
            model.train()
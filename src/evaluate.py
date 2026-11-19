"""Robustness evaluation metrics, structured comparison tables, and feature extraction."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.attacks import fgsm_attack, pgd_attack


def _default_pgd_alpha(epsilon: float, num_steps: int) -> float:
    """Standard heuristic step size: 2.5 * epsilon / num_steps (bounded by epsilon)."""
    if epsilon == 0.0 or num_steps <= 0:
        return 0.0
    return min(epsilon, max(epsilon / 4.0, 2.5 * epsilon / num_steps))


def evaluate_single_setting(
    model: nn.Module,
    test_loader: DataLoader,
    epsilon: float,
    attack_type: str = "pgd",
    pgd_steps: int = 10,
    pgd_alpha: Optional[float] = None,
    pgd_random_start: bool = True,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate a model on clean and adversarial inputs for a single epsilon value.

    Attack Success Rate (ASR) definition:
        ASR = (# samples classified correctly on clean AND misclassified after attack)
              / (# samples classified correctly on clean).
        If the denominator is 0 (no clean samples classified correctly), ASR is set to 0.0.
    """
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    attack_name = attack_type.lower()
    if attack_name not in {"fgsm", "pgd"}:
        raise ValueError(f"Unsupported attack_type '{attack_type}'. Use 'fgsm' or 'pgd'.")

    alpha = pgd_alpha if pgd_alpha is not None else _default_pgd_alpha(epsilon, pgd_steps)

    total_samples = 0
    clean_correct = 0
    adv_correct = 0
    initially_correct_then_fooled = 0
    clean_conf_sum = 0.0
    adv_conf_sum = 0.0

    for images, labels in test_loader:
        images = images.to(torch_device)
        labels = labels.to(torch_device)
        batch_size = labels.size(0)
        total_samples += batch_size

        # 1. Clean forward pass (no gradients required)
        with torch.no_grad():
            clean_logits = model(images)
            clean_probs = F.softmax(clean_logits, dim=1)
            clean_conf, clean_preds = clean_probs.max(dim=1)
            clean_correct_mask = clean_preds == labels
            clean_correct += clean_correct_mask.sum().item()
            clean_conf_sum += clean_conf.sum().item()

        # 2. Adversarial example generation (gradients enabled inside attack functions)
        if epsilon == 0.0:
            adv_images = images
        elif attack_name == "fgsm":
            adv_images = fgsm_attack(model=model, images=images, labels=labels, epsilon=epsilon)
        else:
            adv_images = pgd_attack(
                model=model,
                images=images,
                labels=labels,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=pgd_steps,
                random_start=pgd_random_start,
            )

        # 3. Adversarial forward pass (no gradients required)
        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_probs = F.softmax(adv_logits, dim=1)
            adv_conf, adv_preds = adv_probs.max(dim=1)
            adv_correct_mask = adv_preds == labels
            adv_correct += adv_correct_mask.sum().item()
            adv_conf_sum += adv_conf.sum().item()

            # ASR numerator: clean-correct AND adversarial-incorrect
            fooled_mask = clean_correct_mask & (~adv_correct_mask)
            initially_correct_then_fooled += fooled_mask.sum().item()

    clean_acc = clean_correct / max(total_samples, 1)
    adv_acc = adv_correct / max(total_samples, 1)
    asr = (initially_correct_then_fooled / clean_correct) if clean_correct > 0 else 0.0
    mean_clean_conf = clean_conf_sum / max(total_samples, 1)
    mean_adv_conf = adv_conf_sum / max(total_samples, 1)

    return {
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "attack_success_rate": asr,
        "mean_clean_confidence": mean_clean_conf,
        "mean_adversarial_confidence": mean_adv_conf,
    }


def evaluate_model_across_epsilons(
    model: nn.Module,
    model_name: str,
    test_loader: DataLoader,
    epsilons: List[float],
    attack_types: Tuple[str, ...] = ("pgd", "fgsm"),
    pgd_steps: int = 10,
    pgd_random_start: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate a model across all specified epsilon values and attack types."""
    rows: List[Dict] = []

    for attack_type in attack_types:
        for eps in epsilons:
            metrics = evaluate_single_setting(
                model=model,
                test_loader=test_loader,
                epsilon=eps,
                attack_type=attack_type,
                pgd_steps=pgd_steps,
                pgd_random_start=pgd_random_start,
                device=device,
            )
            row = {
                "model": model_name,
                "attack": attack_type.upper(),
                "epsilon": eps,
                "epsilon_255": f"{int(round(eps * 255))}/255",
                **metrics,
            }
            rows.append(row)
            if verbose:
                print(
                    f"[{model_name} | {attack_type.upper()} | eps={row['epsilon_255']}] "
                    f"Clean Acc: {metrics['clean_accuracy']:.4f} | "
                    f"Adv Acc: {metrics['adversarial_accuracy']:.4f} | "
                    f"ASR: {metrics['attack_success_rate']:.4f} | "
                    f"Conf (Clean/Adv): {metrics['mean_clean_confidence']:.3f}/"
                    f"{metrics['mean_adversarial_confidence']:.3f}"
                )

    return pd.DataFrame(rows)


def collect_predictions_and_confidences(
    model: nn.Module,
    test_loader: DataLoader,
    epsilon: float = 0.0,
    attack_type: str = "pgd",
    pgd_steps: int = 10,
    pgd_random_start: bool = True,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect ground-truth labels, predicted labels, and max softmax confidences."""
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    alpha = _default_pgd_alpha(epsilon, pgd_steps)
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_confs: List[np.ndarray] = []

    for images, labels in test_loader:
        images = images.to(torch_device)
        labels = labels.to(torch_device)

        if epsilon > 0.0:
            if attack_type.lower() == "fgsm":
                eval_images = fgsm_attack(model, images, labels, epsilon=epsilon)
            else:
                eval_images = pgd_attack(
                    model,
                    images,
                    labels,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_steps=pgd_steps,
                    random_start=pgd_random_start,
                )
        else:
            eval_images = images

        with torch.no_grad():
            logits = model(eval_images)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_confs.append(confs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_confs),
    )


def collect_features(
    model: nn.Module,
    test_loader: DataLoader,
    max_samples: int = 500,
    epsilon: float = 0.0,
    pgd_steps: int = 10,
    pgd_random_start: bool = True,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features for up to ``max_samples`` clean or PGD inputs."""
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    alpha = _default_pgd_alpha(epsilon, pgd_steps)
    feats_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    collected = 0

    for images, labels in test_loader:
        if collected >= max_samples:
            break
        remaining = max_samples - collected
        images = images[:remaining].to(torch_device)
        labels = labels[:remaining].to(torch_device)

        if epsilon > 0.0:
            eval_images = pgd_attack(
                model,
                images,
                labels,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=pgd_steps,
                random_start=pgd_random_start,
            )
        else:
            eval_images = images

        with torch.no_grad():
            feats = model.extract_features(eval_images)

        feats_list.append(feats.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        collected += labels.size(0)

    return np.concatenate(feats_list, axis=0), np.concatenate(labels_list, axis=0)
    
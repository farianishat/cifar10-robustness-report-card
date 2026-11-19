"""Publication-quality 300 DPI visualizations for the Robustness Report Card."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from config import CIFAR10_CLASSES
from src.attacks import pgd_attack
from src.data import denormalize
from src.evaluate import _default_pgd_alpha

# Consistent global plot styling
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
PALETTE = {
    "Standard": "#1f77b4",
    "Adversarial": "#d62728",
    "Clean": "#2ca02c",
    "PGD": "#ff7f0e",
}


def _save_and_close(
    fig: plt.Figure,
    save_path: Path | str,
    dpi: int = 300,
    tight_layout_rect: Optional[Tuple[float, float, float, float]] = None,
) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout_rect is not None:
        fig.tight_layout(rect=tight_layout_rect)
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(
    standard_history: Dict[str, List[float]],
    adversarial_history: Dict[str, List[float]],
    save_path: Path | str,
    dpi: int = 300,
) -> Path:
    """Plot training and validation loss/accuracy for standard vs. adversarial training."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    std_epochs = range(1, len(standard_history["train_loss"]) + 1)
    adv_epochs = range(1, len(adversarial_history["train_loss"]) + 1)

    # Loss subplot
    axes[0].plot(
        std_epochs, standard_history["train_loss"], "-o", color=PALETTE["Standard"], label="Standard Train"
    )
    axes[0].plot(
        std_epochs, standard_history["val_loss"], "--o", color=PALETTE["Standard"], alpha=0.75, label="Standard Val"
    )
    axes[0].plot(
        adv_epochs, adversarial_history["train_loss"], "-s", color=PALETTE["Adversarial"], label="Adversarial Train"
    )
    axes[0].plot(
        adv_epochs, adversarial_history["val_loss"], "--s", color=PALETTE["Adversarial"], alpha=0.75, label="Adversarial Val"
    )
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    # Accuracy subplot
    axes[1].plot(
        std_epochs, standard_history["train_acc"], "-o", color=PALETTE["Standard"], label="Standard Train"
    )
    axes[1].plot(
        std_epochs, standard_history["val_acc"], "--o", color=PALETTE["Standard"], alpha=0.75, label="Standard Val"
    )
    axes[1].plot(
        adv_epochs, adversarial_history["train_acc"], "-s", color=PALETTE["Adversarial"], label="Adversarial Train"
    )
    axes[1].plot(
        adv_epochs, adversarial_history["val_acc"], "--s", color=PALETTE["Adversarial"], alpha=0.75, label="Adversarial Val"
    )
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.suptitle("Standard vs. Adversarial Training Dynamics", fontsize=14, y=1.02)
    return _save_and_close(fig, save_path, dpi=dpi)


def plot_accuracy_vs_epsilon(
    results_df: pd.DataFrame,
    save_path: Path | str,
    dpi: int = 300,
) -> Path:
    """Plot accuracy vs. epsilon comparing Standard and Adversarial models under FGSM & PGD."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    styles = {
        ("Standard", "PGD"): {"color": PALETTE["Standard"], "linestyle": "-", "marker": "o"},
        ("Standard", "FGSM"): {"color": PALETTE["Standard"], "linestyle": "--", "marker": "^", "alpha": 0.75},
        ("Adversarial", "PGD"): {"color": PALETTE["Adversarial"], "linestyle": "-", "marker": "s"},
        ("Adversarial", "FGSM"): {"color": PALETTE["Adversarial"], "linestyle": "--", "marker": "d", "alpha": 0.75},
    }

    for (model_name, attack_name), group in results_df.groupby(["model", "attack"]):
        group_sorted = group.sort_values("epsilon")
        eps_255 = group_sorted["epsilon"].values * 255.0
        style = styles.get((model_name, attack_name), {"marker": "o"})
        ax.plot(
            eps_255,
            group_sorted["adversarial_accuracy"].values,
            linewidth=2.2,
            markersize=7,
            label=f"{model_name} Model ({attack_name})",
            **style,
        )

    xticks = sorted(results_df["epsilon"].unique() * 255.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(round(x))}/255" if x > 0 else "0 (Clean)" for x in xticks])
    ax.set_xlabel(r"Perturbation Budget $\epsilon$ ($L_\infty$ in pixel space)")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Model Accuracy vs. Adversarial Perturbation Strength")
    ax.legend(loc="best")

    return _save_and_close(fig, save_path, dpi=dpi)


def plot_adversarial_examples_grid(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    pgd_steps: int,
    save_path: Path | str,
    num_examples: int = 5,
    vis_amplification: float = 10.0,
    device: str = "cpu",
    dpi: int = 300,
) -> Path:
    """Display original image, amplified perturbation, and PGD adversarial image."""
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    n = min(num_examples, images.size(0))
    x = images[:n].to(torch_device)
    y = labels[:n].to(torch_device)
    alpha = _default_pgd_alpha(epsilon, pgd_steps)

    adv_x = pgd_attack(model, x, y, epsilon=epsilon, alpha=alpha, num_steps=pgd_steps)

    with torch.no_grad():
        orig_preds = model(x).argmax(dim=1).cpu().numpy()
        adv_preds = model(adv_x).argmax(dim=1).cpu().numpy()

    x_pixel = denormalize(x).clamp(0.0, 1.0).cpu()
    adv_pixel = denormalize(adv_x).clamp(0.0, 1.0).cpu()
    perturbation = adv_pixel - x_pixel
    max_linf = perturbation.abs().max().item()

    # Scale perturbation visually around 0.5 gray midpoint without altering true perturbation
    pert_vis = (0.5 + perturbation * vis_amplification).clamp(0.0, 1.0)

    fig, axes = plt.subplots(n, 3, figsize=(9.5, 2.7 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Original Image",
        f"Perturbation ({vis_amplification:.0f}x Visual Scale)",
        f"Adversarial Image (PGD, eps={int(round(epsilon * 255))}/255)",
    ]

    for row in range(n):
        true_lbl = CIFAR10_CLASSES[y[row].item()]
        orig_lbl = CIFAR10_CLASSES[orig_preds[row]]
        adv_lbl = CIFAR10_CLASSES[adv_preds[row]]

        imgs_row = [
            x_pixel[row].permute(1, 2, 0).numpy(),
            pert_vis[row].permute(1, 2, 0).numpy(),
            adv_pixel[row].permute(1, 2, 0).numpy(),
        ]
        subtitles = [
            f"True: {true_lbl}\nPred: {orig_lbl}",
            f"Actual max |delta|: {perturbation[row].abs().max().item():.4f}\n(Displayed at {vis_amplification:.0f}x)",
            f"True: {true_lbl}\nAdv Pred: {adv_lbl}",
        ]

        for col in range(3):
            axes[row, col].imshow(imgs_row[col])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"{col_titles[col]}\n{subtitles[col]}", fontsize=9.5)
            else:
                axes[row, col].set_title(subtitles[col], fontsize=9.5)

    fig.suptitle(
        f"PGD Adversarial Examples (eps = {int(round(epsilon * 255))}/255, observed L_inf <= {max_linf:.4f})",
        fontsize=13,
        y=1.01,
    )
    return _save_and_close(fig, save_path, dpi=dpi)


def plot_confusion_matrices(
    y_true: np.ndarray,
    std_clean_preds: np.ndarray,
    std_adv_preds: np.ndarray,
    adv_clean_preds: np.ndarray,
    adv_adv_preds: np.ndarray,
    epsilon: float,
    save_path: Path | str,
    dpi: int = 300,
) -> Path:
    """Plot 2x2 normalized confusion matrices for Clean and PGD inputs across both models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    eps_label = f"{int(round(epsilon * 255))}/255"

    panels = [
        (axes[0, 0], std_clean_preds, "Standard Model — Clean Test Set"),
        (axes[0, 1], std_adv_preds, f"Standard Model — PGD (eps={eps_label})"),
        (axes[1, 0], adv_clean_preds, "Adversarial Model — Clean Test Set"),
        (axes[1, 1], adv_adv_preds, f"Adversarial Model — PGD (eps={eps_label})"),
    ]

    for ax, preds, title in panels:
        cm = confusion_matrix(y_true, preds, labels=list(range(len(CIFAR10_CLASSES))), normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=CIFAR10_CLASSES,
            yticklabels=CIFAR10_CLASSES,
            cbar=False,
            ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Row-Normalized Confusion Matrices: Clean vs. PGD Adversarial Inputs", fontsize=14, y=1.01)
    return _save_and_close(fig, save_path, dpi=dpi)


def plot_confidence_histograms(
    std_clean_conf: np.ndarray,
    std_adv_conf: np.ndarray,
    adv_clean_conf: np.ndarray,
    adv_adv_conf: np.ndarray,
    epsilon: float,
    save_path: Path | str,
    dpi: int = 300,
) -> Path:
    """Compare predicted-class softmax confidence distributions on clean vs. PGD inputs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    eps_label = f"{int(round(epsilon * 255))}/255"

    axes[0].hist(
        std_clean_conf, bins=bins, alpha=0.65, color=PALETTE["Clean"], label=f"Clean (mean={std_clean_conf.mean():.3f})"
    )
    axes[0].hist(
        std_adv_conf, bins=bins, alpha=0.65, color=PALETTE["PGD"], label=f"PGD eps={eps_label} (mean={std_adv_conf.mean():.3f})"
    )
    axes[0].set_title("Standard-Trained Model")
    axes[0].set_xlabel("Max Softmax Confidence")
    axes[0].set_ylabel("Sample Count")
    axes[0].legend()

    axes[1].hist(
        adv_clean_conf, bins=bins, alpha=0.65, color=PALETTE["Clean"], label=f"Clean (mean={adv_clean_conf.mean():.3f})"
    )
    axes[1].hist(
        adv_adv_conf, bins=bins, alpha=0.65, color=PALETTE["PGD"], label=f"PGD eps={eps_label} (mean={adv_adv_conf.mean():.3f})"
    )
    axes[1].set_title("Adversarially Trained Model")
    axes[1].set_xlabel("Max Softmax Confidence")
    axes[1].legend()

    fig.suptitle("Prediction Confidence Distributions: Clean vs. PGD Adversarial Inputs", fontsize=14, y=1.02)
    return _save_and_close(fig, save_path, dpi=dpi)


def plot_tsne_features(
    std_clean_feats: np.ndarray,
    std_adv_feats: np.ndarray,
    adv_clean_feats: np.ndarray,
    adv_adv_feats: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    save_path: Path | str,
    random_state: int = 42,
    dpi: int = 300,
) -> Path:
    """Generate 2x2 t-SNE visualizations of penultimate-layer features for clean vs. PGD inputs.

    Clean and adversarial features for each model are embedded jointly in the same t-SNE
    space so shifts in feature clusters are directly comparable.
    """
    n = len(labels)
    perplexity = min(30, max(5, (n - 1) // 3))

    def _embed_pair(clean_f: np.ndarray, adv_f: np.ndarray) -> Sequence[np.ndarray]:
        combined = np.vstack([clean_f, adv_f])
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        proj = tsne.fit_transform(combined)
        return proj[:n], proj[n:]

    std_clean_2d, std_adv_2d = _embed_pair(std_clean_feats, std_adv_feats)
    adv_clean_2d, adv_adv_2d = _embed_pair(adv_clean_feats, adv_adv_feats)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))
    cmap = plt.get_cmap("tab10")
    eps_label = f"{int(round(epsilon * 255))}/255"

    panels = [
        (axes[0, 0], std_clean_2d, "Standard Model — Clean Features"),
        (axes[0, 1], std_adv_2d, f"Standard Model — PGD Features (eps={eps_label})"),
        (axes[1, 0], adv_clean_2d, "Adversarial Model — Clean Features"),
        (axes[1, 1], adv_adv_2d, f"Adversarial Model — PGD Features (eps={eps_label})"),
    ]

    for ax, coords, title in panels:
        for class_idx, class_name in enumerate(CIFAR10_CLASSES):
            mask = labels == class_idx
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.75,
                color=cmap(class_idx),
                label=class_name,
            )
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        f"Penultimate-Layer Feature Representations (t-SNE, n={n} samples)",
        fontsize=14,
        y=0.99,
    )
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.955),
        frameon=True,
    )

    return _save_and_close(
        fig,
        save_path,
        dpi=dpi,
        tight_layout_rect=(0.0, 0.0, 1.0, 0.90),
    )
    

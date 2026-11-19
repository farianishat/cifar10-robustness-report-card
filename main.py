"""Modular CLI entry point for running the CIFAR-10 Robustness Report Card end-to-end."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from config import ExperimentConfig, get_config, set_seed
from src.attacks import fgsm_attack, pgd_attack
from src.data import denormalize, get_cifar10_dataloaders
from src.evaluate import (
    collect_features,
    collect_predictions_and_confidences,
    evaluate_model_across_epsilons,
)
from src.model import CIFAR10CNN, count_trainable_parameters
from src.plots import (
    plot_accuracy_vs_epsilon,
    plot_adversarial_examples_grid,
    plot_confidence_histograms,
    plot_confusion_matrices,
    plot_training_curves,
    plot_tsne_features,
)
from src.train import train_model


def run_attack_sanity_checks(model: CIFAR10CNN, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
    """Verify FGSM and PGD respect L-infinity bounds, [0, 1] pixel range, and tensor immutability."""
    eps = 8 / 255
    x_before = batch_x.clone()
    x_orig_pixel = denormalize(batch_x).clamp(0.0, 1.0)

    fgsm_x = fgsm_attack(model, batch_x, batch_y, epsilon=eps)
    pgd_x = pgd_attack(model, batch_x, batch_y, epsilon=eps, alpha=2 / 255, num_steps=5)

    assert torch.equal(batch_x, x_before), "Attack modified the original input tensor in-place!"

    for name, adv_x in (("FGSM", fgsm_x), ("PGD", pgd_x)):
        adv_pixel = denormalize(adv_x)
        linf_err = (adv_pixel - x_orig_pixel).abs().max().item()
        assert linf_err <= eps + 1e-5, f"{name} exceeded L_inf budget: {linf_err:.6f} > {eps:.6f}"
        assert adv_pixel.min().item() >= -1e-5 and adv_pixel.max().item() <= 1.0 + 1e-5, (
            f"{name} produced out-of-range pixel values: "
            f"[{adv_pixel.min().item():.4f}, {adv_pixel.max().item():.4f}]"
        )

    print("Sanity checks passed: FGSM and PGD respect L_inf constraints, [0, 1] range, and immutability.")


def write_report_card(
    cfg: ExperimentConfig,
    param_count: int,
    results_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Generate a Markdown Robustness Report Card grounded strictly in experiment outputs."""
    pgd_df = results_df[results_df["attack"] == "PGD"].copy()
    std_clean = pgd_df[(pgd_df["model"] == "Standard") & (pgd_df["epsilon"] == 0.0)]["clean_accuracy"].iloc[0]
    adv_clean = pgd_df[(pgd_df["model"] == "Adversarial") & (pgd_df["epsilon"] == 0.0)]["clean_accuracy"].iloc[0]

    max_eps = max(cfg.epsilons)
    std_max = pgd_df[(pgd_df["model"] == "Standard") & (pgd_df["epsilon"] == max_eps)].iloc[0]
    adv_max = pgd_df[(pgd_df["model"] == "Adversarial") & (pgd_df["epsilon"] == max_eps)].iloc[0]

    clean_winner = "Standard" if std_clean >= adv_clean else "Adversarial"
    robust_winner = (
        "Adversarial"
        if adv_max["adversarial_accuracy"] >= std_max["adversarial_accuracy"]
        else "Standard"
    )

    lines = [
        "# CIFAR-10 Robustness Report Card",
        "",
        "## Model & Experiment Configuration",
        f"- **Architecture:** `CIFAR10CNN` ({param_count:,} trainable parameters)",
        f"- **Device:** `{cfg.device}` | **Seed:** `{cfg.seed}`",
        f"- **Dataset Subsets:** Train+Val = `{cfg.train_subset_size}`, Test = `{cfg.test_subset_size}`",
        f"- **Standard Training:** `{cfg.standard_epochs}` epochs, LR = `{cfg.learning_rate}`, Optimizer = `{cfg.optimizer}`",
        (
            f"- **Adversarial Training (PGD):** `{cfg.adv_epochs}` epochs, "
            f"eps = `{int(round(cfg.adv_train_epsilon * 255))}/255`, "
            f"steps = `{cfg.adv_train_steps}`"
        ),
        "",
        "## Consolidated Evaluation Table (PGD)",
        "",
        pgd_df[
            [
                "model",
                "epsilon_255",
                "clean_accuracy",
                "adversarial_accuracy",
                "attack_success_rate",
                "mean_clean_confidence",
                "mean_adversarial_confidence",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Automated Results Interpretation",
        f"- **Clean Performance:** The **{clean_winner}** model achieved higher clean accuracy "
        f"(Standard: {std_clean:.2%}, Adversarial: {adv_clean:.2%}).",
        f"- **Adversarial Robustness (at eps={std_max['epsilon_255']}):** The **{robust_winner}** model "
        f"achieved higher PGD accuracy (Standard: {std_max['adversarial_accuracy']:.2%}, "
        f"Adversarial: {adv_max['adversarial_accuracy']:.2%}) with Attack Success Rates of "
        f"{std_max['attack_success_rate']:.2%} (Standard) vs. {adv_max['attack_success_rate']:.2%} (Adversarial).",
        f"- **Accuracy–Robustness Trade-Off:** Clean accuracy shifted by {(adv_clean - std_clean):+.2%}, "
        f"while PGD accuracy at eps={std_max['epsilon_255']} shifted by "
        f"{(adv_max['adversarial_accuracy'] - std_max['adversarial_accuracy']):+.2%}.",
    ]
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(mode: str = "default") -> None:
    """Execute the full training, evaluation, plotting, and reporting pipeline."""
    cfg = get_config(mode=mode)
    cfg.ensure_directories()
    set_seed(cfg.seed)

    print(f"Running pipeline in '{mode}' mode on device='{cfg.device}'...")

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        train_subset_size=cfg.train_subset_size,
        test_subset_size=cfg.test_subset_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )

    # Model sanity check & attack verification
    std_model = CIFAR10CNN()
    param_count = count_trainable_parameters(std_model, verbose=True)
    sample_x, sample_y = next(iter(test_loader))
    run_attack_sanity_checks(std_model, sample_x[:8], sample_y[:8])

    # 1. Standard Training
    std_history = train_model(
        model=std_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.standard_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        optimizer_name=cfg.optimizer,
        adversarial_training=False,
        device=cfg.device,
        seed=cfg.seed,
        checkpoint_path=cfg.checkpoints_dir / "standard_cnn.pt",
        history_path=cfg.results_dir / "standard_history.json",
    )

    # 2. Adversarial Training
    adv_model = CIFAR10CNN()
    adv_history = train_model(
        model=adv_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.adv_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        optimizer_name=cfg.optimizer,
        adversarial_training=True,
        pgd_epsilon=cfg.adv_train_epsilon,
        pgd_alpha=cfg.adv_train_alpha,
        pgd_steps=cfg.adv_train_steps,
        pgd_random_start=cfg.adv_train_random_start,
        device=cfg.device,
        seed=cfg.seed,
        checkpoint_path=cfg.checkpoints_dir / "adversarial_cnn.pt",
        history_path=cfg.results_dir / "adversarial_history.json",
    )

    # 3. Evaluation across epsilons
    std_df = evaluate_model_across_epsilons(
        model=std_model,
        model_name="Standard",
        test_loader=test_loader,
        epsilons=cfg.epsilons,
        attack_types=("pgd", "fgsm"),
        pgd_steps=cfg.eval_pgd_steps,
        pgd_random_start=cfg.eval_pgd_random_start,
        device=cfg.device,
    )
    adv_df = evaluate_model_across_epsilons(
        model=adv_model,
        model_name="Adversarial",
        test_loader=test_loader,
        epsilons=cfg.epsilons,
        attack_types=("pgd", "fgsm"),
        pgd_steps=cfg.eval_pgd_steps,
        pgd_random_start=cfg.eval_pgd_random_start,
        device=cfg.device,
    )
    results_df = pd.concat([std_df, adv_df], ignore_index=True)
    results_df.to_csv(cfg.results_dir / "evaluation_metrics.csv", index=False)

    # 4. Visualizations (300 DPI PNGs)
    vis_eps = cfg.eval_vis_epsilon
    plot_training_curves(
        std_history, adv_history, cfg.figures_dir / "training_curves.png", dpi=cfg.figure_dpi
    )
    plot_accuracy_vs_epsilon(
        results_df, cfg.figures_dir / "accuracy_vs_epsilon.png", dpi=cfg.figure_dpi
    )
    plot_adversarial_examples_grid(
        model=std_model,
        images=sample_x,
        labels=sample_y,
        epsilon=vis_eps,
        pgd_steps=cfg.eval_pgd_steps,
        save_path=cfg.figures_dir / "adversarial_examples_grid.png",
        device=cfg.device,
        dpi=cfg.figure_dpi,
    )

    y_true, std_clean_preds, std_clean_conf = collect_predictions_and_confidences(
        std_model, test_loader, epsilon=0.0, device=cfg.device
    )
    _, std_adv_preds, std_adv_conf = collect_predictions_and_confidences(
        std_model, test_loader, epsilon=vis_eps, pgd_steps=cfg.eval_pgd_steps, device=cfg.device
    )
    _, adv_clean_preds, adv_clean_conf = collect_predictions_and_confidences(
        adv_model, test_loader, epsilon=0.0, device=cfg.device
    )
    _, adv_adv_preds, adv_adv_conf = collect_predictions_and_confidences(
        adv_model, test_loader, epsilon=vis_eps, pgd_steps=cfg.eval_pgd_steps, device=cfg.device
    )

    plot_confusion_matrices(
        y_true,
        std_clean_preds,
        std_adv_preds,
        adv_clean_preds,
        adv_adv_preds,
        epsilon=vis_eps,
        save_path=cfg.figures_dir / "confusion_matrices.png",
        dpi=cfg.figure_dpi,
    )
    plot_confidence_histograms(
        std_clean_conf,
        std_adv_conf,
        adv_clean_conf,
        adv_adv_conf,
        epsilon=vis_eps,
        save_path=cfg.figures_dir / "confidence_histograms.png",
        dpi=cfg.figure_dpi,
    )

    std_clean_f, tsne_labels = collect_features(
        std_model, test_loader, max_samples=cfg.tsne_samples, epsilon=0.0, device=cfg.device
    )
    std_adv_f, _ = collect_features(
        std_model,
        test_loader,
        max_samples=cfg.tsne_samples,
        epsilon=vis_eps,
        pgd_steps=cfg.eval_pgd_steps,
        device=cfg.device,
    )
    adv_clean_f, _ = collect_features(
        adv_model, test_loader, max_samples=cfg.tsne_samples, epsilon=0.0, device=cfg.device
    )
    adv_adv_f, _ = collect_features(
        adv_model,
        test_loader,
        max_samples=cfg.tsne_samples,
        epsilon=vis_eps,
        pgd_steps=cfg.eval_pgd_steps,
        device=cfg.device,
    )
    plot_tsne_features(
        std_clean_f,
        std_adv_f,
        adv_clean_f,
        adv_adv_f,
        labels=tsne_labels,
        epsilon=vis_eps,
        save_path=cfg.figures_dir / "tsne_features.png",
        random_state=cfg.seed,
        dpi=cfg.figure_dpi,
    )

    # 5. Consolidated Report Card
    write_report_card(cfg, param_count, results_df, cfg.results_dir / "robustness_report_card.md")
    print(f"Pipeline complete. Results saved in '{cfg.results_dir}' and figures in '{cfg.figures_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CIFAR-10 Robustness Report Card pipeline.")
    parser.add_argument(
        "--mode",
        choices=["default", "sanity_check"],
        default="default",
        help="Choose 'sanity_check' for a fast CPU smoke test or 'default' for the full run.",
    )
    args = parser.parse_args()
    run_pipeline(mode=args.mode)
    
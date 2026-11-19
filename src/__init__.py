"""Core modules for the CIFAR-10 Robustness Report Card project."""

from .attacks import fgsm_attack, pgd_attack
from .data import denormalize, get_cifar10_dataloaders, normalize
from .evaluate import (
    collect_features,
    collect_predictions_and_confidences,
    evaluate_model_across_epsilons,
    evaluate_single_setting,
)
from .model import CIFAR10CNN, count_trainable_parameters
from .plots import (
    plot_accuracy_vs_epsilon,
    plot_adversarial_examples_grid,
    plot_confidence_histograms,
    plot_confusion_matrices,
    plot_training_curves,
    plot_tsne_features,
)
from .train import load_checkpoint, train_model

__all__ = [
    "CIFAR10CNN",
    "count_trainable_parameters",
    "normalize",
    "denormalize",
    "get_cifar10_dataloaders",
    "fgsm_attack",
    "pgd_attack",
    "train_model",
    "load_checkpoint",
    "evaluate_single_setting",
    "evaluate_model_across_epsilons",
    "collect_predictions_and_confidences",
    "collect_features",
    "plot_training_curves",
    "plot_accuracy_vs_epsilon",
    "plot_adversarial_examples_grid",
    "plot_confusion_matrices",
    "plot_confidence_histograms",
    "plot_tsne_features",
]

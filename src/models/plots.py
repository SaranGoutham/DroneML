"""Reusable plots for DroneML binary model evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

LOGGER = logging.getLogger(__name__)
FIGURE_DPI = 300


def _save_figure(figure: plt.Figure, output_path: Path) -> Path:
    """Apply the common layout and save settings to a figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved plot: %s", output_path)
    return output_path


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: list[str],
    output_dir: Path,
) -> Path:
    """Create and save ``confusion_matrix.png``."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
    )
    axis.set_title("Confusion Matrix")
    return _save_figure(figure, output_dir / "confusion_matrix.png")


def plot_roc_curve(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    output_dir: Path,
) -> Path:
    """Create and save ``roc_curve.png``."""
    labels = np.asarray(y_true)
    figure, axis = plt.subplots(figsize=(7, 6))
    if np.unique(labels).size > 1:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            labels, y_probability
        )
        score = auc(false_positive_rate, true_positive_rate)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"ROC curve (AUC = {score:.4f})",
        )
    else:
        axis.text(0.5, 0.5, "ROC curve requires both classes", ha="center")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    axis.legend(loc="lower right")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Receiver Operating Characteristic Curve")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    return _save_figure(figure, output_dir / "roc_curve.png")


def plot_precision_recall_curve(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    output_dir: Path,
) -> Path:
    """Create and save ``precision_recall_curve.png``."""
    curve_precision, curve_recall, _ = precision_recall_curve(
        y_true, y_probability
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(curve_recall, curve_precision, linewidth=2)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curve")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    return _save_figure(figure, output_dir / "precision_recall_curve.png")


def plot_confidence_histogram(
    confidences: ArrayLike,
    output_dir: Path,
) -> Path:
    """Create and save ``confidence_histogram.png``."""
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.hist(confidences, bins=20, range=(0.5, 1.0), edgecolor="black")
    axis.set_xlabel("Prediction Confidence")
    axis.set_ylabel("Number of Predictions")
    axis.set_title("Prediction Confidence Distribution")
    return _save_figure(figure, output_dir / "confidence_histogram.png")


__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_confidence_histogram",
]

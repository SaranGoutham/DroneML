"""Reusable binary-classification metrics for DroneML model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return positive-class precision, using zero for an undefined result."""
    return float(precision_score(y_true, y_pred, zero_division=0))


def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return positive-class recall, using zero for an undefined result."""
    return float(recall_score(y_true, y_pred, zero_division=0))


def f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the positive-class F1 score."""
    return float(f1_score(y_true, y_pred, zero_division=0))


def roc_auc(y_true: ArrayLike, y_probability: ArrayLike) -> float:
    """Return ROC-AUC from positive-class probabilities.

    ROC-AUC is undefined when the input contains only one class. In that case
    ``nan`` is returned so callers can explicitly represent the missing value.
    """
    labels = np.asarray(y_true)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, y_probability))


def specificity(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the true-negative rate (specificity)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


def sensitivity(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the true-positive rate (sensitivity)."""
    return recall(y_true, y_pred)


def balanced_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return balanced accuracy."""
    return float(balanced_accuracy_score(y_true, y_pred))


def matthews_correlation_coefficient(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Return the Matthews correlation coefficient (MCC)."""
    return float(matthews_corrcoef(y_true, y_pred))


def calculate_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_probability: ArrayLike,
) -> dict[str, Any]:
    """Compute and return the complete DroneML classification metric suite."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred),
        "f1_score": f1(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_probability),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "mcc": matthews_correlation_coefficient(y_true, y_pred),
    }


__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "specificity",
    "sensitivity",
    "balanced_accuracy",
    "matthews_correlation_coefficient",
    "calculate_metrics",
]

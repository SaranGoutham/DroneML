# -*- coding: utf-8 -*-
"""metrics.py – DroneML  [src/evaluation]
==========================================

Phase 10 – Comprehensive Evaluation
--------------------------------------
Loads each saved final model and produces a full evaluation suite on the
held-out test set:

  Metrics computed
  ----------------
    - Accuracy, Loss  (Keras evaluate)
    - Precision, Recall  (Keras evaluate)
    - F1-score  (sklearn)
    - ROC-AUC  (sklearn)
    - Confusion Matrix  (sklearn + matplotlib)
    - Classification Report  (sklearn)

  Outputs  (all written to  results/evaluation/)
  -----------------------------------------------
    confusion_matrix_<model>.png   – per-model confusion matrix
    roc_curve.png                  – all models on one ROC plot
    metrics_<model>.json           – per-model metric dict
    metrics.json                   – combined dict for all models

Usage
-----
  python -m src.evaluation.metrics
  python -m src.evaluation.metrics --model resnet50
"""

from __future__ import annotations

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import PROCESSED_DIR
from src.preprocessing.create_tf_dataset import create_datasets
from src.models.config import (
    EVALUATION_DIR,
    ROC_CURVE_PNG,
    ALL_METRICS_JSON,
    MOBILENETEV3_CONFUSION_PNG,
    RESNET50_CONFUSION_PNG,
    EFFICIENTNETB0_CONFUSION_PNG,
    MOBILENETV3_METRICS_JSON,
    RESNET50_METRICS_JSON,
    EFFICIENTNETB0_METRICS_JSON,
    MOBILENETV3_FINAL_MODEL_PATH,
    RESNET50_FINAL_MODEL_PATH,
    EFFICIENTNETB0_FINAL_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi"       : 130,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})

# Threshold for binary classification
THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "key"           : "mobilenetv3",
        "display_name"  : "MobileNetV3-Large",
        "model_path"    : MOBILENETV3_FINAL_MODEL_PATH,
        "confusion_png" : MOBILENETEV3_CONFUSION_PNG,
        "metrics_json"  : MOBILENETV3_METRICS_JSON,
        "color"         : "#4C72B0",
        "linestyle"     : "-",
    },
    {
        "key"           : "resnet50",
        "display_name"  : "ResNet50",
        "model_path"    : RESNET50_FINAL_MODEL_PATH,
        "confusion_png" : RESNET50_CONFUSION_PNG,
        "metrics_json"  : RESNET50_METRICS_JSON,
        "color"         : "#DD8452",
        "linestyle"     : "--",
    },
    {
        "key"           : "efficientnetb0",
        "display_name"  : "EfficientNetB0",
        "model_path"    : EFFICIENTNETB0_FINAL_MODEL_PATH,
        "confusion_png" : EFFICIENTNETB0_CONFUSION_PNG,
        "metrics_json"  : EFFICIENTNETB0_METRICS_JSON,
        "color"         : "#55A868",
        "linestyle"     : "-.",
    },
]


# ---------------------------------------------------------------------------
# Helper: collect predictions from a dataset
# ---------------------------------------------------------------------------

def _collect_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over the whole dataset.

    Returns
    -------
    y_true : np.ndarray  shape (N,)
    y_pred_prob : np.ndarray  shape (N,)  – sigmoid probabilities
    """
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []

    for images, labels in dataset:
        probs = model(images, training=False).numpy().flatten()
        y_prob_list.append(probs)
        y_true_list.append(labels.numpy().flatten())

    return np.concatenate(y_true_list), np.concatenate(y_prob_list)


# ---------------------------------------------------------------------------
# Compute all metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    keras_results: dict,
    threshold: float = THRESHOLD,
) -> dict:
    """Compute the full metric suite and return as a dict.

    Parameters
    ----------
    y_true : ground-truth binary labels  (0 / 1)
    y_pred_prob : model sigmoid probabilities  [0, 1]
    keras_results : dict returned by model.evaluate(return_dict=True)
    threshold : decision threshold

    Returns
    -------
    dict with keys:
        loss, accuracy, precision_keras, recall_keras, auc_keras,
        precision_sklearn, recall_sklearn, f1_score, roc_auc_sklearn
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics: dict[str, float] = {}

    # Keras metrics (from model.evaluate)
    metrics["loss"]           = float(keras_results.get("loss",      0))
    metrics["accuracy_keras"] = float(keras_results.get("accuracy",  0))
    metrics["precision_keras"]= float(keras_results.get("precision", 0))
    metrics["recall_keras"]   = float(keras_results.get("recall",    0))
    metrics["auc_keras"]      = float(keras_results.get("auc",       0))

    # Sklearn metrics
    metrics["accuracy_sklearn"]  = float(accuracy_score (y_true, y_pred))
    metrics["precision_sklearn"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall_sklearn"]    = float(recall_score   (y_true, y_pred, zero_division=0))
    metrics["f1_score"]          = float(f1_score       (y_true, y_pred, zero_division=0))

    try:
        metrics["roc_auc_sklearn"] = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        metrics["roc_auc_sklearn"] = float("nan")

    # Threshold used
    metrics["threshold"] = threshold

    return metrics


# ---------------------------------------------------------------------------
# Plot: Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: list[str],
    display_name: str,
    out_path: Path,
    threshold: float = THRESHOLD,
) -> None:
    """Save a confusion matrix PNG for one model."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    disp   = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = class_names,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix – {display_name}",
        fontsize=13,
        fontweight="bold",
    )
    # Annotate with percentages
    total = cm.sum()
    for text_obj in ax.texts:
        val = int(text_obj.get_text())
        pct = 100 * val / total if total > 0 else 0
        text_obj.set_text(f"{val}\n({pct:.1f}%)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("Confusion matrix saved → %s", out_path)


# ---------------------------------------------------------------------------
# Plot: Combined ROC Curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
) -> None:
    """Plot all models' ROC curves on one figure.

    Parameters
    ----------
    roc_data : {model_display_name: (fpr, tpr, roc_auc)}
    out_path : where to save the PNG
    """
    color_map   = {e["display_name"]: e["color"]     for e in MODEL_REGISTRY}
    ls_map      = {e["display_name"]: e["linestyle"]  for e in MODEL_REGISTRY}

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("ROC Curves – Test Set", fontsize=14, fontweight="bold")

    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(
            fpr, tpr,
            color     = color_map.get(name, "gray"),
            linestyle = ls_map.get(name, "-"),
            linewidth = 2.5,
            label     = f"{name}  (AUC={roc_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("ROC curves saved → %s", out_path)


# ---------------------------------------------------------------------------
# Print classification report
# ---------------------------------------------------------------------------

def print_classification_report(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: list[str],
    display_name: str,
    threshold: float = THRESHOLD,
) -> str:
    """Print and return the sklearn classification report."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    print(f"\nClassification Report – {display_name}")
    print("=" * 55)
    print(report)
    return report


# ---------------------------------------------------------------------------
# Evaluate a single model
# ---------------------------------------------------------------------------

def evaluate_single(
    entry: dict[str, Any],
    test_ds: tf.data.Dataset,
    class_names: list[str],
) -> dict | None:
    """Full evaluation pipeline for one model.

    Returns
    -------
    dict with keys: metrics, classification_report, roc_fpr, roc_tpr, roc_auc
    or None if the model file is missing.
    """
    model_path = entry["model_path"]
    if not model_path.exists():
        logging.warning("Model not found, skipping: %s", model_path)
        return None

    logging.info("Loading %s from %s ...", entry["display_name"], model_path)
    model: keras.Model = keras.models.load_model(str(model_path))

    # Keras evaluate
    keras_results = model.evaluate(test_ds, return_dict=True, verbose=1)

    # Predictions
    y_true, y_pred_prob = _collect_predictions(model, test_ds)

    # Full metrics
    metrics = compute_metrics(y_true, y_pred_prob, keras_results)

    # Print metrics table
    print(f"\n{'=' * 55}")
    print(f"  METRICS – {entry['display_name']}")
    print(f"{'=' * 55}")
    print(f"  {'Loss':<25}: {metrics['loss']:.4f}")
    print(f"  {'Accuracy (Keras)':<25}: {metrics['accuracy_keras']:.4f}")
    print(f"  {'Accuracy (sklearn)':<25}: {metrics['accuracy_sklearn']:.4f}")
    print(f"  {'Precision (Keras)':<25}: {metrics['precision_keras']:.4f}")
    print(f"  {'Precision (sklearn)':<25}: {metrics['precision_sklearn']:.4f}")
    print(f"  {'Recall (Keras)':<25}: {metrics['recall_keras']:.4f}")
    print(f"  {'Recall (sklearn)':<25}: {metrics['recall_sklearn']:.4f}")
    print(f"  {'F1-Score':<25}: {metrics['f1_score']:.4f}")
    print(f"  {'AUC (Keras)':<25}: {metrics['auc_keras']:.4f}")
    print(f"  {'ROC-AUC (sklearn)':<25}: {metrics['roc_auc_sklearn']:.4f}")
    print(f"{'=' * 55}")

    # Classification report
    report_str = print_classification_report(
        y_true, y_pred_prob, class_names, entry["display_name"]
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_true       = y_true,
        y_pred_prob  = y_pred_prob,
        class_names  = class_names,
        display_name = entry["display_name"],
        out_path     = entry["confusion_png"],
    )

    # ROC data
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)

    # Save per-model metrics JSON
    entry["metrics_json"].parent.mkdir(parents=True, exist_ok=True)
    with open(entry["metrics_json"], "w", encoding="utf-8") as f:
        json.dump(
            {"model": entry["display_name"], **metrics},
            f, indent=4,
        )
    logging.info("Metrics JSON saved → %s", entry["metrics_json"])

    del model
    tf.keras.backend.clear_session()

    return {
        "metrics"               : metrics,
        "classification_report" : report_str,
        "roc_fpr"               : fpr.tolist(),
        "roc_tpr"               : tpr.tolist(),
        "roc_auc"               : roc_auc_val,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_all(model_filter: str | None = None) -> None:
    """Run full evaluation for all (or one) model(s).

    Parameters
    ----------
    model_filter : optional model key to evaluate only one model.
    """
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading test dataset ...")
    _, _, test_ds, class_names = create_datasets(PROCESSED_DIR)
    logging.info("Classes: %s", class_names)

    registry = MODEL_REGISTRY
    if model_filter:
        registry = [e for e in MODEL_REGISTRY if e["key"] == model_filter.lower()]
        if not registry:
            logging.error("Unknown model key: '%s'", model_filter)
            return

    all_results: dict[str, dict] = {}
    roc_data: dict[str, tuple] = {}

    for entry in registry:
        result = evaluate_single(entry, test_ds, class_names)
        if result is None:
            continue
        name = entry["display_name"]
        all_results[name] = result
        roc_data[name]    = (
            np.array(result["roc_fpr"]),
            np.array(result["roc_tpr"]),
            result["roc_auc"],
        )

    # Combined ROC plot
    if roc_data:
        plot_roc_curves(roc_data, ROC_CURVE_PNG)

    # Save combined metrics.json
    combined: dict[str, Any] = {}
    for name, res in all_results.items():
        combined[name] = res["metrics"]

    with open(ALL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=4)
    logging.info("Combined metrics saved → %s", ALL_METRICS_JSON)

    # Summary table
    if all_results:
        best = max(all_results, key=lambda n: all_results[n]["metrics"].get("f1_score", 0))
        print("\n" + "=" * 72)
        print("  EVALUATION SUMMARY")
        print("=" * 72)
        print(
            f"  {'Model':<22} | {'Accuracy':>10} | {'Precision':>10} | "
            f"{'Recall':>8} | {'F1':>8} | {'ROC-AUC':>8}"
        )
        print("-" * 72)
        for name, res in all_results.items():
            m    = res["metrics"]
            flag = " 🏆" if name == best else ""
            print(
                f"  {name:<22} | {m['accuracy_sklearn']:>10.4f} | "
                f"{m['precision_sklearn']:>10.4f} | {m['recall_sklearn']:>8.4f} | "
                f"{m['f1_score']:>8.4f} | {m['roc_auc_sklearn']:>8.4f}{flag}"
            )
        print("=" * 72)
        print(f"\n  Outputs saved to → {EVALUATION_DIR}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 – Comprehensive evaluation of DroneML models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mobilenetv3", "resnet50", "efficientnetb0"],
        help="Evaluate only this model (default: all models).",
    )
    args = parser.parse_args()
    evaluate_all(model_filter=args.model)


if __name__ == "__main__":
    main()

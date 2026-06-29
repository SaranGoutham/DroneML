# -*- coding: utf-8 -*-
"""compare_models.py – DroneML
================================

Phase 12 – Model Comparison & Visualisation
---------------------------------------------
Loads saved training histories and test results for all models and
generates a comprehensive comparison dashboard:

  1. Training Loss Curves         – all models on one plot
  2. Training Accuracy Curves     – all models on one plot
  3. Metric Bar Chart             – Accuracy / AUC / Precision / Recall
  4. ROC Curves                   – AUC on the test set per model
  5. Confusion Matrices           – one per model
  6. Markdown Report              – winner table + narrative

All figures are saved to  results/comparison/.

Usage
-----
  python -m src.models.compare_models
"""

from __future__ import annotations

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
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
    COMPARISON_DIR,
    COMPARISON_RESULTS_JSON,
    COMPARISON_REPORT_MD,
    MOBILENETV3_HISTORY_PKL,
    MOBILENETV3_FINAL_MODEL_PATH,
    RESNET50_HISTORY_PKL,
    RESNET50_FINAL_MODEL_PATH,
    EFFICIENTNETB0_HISTORY_PKL,
    EFFICIENTNETB0_FINAL_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Colour palette  (one distinct colour per model)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "name"        : "MobileNetV3-Large",
        "short"       : "MobileNetV3",
        "history_pkl" : MOBILENETV3_HISTORY_PKL,
        "model_path"  : MOBILENETV3_FINAL_MODEL_PATH,
        "color"       : "#4C72B0",   # steel blue
        "linestyle"   : "-",
    },
    {
        "name"        : "ResNet50",
        "short"       : "ResNet50",
        "history_pkl" : RESNET50_HISTORY_PKL,
        "model_path"  : RESNET50_FINAL_MODEL_PATH,
        "color"       : "#DD8452",   # burnt orange
        "linestyle"   : "--",
    },
    {
        "name"        : "EfficientNetB0",
        "short"       : "EffNetB0",
        "history_pkl" : EFFICIENTNETB0_HISTORY_PKL,
        "model_path"  : EFFICIENTNETB0_FINAL_MODEL_PATH,
        "color"       : "#55A868",   # forest green
        "linestyle"   : "-.",
    },
]

# Matplotlib style
plt.rcParams.update({
    "figure.dpi"        : 120,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "font.size"         : 11,
    "legend.fontsize"   : 9,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
})


# ---------------------------------------------------------------------------
# Helper: load history
# ---------------------------------------------------------------------------

def load_history(pkl_path: Path) -> dict | None:
    """Load training history from a pickle file, return None if missing."""
    if not pkl_path.exists():
        logging.warning("History file not found: %s", pkl_path)
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Helper: load model
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> keras.Model | None:
    """Load a saved Keras model, return None if missing."""
    if not model_path.exists():
        logging.warning("Model file not found: %s", model_path)
        return None
    logging.info("Loading model from %s ...", model_path)
    return keras.models.load_model(str(model_path))


# ---------------------------------------------------------------------------
# Plot 1 & 2: Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    histories: dict[str, dict],
    out_dir: Path,
) -> None:
    """Plot loss and accuracy curves for all models side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves – All Models", fontsize=14, fontweight="bold")

    for entry in MODEL_REGISTRY:
        name = entry["name"]
        hist = histories.get(name)
        if hist is None:
            continue
        color = entry["color"]
        ls    = entry["linestyle"]
        short = entry["short"]

        epochs = range(1, len(hist["loss"]) + 1)

        # Loss
        axes[0].plot(epochs, hist["loss"],     color=color, ls=ls,  lw=2,
                     label=f"{short} train")
        axes[0].plot(epochs, hist["val_loss"], color=color, ls=":", lw=1.5,
                     label=f"{short} val",    alpha=0.7)

        # Accuracy
        axes[1].plot(epochs, hist["accuracy"],     color=color, ls=ls,  lw=2,
                     label=f"{short} train")
        axes[1].plot(epochs, hist["val_accuracy"], color=color, ls=":", lw=1.5,
                     label=f"{short} val",         alpha=0.7)

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right", ncol=2)

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Plot 3: Metric bar chart
# ---------------------------------------------------------------------------

def plot_metric_bars(
    comparison: dict[str, dict],
    out_dir: Path,
) -> None:
    """Bar chart comparing Accuracy / AUC / Precision / Recall per model."""
    metrics  = ["accuracy", "auc", "precision", "recall"]
    labels   = ["Accuracy", "AUC", "Precision", "Recall"]
    models   = [e["name"] for e in MODEL_REGISTRY if e["name"] in comparison]
    colors   = {e["name"]: e["color"] for e in MODEL_REGISTRY}

    x     = np.arange(len(labels))
    width = 0.22
    n     = len(models)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title("Test-Set Metric Comparison", fontsize=14, fontweight="bold")

    for offset, model_name in zip(offsets, models):
        vals = [
            comparison[model_name]["test_results"].get(m, 0)
            for m in metrics
        ]
        bars = ax.bar(
            x + offset, vals,
            width=width,
            label=model_name,
            color=colors[model_name],
            edgecolor="white",
            linewidth=0.8,
        )
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    ax.axhline(y=1.0, color="gray", lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    out_path = out_dir / "metric_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Plot 4: ROC Curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models_dict: dict[str, keras.Model],
    test_ds: tf.data.Dataset,
    out_dir: Path,
) -> dict[str, float]:
    """Compute and plot ROC curves for all models on the test set."""
    colors = {e["name"]: e["color"] for e in MODEL_REGISTRY}
    ls_map = {e["name"]: e["linestyle"] for e in MODEL_REGISTRY}

    # Collect true labels from the test dataset
    y_true_all: list[np.ndarray] = []
    for _, labels in test_ds:
        y_true_all.append(labels.numpy().flatten())
    y_true = np.concatenate(y_true_all)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("ROC Curves – Test Set", fontsize=14, fontweight="bold")

    roc_aucs: dict[str, float] = {}

    for name, model in models_dict.items():
        if model is None:
            continue
        y_pred_all: list[np.ndarray] = []
        for images, _ in test_ds:
            preds = model(images, training=False).numpy().flatten()
            y_pred_all.append(preds)
        y_pred = np.concatenate(y_pred_all)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        roc_aucs[name] = roc_auc

        ax.plot(
            fpr, tpr,
            label=f"{name}  (AUC={roc_auc:.4f})",
            color=colors.get(name, "gray"),
            lw=2.5,
            linestyle=ls_map.get(name, "-"),
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    plt.tight_layout()
    out_path = out_dir / "roc_curves.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)

    return roc_aucs


# ---------------------------------------------------------------------------
# Plot 5: Confusion Matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    models_dict: dict[str, keras.Model],
    test_ds: tf.data.Dataset,
    out_dir: Path,
    class_names: list[str],
) -> None:
    """Generate and save a confusion matrix PNG for each model."""
    y_true_all: list[np.ndarray] = []
    for _, labels in test_ds:
        y_true_all.append(labels.numpy().flatten())
    y_true = np.concatenate(y_true_all)

    colors = {e["name"]: e["color"] for e in MODEL_REGISTRY}

    for name, model in models_dict.items():
        if model is None:
            continue

        y_pred_all: list[np.ndarray] = []
        for images, _ in test_ds:
            preds = model(images, training=False).numpy().flatten()
            y_pred_all.append(preds)
        y_pred_prob = np.concatenate(y_pred_all)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        cm   = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names,
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix – {name}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        safe_name = name.replace(" ", "_").replace("-", "_").lower()
        out_path  = out_dir / f"confusion_matrix_{safe_name}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logging.info("Saved → %s", out_path)

        # Also print classification report
        print(f"\nClassification Report – {name}")
        print(classification_report(y_true, y_pred, target_names=class_names))


# ---------------------------------------------------------------------------
# Markdown Report
# ---------------------------------------------------------------------------

def generate_report(
    comparison: dict[str, dict],
    roc_aucs: dict[str, float],
    out_dir: Path,
) -> None:
    """Write a human-readable Markdown comparison report."""

    # Determine winner by test accuracy
    best_model = max(
        comparison,
        key=lambda n: comparison[n]["test_results"].get("accuracy", 0),
    )
    best_acc = comparison[best_model]["test_results"].get("accuracy", 0)
    best_auc = comparison[best_model]["test_results"].get("auc", 0)

    lines: list[str] = []
    lines.append("# DroneML – Model Comparison Report\n")
    lines.append(
        "Comparison of **MobileNetV3-Large**, **ResNet50**, and **EfficientNetB0** "
        "trained on the thermal UAV landmine detection dataset.\n"
    )
    lines.append("---\n")

    # Winner
    lines.append(f"## 🏆 Winner: {best_model}\n")
    lines.append(
        f"**{best_model}** achieved the highest test accuracy of "
        f"**{best_acc:.4f}** (AUC = {best_auc:.4f}).\n"
    )
    lines.append("---\n")

    # Metric table
    lines.append("## Metric Comparison Table\n")
    lines.append(
        "| Model | Accuracy | AUC | Precision | Recall | "
        "Val Accuracy | Val AUC |\n"
    )
    lines.append("|---|---|---|---|---|---|---|\n")

    for name, info in comparison.items():
        tr      = info["test_results"]
        acc     = tr.get("accuracy",  0)
        auc_v   = tr.get("auc",       0)
        prec    = tr.get("precision", 0)
        rec     = tr.get("recall",    0)
        val_acc = info.get("final_val_accuracy", 0)
        val_auc = info.get("final_val_auc",      0)
        crown   = " 🏆" if name == best_model else ""
        lines.append(
            f"| **{name}**{crown} | {acc:.4f} | {auc_v:.4f} | "
            f"{prec:.4f} | {rec:.4f} | {val_acc:.4f} | {val_auc:.4f} |\n"
        )

    lines.append("\n---\n")

    # Training phases
    lines.append("## Training Details\n")
    lines.append("| Model | Phase A Best Val Acc | Phase B Best Val Acc |\n")
    lines.append("|---|---|---|\n")
    for name, info in comparison.items():
        pa = info.get("phase_a_best_val_accuracy", 0)
        pb = info.get("phase_b_best_val_accuracy", 0)
        lines.append(f"| **{name}** | {pa:.4f} | {pb:.4f} |\n")
    lines.append("\n---\n")

    # ROC AUCs
    if roc_aucs:
        lines.append("## ROC AUC (Test Set)\n")
        lines.append("| Model | ROC AUC |\n")
        lines.append("|---|---|\n")
        for name, roc_v in roc_aucs.items():
            lines.append(f"| **{name}** | {roc_v:.4f} |\n")
        lines.append("\n---\n")

    # Figures
    lines.append("## Generated Figures\n")
    lines.append("| Figure | Description |\n")
    lines.append("|---|---|\n")
    lines.append("| `training_curves.png` | Loss & Accuracy curves for all models |\n")
    lines.append("| `metric_comparison.png` | Bar chart: Accuracy / AUC / Precision / Recall |\n")
    lines.append("| `roc_curves.png` | ROC/AUC curves on the test set |\n")
    lines.append("| `confusion_matrix_mobilenetv3_large.png` | Confusion matrix – MobileNetV3 |\n")
    lines.append("| `confusion_matrix_resnet50.png` | Confusion matrix – ResNet50 |\n")
    lines.append("| `confusion_matrix_efficientnetb0.png` | Confusion matrix – EfficientNetB0 |\n")
    lines.append("\n---\n")

    # Recommendation
    lines.append("## Recommendation\n")
    lines.append(
        f"Based on test accuracy and AUC, **{best_model}** is the recommended "
        "model for deployment on this thermal UAV landmine detection task.\n"
    )

    out_path = out_dir / "comparison_report.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    logging.info("Report saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare_models() -> None:
    """Load all histories + models, generate all comparison artefacts."""
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load histories
    # ----------------------------------------------------------------
    logging.info("Loading training histories ...")
    histories: dict[str, dict | None] = {}
    for entry in MODEL_REGISTRY:
        histories[entry["name"]] = load_history(entry["history_pkl"])

    available = [n for n, h in histories.items() if h is not None]
    if not available:
        logging.error(
            "No history files found. Run `python -m src.models.train_all` first."
        )
        return

    # ----------------------------------------------------------------
    # Load comparison JSON (test metrics)
    # ----------------------------------------------------------------
    if not COMPARISON_RESULTS_JSON.exists():
        logging.error(
            "comparison_results.json not found. "
            "Run `python -m src.models.train_all` first."
        )
        return

    with open(COMPARISON_RESULTS_JSON, encoding="utf-8") as f:
        comparison: dict[str, dict] = json.load(f)

    # ----------------------------------------------------------------
    # Load test dataset
    # ----------------------------------------------------------------
    logging.info("Loading test dataset ...")
    _, _, test_ds, class_names = create_datasets(PROCESSED_DIR)

    # ----------------------------------------------------------------
    # Load saved models
    # ----------------------------------------------------------------
    logging.info("Loading trained models ...")
    models_dict: dict[str, keras.Model | None] = {}
    for entry in MODEL_REGISTRY:
        if entry["name"] in available:
            models_dict[entry["name"]] = load_model(entry["model_path"])
        else:
            models_dict[entry["name"]] = None

    # ----------------------------------------------------------------
    # Generate plots
    # ----------------------------------------------------------------
    logging.info("Generating training curves ...")
    plot_training_curves(histories, COMPARISON_DIR)

    logging.info("Generating metric bar chart ...")
    plot_metric_bars(comparison, COMPARISON_DIR)

    logging.info("Generating ROC curves ...")
    roc_aucs = plot_roc_curves(models_dict, test_ds, COMPARISON_DIR)

    logging.info("Generating confusion matrices ...")
    plot_confusion_matrices(
        models_dict, test_ds, COMPARISON_DIR, class_names
    )

    # ----------------------------------------------------------------
    # Generate Markdown report
    # ----------------------------------------------------------------
    logging.info("Generating comparison report ...")
    generate_report(comparison, roc_aucs, COMPARISON_DIR)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    best_model = max(
        comparison,
        key=lambda n: comparison[n]["test_results"].get("accuracy", 0),
    )
    best_acc = comparison[best_model]["test_results"].get("accuracy", 0)

    print("\n" + "=" * 60)
    print("  COMPARISON COMPLETE")
    print("=" * 60)
    print(f"  All figures saved to : {COMPARISON_DIR}")
    print(f"  Report               : {COMPARISON_DIR / 'comparison_report.md'}")
    print(f"\n  🏆  WINNER : {best_model}  (accuracy={best_acc:.4f})")
    print("=" * 60 + "\n")


def main() -> None:
    compare_models()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""plot_training.py – DroneML  [src/evaluation]
================================================

Phase 9b – Training Curve Visualisation
-----------------------------------------
Loads saved per-model training histories and generates publication-quality
training curves for all three models:

    1. Loss curves  (train + validation, all models)
    2. Accuracy curves  (train + validation, all models)
    3. AUC curves  (train + validation, all models)
    4. Precision + Recall curves  (validation only, all models)
    5. Per-model individual curve sets (Phase A frozen + Phase B fine-tune
       boundary marked with a vertical dashed line)

All figures are saved to  results/figures/.

Usage
-----
  python -m src.evaluation.plot_training
"""

from __future__ import annotations

import sys
import pickle
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import (
    FIGURES_DIR,
    MOBILENETV3_HISTORY_PKL,
    RESNET50_HISTORY_PKL,
    EFFICIENTNETB0_HISTORY_PKL,
    FINE_TUNE_EPOCHS,
)

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi"        : 130,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "font.size"         : 11,
    "legend.fontsize"   : 9,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "lines.linewidth"   : 2.0,
})

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "name"        : "MobileNetV3-Large",
        "short"       : "MobileNetV3",
        "history_pkl" : MOBILENETV3_HISTORY_PKL,
        "color"       : "#4C72B0",
        "linestyle"   : "-",
    },
    {
        "name"        : "ResNet50",
        "short"       : "ResNet50",
        "history_pkl" : RESNET50_HISTORY_PKL,
        "color"       : "#DD8452",
        "linestyle"   : "--",
    },
    {
        "name"        : "EfficientNetB0",
        "short"       : "EffNetB0",
        "history_pkl" : EFFICIENTNETB0_HISTORY_PKL,
        "color"       : "#55A868",
        "linestyle"   : "-.",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_history(pkl_path: Path) -> dict | None:
    """Load a pickle history file; return None if missing."""
    if not pkl_path.exists():
        logging.warning("History not found: %s", pkl_path)
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _phase_boundary(hist: dict) -> int | None:
    """Return the epoch index where fine-tuning begins, estimated from
    the total number of epochs and FINE_TUNE_EPOCHS config value.

    Returns None if history is shorter than expected (fine-tune may not
    have run).
    """
    total = len(hist.get("loss", []))
    if total > FINE_TUNE_EPOCHS:
        return total - FINE_TUNE_EPOCHS
    return None


def _annotate_phase_boundary(ax: plt.Axes, boundary: int | None) -> None:
    """Draw a vertical dashed line at the frozen/fine-tune boundary."""
    if boundary is not None:
        ax.axvline(
            x=boundary,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.text(
            boundary + 0.3,
            ax.get_ylim()[1] * 0.98,
            "Fine-tune →",
            fontsize=7,
            color="gray",
            va="top",
        )


# ---------------------------------------------------------------------------
# Plot 1 & 2: Combined loss + accuracy for all models
# ---------------------------------------------------------------------------

def plot_combined_curves(
    histories: dict[str, dict | None],
    out_dir: Path,
) -> None:
    """Plot loss and accuracy for all models side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        "Training Curves – MobileNetV3 / ResNet50 / EfficientNetB0",
        fontsize=14,
        fontweight="bold",
    )

    for entry in MODEL_REGISTRY:
        name  = entry["name"]
        hist  = histories.get(name)
        if hist is None:
            continue
        short = entry["short"]
        color = entry["color"]
        ls    = entry["linestyle"]
        epochs = range(1, len(hist["loss"]) + 1)

        # Loss
        axes[0].plot(epochs, hist["loss"],     color=color, ls=ls,  label=f"{short} train")
        axes[0].plot(epochs, hist["val_loss"], color=color, ls=":", alpha=0.65, label=f"{short} val")

        # Accuracy
        axes[1].plot(epochs, hist["accuracy"],     color=color, ls=ls,  label=f"{short} train")
        axes[1].plot(epochs, hist["val_accuracy"], color=color, ls=":", alpha=0.65, label=f"{short} val")

    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(ncol=2, loc="upper right")

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(ncol=2, loc="lower right")

    plt.tight_layout()
    out_path = out_dir / "training_curves_combined.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Plot 3: AUC curves for all models
# ---------------------------------------------------------------------------

def plot_auc_curves(
    histories: dict[str, dict | None],
    out_dir: Path,
) -> None:
    """Plot validation AUC over epochs for all models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Validation AUC over Epochs", fontsize=13, fontweight="bold")

    for entry in MODEL_REGISTRY:
        name  = entry["name"]
        hist  = histories.get(name)
        if hist is None or "val_auc" not in hist:
            continue
        epochs = range(1, len(hist["val_auc"]) + 1)
        ax.plot(
            epochs,
            hist["val_auc"],
            color=entry["color"],
            ls=entry["linestyle"],
            label=entry["short"],
        )
        boundary = _phase_boundary(hist)
        _annotate_phase_boundary(ax, boundary)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    out_path = out_dir / "auc_curves.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Plot 4: Precision + Recall curves (validation)
# ---------------------------------------------------------------------------

def plot_precision_recall_curves(
    histories: dict[str, dict | None],
    out_dir: Path,
) -> None:
    """Plot validation precision and recall over epochs for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Validation Precision & Recall over Epochs",
        fontsize=13,
        fontweight="bold",
    )

    for entry in MODEL_REGISTRY:
        name  = entry["name"]
        hist  = histories.get(name)
        if hist is None:
            continue
        color  = entry["color"]
        ls     = entry["linestyle"]
        short  = entry["short"]
        epochs = range(1, len(hist.get("val_precision", [])) + 1)

        if "val_precision" in hist:
            axes[0].plot(epochs, hist["val_precision"], color=color, ls=ls, label=short)
        if "val_recall" in hist:
            axes[1].plot(epochs, hist["val_recall"],    color=color, ls=ls, label=short)

    axes[0].set_title("Precision")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Precision")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].set_title("Recall")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    out_path = out_dir / "precision_recall_curves.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Plot 5: Individual per-model curves with phase boundary
# ---------------------------------------------------------------------------

def plot_individual_curves(
    histories: dict[str, dict | None],
    out_dir: Path,
) -> None:
    """For each model, plot a 2×2 grid: loss, accuracy, AUC, precision/recall."""
    for entry in MODEL_REGISTRY:
        name  = entry["name"]
        hist  = histories.get(name)
        if hist is None:
            continue
        color    = entry["color"]
        short    = entry["short"]
        epochs   = range(1, len(hist["loss"]) + 1)
        boundary = _phase_boundary(hist)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(
            f"Training History – {name}",
            fontsize=14,
            fontweight="bold",
        )

        # Loss
        axes[0, 0].plot(epochs, hist["loss"],     color=color, label="Train")
        axes[0, 0].plot(epochs, hist["val_loss"], color=color, ls="--", alpha=0.65, label="Val")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_ylabel("BCE Loss")
        axes[0, 0].legend()
        _annotate_phase_boundary(axes[0, 0], boundary)

        # Accuracy
        axes[0, 1].plot(epochs, hist["accuracy"],     color=color, label="Train")
        axes[0, 1].plot(epochs, hist["val_accuracy"], color=color, ls="--", alpha=0.65, label="Val")
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].legend()
        _annotate_phase_boundary(axes[0, 1], boundary)

        # AUC
        if "val_auc" in hist:
            axes[1, 0].plot(epochs, hist.get("auc", []),     color=color, label="Train")
            axes[1, 0].plot(epochs, hist["val_auc"],          color=color, ls="--", alpha=0.65, label="Val")
        axes[1, 0].set_title("AUC")
        axes[1, 0].set_ylabel("AUC")
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].legend()
        _annotate_phase_boundary(axes[1, 0], boundary)

        # Precision & Recall
        if "val_precision" in hist:
            axes[1, 1].plot(
                range(1, len(hist["val_precision"]) + 1),
                hist["val_precision"],
                color=color, ls="-", label="Val Precision",
            )
        if "val_recall" in hist:
            axes[1, 1].plot(
                range(1, len(hist["val_recall"]) + 1),
                hist["val_recall"],
                color=color, ls="--", alpha=0.65, label="Val Recall",
            )
        axes[1, 1].set_title("Precision & Recall (Val)")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].legend()
        _annotate_phase_boundary(axes[1, 1], boundary)

        for ax in axes.flat:
            ax.set_xlabel("Epoch")

        plt.tight_layout()
        safe = name.replace(" ", "_").replace("-", "_").lower()
        out_path = out_dir / f"training_history_{safe}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logging.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_all() -> None:
    """Load all histories and generate all training curve figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading training histories ...")
    histories: dict[str, dict | None] = {}
    for entry in MODEL_REGISTRY:
        histories[entry["name"]] = _load_history(entry["history_pkl"])

    available = [n for n, h in histories.items() if h is not None]
    if not available:
        logging.error(
            "No history files found. Run `python -m src.models.train_all` first."
        )
        return

    logging.info("Available histories : %s", available)

    logging.info("Generating combined loss + accuracy curves ...")
    plot_combined_curves(histories, FIGURES_DIR)

    logging.info("Generating AUC curves ...")
    plot_auc_curves(histories, FIGURES_DIR)

    logging.info("Generating precision + recall curves ...")
    plot_precision_recall_curves(histories, FIGURES_DIR)

    logging.info("Generating individual per-model curves ...")
    plot_individual_curves(histories, FIGURES_DIR)

    print(f"\n[DONE] All training curves saved → {FIGURES_DIR}")


def main() -> None:
    plot_all()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""evaluate_models.py – DroneML
==================================

Phase 12b – Standalone Test-Set Evaluation
--------------------------------------------
Loads each saved final model and runs a full evaluation on the test set,
producing per-model metrics and confusion matrices without needing to
re-train.

This script is useful for:
  - Quick re-evaluation after loading saved models
  - Generating fresh confusion matrices / classification reports
  - Updating comparison_results.json with new test metrics

Usage
-----
  python -m src.models.evaluate_models
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
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
    MOBILENETV3_FINAL_MODEL_PATH,
    RESNET50_FINAL_MODEL_PATH,
    EFFICIENTNETB0_FINAL_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = [
    {
        "name"       : "MobileNetV3-Large",
        "model_path" : MOBILENETV3_FINAL_MODEL_PATH,
        "color"      : "#4C72B0",
    },
    {
        "name"       : "ResNet50",
        "model_path" : RESNET50_FINAL_MODEL_PATH,
        "color"      : "#DD8452",
    },
    {
        "name"       : "EfficientNetB0",
        "model_path" : EFFICIENTNETB0_FINAL_MODEL_PATH,
        "color"      : "#55A868",
    },
]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_single(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    model_name: str,
    class_names: list[str],
    out_dir: Path,
) -> dict:
    """Evaluate one model on the test set.

    Returns
    -------
    dict
        Test metrics dictionary.
    """
    # Keras evaluate
    results = model.evaluate(test_ds, return_dict=True, verbose=1)

    # Collect predictions + labels
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    for images, labels in test_ds:
        preds = model(images, training=False).numpy().flatten()
        y_pred_list.append(preds)
        y_true_list.append(labels.numpy().flatten())

    y_true = np.concatenate(y_true_list)
    y_pred_prob = np.concatenate(y_pred_list)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # ROC-AUC (sklearn)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except Exception:
        roc_auc = float("nan")

    results["roc_auc_sklearn"] = roc_auc

    # Print metrics
    print(f"\n{'=' * 55}")
    print(f"  EVALUATION – {model_name}")
    print(f"{'=' * 55}")
    for k, v in results.items():
        print(f"  {k:<25}: {v:.4f}")
    print(f"{'=' * 55}")

    # Classification report
    print(f"\nClassification Report – {model_name}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("-", "_").lower()
    cm_path   = out_dir / f"confusion_matrix_{safe_name}.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    logging.info("Confusion matrix saved → %s", cm_path)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_all() -> None:
    """Evaluate all saved models and update comparison_results.json."""
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    logging.info("Loading test dataset ...")
    _, _, test_ds, class_names = create_datasets(PROCESSED_DIR)
    logging.info("Classes : %s", class_names)

    # Load existing comparison JSON (if any)
    if COMPARISON_RESULTS_JSON.exists():
        with open(COMPARISON_RESULTS_JSON, encoding="utf-8") as f:
            comparison: dict = json.load(f)
    else:
        comparison = {}

    all_results: dict[str, dict] = {}

    for entry in MODEL_REGISTRY:
        name       = entry["name"]
        model_path = entry["model_path"]

        if not model_path.exists():
            logging.warning("Model not found, skipping: %s", model_path)
            continue

        logging.info("Loading model: %s from %s", name, model_path)
        model = keras.models.load_model(str(model_path))

        results = evaluate_single(
            model      = model,
            test_ds    = test_ds,
            model_name = name,
            class_names = class_names,
            out_dir    = COMPARISON_DIR,
        )
        all_results[name] = results

        # Update comparison dict
        if name not in comparison:
            comparison[name] = {}
        comparison[name]["test_results"] = results

        del model
        tf.keras.backend.clear_session()

    # Save updated comparison JSON
    with open(COMPARISON_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)
    logging.info("Updated comparison_results.json → %s", COMPARISON_RESULTS_JSON)

    # Final summary table
    if all_results:
        best = max(all_results, key=lambda n: all_results[n].get("accuracy", 0))
        print("\n" + "=" * 65)
        print("  EVALUATION SUMMARY")
        print("=" * 65)
        print(f"  {'Model':<22} | {'Accuracy':>10} | {'AUC':>8} | {'ROC-AUC':>8}")
        print("-" * 65)
        for name, res in all_results.items():
            acc     = res.get("accuracy",           0)
            auc_v   = res.get("auc",                0)
            roc_auc = res.get("roc_auc_sklearn",    0)
            flag    = " 🏆" if name == best else ""
            print(f"  {name:<22} | {acc:>10.4f} | {auc_v:>8.4f} | {roc_auc:>8.4f}{flag}")
        print("=" * 65 + "\n")


def main() -> None:
    evaluate_all()


if __name__ == "__main__":
    main()

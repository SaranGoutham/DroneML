# -*- coding: utf-8 -*-
"""Evaluate saved MineInsight models on the held-out test split.

Usage:
  python -m src.models.mineinsight.evaluate_mineinsight --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import BATCH_SIZE, RANDOM_STATE
from src.models.training_utils import RunPaths, create_image_datasets, select_model_specs
from src.preprocessing.mineinsight.mineinsight_config import (
    PROCESSED_DIR as MINEINSIGHT_PROCESSED_DIR,
    RESULTS_DIR as MINEINSIGHT_RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["No_Mine", "Mine"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved MineInsight models on the test split."
    )
    parser.add_argument("--data-dir", type=Path, default=MINEINSIGHT_PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=MINEINSIGHT_RESULTS_DIR)
    parser.add_argument(
        "--model",
        choices=["all", "mobilenetv3", "resnet50", "efficientnetb0"],
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classifying Mine.",
    )
    return parser.parse_args()


def collect_labels_and_scores(
    model: keras.Model,
    test_ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_batches: list[np.ndarray] = []
    y_score_batches: list[np.ndarray] = []
    for images, labels in test_ds:
        y_true_batches.append(labels.numpy().reshape(-1))
        y_score_batches.append(model(images, training=False).numpy().reshape(-1))
    return (
        np.concatenate(y_true_batches).astype(int),
        np.concatenate(y_score_batches),
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_mineinsight(args: argparse.Namespace) -> dict[str, Any]:
    run_paths = RunPaths.from_root(args.results_dir)
    specs = select_model_specs(args.model)

    _, _, test_ds, class_names = create_image_datasets(
        args.data_dir,
        class_names=CLASS_NAMES,
        batch_size=args.batch_size,
        random_seed=RANDOM_STATE,
        augment_train=False,
        cache=False,
    )

    all_metrics: dict[str, Any] = {}
    for spec in specs:
        model_path = run_paths.models / f"{spec.key}_final.keras"
        if not model_path.exists():
            logger.warning("Skipping %s; model not found: %s", spec.key, model_path)
            continue

        logger.info("Loading %s", model_path)
        model = keras.models.load_model(str(model_path))
        eval_results = model.evaluate(test_ds, return_dict=True, verbose=1)
        y_true, y_score = collect_labels_and_scores(model, test_ds)
        y_pred = (y_score >= args.threshold).astype(int)

        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        roc_auc = roc_auc_score(y_true, y_score)
        cm_path = run_paths.evaluation / f"confusion_matrix_{spec.key}.png"
        save_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            cm_path,
            title=f"MineInsight Confusion Matrix - {spec.display_name}",
        )

        metrics = {
            "display_name": spec.display_name,
            "model_path": str(model_path),
            "class_order": class_names,
            "positive_class": "Mine",
            "threshold": args.threshold,
            "keras_evaluate": eval_results,
            "roc_auc": float(roc_auc),
            "classification_report": report,
            "confusion_matrix_png": str(cm_path),
        }
        metrics_path = run_paths.evaluation / f"metrics_{spec.key}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        all_metrics[spec.key] = metrics

    combined_path = run_paths.evaluation / "metrics_mineinsight.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\nMineInsight evaluation complete: {combined_path}\n")
    return all_metrics


def main() -> None:
    evaluate_mineinsight(parse_args())


if __name__ == "__main__":
    main()

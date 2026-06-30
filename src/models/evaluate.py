"""Evaluate a configured DroneML CNN on the complete held-out test set.

Run with ``python -m src.models.evaluate``. Model selection, checkpoint phase,
dataset paths, output paths, and the decision threshold are defined centrally
in :mod:`src.models.config`.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from tensorflow import keras
from tqdm.auto import tqdm

from src.models.config import (
    EVALUATION_DIR,
    EVALUATION_MODEL_CONFIGS,
    EVALUATION_MODEL_NAME,
    EVALUATION_THRESHOLD,
    EVALUATION_USE_FINETUNED_MODEL,
    PROCESSED_DIR,
    TEST_DIR,
)
from src.models.metrics import calculate_metrics
from src.models.plots import (
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.preprocessing.create_tf_dataset import create_datasets

LOGGER = logging.getLogger(__name__)
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


def _print_heading() -> None:
    """Print the evaluation console heading."""
    print("=" * 38)
    print("MODEL EVALUATION")
    print("=" * 38)


def _resolve_model() -> tuple[str, Path]:
    """Return the configured display name and selected checkpoint path."""
    key = EVALUATION_MODEL_NAME.strip().lower()
    if key not in EVALUATION_MODEL_CONFIGS:
        valid_names = ", ".join(sorted(EVALUATION_MODEL_CONFIGS))
        raise ValueError(
            f"Unknown EVALUATION_MODEL_NAME '{EVALUATION_MODEL_NAME}'. "
            f"Choose one of: {valid_names}."
        )

    model_config = EVALUATION_MODEL_CONFIGS[key]
    path_key = (
        "finetuned_model_path"
        if EVALUATION_USE_FINETUNED_MODEL
        else "best_model_path"
    )
    return str(model_config["display_name"]), Path(model_config[path_key])


def _load_model(model_path: Path) -> keras.Model:
    """Load a Keras model checkpoint after validating its path."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Configured model checkpoint not found: {model_path}")
    LOGGER.info("Loading model checkpoint: %s", model_path)
    return keras.models.load_model(model_path, compile=False)


def _test_filenames(test_dir: Path) -> list[str]:
    """Return filenames in the same deterministic order as the test dataset."""
    paths = sorted(
        path
        for path in test_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    return [path.name for path in paths]


def _positive_probabilities(predictions: NDArray[np.floating]) -> NDArray[np.float64]:
    """Normalize sigmoid or two-class softmax output to positive probabilities."""
    values = np.asarray(predictions)
    if values.ndim == 1:
        probabilities = values
    elif values.ndim == 2 and values.shape[1] == 1:
        probabilities = values[:, 0]
    elif values.ndim == 2 and values.shape[1] == 2:
        probabilities = values[:, 1]
    else:
        raise ValueError(
            "Expected binary model output with shape (N,), (N, 1), or (N, 2); "
            f"received {values.shape}."
        )
    return np.clip(probabilities.astype(np.float64), 0.0, 1.0)


def collect_predictions(
    model: keras.Model,
    test_ds: tf.data.Dataset,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Run inference over every test batch and collect labels/probabilities."""
    true_batches: list[NDArray[np.int64]] = []
    probability_batches: list[NDArray[np.float64]] = []
    cardinality = int(tf.data.experimental.cardinality(test_ds).numpy())
    total = cardinality if cardinality >= 0 else None

    for images, labels in tqdm(test_ds, total=total, desc="Predicting", unit="batch"):
        outputs = model(images, training=False).numpy()
        true_batches.append(np.asarray(labels).reshape(-1).astype(np.int64))
        probability_batches.append(_positive_probabilities(outputs))

    if not true_batches:
        raise ValueError("The configured test dataset is empty.")
    return np.concatenate(true_batches), np.concatenate(probability_batches)


def _classification_rows(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_names: list[str],
) -> list[dict[str, Any]]:
    """Build per-class classification report rows."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return [
        {
            "Class": class_name,
            "Precision": report[class_name]["precision"],
            "Recall": report[class_name]["recall"],
            "F1-score": report[class_name]["f1-score"],
            "Support": int(report[class_name]["support"]),
        }
        for class_name in class_names
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write dictionaries to a UTF-8 CSV with a fixed column order."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved report: %s", path)


def _save_classification_report(
    output_dir: Path,
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_names: list[str],
) -> Path:
    """Save the per-class classification report CSV."""
    output_path = output_dir / "classification_report.csv"
    columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
    _write_csv(
        output_path,
        _classification_rows(y_true, y_pred, class_names),
        columns,
    )
    return output_path


def _save_predictions(
    output_dir: Path,
    filenames: list[str],
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    confidences: NDArray[np.float64],
    class_names: list[str],
) -> Path:
    """Save one prediction record per test image."""
    if len(filenames) != len(y_true):
        raise ValueError(
            "Test filename count does not match prediction count: "
            f"{len(filenames)} files versus {len(y_true)} predictions."
        )
    rows = [
        {
            "Filename": filename,
            "True Label": class_names[int(true_label)],
            "Predicted Label": class_names[int(predicted_label)],
            "Confidence": float(confidence),
            "Correct Prediction (Yes/No)": (
                "Yes" if true_label == predicted_label else "No"
            ),
        }
        for filename, true_label, predicted_label, confidence in zip(
            filenames, y_true, y_pred, confidences
        )
    ]
    output_path = output_dir / "predictions.csv"
    columns = [
        "Filename",
        "True Label",
        "Predicted Label",
        "Confidence",
        "Correct Prediction (Yes/No)",
    ]
    _write_csv(output_path, rows, columns)
    return output_path


def _json_value(value: float) -> float | None:
    """Convert non-finite metric values to JSON-compatible null values."""
    return float(value) if math.isfinite(value) else None


def _save_metrics(output_dir: Path, metrics: dict[str, float]) -> Path:
    """Save evaluation metrics as standards-compliant JSON."""
    output_path = output_dir / "metrics.json"
    serializable = {key: _json_value(value) for key, value in metrics.items()}
    output_path.write_text(
        json.dumps(serializable, indent=4, allow_nan=False),
        encoding="utf-8",
    )
    LOGGER.info("Saved metrics: %s", output_path)
    return output_path


def _save_summary(
    output_dir: Path,
    model_name: str,
    sample_count: int,
    metrics: dict[str, float],
    elapsed_seconds: float,
    saved_files: list[Path],
) -> Path:
    """Create the human-readable evaluation summary."""
    separator = "=" * 39

    def formatted(metric_name: str) -> str:
        value = metrics[metric_name]
        return f"{value:.6f}" if math.isfinite(value) else "N/A"

    lines = [
        separator,
        "MODEL EVALUATION",
        separator,
        f"Model Name                         : {model_name}",
        f"Test Samples                       : {sample_count}",
        f"Accuracy                           : {formatted('accuracy')}",
        f"Precision                          : {formatted('precision')}",
        f"Recall                             : {formatted('recall')}",
        f"Specificity                        : {formatted('specificity')}",
        f"Sensitivity                        : {formatted('sensitivity')}",
        f"F1 Score                           : {formatted('f1_score')}",
        f"ROC AUC                            : {formatted('roc_auc')}",
        f"Balanced Accuracy                  : {formatted('balanced_accuracy')}",
        f"Matthews Correlation Coefficient   : {formatted('mcc')}",
        f"Evaluation Time                    : {elapsed_seconds:.2f} seconds",
        "Saved Files:",
        *[f"  - {path.name}" for path in saved_files],
        separator,
    ]
    output_path = output_dir / "evaluation_summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved summary: %s", output_path)
    return output_path


def evaluate() -> dict[str, float]:
    """Execute the complete evaluation pipeline for the configured model."""
    started_at = time.perf_counter()
    model_name, model_path = _resolve_model()
    output_dir = EVALUATION_DIR / EVALUATION_MODEL_NAME.strip().lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_heading()
    print("Loading model...")
    model = _load_model(model_path)

    print("Loading test dataset...")
    _, _, test_ds, class_names = create_datasets(PROCESSED_DIR)
    if len(class_names) != 2:
        raise ValueError(
            f"This evaluation pipeline expects two classes; found {class_names}."
        )

    print("Predicting...")
    y_true, y_probability = collect_predictions(model, test_ds)

    # Temporary diagnostics for validating binary probability interpretation.
    LOGGER.info("Class names (label order): %s", class_names)
    LOGGER.info("Prediction threshold: %.6f", EVALUATION_THRESHOLD)
    LOGGER.info("Minimum probability: %.6f", float(np.min(y_probability)))
    LOGGER.info("Maximum probability: %.6f", float(np.max(y_probability)))
    LOGGER.info("Mean probability: %.6f", float(np.mean(y_probability)))
    LOGGER.info(
        "First 20 probabilities: %s",
        np.array2string(y_probability[:20], precision=6, separator=", "),
    )

    y_pred = (y_probability >= EVALUATION_THRESHOLD).astype(np.int64)
    LOGGER.info("Predictions for label 0: %d", int(np.sum(y_pred == 0)))
    LOGGER.info("Predictions for label 1: %d", int(np.sum(y_pred == 1)))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    LOGGER.info(
        "Confusion matrix [[TN, FP], [FN, TP]]: %s",
        matrix.tolist(),
    )
    confidences = np.where(y_pred == 1, y_probability, 1.0 - y_probability)

    print("Computing metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_probability)
    metrics = {
        "loss": float(log_loss(y_true, y_probability, labels=[0, 1])),
        **metrics,
    }

    print("Generating plots...")
    plot_paths = [
        plot_confusion_matrix(y_true, y_pred, class_names, output_dir),
        plot_roc_curve(y_true, y_probability, output_dir),
        plot_precision_recall_curve(y_true, y_probability, output_dir),
        plot_confidence_histogram(confidences, output_dir),
    ]

    print("Saving reports...")
    report_paths = [
        _save_classification_report(
            output_dir, y_true, y_pred, class_names
        ),
        _save_predictions(
            output_dir,
            _test_filenames(TEST_DIR),
            y_true,
            y_pred,
            confidences,
            class_names,
        ),
        _save_metrics(output_dir, metrics),
    ]
    elapsed_seconds = time.perf_counter() - started_at
    _save_summary(
        output_dir,
        model_name,
        len(y_true),
        metrics,
        elapsed_seconds,
        report_paths + plot_paths + [output_dir / "evaluation_summary.txt"],
    )

    print("Evaluation Complete.")
    print(f"Results saved to: {output_dir}")
    return metrics


def main() -> None:
    """Configure logging and run evaluation from the command line."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    evaluate()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""train_all.py – DroneML
==========================

Phase 11 – Multi-Model Training Runner
----------------------------------------
Trains ALL configured models on the thermal mine detection dataset and
saves per-model results. Currently supports:

    1. MobileNetV3-Large  (mobilenetv3.py)
    2. ResNet50           (resnet50.py)
    3. EfficientNetB0     (efficientnetb0.py)

Training Strategy (per model)
------------------------------
  Phase A – Frozen backbone training
      Optimizer : Adam (lr=model_config.LEARNING_RATE)
      Max Epochs: EPOCHS (EarlyStopping with patience=PATIENCE)
      Backbone  : Frozen (only classifier head trained)

  Phase B – Fine-tuning
      Optimizer : Adam (lr=FINE_TUNE_LEARNING_RATE)
      Max Epochs: FINE_TUNE_EPOCHS
      Backbone  : Top FINE_TUNE_UNFREEZE_LAYERS layers unfrozen

Outputs (per model)
--------------------
  results/checkpoints/<model>_best.keras       – best frozen checkpoint
  results/checkpoints/<model>_finetuned.keras  – best fine-tuned checkpoint
  results/models/<model>_final.keras           – final saved model
  results/history/<model>_history.json         – per-epoch metrics

Final comparison output
------------------------
  results/comparison/comparison_results.json   – all model test metrics

Usage
-----
  python -m src.models.train_all
"""

from __future__ import annotations

import sys
import pickle
import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import PROCESSED_DIR, RANDOM_STATE
from src.preprocessing.create_tf_dataset import create_datasets

# Model builders
from src.models.mobilenetv3 import (
    build_model as build_mobilenetv3,
    INPUT_SHAPE as MOBILENETV3_INPUT_SHAPE,
    unfreeze_top_layers as mobilenetv3_unfreeze_top,
)
from src.models.resnet50 import (
    build_model as build_resnet50,
    INPUT_SHAPE as RESNET50_INPUT_SHAPE,
    unfreeze_top_layers as resnet50_unfreeze_top,
)
from src.models.efficientnetb0 import (
    build_model as build_efficientnetb0,
    INPUT_SHAPE as EFFICIENTNETB0_INPUT_SHAPE,
    unfreeze_top_layers as efficientnetb0_unfreeze_top,
)

# Config
from src.models.config import (
    # Shared training
    EPOCHS,
    PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LEARNING_RATE,
    TENSORBOARD_UPDATE_FREQ,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    FINE_TUNE_UNFREEZE_LAYERS,
    # MobileNetV3
    MOBILENETV3_LEARNING_RATE,
    MOBILENETV3_BEST_MODEL_PATH,
    MOBILENETV3_FINETUNED_MODEL_PATH,
    MOBILENETV3_FINAL_MODEL_PATH,
    MOBILENETV3_HISTORY_JSON,
    MOBILENETV3_HISTORY_PKL,
    # ResNet50
    RESNET50_LEARNING_RATE,
    RESNET50_BEST_MODEL_PATH,
    RESNET50_FINETUNED_MODEL_PATH,
    RESNET50_FINAL_MODEL_PATH,
    RESNET50_HISTORY_JSON,
    RESNET50_HISTORY_PKL,
    # EfficientNetB0
    EFFICIENTNETB0_LEARNING_RATE,
    EFFICIENTNETB0_BEST_MODEL_PATH,
    EFFICIENTNETB0_FINETUNED_MODEL_PATH,
    EFFICIENTNETB0_FINAL_MODEL_PATH,
    EFFICIENTNETB0_HISTORY_JSON,
    EFFICIENTNETB0_HISTORY_PKL,
    # Directories & comparison
    TENSORBOARD_LOG_DIR,
    COMPARISON_RESULTS_JSON,
    COMPARISON_DIR,
)

# Total number of models being trained
NUM_MODELS = 3


# ---------------------------------------------------------------------------
# Helper: Count images in folder
# ---------------------------------------------------------------------------

def count_images(folder: Path) -> int:
    """Count all supported image files inside a folder."""
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    return sum(len(list(folder.glob(ext))) for ext in extensions)


# ---------------------------------------------------------------------------
# Helper: Compute class weights
# ---------------------------------------------------------------------------

def compute_class_weights(train_dir: Path) -> dict:
    """Compute balanced class weights from training folder counts.

    Returns
    -------
    dict
        ``{0: weight_mine, 1: weight_no_mine}``
    """
    mine_count    = count_images(train_dir / "Mine")
    no_mine_count = count_images(train_dir / "No_Mine")

    if mine_count == 0 or no_mine_count == 0:
        raise ValueError(
            "Training dataset must contain both Mine and No_Mine images."
        )

    total       = mine_count + no_mine_count
    num_classes = 2
    w_mine    = round(total / (num_classes * mine_count), 4)
    w_no_mine = round(total / (num_classes * no_mine_count), 4)

    logging.info(
        "Class Weights  Mine=%s  No_Mine=%s  (Mine=%d  No_Mine=%d  Total=%d)",
        w_mine, w_no_mine, mine_count, no_mine_count, total,
    )
    return {0: w_mine, 1: w_no_mine}


# ---------------------------------------------------------------------------
# Helper: Build callbacks
# ---------------------------------------------------------------------------

def _tensorboard_callback(
    log_dir: Path,
    histogram_freq: int = 0,
) -> keras.callbacks.Callback | None:
    """Return a TensorBoard callback if tensorboard is installed, else None."""
    try:
        import tensorboard  # noqa: F401  (just checking it is importable)
        return keras.callbacks.TensorBoard(
            log_dir        = str(log_dir),
            histogram_freq = histogram_freq,
            write_graph    = True,
            update_freq    = TENSORBOARD_UPDATE_FREQ,
        )
    except ImportError:
        logging.warning(
            "TensorBoard not installed – skipping TensorBoard callback. "
            "Install with: pip install tensorboard"
        )
        return None


def build_callbacks(model_name: str, best_model_path: Path) -> list:
    """Construct Keras callbacks for frozen-backbone training."""
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = TENSORBOARD_LOG_DIR / f"{model_name}_{run_id}"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = PATIENCE,
            restore_best_weights = True,
            verbose              = 1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = REDUCE_LR_FACTOR,
            patience = max(2, PATIENCE // 2),
            min_lr   = MIN_LEARNING_RATE,
            verbose  = 1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath          = str(best_model_path),
            monitor           = "val_loss",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1,
        ),
    ]
    tb = _tensorboard_callback(run_dir, histogram_freq=1)
    if tb is not None:
        callbacks.append(tb)
    return callbacks


def build_finetune_callbacks(
    model_name: str,
    finetuned_model_path: Path,
) -> list:
    """Construct Keras callbacks for fine-tuning phase (smaller patience)."""
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = TENSORBOARD_LOG_DIR / f"{model_name}_ft_{run_id}"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = REDUCE_LR_FACTOR,
            patience = 3,
            min_lr   = MIN_LEARNING_RATE,
            verbose  = 1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath          = str(finetuned_model_path),
            monitor           = "val_loss",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1,
        ),
    ]
    tb = _tensorboard_callback(run_dir, histogram_freq=0)
    if tb is not None:
        callbacks.append(tb)
    return callbacks


# ---------------------------------------------------------------------------
# Helper: Evaluate on test set
# ---------------------------------------------------------------------------

def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    model_name: str,
) -> dict[str, float]:
    """Run model.evaluate() on the test set and return a results dict."""
    logging.info("Evaluating %s on test set ...", model_name)
    results = model.evaluate(test_ds, return_dict=True, verbose=1)

    print(f"\n{'=' * 55}")
    print(f"  TEST RESULTS  – {model_name}")
    print(f"{'=' * 55}")
    for metric, value in results.items():
        print(f"  {metric:<22}: {value:.4f}")
    print(f"{'=' * 55}\n")

    return results


# ---------------------------------------------------------------------------
# Core: Train a single model (Phase A + Phase B)
# ---------------------------------------------------------------------------

def train_single_model(
    model_name: str,
    model_idx: int,
    model: keras.Model,
    backbone: keras.Model,
    unfreeze_fn,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    class_weights: dict,
    best_model_path: Path,
    finetuned_model_path: Path,
    final_model_path: Path,
    json_path: Path,
    pkl_path: Path,
    frozen_lr: float,
    finetune_lr: float,
) -> dict[str, Any]:
    """Full two-phase training pipeline for one model.

    Returns
    -------
    dict
        ``{'history_merged': dict, 'history_a': dict,
           'history_b': dict, 'test_results': dict}``
    """
    sep = "=" * 60
    logging.info("%s", sep)
    logging.info("MODEL %d / %d : %s", model_idx, NUM_MODELS, model_name)
    logging.info("%s", sep)

    # ----------------------------------------------------------------
    # Phase A – Frozen backbone
    # ----------------------------------------------------------------
    logging.info("[Phase A] Frozen-backbone training  lr=%.0e ...", frozen_lr)
    callbacks_a = build_callbacks(model_name, best_model_path)

    history_a = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = callbacks_a,
        class_weight    = class_weights,
        verbose         = 1,
    )

    logging.info(
        "[Phase A] Done. Best val_accuracy=%.4f  val_auc=%.4f",
        max(history_a.history.get("val_accuracy", [0])),
        max(history_a.history.get("val_auc", [0])),
    )

    # ----------------------------------------------------------------
    # Phase B – Fine-tuning
    # ----------------------------------------------------------------
    logging.info(
        "[Phase B] Fine-tuning top %d layers  lr=%.0e ...",
        FINE_TUNE_UNFREEZE_LAYERS, finetune_lr,
    )
    unfreeze_fn(backbone, FINE_TUNE_UNFREEZE_LAYERS)

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=finetune_lr),
        loss      = "binary_crossentropy",
        metrics   = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks_b = build_finetune_callbacks(model_name, finetuned_model_path)

    history_b = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = FINE_TUNE_EPOCHS,
        callbacks       = callbacks_b,
        class_weight    = class_weights,
        verbose         = 1,
    )

    logging.info(
        "[Phase B] Done. Best val_accuracy=%.4f  val_auc=%.4f",
        max(history_b.history.get("val_accuracy", [0])),
        max(history_b.history.get("val_auc", [0])),
    )

    # ----------------------------------------------------------------
    # Merge and save history
    # ----------------------------------------------------------------
    merged: dict[str, list] = {}
    for key in history_a.history:
        merged[key] = (
            history_a.history[key] + history_b.history.get(key, [])
        )

    with open(pkl_path, "wb") as f:
        pickle.dump(merged, f)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4)
    logging.info("Merged history saved → %s", json_path)

    # ----------------------------------------------------------------
    # Test evaluation
    # ----------------------------------------------------------------
    test_results = evaluate_model(model, test_ds, model_name)

    # ----------------------------------------------------------------
    # Save final model
    # ----------------------------------------------------------------
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)
    logging.info("Final model saved → %s", final_model_path)

    return {
        "history_merged": merged,
        "history_a"     : history_a.history,
        "history_b"     : history_b.history,
        "test_results"  : test_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_all() -> dict[str, Any]:
    """Train all three models and produce a comparison JSON.

    Returns
    -------
    dict
        ``{model_name: {history_merged, history_a, history_b, test_results}}``
    """
    # Reproducibility
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # ----------------------------------------------------------------
    # Dataset (loaded once, shared across all models)
    # ----------------------------------------------------------------
    logging.info("Loading datasets from %s ...", PROCESSED_DIR)
    train_ds, val_ds, test_ds, class_names = create_datasets(PROCESSED_DIR)
    logging.info("Classes : %s", class_names)

    train_dir    = PROCESSED_DIR / "train"
    class_weights = compute_class_weights(train_dir)

    all_results: dict[str, Any] = {}

    # ================================================================
    # MODEL 1 – MobileNetV3-Large
    # ================================================================
    print("\n" + "#" * 60)
    print(f"  MODEL 1 / {NUM_MODELS} : MobileNetV3-Large")
    print("#" * 60 + "\n")

    model_mnv3, backbone_mnv3 = build_mobilenetv3(
        input_shape        = MOBILENETV3_INPUT_SHAPE,
        learning_rate      = MOBILENETV3_LEARNING_RATE,
        trainable_backbone = False,
        random_seed        = RANDOM_STATE,
    )

    all_results["MobileNetV3-Large"] = train_single_model(
        model_name           = "mobilenetv3",
        model_idx            = 1,
        model                = model_mnv3,
        backbone             = backbone_mnv3,
        unfreeze_fn          = mobilenetv3_unfreeze_top,
        train_ds             = train_ds,
        val_ds               = val_ds,
        test_ds              = test_ds,
        class_weights        = class_weights,
        best_model_path      = MOBILENETV3_BEST_MODEL_PATH,
        finetuned_model_path = MOBILENETV3_FINETUNED_MODEL_PATH,
        final_model_path     = MOBILENETV3_FINAL_MODEL_PATH,
        json_path            = MOBILENETV3_HISTORY_JSON,
        pkl_path             = MOBILENETV3_HISTORY_PKL,
        frozen_lr            = MOBILENETV3_LEARNING_RATE,
        finetune_lr          = FINE_TUNE_LEARNING_RATE,
    )

    del model_mnv3, backbone_mnv3
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # ================================================================
    # MODEL 2 – ResNet50
    # ================================================================
    print("\n" + "#" * 60)
    print(f"  MODEL 2 / {NUM_MODELS} : ResNet50")
    print("#" * 60 + "\n")

    model_rn50, backbone_rn50 = build_resnet50(
        input_shape        = RESNET50_INPUT_SHAPE,
        learning_rate      = RESNET50_LEARNING_RATE,
        trainable_backbone = False,
        random_seed        = RANDOM_STATE,
    )

    all_results["ResNet50"] = train_single_model(
        model_name           = "resnet50",
        model_idx            = 2,
        model                = model_rn50,
        backbone             = backbone_rn50,
        unfreeze_fn          = resnet50_unfreeze_top,
        train_ds             = train_ds,
        val_ds               = val_ds,
        test_ds              = test_ds,
        class_weights        = class_weights,
        best_model_path      = RESNET50_BEST_MODEL_PATH,
        finetuned_model_path = RESNET50_FINETUNED_MODEL_PATH,
        final_model_path     = RESNET50_FINAL_MODEL_PATH,
        json_path            = RESNET50_HISTORY_JSON,
        pkl_path             = RESNET50_HISTORY_PKL,
        frozen_lr            = RESNET50_LEARNING_RATE,
        finetune_lr          = FINE_TUNE_LEARNING_RATE,
    )

    del model_rn50, backbone_rn50
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # ================================================================
    # MODEL 3 – EfficientNetB0
    # ================================================================
    print("\n" + "#" * 60)
    print(f"  MODEL 3 / {NUM_MODELS} : EfficientNetB0")
    print("#" * 60 + "\n")

    model_efn, backbone_efn = build_efficientnetb0(
        input_shape        = EFFICIENTNETB0_INPUT_SHAPE,
        learning_rate      = EFFICIENTNETB0_LEARNING_RATE,
        trainable_backbone = False,
        random_seed        = RANDOM_STATE,
    )

    all_results["EfficientNetB0"] = train_single_model(
        model_name           = "efficientnetb0",
        model_idx            = 3,
        model                = model_efn,
        backbone             = backbone_efn,
        unfreeze_fn          = efficientnetb0_unfreeze_top,
        train_ds             = train_ds,
        val_ds               = val_ds,
        test_ds              = test_ds,
        class_weights        = class_weights,
        best_model_path      = EFFICIENTNETB0_BEST_MODEL_PATH,
        finetuned_model_path = EFFICIENTNETB0_FINETUNED_MODEL_PATH,
        final_model_path     = EFFICIENTNETB0_FINAL_MODEL_PATH,
        json_path            = EFFICIENTNETB0_HISTORY_JSON,
        pkl_path             = EFFICIENTNETB0_HISTORY_PKL,
        frozen_lr            = EFFICIENTNETB0_LEARNING_RATE,
        finetune_lr          = FINE_TUNE_LEARNING_RATE,
    )

    del model_efn, backbone_efn
    tf.keras.backend.clear_session()

    # ================================================================
    # Save comparison JSON
    # ================================================================
    comparison: dict[str, Any] = {}
    for name, result in all_results.items():
        comparison[name] = {
            "test_results": result["test_results"],
            "phase_a_best_val_accuracy": max(
                result["history_a"].get("val_accuracy", [0])
            ),
            "phase_b_best_val_accuracy": max(
                result["history_b"].get("val_accuracy", [0])
            ),
            "final_val_accuracy": max(
                result["history_merged"].get("val_accuracy", [0])
            ),
            "final_val_auc": max(
                result["history_merged"].get("val_auc", [0])
            ),
            "final_val_precision": max(
                result["history_merged"].get("val_precision", [0])
            ),
            "final_val_recall": max(
                result["history_merged"].get("val_recall", [0])
            ),
        }

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)
    logging.info("Comparison results saved → %s", COMPARISON_RESULTS_JSON)

    # ================================================================
    # Print final comparison table
    # ================================================================
    best_model = max(
        comparison,
        key=lambda n: comparison[n]["test_results"].get("accuracy", 0),
    )
    best_acc = comparison[best_model]["test_results"].get("accuracy", 0)

    print("\n" + "=" * 65)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<22} | {'Accuracy':>10} | {'AUC':>8} | {'Precision':>10} | {'Recall':>8}")
    print("-" * 65)
    for name, info in comparison.items():
        tr   = info["test_results"]
        acc  = tr.get("accuracy", 0)
        auc  = tr.get("auc", 0)
        prec = tr.get("precision", 0)
        rec  = tr.get("recall", 0)
        flag = " 🏆" if name == best_model else ""
        print(
            f"  {name:<22} | {acc:>10.4f} | {auc:>8.4f} | {prec:>10.4f} | {rec:>8.4f}{flag}"
        )
    print("=" * 65)
    print(f"\n  🏆  WINNER : {best_model}  (test accuracy = {best_acc:.4f})")
    print("=" * 65 + "\n")

    logging.info(
        "ALL MODELS TRAINED. Run `python -m src.models.compare_models` "
        "for full visualisations."
    )

    return all_results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    train_all()


if __name__ == "__main__":
    main()

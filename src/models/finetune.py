# -*- coding: utf-8 -*-
"""finetune.py – DroneML
=========================

Phase 9 – Standalone Fine-Tuning
----------------------------------
Loads each model's best frozen-backbone checkpoint, unfreezes the top
FINE_TUNE_UNFREEZE_LAYERS layers of the backbone, recompiles with a lower
learning rate, and continues training.

This script runs **independently** from train_all.py.  It is designed for
the case where you have already run Phase 8 / baseline training and now
want to fine-tune without repeating the frozen phase.

Training Strategy
-----------------
  Input  : results/checkpoints/<model>_best.keras   (frozen-phase checkpoint)
  Phase B: top FINE_TUNE_UNFREEZE_LAYERS unfrozen, Adam lr=FINE_TUNE_LEARNING_RATE
  Output : results/checkpoints/<model>_finetuned.keras
         : results/models/<model>_final.keras
         : results/history/<model>_finetune_history.json

Callbacks
---------
  - EarlyStopping   : patience=5, monitor=val_loss, restore_best_weights
  - ReduceLROnPlateau : patience=3, factor=0.2, min_lr=1e-7
  - ModelCheckpoint : saves best fine-tuned model

Usage
-----
  python -m src.models.finetune
  python -m src.models.finetune --model mobilenetv3   # single model only
"""

from __future__ import annotations

import sys
import json
import pickle
import logging
import argparse
import datetime
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import PROCESSED_DIR, RANDOM_STATE
from src.preprocessing.create_tf_dataset import create_datasets

from src.models.mobilenetv3 import unfreeze_top_layers as mnv3_unfreeze
from src.models.resnet50     import unfreeze_top_layers as rn50_unfreeze
from src.models.efficientnetb0 import unfreeze_top_layers as efn_unfreeze

from src.models.config import (
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    FINE_TUNE_UNFREEZE_LAYERS,
    REDUCE_LR_FACTOR,
    MIN_LEARNING_RATE,
    TENSORBOARD_UPDATE_FREQ,
    PATIENCE,
    REDUCE_LR_PATIENCE,
    # Checkpoint paths (input – best frozen model)
    MOBILENETV3_BEST_MODEL_PATH,
    RESNET50_BEST_MODEL_PATH,
    EFFICIENTNETB0_BEST_MODEL_PATH,
    # Fine-tuned checkpoint paths (output)
    MOBILENETV3_FINETUNED_MODEL_PATH,
    RESNET50_FINETUNED_MODEL_PATH,
    EFFICIENTNETB0_FINETUNED_MODEL_PATH,
    # Final model paths (output)
    MOBILENETV3_FINAL_MODEL_PATH,
    RESNET50_FINAL_MODEL_PATH,
    EFFICIENTNETB0_FINAL_MODEL_PATH,
    # History paths (fine-tune only)
    HISTORY_DIR,
    LOGS_DIR,
    TENSORBOARD_LOG_DIR,
)

# Fine-tune-only history JSON files
MOBILENETV3_FT_HISTORY_JSON  = HISTORY_DIR / "mobilenetv3_finetune_history.json"
RESNET50_FT_HISTORY_JSON     = HISTORY_DIR / "resnet50_finetune_history.json"
EFFICIENTNETB0_FT_HISTORY_JSON = HISTORY_DIR / "efficientnetb0_finetune_history.json"

# ---------------------------------------------------------------------------
# Registry – one entry per model
# ---------------------------------------------------------------------------

MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "key"                  : "mobilenetv3",
        "display_name"         : "MobileNetV3-Large",
        "best_ckpt"            : MOBILENETV3_BEST_MODEL_PATH,
        "finetuned_ckpt"       : MOBILENETV3_FINETUNED_MODEL_PATH,
        "final_model"          : MOBILENETV3_FINAL_MODEL_PATH,
        "ft_history_json"      : MOBILENETV3_FT_HISTORY_JSON,
        "unfreeze_fn"          : mnv3_unfreeze,
        "backbone_layer_name"  : "MobileNetV3Large",   # used for locating backbone sub-model
    },
    {
        "key"                  : "resnet50",
        "display_name"         : "ResNet50",
        "best_ckpt"            : RESNET50_BEST_MODEL_PATH,
        "finetuned_ckpt"       : RESNET50_FINETUNED_MODEL_PATH,
        "final_model"          : RESNET50_FINAL_MODEL_PATH,
        "ft_history_json"      : RESNET50_FT_HISTORY_JSON,
        "unfreeze_fn"          : rn50_unfreeze,
        "backbone_layer_name"  : "resnet50",
    },
    {
        "key"                  : "efficientnetb0",
        "display_name"         : "EfficientNetB0",
        "best_ckpt"            : EFFICIENTNETB0_BEST_MODEL_PATH,
        "finetuned_ckpt"       : EFFICIENTNETB0_FINETUNED_MODEL_PATH,
        "final_model"          : EFFICIENTNETB0_FINAL_MODEL_PATH,
        "ft_history_json"      : EFFICIENTNETB0_FT_HISTORY_JSON,
        "unfreeze_fn"          : efn_unfreeze,
        "backbone_layer_name"  : "efficientnetb0",
    },
]


# ---------------------------------------------------------------------------
# Helper: count images
# ---------------------------------------------------------------------------

def _count_images(folder: Path) -> int:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    return sum(len(list(folder.glob(e))) for e in exts)


# ---------------------------------------------------------------------------
# Helper: compute class weights
# ---------------------------------------------------------------------------

def _compute_class_weights(train_dir: Path) -> dict:
    mine    = _count_images(train_dir / "Mine")
    no_mine = _count_images(train_dir / "No_Mine")
    total   = mine + no_mine
    return {0: round(total / (2 * mine), 4), 1: round(total / (2 * no_mine), 4)}


# ---------------------------------------------------------------------------
# Helper: optional TensorBoard callback
# ---------------------------------------------------------------------------

def _tb_callback(log_dir: Path) -> keras.callbacks.Callback | None:
    try:
        import tensorboard  # noqa: F401
        return keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,
            update_freq=TENSORBOARD_UPDATE_FREQ,
        )
    except ImportError:
        logging.warning("TensorBoard not installed – skipping TB callback.")
        return None


# ---------------------------------------------------------------------------
# Helper: build fine-tuning callbacks
# ---------------------------------------------------------------------------

def _build_ft_callbacks(
    model_key: str,
    finetuned_ckpt: Path,
) -> list:
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = TENSORBOARD_LOG_DIR / f"{model_key}_ft_{datetime.datetime.now():%Y%m%d-%H%M%S}"

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
            patience = REDUCE_LR_PATIENCE,
            min_lr   = MIN_LEARNING_RATE,
            verbose  = 1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath          = str(finetuned_ckpt),
            monitor           = "val_loss",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1,
        ),
    ]
    tb = _tb_callback(run_dir)
    if tb is not None:
        callbacks.append(tb)
    return callbacks


# ---------------------------------------------------------------------------
# Core: fine-tune a single model
# ---------------------------------------------------------------------------

def _get_backbone(model: keras.Model, backbone_name_fragment: str) -> keras.Model | None:
    """Find the backbone sub-model by searching layer names for a fragment."""
    for layer in model.layers:
        if backbone_name_fragment.lower() in layer.name.lower():
            return layer
    return None


def finetune_single(
    entry: dict[str, Any],
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    class_weights: dict,
) -> dict:
    """Load the best frozen checkpoint, unfreeze top layers, and fine-tune.

    Returns
    -------
    dict
        Fine-tuning history dict and test results.
    """
    sep = "=" * 60
    logging.info("%s", sep)
    logging.info("FINE-TUNING : %s", entry["display_name"])
    logging.info("%s", sep)

    best_ckpt = entry["best_ckpt"]
    if not best_ckpt.exists():
        logging.warning(
            "Best checkpoint not found: %s – skipping %s",
            best_ckpt, entry["display_name"],
        )
        return {}

    # ----------------------------------------------------------------
    # Load best frozen-phase model
    # ----------------------------------------------------------------
    logging.info("Loading checkpoint from %s ...", best_ckpt)
    model: keras.Model = keras.models.load_model(str(best_ckpt))

    # ----------------------------------------------------------------
    # Locate backbone sub-model and unfreeze top layers
    # ----------------------------------------------------------------
    backbone = _get_backbone(model, entry["backbone_layer_name"])
    if backbone is None:
        logging.warning(
            "Could not locate backbone layer containing '%s'. "
            "Attempting full model fine-tuning with top-%d layers.",
            entry["backbone_layer_name"], FINE_TUNE_UNFREEZE_LAYERS,
        )
        # Fallback: unfreeze the last N layers of the full model
        for layer in model.layers[-FINE_TUNE_UNFREEZE_LAYERS:]:
            layer.trainable = True
    else:
        entry["unfreeze_fn"](backbone, FINE_TUNE_UNFREEZE_LAYERS)
        logging.info(
            "Unfrozen top %d layers of backbone '%s'.",
            FINE_TUNE_UNFREEZE_LAYERS, backbone.name,
        )

    # ----------------------------------------------------------------
    # Recompile with lower learning rate
    # ----------------------------------------------------------------
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss      = "binary_crossentropy",
        metrics   = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    logging.info(
        "Trainable parameters after unfreezing: %s", f"{trainable_params:,}"
    )

    # ----------------------------------------------------------------
    # Fine-tune
    # ----------------------------------------------------------------
    callbacks = _build_ft_callbacks(entry["key"], entry["finetuned_ckpt"])

    logging.info(
        "Starting fine-tuning  (lr=%.0e  max_epochs=%d) ...",
        FINE_TUNE_LEARNING_RATE, FINE_TUNE_EPOCHS,
    )
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = FINE_TUNE_EPOCHS,
        callbacks       = callbacks,
        class_weight    = class_weights,
        verbose         = 1,
    )
    training_time = time.time() - start_time

    logging.info(
        "Training Time : %.2f minutes",
        training_time / 60,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    best_val_auc = max(history.history.get("val_auc",      [0]))
    logging.info(
        "Fine-tuning complete.  best val_accuracy=%.4f  best val_auc=%.4f",
        best_val_acc, best_val_auc,
    )

    # ----------------------------------------------------------------
    # Evaluate on test set
    # ----------------------------------------------------------------
    logging.info("Evaluating on test set ...")
    test_results = model.evaluate(test_ds, return_dict=True, verbose=1)

    print(f"\n{'=' * 55}")
    print(f"  TEST RESULTS (fine-tuned) – {entry['display_name']}")
    print(f"{'=' * 55}")
    for k, v in test_results.items():
        print(f"  {k:<22}: {v:.4f}")
    print(f"{'=' * 55}\n")

    # ----------------------------------------------------------------
    # Save history + final model
    # ----------------------------------------------------------------
    ft_hist_json = entry["ft_history_json"]
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(ft_hist_json, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=4)
        ft_hist_pkl = HISTORY_DIR / f"{entry['key']}_finetune_history.pkl"

    with open(ft_hist_pkl, "wb") as f:
        pickle.dump(history.history, f)

    logging.info("Fine-tune history saved → %s", ft_hist_pkl)
    logging.info("Fine-tune history saved → %s", ft_hist_json)

    entry["final_model"].parent.mkdir(parents=True, exist_ok=True)
    model.save(str(entry["final_model"]))
    logging.info("Final model saved → %s", entry["final_model"])

    del model
    tf.keras.backend.clear_session()

    return {
        "history"      : history.history,
        "test_results" : test_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def finetune(model_filter: str | None = None) -> dict[str, dict]:
    """Fine-tune all (or one) model(s).

    Parameters
    ----------
    model_filter:
        If provided, only the model whose key matches this string is tuned.
        Accepts: ``"mobilenetv3"``, ``"resnet50"``, ``"efficientnetb0"``.

    Returns
    -------
    dict
        ``{model_display_name: {history, test_results}}``
    """
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # ----------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------
    logging.info("Loading datasets ...")
    train_ds, val_ds, test_ds, class_names = create_datasets(PROCESSED_DIR)
    logging.info("Classes: %s", class_names)

    class_weights = _compute_class_weights(PROCESSED_DIR / "train")

    all_results: dict[str, dict] = {}

    registry = MODEL_REGISTRY
    if model_filter:
        registry = [e for e in MODEL_REGISTRY if e["key"] == model_filter.lower()]
        if not registry:
            logging.error(
                "Unknown model key '%s'. Choose from: %s",
                model_filter,
                [e["key"] for e in MODEL_REGISTRY],
            )
            return {}

    total = len(registry)
    for idx, entry in enumerate(registry, start=1):
        print(f"\n{'#' * 60}")
        print(f"  MODEL {idx} / {total} : {entry['display_name']}")
        print(f"{'#' * 60}\n")

        result = finetune_single(
            entry         = entry,
            train_ds      = train_ds,
            val_ds        = val_ds,
            test_ds       = test_ds,
            class_weights = class_weights,
        )
        if result:
            all_results[entry["display_name"]] = result

        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

    # ----------------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------------
    if all_results:
        best = max(
            all_results,
            key=lambda n: all_results[n]["test_results"].get("accuracy", 0),
        )
        print("\n" + "=" * 60)
        print("  FINE-TUNING SUMMARY")
        print("=" * 60)
        for name, res in all_results.items():
            acc = res["test_results"].get("accuracy", 0)
            auc = res["test_results"].get("auc", 0)
            tag = " ← best" if name == best else ""
            print(f"  {name:<25} | acc={acc:.4f}  auc={auc:.4f}{tag}")
        print("=" * 60 + "\n")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 9 – Fine-tune pre-trained DroneML models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mobilenetv3", "resnet50", "efficientnetb0"],
        help="Fine-tune only this model (default: all models).",
    )
    args = parser.parse_args()
    finetune(model_filter=args.model)


if __name__ == "__main__":
    main()

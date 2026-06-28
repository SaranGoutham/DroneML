# -*- coding: utf-8 -*-
"""train.py - DroneML
=======================

Phase 9 - Model Training (Baseline)
--------------------------------------
Trains the MobileNetV3-Large baseline model built in Phase 8 on the
thermal mine dataset.

Training Configuration
----------------------
  Optimizer       : Adam (lr=1e-3)
  Loss            : binary_crossentropy
  Batch Size      : 32
  Max Epochs      : 100 (EarlyStopping decides actual cutoff)
  Backbone        : Frozen (only head is trained)

Callbacks
---------
  - EarlyStopping       : patience=10, monitor=val_loss, restore_best_weights
  - ReduceLROnPlateau   : patience=5,  factor=0.2, min_lr=1e-7
  - ModelCheckpoint     : saves best_model.keras on val_loss improvement
  - TensorBoard         : logs to results/logs/

Outputs
-------
  results/best_model.keras        - Best model at lowest val_loss
  results/training_history.pkl    - Full per-epoch loss/metric curves
  results/logs/                   - TensorBoard logs

Usage
-----
  python -m src.models.train
"""

from __future__ import annotations

import sys
import pickle
import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import PROCESSED_DIR, RESULTS_DIR, RANDOM_STATE
from src.preprocessing.create_tf_dataset import create_datasets
from src.models.mobilevnetv3 import build_model, INPUT_SHAPE, LEARNING_RATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Output paths
BEST_MODEL_PATH   = RESULTS_DIR / "best_model.keras"
HISTORY_PATH      = RESULTS_DIR / "training_history.pkl"
TENSORBOARD_DIR   = RESULTS_DIR / "logs"

# Training config
MAX_EPOCHS        = 100
EARLY_STOP_PAT    = 10    # EarlyStopping patience
REDUCE_LR_PAT     = 5     # ReduceLROnPlateau patience
REDUCE_LR_FACTOR  = 0.2   # Multiply LR by this on plateau
MIN_LR            = 1e-7  # Floor for learning rate

# Class weights (computed from train split: Mine=220, No_Mine=30, Total=250)
# Formula: weight_class = total / (num_classes * count_class)
CLASS_WEIGHT: dict = {
    0: 4.1667,   # No_Mine (rare  -> penalise misclassification more)
    1: 0.5682,   # Mine    (common -> penalise less)
}

# ---------------------------------------------------------------------------
# Helper: Compute class weights dynamically from the dataset folder
# ---------------------------------------------------------------------------

def compute_class_weights(train_dir: Path) -> dict:
    """Compute class weights from the training split folder counts.

    Returns
    -------
    dict
        {0: weight_no_mine, 1: weight_mine}
    """
    mine_count    = len(list((train_dir / "Mine").glob("*.png")))
    no_mine_count = len(list((train_dir / "No_Mine").glob("*.png")))
    total         = mine_count + no_mine_count
    num_classes   = 2

    w_no_mine = round(total / (num_classes * no_mine_count), 4)
    w_mine    = round(total / (num_classes * mine_count),    4)

    return {0: w_no_mine, 1: w_mine}


# ---------------------------------------------------------------------------
# Helper: Build callbacks
# ---------------------------------------------------------------------------

def build_callbacks(log_dir: Path, model_path: Path) -> list:
    """Construct the list of Keras callbacks for training.

    Parameters
    ----------
    log_dir:
        Directory for TensorBoard logs.
    model_path:
        File path to save the best model checkpoint.

    Returns
    -------
    list
        List of keras.callbacks instances.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped subdirectory so each run has its own TensorBoard log
    run_id  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = log_dir / run_id

    callbacks = [

        # Stop training when val_loss stops improving
        keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = EARLY_STOP_PAT,
            restore_best_weights = True,   # revert to best epoch at the end
            verbose              = 1,
        ),

        # Reduce learning rate when val_loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = REDUCE_LR_FACTOR,
            patience = REDUCE_LR_PAT,
            min_lr   = MIN_LR,
            verbose  = 1,
        ),

        # Save the model only when val_loss improves
        keras.callbacks.ModelCheckpoint(
            filepath          = str(model_path),
            monitor           = "val_loss",
            save_best_only    = True,
            save_weights_only = False,   # save full model (architecture + weights)
            verbose           = 1,
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir          = str(run_dir),
            histogram_freq   = 1,
            write_graph      = True,
            update_freq      = "epoch",
        ),
    ]

    return callbacks


# ---------------------------------------------------------------------------
# Helper: Print training config
# ---------------------------------------------------------------------------

def print_config(class_weights: dict) -> None:
    """Print a summary of the training configuration."""
    print("\n" + "=" * 60)
    print("  PHASE 9 - MODEL TRAINING : MobileNetV3-Large (Baseline)")
    print("=" * 60)
    print(f"\n  Max Epochs          : {MAX_EPOCHS}")
    print(f"  EarlyStopping pat.  : {EARLY_STOP_PAT}")
    print(f"  ReduceLR patience   : {REDUCE_LR_PAT}")
    print(f"  ReduceLR factor     : {REDUCE_LR_FACTOR}")
    print(f"  Min LR              : {MIN_LR}")
    print(f"  Learning Rate       : {LEARNING_RATE}")
    print(f"  Backbone            : Frozen")
    print(f"  Class Weights       : {class_weights}")
    print(f"  Best Model Path     : {BEST_MODEL_PATH}")
    print(f"  History Path        : {HISTORY_PATH}")
    print(f"  TensorBoard Logs    : {TENSORBOARD_DIR}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Helper: Save training history
# ---------------------------------------------------------------------------

def save_history(history: keras.callbacks.History) -> None:
    """Serialise the training history dictionary to a pickle file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)
    print(f"\n[INFO] Training history saved -> {HISTORY_PATH}")


# ---------------------------------------------------------------------------
# Helper: Print training summary
# ---------------------------------------------------------------------------

def print_summary(history: keras.callbacks.History) -> None:
    """Print a concise summary of the training run."""
    hist      = history.history
    epochs    = len(hist["loss"])
    best_val  = min(hist["val_loss"])
    best_ep   = hist["val_loss"].index(best_val) + 1
    best_acc  = hist["val_accuracy"][best_ep - 1]
    best_auc  = hist["val_auc"][best_ep - 1]

    print("\n" + "-" * 60)
    print("  TRAINING SUMMARY")
    print("-" * 60)
    print(f"  Total Epochs Run    : {epochs}")
    print(f"  Best Epoch          : {best_ep}")
    print(f"  Best val_loss       : {best_val:.4f}")
    print(f"  Best val_accuracy   : {best_acc:.4f}")
    print(f"  Best val_auc        : {best_auc:.4f}")
    print("-" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train() -> keras.callbacks.History:
    """Run the full baseline training pipeline.

    Steps
    -----
    1. Set random seeds for reproducibility.
    2. Compute class weights from training folder.
    3. Load and prepare tf.data.Dataset objects.
    4. Build and compile the model (Phase 8).
    5. Print training config.
    6. Build callbacks.
    7. Run model.fit().
    8. Save training history.
    9. Print summary.

    Returns
    -------
    keras.callbacks.History
        Keras History object containing per-epoch metrics.
    """
    # 1. Reproducibility
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # 2. Class weights (dynamic - computed from actual folder counts)
    train_dir    = PROCESSED_DIR / "train"
    class_weights = compute_class_weights(train_dir)

    # 3. Load datasets
    print("[1/5] Loading datasets ...")
    train_ds, val_ds, test_ds, class_names = create_datasets(PROCESSED_DIR)
    print(f"      Classes : {class_names}")

    # 4. Build model
    print("[2/5] Building model ...")
    model, _backbone = build_model(
        input_shape        = INPUT_SHAPE,
        learning_rate      = LEARNING_RATE,
        trainable_backbone = False,   # baseline: backbone frozen
        random_seed        = RANDOM_STATE,
    )

    # 5. Print config
    print_config(class_weights)

    # 6. Build callbacks
    print("[3/5] Setting up callbacks ...")
    callbacks = build_callbacks(TENSORBOARD_DIR, BEST_MODEL_PATH)

    # 7. Train
    print("[4/5] Training ...\n")
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = MAX_EPOCHS,
        callbacks       = callbacks,
        class_weight    = class_weights,
        verbose         = 1,
    )

    # 8. Save history
    print("[5/5] Saving training history ...")
    save_history(history)

    # 9. Summary
    print_summary(history)

    print("[DONE] Phase 9 complete.")
    print(f"       Best model saved -> {BEST_MODEL_PATH}")
    print("=" * 60 + "\n")

    return history


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    train()


if __name__ == "__main__":
    main()

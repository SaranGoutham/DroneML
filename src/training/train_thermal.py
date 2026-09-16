"""
Master Training Entry Point for DroneML Thermal classification_v2
=================================================================

Trains MobileNetV3-Large 3-class baseline models on canonical classification_v2
using manifest-driven streaming from Protocol 1 or Protocol 2 split files.

Enforces remote-first safety:
- Explicit GPU visibility detection; halts immediately if running on CPU unless
  '--allow-cpu' is explicitly specified for local mock testing.
- Remote Smoke-Test mode ('--smoke-test') for fast, cheap pre-run verification
  on Lightning AI before committing compute credits.
- Supports mixed precision, early stopping, LR scheduling, class weighting,
  model checkpointing, and comprehensive multi-class evaluation.
"""

import os
import sys

# Suppress protobuf version mismatch if present in environment
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"

import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

from src.training.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    INPUT_SHAPE,
    PROTOCOL1_SPLIT_CSV,
    PROTOCOL2_SPLIT_CSV,
    DEFAULT_RESULTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OPTIMIZER,
    DEFAULT_SEED,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_REDUCE_LR_PATIENCE,
    DEFAULT_REDUCE_LR_FACTOR,
    DEFAULT_MIN_LR,
    SMOKE_TEST_EPOCHS,
)
from src.training.dataset import create_manifest_datasets
from src.training.model import build_mobilenetv3_large, unfreeze_backbone_top_layers
from src.training.metrics import (
    compute_comprehensive_metrics,
    format_metrics_summary_table,
    export_evaluation_artifacts,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def verify_gpu_environment(allow_cpu: bool = False) -> None:
    """
    Ensure TensorFlow has direct access to a dedicated GPU.
    Refuses to silently fall back to heavy local CPU execution unless explicitly overridden.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if not allow_cpu:
            sep = "=" * 76
            error_message = (
                f"\n{sep}\n"
                "CRITICAL ERROR: No GPU device detected by TensorFlow!\n\n"
                "Heavy thermal model training on CPU is strictly prohibited to avoid\n"
                "system exhaustion and days of unnecessary execution.\n"
                "Training must run on a remote GPU container (e.g. Lightning AI / CUDA).\n\n"
                "To execute a local CLI syntax check or lightweight unit test, pass:\n"
                "  --allow-cpu\n"
                f"{sep}\n"
            )
            logger.error(error_message)
            sys.exit(1)
        else:
            logger.warning(
                "Running in CPU mode (--allow-cpu passed). Full training on CPU is strongly discouraged."
            )
    else:
        logger.info(f"Verified GPU visibility: {len(gpus)} physical device(s) found:")
        for gpu in gpus:
            logger.info(f"  -> {gpu.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DroneML thermal classification_v2 Remote Training Pipeline"
    )
    # Split selection
    parser.add_argument(
        "--protocol",
        choices=["protocol1", "protocol2"],
        default="protocol1",
        help="Evaluation protocol: 'protocol1' (instance-disjoint) or 'protocol2' (flight-holdout)",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Optional custom path to split manifest CSV (overrides --protocol default)",
    )

    # Core hyperparameters
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default=DEFAULT_OPTIMIZER,
        help="Optimizer algorithm",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Class weighting
    parser.add_argument(
        "--class-weights",
        choices=["balanced", "none"],
        default="balanced",
        help="Apply balanced inverse-frequency class weights (recommended for Protocol 1)",
    )

    # Optimization & Regularization
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--reduce-lr-patience", type=int, default=DEFAULT_REDUCE_LR_PATIENCE)
    parser.add_argument("--reduce-lr-factor", type=float, default=DEFAULT_REDUCE_LR_FACTOR)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed_float16 computation for faster GPU throughput",
    )

    # Output & Checkpoints
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save models, checkpoints, logs, and evaluation metrics",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a pre-existing checkpoint to resume training from",
    )

    # Verification & Remote Safety
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute ultra-fast remote smoke test (few batches, 2 epochs) to verify pipeline before spending credits",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Explicitly permit CPU execution (for mock testing/debugging only)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training augmentations",
    )

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute complete thermal training, checkpointing, and evaluation workflow."""
    start_time = time.time()
    logger.info("=" * 76)
    logger.info("  DRONEML THERMAL CLASSIFICATION_V2 TRAINING PIPELINE")
    logger.info("=" * 76)

    # 1. Environment & Hardware Validation
    verify_gpu_environment(allow_cpu=args.allow_cpu)

    if args.mixed_precision:
        policy = keras.mixed_precision.Policy("mixed_float16")
        keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Mixed precision policy activated: {policy.compute_dtype}/{policy.variable_dtype}")

    # Set seeds
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # 2. Select Split Manifest
    if args.split_csv is not None:
        split_path = args.split_csv
    elif args.protocol == "protocol1":
        split_path = PROTOCOL1_SPLIT_CSV
    else:
        split_path = PROTOCOL2_SPLIT_CSV

    logger.info(f"Target Evaluation Protocol: {args.protocol.upper()} ({split_path.name})")

    # 3. Setup Results Directories
    results_dir: Path = args.results_dir
    checkpoints_dir = results_dir / "checkpoints"
    models_dir = results_dir / "models"
    history_dir = results_dir / "history"
    eval_dir = results_dir / "evaluation"

    for d in [checkpoints_dir, models_dir, history_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 4. Build Dataset Pipelines
    logger.info("Initializing manifest-driven tf.data streaming pipelines...")
    train_ds, val_ds, test_ds, meta = create_manifest_datasets(
        split_csv_path=split_path,
        batch_size=args.batch_size,
        augment_train=not args.no_augment,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )

    logger.info(
        f"Dataset Ready: Train: {meta['train_samples']:,} samples | "
        f"Val: {meta['val_samples']:,} samples | Test: {meta['test_samples']:,} samples"
    )

    # Class Weights setup
    if args.class_weights == "balanced":
        class_weights = meta["class_weights"]
        logger.info(f"Applying balanced class weights: {class_weights}")
    else:
        class_weights = None
        logger.info("Class weights disabled (uniform loss weighting).")

    # 5. Model Initialization
    if args.resume is not None:
        logger.info(f"Loading checkpoint to resume: {args.resume}")
        model = keras.models.load_model(args.resume)
    else:
        logger.info("Building MobileNetV3-Large baseline model (ImageNet weights, 3 classes)...")
        model = build_mobilenetv3_large(
            num_classes=NUM_CLASSES,
            input_shape=INPUT_SHAPE,
            weights="imagenet",
            learning_rate=args.lr,
            optimizer_name=args.optimizer,
            random_seed=args.seed,
        )

    # 6. Callbacks Configuration
    epochs = SMOKE_TEST_EPOCHS if args.smoke_test else args.epochs
    best_checkpoint_path = checkpoints_dir / "best_model.keras"
    history_csv_path = history_dir / "training_log.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience if not args.smoke_test else 2,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.min_lr,
            verbose=1,
        ),
        CSVLogger(filename=str(history_csv_path)),
    ]

    # Save run configuration
    run_config = {
        "protocol": args.protocol,
        "split_manifest": str(split_path),
        "smoke_test": args.smoke_test,
        "batch_size": args.batch_size,
        "epochs": epochs,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "class_weights": class_weights,
        "seed": args.seed,
        "mixed_precision": args.mixed_precision,
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
    }
    with open(results_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    # 7. Model Training
    logger.info(f"Starting training run ({epochs} epochs)...")
    train_start = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    train_duration = time.time() - train_start
    logger.info(f"Training loop completed in {train_duration:.2f} seconds ({train_duration/60:.2f} min).")

    # Save final model
    final_model_path = models_dir / "final_model.keras"
    model.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # Export history to JSON
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=4)

    # 8. Evaluation on Test Set
    logger.info("Evaluating best model on held-out test partition...")
    if best_checkpoint_path.exists():
        eval_model = keras.models.load_model(best_checkpoint_path)
    else:
        eval_model = model

    all_preds = []
    all_targets = []
    for imgs, lbls in test_ds:
        preds = eval_model.predict_on_batch(imgs)
        all_preds.append(preds)
        all_targets.append(lbls.numpy())

    if all_preds:
        y_pred_probs = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        eval_results = compute_comprehensive_metrics(
            y_true=y_true,
            y_pred_probs=y_pred_probs,
            class_names=CLASS_NAMES,
        )

        export_evaluation_artifacts(eval_results, eval_dir)
        summary_display = format_metrics_summary_table(eval_results)
        print(summary_display)
    else:
        eval_results = {}
        logger.warning("No test predictions collected.")

    total_time = time.time() - start_time
    logger.info(f"Complete run finished in {total_time:.2f} s ({total_time/60:.2f} min).")
    return {
        "status": "COMPLETED",
        "run_config": run_config,
        "evaluation": eval_results,
        "train_duration_sec": train_duration,
        "total_duration_sec": total_time,
    }


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

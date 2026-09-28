"""
Kaggle Training Entry Point for DroneML Thermal classification_v2

Trains MobileNetV3-Large 3-class baseline models on canonical
classification_v2 using manifest-driven streaming from Protocol 1
or Protocol 2 split files.

Kaggle-specific support:
- Explicit --data-root for the Kaggle dataset
- Automatic multi-GPU support through MirroredStrategy
- Mixed precision support
- GPU visibility validation
- Smoke-test mode
- Early stopping
- Learning-rate scheduling
- Class weighting
- Model checkpointing
- Comprehensive multi-class evaluation

Expected Kaggle dataset:

/kaggle/input/datasets/sarangoutham/mineinsight-preprocessed/
├── classification_v2/
│   ├── Mine/
│   ├── Known/
│   └── Background/
└── reports_v2/
    ├── protocol1_split.csv
    └── protocol2_split.csv
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Suppress protobuf version mismatch if present in environment
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"

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

from src.training.model import (
    build_mobilenetv3_large,
    unfreeze_backbone_top_layers,
)

from src.training.metrics import (
    compute_comprehensive_metrics,
    format_metrics_summary_table,
    export_evaluation_artifacts,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU / Hardware Validation
# ---------------------------------------------------------------------------

def verify_gpu_environment(allow_cpu: bool = False) -> None:
    """
    Ensure TensorFlow has direct access to GPU(s).

    Kaggle is expected to provide 2x NVIDIA T4 GPUs.
    """

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        if not allow_cpu:
            separator = "=" * 76

            error_message = (
                f"\n{separator}\n"
                "CRITICAL ERROR: No GPU device detected by TensorFlow!\n\n"
                "Training is configured for remote GPU execution.\n"
                "CPU training is disabled unless --allow-cpu is explicitly used.\n\n"
                "Expected environment: Kaggle GPU runtime.\n"
                f"{separator}"
            )

            logger.error(error_message)
            sys.exit(1)

        logger.warning(
            "Running in CPU mode (--allow-cpu passed). "
            "Full training on CPU is strongly discouraged."
        )

    else:
        logger.info(
            f"Verified GPU visibility: {len(gpus)} physical device(s) found:"
        )

        for gpu in gpus:
            logger.info(f"  -> {gpu.name}")


# ---------------------------------------------------------------------------
# Multi-GPU Strategy
# ---------------------------------------------------------------------------

def create_distribution_strategy() -> tf.distribute.Strategy:
    """
    Create the appropriate TensorFlow distribution strategy.

    If multiple GPUs are available, MirroredStrategy is used so that
    training is distributed across all visible GPUs.

    With Kaggle 2x T4:

        T4 GPU 0
            +
        T4 GPU 1
            ↓
        MirroredStrategy
            ↓
        synchronized training
    """

    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()

        logger.info(
            "Using TensorFlow MirroredStrategy across "
            f"{strategy.num_replicas_in_sync} GPUs."
        )

        return strategy

    if len(gpus) == 1:
        strategy = tf.distribute.get_strategy()

        logger.info(
            "Using single-GPU TensorFlow strategy."
        )

        return strategy

    logger.warning(
        "No GPU detected. Using default TensorFlow strategy."
    )

    return tf.distribute.get_strategy()


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DroneML thermal classification_v2 Kaggle training pipeline"
    )

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root directory of the MineInsight dataset. "
            "For Kaggle use: "
            "/kaggle/input/datasets/sarangoutham/mineinsight-preprocessed"
        ),
    )

    # -----------------------------------------------------------------------
    # Split selection
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--protocol",
        choices=["protocol1", "protocol2"],
        default="protocol1",
        help=(
            "Evaluation protocol: "
            "'protocol1' (instance-disjoint) or "
            "'protocol2' (flight-holdout)"
        ),
    )

    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help=(
            "Optional custom path to split manifest CSV "
            "(overrides --protocol default)"
        ),
    )

    # -----------------------------------------------------------------------
    # Core hyperparameters
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default=DEFAULT_OPTIMIZER,
        help="Optimizer algorithm",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    # -----------------------------------------------------------------------
    # Class weighting
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--class-weights",
        choices=["balanced", "none"],
        default="balanced",
        help=(
            "Apply balanced inverse-frequency class weights "
            "(recommended for Protocol 1)"
        ),
    )

    # -----------------------------------------------------------------------
    # Optimization / Regularization
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
    )

    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=DEFAULT_REDUCE_LR_PATIENCE,
    )

    parser.add_argument(
        "--reduce-lr-factor",
        type=float,
        default=DEFAULT_REDUCE_LR_FACTOR,
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=DEFAULT_MIN_LR,
    )

    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed_float16 computation for faster T4 GPU throughput",
    )

    # -----------------------------------------------------------------------
    # Output / Checkpoints
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory to save models, checkpoints, logs, "
            "and evaluation metrics"
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a pre-existing checkpoint to resume training from",
    )

    # -----------------------------------------------------------------------
    # Verification / Safety
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Execute ultra-fast smoke test "
            "to verify the pipeline before full training"
        ),
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Explicitly permit CPU execution (for debugging only)",
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training augmentations",
    )

    # IMPORTANT FOR KAGGLE/JUPYTER:
    # Ignore arguments injected by the notebook kernel,
    # such as:
    # -f /root/.local/share/jupyter/runtime/kernel-xxxx.json
    args, unknown = parser.parse_known_args()

    return args


# ---------------------------------------------------------------------------
# Dataset Root Resolution
# ---------------------------------------------------------------------------

def resolve_dataset_root(
    args: argparse.Namespace,
) -> Path:
    """
    Resolve the dataset root.

    Priority:

    1. --data-root if explicitly provided.
    2. Kaggle dataset path if it exists.
    3. Existing repository DATA_DIR from config.py.
    """

    if args.data_root is not None:
        data_root = args.data_root

    else:

        kaggle_default = Path(
            "/kaggle/input/datasets/"
            "sarangoutham/mineinsight-preprocessed"
        )

        if kaggle_default.exists():
            data_root = kaggle_default

        else:
            from src.training.config import DATA_DIR

            data_root = DATA_DIR

    data_root = Path(data_root).resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist:\n{data_root}"
        )

    logger.info(
        f"Using dataset root: {data_root}"
    )

    return data_root


# ---------------------------------------------------------------------------
# Dataset Validation
# ---------------------------------------------------------------------------

def validate_dataset_structure(
    data_root: Path,
) -> None:
    """
    Validate the expected MineInsight dataset structure.
    """

    required_paths = [
        data_root / "classification_v2",
        data_root / "classification_v2" / "Mine",
        data_root / "classification_v2" / "Known",
        data_root / "classification_v2" / "Background",
        data_root / "reports_v2",
        data_root / "reports_v2" / "protocol1_split.csv",
        data_root / "reports_v2" / "protocol2_split.csv",
    ]

    logger.info("Validating dataset structure...")

    missing = []

    for path in required_paths:

        if path.exists():
            logger.info(f"  OK   {path}")

        else:
            logger.error(f"  MISS {path}")
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Dataset validation failed. "
            f"{len(missing)} required path(s) are missing."
        )

    logger.info("Dataset structure validated successfully.")


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

def run_training(
    args: argparse.Namespace,
) -> Dict[str, Any]:

    start_time = time.time()

    logger.info("=" * 76)
    logger.info(
        "  DRONEML THERMAL CLASSIFICATION_V2 TRAINING PIPELINE"
    )
    logger.info("=" * 76)

    # -----------------------------------------------------------------------
    # 1. Environment / Hardware Validation
    # -----------------------------------------------------------------------

    verify_gpu_environment(
        allow_cpu=args.allow_cpu
    )

    # -----------------------------------------------------------------------
    # 2. Dataset Root
    # -----------------------------------------------------------------------

    data_root = resolve_dataset_root(args)

    validate_dataset_structure(data_root)

    # -----------------------------------------------------------------------
    # 3. Mixed Precision
    # -----------------------------------------------------------------------

    if args.mixed_precision:

        policy = keras.mixed_precision.Policy(
            "mixed_float16"
        )

        keras.mixed_precision.set_global_policy(
            policy
        )

        logger.info(
            "Mixed precision policy activated: "
            f"{policy.compute_dtype}/{policy.variable_dtype}"
        )

    # -----------------------------------------------------------------------
    # 4. Seeds
    # -----------------------------------------------------------------------

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # 5. Select Split Manifest
    # -----------------------------------------------------------------------

    if args.split_csv is not None:

        split_path = args.split_csv

    elif args.protocol == "protocol1":

        split_path = (
            data_root
            / "reports_v2"
            / "protocol1_split.csv"
        )

    else:

        split_path = (
            data_root
            / "reports_v2"
            / "protocol2_split.csv"
        )

    split_path = Path(split_path).resolve()

    if not split_path.exists():

        raise FileNotFoundError(
            f"Split manifest not found:\n{split_path}"
        )

    logger.info(
        f"Target Evaluation Protocol: "
        f"{args.protocol.upper()} "
        f"({split_path.name})"
    )

    # -----------------------------------------------------------------------
    # 6. Setup Results Directories
    # -----------------------------------------------------------------------

    results_dir: Path = args.results_dir

    checkpoints_dir = (
        results_dir / "checkpoints"
    )

    models_dir = (
        results_dir / "models"
    )

    history_dir = (
        results_dir / "history"
    )

    eval_dir = (
        results_dir / "evaluation"
    )

    for directory in [
        checkpoints_dir,
        models_dir,
        history_dir,
        eval_dir,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    logger.info(
        f"Results directory: {results_dir.resolve()}"
    )

    # -----------------------------------------------------------------------
    # 7. Build Dataset Pipelines
    # -----------------------------------------------------------------------

    logger.info(
        "Initializing manifest-driven "
        "tf.data streaming pipelines..."
    )

    train_ds, val_ds, test_ds, meta = (
        create_manifest_datasets(
            split_csv_path=split_path,
            batch_size=args.batch_size,
            augment_train=not args.no_augment,
            seed=args.seed,
            smoke_test=args.smoke_test,
            data_root=data_root,
        )
    )

    logger.info(
        f"Dataset Ready: "
        f"Train: {meta['train_samples']:,} samples | "
        f"Val: {meta['val_samples']:,} samples | "
        f"Test: {meta['test_samples']:,} samples"
    )

    # -----------------------------------------------------------------------
    # 8. Class Weights
    # -----------------------------------------------------------------------

    if args.class_weights == "balanced":

        class_weights = meta["class_weights"]

        logger.info(
            f"Applying balanced class weights: "
            f"{class_weights}"
        )

    else:

        class_weights = None

        logger.info(
            "Class weights disabled "
            "(uniform loss weighting)."
        )

    # -----------------------------------------------------------------------
    # 9. Distribution Strategy
    # -----------------------------------------------------------------------

    strategy = create_distribution_strategy()

    # -----------------------------------------------------------------------
    # 10. Model Initialization
    # -----------------------------------------------------------------------

    with strategy.scope():

        if args.resume is not None:

            logger.info(
                f"Loading checkpoint to resume: "
                f"{args.resume}"
            )

            model = keras.models.load_model(
                args.resume
            )

        else:

            logger.info(
                "Building MobileNetV3-Large baseline model "
                "(ImageNet weights, 3 classes)..."
            )

            model = build_mobilenetv3_large(
                num_classes=NUM_CLASSES,
                input_shape=INPUT_SHAPE,
                weights="imagenet",
                learning_rate=args.lr,
                optimizer_name=args.optimizer,
                random_seed=args.seed,
            )

    # -----------------------------------------------------------------------
    # 11. Callbacks
    # -----------------------------------------------------------------------

    epochs = (
        SMOKE_TEST_EPOCHS
        if args.smoke_test
        else args.epochs
    )

    best_checkpoint_path = (
        checkpoints_dir / "best_model.keras"
    )

    history_csv_path = (
        history_dir / "training_log.csv"
    )

    callbacks = [

        ModelCheckpoint(
            filepath=str(
                best_checkpoint_path
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=(
                args.early_stopping_patience
                if not args.smoke_test
                else 2
            ),
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

        CSVLogger(
            filename=str(
                history_csv_path
            )
        ),
    ]

    # -----------------------------------------------------------------------
    # 12. Save Run Configuration
    # -----------------------------------------------------------------------

    run_config = {

        "platform": "kaggle",

        "dataset_root": str(
            data_root
        ),

        "protocol": args.protocol,

        "split_manifest": str(
            split_path
        ),

        "smoke_test": args.smoke_test,

        "batch_size": args.batch_size,

        "epochs": epochs,

        "learning_rate": args.lr,

        "optimizer": args.optimizer,

        "class_weights": class_weights,

        "seed": args.seed,

        "mixed_precision": args.mixed_precision,

        "input_shape": list(
            INPUT_SHAPE
        ),

        "num_classes": NUM_CLASSES,

        "class_names": CLASS_NAMES,

        "num_gpus": len(
            tf.config.list_physical_devices(
                "GPU"
            )
        ),

        "num_replicas": (
            strategy.num_replicas_in_sync
        ),
    }

    with open(
        results_dir / "run_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_config,
            f,
            indent=4,
        )

    # -----------------------------------------------------------------------
    # 13. Model Training
    # -----------------------------------------------------------------------

    logger.info(
        f"Starting training run "
        f"({epochs} epochs)..."
    )

    train_start = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    train_duration = (
        time.time() - train_start
    )

    logger.info(
        "Training loop completed in "
        f"{train_duration:.2f} seconds "
        f"({train_duration / 60:.2f} min)."
    )

    # -----------------------------------------------------------------------
    # 14. Save Final Model
    # -----------------------------------------------------------------------

    final_model_path = (
        models_dir / "final_model.keras"
    )

    model.save(
        final_model_path
    )

    logger.info(
        f"Saved final model to "
        f"{final_model_path}"
    )

    # -----------------------------------------------------------------------
    # 15. Export Training History
    # -----------------------------------------------------------------------

    hist_dict = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    with open(
        history_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            hist_dict,
            f,
            indent=4,
        )

    # -----------------------------------------------------------------------
    # 16. Evaluation
    # -----------------------------------------------------------------------

    logger.info(
        "Evaluating best model on "
        "held-out test partition..."
    )

    if best_checkpoint_path.exists():

        eval_model = keras.models.load_model(
            best_checkpoint_path
        )

    else:

        eval_model = model

    all_preds = []
    all_targets = []

    for imgs, lbls in test_ds:

        preds = eval_model.predict_on_batch(
            imgs
        )

        all_preds.append(
            preds
        )

        all_targets.append(
            lbls.numpy()
        )

    if all_preds:

        y_pred_probs = np.concatenate(
            all_preds,
            axis=0,
        )

        y_true = np.concatenate(
            all_targets,
            axis=0,
        )

        eval_results = (
            compute_comprehensive_metrics(
                y_true=y_true,
                y_pred_probs=y_pred_probs,
                class_names=CLASS_NAMES,
            )
        )

        export_evaluation_artifacts(
            eval_results,
            eval_dir,
        )

        summary_display = (
            format_metrics_summary_table(
                eval_results
            )
        )

        print(summary_display)

    else:

        eval_results = {}

        logger.warning(
            "No test predictions collected."
        )

    # -----------------------------------------------------------------------
    # 17. Complete
    # -----------------------------------------------------------------------

    total_time = (
        time.time() - start_time
    )

    logger.info(
        "Complete run finished in "
        f"{total_time:.2f} seconds "
        f"({total_time / 60:.2f} min)."
    )

    logger.info(
        f"Results saved to: "
        f"{results_dir.resolve()}"
    )

    return {

        "status": "COMPLETED",

        "run_config": run_config,

        "evaluation": eval_results,

        "train_duration_sec": (
            train_duration
        ),

        "total_duration_sec": (
            total_time
        ),
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
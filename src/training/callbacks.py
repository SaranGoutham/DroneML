"""callbacks.py

Utility module for creating TensorFlow/Keras training callbacks.

This module is deliberately lightweight: it imports configuration values from
`src.preprocessing.config` and assembles a list of callbacks required during
model training. It does **not** build models or load data, making it reusable
by any training script (e.g., `train.py`, fine‑tuning scripts, or notebooks).

The public API consists of a single function:

    create_callbacks() -> list[tf.keras.callbacks.Callback]

which returns a ready‑to‑use list of callbacks.
"""

import logging
from pathlib import Path
from typing import List

import tensorflow as tf

# Import configuration values (paths and training hyper‑parameters)
from src.preprocessing.config import (
    BEST_MODEL_PATH,
    TENSORBOARD_LOG_DIR,
    TRAINING_LOG_CSV,
    PATIENCE,
)

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    """Create the parent directory for *path* if it does not exist.

    Args:
        path: The file path whose parent directory should exist.
    """
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory %s", parent)


def create_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Create and return the list of Keras training callbacks.

    The callbacks are configured using values imported from
    ``src.preprocessing.config``. All required directories are created
    beforehand. If any callback cannot be instantiated, the exception is logged
    and re‑raised.

    Returns:
        List[tf.keras.callbacks.Callback]: A list containing the following
        callbacks in order:

        1. ``EarlyStopping`` – stops training when ``val_loss`` stops improving.
        2. ``ModelCheckpoint`` – saves the best model checkpoint based on
           ``val_loss``.
        3. ``ReduceLROnPlateau`` – reduces the learning rate when
           ``val_loss`` plateaus.
        4. ``TensorBoard`` – writes logs for TensorBoard visualisation.
        5. ``CSVLogger`` – records epoch‑wise metrics to a CSV file.
    """
    try:
        logger.info("Creating callbacks…")

        # EarlyStopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        )
        logger.info("✓ EarlyStopping")

        # ModelCheckpoint
        best_path = Path(BEST_MODEL_PATH)
        _ensure_dir(best_path)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        )
        logger.info("✓ ModelCheckpoint")

        # ReduceLROnPlateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=max(2, PATIENCE // 2),
            min_lr=1e-7,
        )
        logger.info("✓ ReduceLROnPlateau")

        # TensorBoard
        tb_dir = Path(TENSORBOARD_LOG_DIR)
        _ensure_dir(tb_dir)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(tb_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        )
        logger.info("✓ TensorBoard")

        # CSVLogger
        csv_path = Path(TRAINING_LOG_CSV)
        _ensure_dir(csv_path)
        csv_logger = tf.keras.callbacks.CSVLogger(
            filename=str(csv_path),
            append=False,
        )
        logger.info("✓ CSVLogger")

        logger.info("Callbacks Ready")
        return [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger]
    except Exception as exc:
        logger.exception("Failed to create callbacks: %s", exc)
        raise

__all__ = ["create_callbacks"]

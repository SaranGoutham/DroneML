# create_tf_dataset.py – DroneML Specification
"""
Utility module to build TensorFlow `tf.data.Dataset` pipelines from the verified
DroneML dataset.

The module loads the images from the standard processed directory structure,
normalizes them to the range [0, 1], optionally applies the training augmentation
pipeline, caches and prefetches the data for optimal performance, and provides
convenient helpers to inspect the created datasets.

No image resizing is performed because all images are already 224×224.
"""

from __future__ import annotations

import tensorflow as tf
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The training augmentation pipeline defined in `augment.py`.
# Import is delayed until runtime to avoid circular imports when the module is
# executed as a script.
from src.preprocessing.augment import build_train_augmentation
from src.preprocessing.config import IMAGE_SIZE, PROCESSED_DIR, RANDOM_STATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Build augmentation pipeline once
TRAIN_AUGMENTATION = build_train_augmentation()

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_split(split_path: Path, *, shuffle: bool = False) -> tf.data.Dataset:
    """
    Load a dataset split from ``split_path`` using ``tf.keras.utils.image_dataset_from_directory``.

    Parameters
    ----------
    split_path:
        Path to the split directory (e.g. ``processed_dataset/train``).
    shuffle:
        Whether to shuffle the dataset. ``True`` for the training split,
        ``False`` for validation and test splits.

    Returns
    -------
    tf.data.Dataset
        A dataset yielding ``(image, label)`` batches.
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory=str(split_path),
        labels="inferred",
        label_mode="binary",  # Mine = 1, No_Mine = 0
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=RANDOM_STATE,
        validation_split=None,
        subset=None,
    )

def prepare_dataset(dataset: tf.data.Dataset, *, training: bool = False) -> tf.data.Dataset:
    """Normalize, optionally augment, cache and prefetch a dataset.

    The pipeline applied is:
        1. Cast to ``tf.float32`` and divide by 255 → range ``[0, 1]``.
        2. If ``training`` is ``True``, apply the augmentation pipeline
           returned by :func:`build_train_augmentation`.
        3. ``cache()`` – caches the dataset in memory for faster reuse.
        4. ``prefetch(AUTOTUNE)`` – overlaps preprocessing and training.
    """
    # 1. Normalization
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE,
    )

    # 2. Optional augmentation (training only)
    if training:
        dataset = dataset.map(
            lambda x, y: (TRAIN_AUGMENTATION(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # 3. Cache and prefetch
    dataset = dataset.cache().prefetch(AUTOTUNE)
    return dataset

def create_datasets(base_path: Path = PROCESSED_DIR) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset,List[str]]:
    """Create training, validation and test datasets.

    The function loads each split, applies the appropriate preprocessing/augmentation
    and returns three ready‑to‑use ``tf.data.Dataset`` objects.
    """
    train_path = base_path / "train"
    val_path = base_path / "validation"
    test_path = base_path / "test"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset split not found: {path}")

    # Load raw datasets
    train_raw = load_split(train_path, shuffle=True)
    class_names = train_raw.class_names
    val_raw = load_split(val_path, shuffle=False)
    test_raw = load_split(test_path, shuffle=False)

    # Apply preparation pipelines
    train_ds = prepare_dataset(train_raw, training=True)
    val_ds = prepare_dataset(val_raw, training=False)
    test_ds = prepare_dataset(test_raw, training=False)

    return train_ds, val_ds, test_ds, class_names

def print_dataset_info(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset,class_names: List[str]) -> None:
    """Print a concise summary of the created datasets.
    """
    print("=" * 39)
    print("TF DATASET CREATION")
    print("=" * 39)
    print(f"Batch Size      : {BATCH_SIZE}")
    print(f"Image Size      : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    # Class names are inferred from the directory structure; we can extract them
    # from one of the raw datasets (they share the same ordering).
    print(f"Class Names     : {class_names}")
    print(f"Number of Classes : {len(class_names)}")
    print(f"Train Batches   : {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Validation Batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Test Batches    : {tf.data.experimental.cardinality(test_ds).numpy()}")

def main() -> None:
    """Run the full dataset creation pipeline and display information.
    """
    base_path = PROCESSED_DIR
    train_ds, val_ds, test_ds,class_names = create_datasets(base_path)
    print_dataset_info(train_ds, val_ds, test_ds,class_names)
    print("Dataset Ready")

if __name__ == "__main__":
    main()

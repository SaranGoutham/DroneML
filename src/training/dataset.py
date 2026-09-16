"""
Manifest-Driven tf.data Dataset Pipeline for classification_v2
==============================================================

Streams 224x224 RGB crops directly from split manifests (protocol1_split.csv
or protocol2_split.csv) without dataset duplication. Implements high-throughput
tf.data prefetching, parallel decoding, augmentation, and class weighting.
"""

import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import tensorflow as tf

from src.training.config import (
    DATA_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IMAGE_SIZE,
    SMOKE_TEST_BATCHES_TRAIN,
    SMOKE_TEST_BATCHES_VAL,
    SMOKE_TEST_BATCHES_TEST,
)

logger = logging.getLogger(__name__)


def load_split_manifest(
    split_csv_path: Path,
    split_name: str,
    data_root: Path = DATA_DIR,
) -> Tuple[List[str], List[int]]:
    """
    Parse a split CSV and extract absolute image paths and class integer labels.
    
    Args:
        split_csv_path: Path to protocol1_split.csv or protocol2_split.csv.
        split_name: Target partition ('train', 'validation', or 'test').
        data_root: Root data directory where 'classification_v2/...' is located.
        
    Returns:
        (file_paths, labels)
    """
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_csv_path}")

    file_paths: List[str] = []
    labels: List[int] = []

    with open(split_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"].lower() != split_name.lower():
                continue
            cat = row["category"]
            if cat not in CLASS_TO_IDX:
                continue

            rel_path = row["crop_rel_path"].replace("\\", "/")
            full_path = data_root / rel_path

            file_paths.append(str(full_path))
            labels.append(CLASS_TO_IDX[cat])

    logger.info(
        f"Loaded {len(file_paths):,} samples for split '{split_name}' from {split_csv_path.name}"
    )
    return file_paths, labels


def compute_balanced_class_weights(
    labels: List[int],
    num_classes: int = len(CLASS_NAMES),
) -> Dict[int, float]:
    """
    Compute balanced inverse-frequency class weights:
    w_c = N / (K * N_c)
    
    Ensures that gradient updates are normalized so under-represented classes
    receive proportionally larger loss penalties.
    """
    total_samples = len(labels)
    if total_samples == 0:
        return {c: 1.0 for c in range(num_classes)}

    counts = Counter(labels)
    weights: Dict[int, float] = {}

    for c in range(num_classes):
        c_count = counts.get(c, 0)
        if c_count > 0:
            w = total_samples / (float(num_classes) * float(c_count))
        else:
            w = 1.0
        weights[c] = round(w, 4)

    logger.info(f"Computed balanced class weights: {weights}")
    return weights


def parse_and_decode_image(
    file_path: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read JPEG from disk, decode as 3-channel RGB float32 in [0, 255]."""
    img_bytes = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def augment_image(
    image: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply safe thermal data augmentations preserving 1:1 geometry."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=15.0)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def build_tf_dataset(
    file_paths: List[str],
    labels: List[int],
    batch_size: int = 64,
    is_training: bool = True,
    augment: bool = True,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    smoke_test_limit: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Construct an optimized tf.data.Dataset pipeline.
    
    Args:
        file_paths: List of absolute image paths.
        labels: Corresponding integer labels in [0, NUM_CLASSES - 1].
        batch_size: Batch size.
        is_training: True for training set (enables shuffle).
        augment: Apply augmentations if training.
        shuffle_buffer: Shuffle buffer size.
        seed: Random seed.
        smoke_test_limit: If set, restricts dataset to a tiny slice for smoke-testing.
    """
    if smoke_test_limit is not None and smoke_test_limit > 0:
        file_paths = file_paths[:smoke_test_limit]
        labels = labels[:smoke_test_limit]

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        ds = ds.shuffle(
            buffer_size=min(len(file_paths), shuffle_buffer),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    # Parallel decode & resize
    ds = ds.map(
        parse_and_decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training,
    )

    if is_training and augment:
        ds = ds.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_manifest_datasets(
    split_csv_path: Path,
    batch_size: int = 64,
    augment_train: bool = True,
    seed: int = 42,
    smoke_test: bool = False,
    data_root: Path = DATA_DIR,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]:
    """
    Create train, validation, and test tf.data.Datasets directly from split CSV.
    
    Returns:
        (train_ds, val_ds, test_ds, metadata_summary)
    """
    train_paths, train_labels = load_split_manifest(split_csv_path, "train", data_root)
    val_paths, val_labels = load_split_manifest(split_csv_path, "validation", data_root)
    test_paths, test_labels = load_split_manifest(split_csv_path, "test", data_root)

    train_limit = (SMOKE_TEST_BATCHES_TRAIN * batch_size) if smoke_test else None
    val_limit = (SMOKE_TEST_BATCHES_VAL * batch_size) if smoke_test else None
    test_limit = (SMOKE_TEST_BATCHES_TEST * batch_size) if smoke_test else None

    if smoke_test:
        logger.info(
            f"SMOKE TEST MODE: Slicing datasets (train: {train_limit}, val: {val_limit}, test: {test_limit})"
        )

    train_ds = build_tf_dataset(
        train_paths,
        train_labels,
        batch_size=batch_size,
        is_training=True,
        augment=augment_train,
        seed=seed,
        smoke_test_limit=train_limit,
    )

    val_ds = build_tf_dataset(
        val_paths,
        val_labels,
        batch_size=batch_size,
        is_training=False,
        augment=False,
        seed=seed,
        smoke_test_limit=val_limit,
    )

    test_ds = build_tf_dataset(
        test_paths,
        test_labels,
        batch_size=batch_size,
        is_training=False,
        augment=False,
        seed=seed,
        smoke_test_limit=test_limit,
    )

    class_weights = compute_balanced_class_weights(train_labels)

    metadata = {
        "split_csv": str(split_csv_path),
        "smoke_test": smoke_test,
        "batch_size": batch_size,
        "train_samples": len(train_paths) if not smoke_test else min(len(train_paths), train_limit or 0),
        "val_samples": len(val_paths) if not smoke_test else min(len(val_paths), val_limit or 0),
        "test_samples": len(test_paths) if not smoke_test else min(len(test_paths), test_limit or 0),
        "train_class_counts": dict(Counter(train_labels)),
        "class_weights": class_weights,
        "class_names": CLASS_NAMES,
    }

    return train_ds, val_ds, test_ds, metadata

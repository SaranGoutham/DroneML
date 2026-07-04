# -*- coding: utf-8 -*-
"""Shared training helpers for DroneML model runners."""

from __future__ import annotations

import datetime
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.config import (
    EFFICIENTNETB0_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    FINE_TUNE_UNFREEZE_LAYERS,
    IMAGE_SIZE,
    MIN_LEARNING_RATE,
    MOBILENETV3_LEARNING_RATE,
    PATIENCE,
    RANDOM_STATE,
    REDUCE_LR_FACTOR,
    RESNET50_LEARNING_RATE,
    TENSORBOARD_UPDATE_FREQ,
)
from src.models.efficientnetb0 import (
    INPUT_SHAPE as EFFICIENTNETB0_INPUT_SHAPE,
    build_model as build_efficientnetb0,
    unfreeze_top_layers as efficientnetb0_unfreeze_top,
)
from src.models.mobilenetv3 import (
    INPUT_SHAPE as MOBILENETV3_INPUT_SHAPE,
    build_model as build_mobilenetv3,
    unfreeze_top_layers as mobilenetv3_unfreeze_top,
)
from src.models.resnet50 import (
    INPUT_SHAPE as RESNET50_INPUT_SHAPE,
    build_model as build_resnet50,
    unfreeze_top_layers as resnet50_unfreeze_top,
)
from src.preprocessing.augment import build_train_augmentation

logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")


@dataclass(frozen=True)
class ModelSpec:
    """Static information required to train one architecture."""

    key: str
    display_name: str
    build_fn: Callable[..., tuple[keras.Model, keras.Model]]
    input_shape: tuple[int, int, int]
    unfreeze_fn: Callable[[keras.Model, int], None]
    learning_rate: float


@dataclass(frozen=True)
class RunPaths:
    """Output locations for one dataset-specific experiment family."""

    root: Path
    checkpoints: Path
    models: Path
    history: Path
    logs: Path
    evaluation: Path
    comparison: Path

    @classmethod
    def from_root(cls, root: Path) -> "RunPaths":
        paths = cls(
            root=root,
            checkpoints=root / "checkpoints",
            models=root / "models",
            history=root / "history",
            logs=root / "logs",
            evaluation=root / "evaluation",
            comparison=root / "comparison",
        )
        for path in (
            paths.root,
            paths.checkpoints,
            paths.models,
            paths.history,
            paths.logs,
            paths.evaluation,
            paths.comparison,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return paths


MODEL_SPECS: dict[str, ModelSpec] = {
    "mobilenetv3": ModelSpec(
        key="mobilenetv3",
        display_name="MobileNetV3-Large",
        build_fn=build_mobilenetv3,
        input_shape=MOBILENETV3_INPUT_SHAPE,
        unfreeze_fn=mobilenetv3_unfreeze_top,
        learning_rate=MOBILENETV3_LEARNING_RATE,
    ),
    "resnet50": ModelSpec(
        key="resnet50",
        display_name="ResNet50",
        build_fn=build_resnet50,
        input_shape=RESNET50_INPUT_SHAPE,
        unfreeze_fn=resnet50_unfreeze_top,
        learning_rate=RESNET50_LEARNING_RATE,
    ),
    "efficientnetb0": ModelSpec(
        key="efficientnetb0",
        display_name="EfficientNetB0",
        build_fn=build_efficientnetb0,
        input_shape=EFFICIENTNETB0_INPUT_SHAPE,
        unfreeze_fn=efficientnetb0_unfreeze_top,
        learning_rate=EFFICIENTNETB0_LEARNING_RATE,
    ),
}


def select_model_specs(model_key: str) -> list[ModelSpec]:
    """Return one or all model specs for a CLI selection."""
    if model_key == "all":
        return list(MODEL_SPECS.values())
    if model_key not in MODEL_SPECS:
        valid = ", ".join(["all", *MODEL_SPECS])
        raise ValueError(f"Unknown model '{model_key}'. Valid choices: {valid}")
    return [MODEL_SPECS[model_key]]


def count_images(folder: Path) -> int:
    """Count supported images recursively under ``folder``."""
    if not folder.exists():
        return 0
    return sum(1 for pattern in IMAGE_PATTERNS for _ in folder.rglob(pattern))


def validate_split_dataset(
    base_dir: Path,
    class_names: Iterable[str],
    split_names: Iterable[str] = ("train", "validation", "test"),
) -> None:
    """Ensure the dataset has split/class directories with at least one image."""
    missing: list[str] = []
    empty: list[str] = []
    for split in split_names:
        for class_name in class_names:
            class_dir = base_dir / split / class_name
            if not class_dir.exists():
                missing.append(str(class_dir))
            elif count_images(class_dir) == 0:
                empty.append(str(class_dir))

    if missing:
        raise FileNotFoundError(
            "Missing dataset directories:\n" + "\n".join(missing)
        )
    if empty:
        raise ValueError("Empty dataset directories:\n" + "\n".join(empty))


def compute_class_weights(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    """Compute balanced Keras class weights from a split directory."""
    counts = [count_images(train_dir / class_name) for class_name in class_names]
    if any(count == 0 for count in counts):
        pairs = ", ".join(f"{name}={count}" for name, count in zip(class_names, counts))
        raise ValueError(f"Training split must contain every class. Counts: {pairs}")

    total = sum(counts)
    num_classes = len(class_names)
    weights = {
        idx: round(total / (num_classes * count), 4)
        for idx, count in enumerate(counts)
    }
    logger.info(
        "Class counts: %s",
        ", ".join(f"{name}={count}" for name, count in zip(class_names, counts)),
    )
    logger.info("Class weights: %s", weights)
    return weights


def create_image_datasets(
    base_dir: Path,
    *,
    class_names: list[str],
    batch_size: int,
    image_size: tuple[int, int] = IMAGE_SIZE,
    random_seed: int = RANDOM_STATE,
    augment_train: bool = True,
    cache: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Create normalized train/validation/test datasets from folders."""
    validate_split_dataset(base_dir, class_names)
    train_aug = build_train_augmentation()

    def load(split: str, shuffle: bool) -> tf.data.Dataset:
        return tf.keras.utils.image_dataset_from_directory(
            directory=str(base_dir / split),
            labels="inferred",
            label_mode="binary",
            class_names=class_names,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            seed=random_seed,
        )

    def prepare(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=AUTOTUNE,
        )
        if cache:
            dataset = dataset.cache()
        if training and augment_train:
            dataset = dataset.map(
                lambda x, y: (train_aug(x, training=True), y),
                num_parallel_calls=AUTOTUNE,
            )
        return dataset.prefetch(AUTOTUNE)

    train_raw = load("train", shuffle=True)
    val_raw = load("validation", shuffle=False)
    test_raw = load("test", shuffle=False)

    return (
        prepare(train_raw, training=True),
        prepare(val_raw, training=False),
        prepare(test_raw, training=False),
        train_raw.class_names,
    )


def build_callbacks(
    *,
    model_name: str,
    checkpoint_path: Path,
    log_root: Path,
    patience: int = PATIENCE,
) -> list[keras.callbacks.Callback]:
    """Build common callbacks for one training phase."""
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = log_root / f"{model_name}_{run_id}"

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=max(2, patience // 2),
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    try:
        import tensorboard  # noqa: F401

        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(run_dir),
                histogram_freq=0,
                write_graph=True,
                update_freq=TENSORBOARD_UPDATE_FREQ,
            )
        )
    except ImportError:
        logger.warning("TensorBoard is not installed; skipping TensorBoard callback.")

    return callbacks


def save_history(history: dict[str, list], json_path: Path, pkl_path: Path) -> None:
    """Persist a Keras history dictionary as JSON and pickle."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    with open(pkl_path, "wb") as f:
        pickle.dump(history, f)


def train_two_phase_model(
    *,
    spec: ModelSpec,
    run_paths: RunPaths,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    class_weights: dict[int, float],
    epochs: int,
    fine_tune_epochs: int,
    random_seed: int = RANDOM_STATE,
    fine_tune: bool = True,
    fine_tune_learning_rate: float = FINE_TUNE_LEARNING_RATE,
    unfreeze_layers: int = FINE_TUNE_UNFREEZE_LAYERS,
) -> dict[str, Any]:
    """Train one model with frozen-backbone training and optional fine-tuning."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    model, backbone = spec.build_fn(
        input_shape=spec.input_shape,
        learning_rate=spec.learning_rate,
        trainable_backbone=False,
        random_seed=random_seed,
    )

    logger.info("Training %s: frozen phase for %d epochs", spec.display_name, epochs)
    frozen_checkpoint = run_paths.checkpoints / f"{spec.key}_best.keras"
    history_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=build_callbacks(
            model_name=f"{spec.key}_frozen",
            checkpoint_path=frozen_checkpoint,
            log_root=run_paths.logs,
        ),
        class_weight=class_weights,
        verbose=1,
    )

    history_b_dict: dict[str, list] = {}
    if fine_tune and fine_tune_epochs > 0:
        logger.info(
            "Fine-tuning %s: top %d layers for %d epochs",
            spec.display_name,
            unfreeze_layers,
            fine_tune_epochs,
        )
        spec.unfreeze_fn(backbone, unfreeze_layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        finetuned_checkpoint = run_paths.checkpoints / f"{spec.key}_finetuned.keras"
        history_b = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=build_callbacks(
                model_name=f"{spec.key}_finetune",
                checkpoint_path=finetuned_checkpoint,
                log_root=run_paths.logs,
                patience=max(3, PATIENCE // 2),
            ),
            class_weight=class_weights,
            verbose=1,
        )
        history_b_dict = history_b.history

    merged_history: dict[str, list] = {}
    for key, values in history_a.history.items():
        merged_history[key] = values + history_b_dict.get(key, [])

    history_json = run_paths.history / f"{spec.key}_history.json"
    history_pkl = run_paths.history / f"{spec.key}_history.pkl"
    save_history(merged_history, history_json, history_pkl)

    test_results = model.evaluate(test_ds, return_dict=True, verbose=1)
    final_model_path = run_paths.models / f"{spec.key}_final.keras"
    model.save(final_model_path)

    return {
        "display_name": spec.display_name,
        "model_key": spec.key,
        "history_json": str(history_json),
        "history_pkl": str(history_pkl),
        "best_checkpoint": str(frozen_checkpoint),
        "final_model": str(final_model_path),
        "phase_a": history_a.history,
        "phase_b": history_b_dict,
        "history": merged_history,
        "test_results": test_results,
    }

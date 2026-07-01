# -*- coding: utf-8 -*-
"""efficientnetb0.py – DroneML
================================

Phase 10b – EfficientNetB0 Model
----------------------------------
Builds and compiles a **binary classification** model using EfficientNetB0
as a frozen feature-extractor (ImageNet weights) with a custom classifier
head appended on top.

Architecture Overview
---------------------
  Input  →  EfficientNetB0 (frozen backbone, ImageNet weights)
         →  GlobalAveragePooling2D
         →  BatchNormalization
         →  Dense(256, activation='relu') + L2 + Dropout(0.4)
         →  Dense(128, activation='relu') + L2 + Dropout(0.3)
         →  Dense(1,  activation='sigmoid')   ← Binary output

Label convention (inherited from dataset_loader / create_tf_dataset):
    1 → No_Mine
    0 → Mine

Note on preprocessing
---------------------
EfficientNetB0 includes its own internal preprocessing (rescaling from
[0,255] to the network's expected range). Our data pipeline produces
images in [0, 1], so we add a ``Rescaling(255.0)`` layer before the
backbone so the internal preprocessing of EfficientNetB0 works correctly.

Public API
----------
- ``build_model``          – construct and return the compiled Keras model.
- ``get_model_summary``    – print the model summary to stdout.
- ``verify_output_shape``  – run a dummy forward-pass for shape verification.
- ``freeze_backbone``      – freeze all backbone layers.
- ``unfreeze_backbone``    – unfreeze all backbone layers.
- ``unfreeze_top_layers``  – selectively unfreeze the top N layers.
- ``save_model_artifacts`` – save architecture JSON and initial weights.
- ``save_model_diagram``   – save an architecture diagram PNG (requires pydot).
- ``main``                 – CLI entry point.
"""

from __future__ import annotations
import random
import numpy as np
import sys
from pathlib import Path
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Configure module-level logger
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

from src.models.config import (
    IMAGE_SIZE,
    RESULTS_DIR,
    RANDOM_STATE,
    EFFICIENTNETB0_LEARNING_RATE,
    EFFICIENTNETB0_DENSE_UNITS_1,
    EFFICIENTNETB0_DENSE_UNITS_2,
    EFFICIENTNETB0_DROPOUT_RATE_1,
    EFFICIENTNETB0_DROPOUT_RATE_2,
    EFFICIENTNETB0_L2_WEIGHT,
    FIGURES_DIR,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SHAPE: tuple[int, int, int] = (*IMAGE_SIZE, 3)    # (224, 224, 3)
NUM_CLASSES: int = 1                                      # Binary → sigmoid

# Classifier head hyper-parameters (imported from config)
DENSE_UNITS_1: int    = EFFICIENTNETB0_DENSE_UNITS_1
DENSE_UNITS_2: int    = EFFICIENTNETB0_DENSE_UNITS_2
DROPOUT_RATE_1: float = EFFICIENTNETB0_DROPOUT_RATE_1
DROPOUT_RATE_2: float = EFFICIENTNETB0_DROPOUT_RATE_2
L2_WEIGHT: float      = EFFICIENTNETB0_L2_WEIGHT

# Optimiser
LEARNING_RATE: float = EFFICIENTNETB0_LEARNING_RATE   # frozen-backbone phase


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    trainable_backbone: bool = False,
    random_seed: int = RANDOM_STATE,
) -> tuple[keras.Model, keras.Model]:
    """Construct and compile the EfficientNetB0 model.

    Parameters
    ----------
    input_shape:
        Input tensor shape ``(height, width, channels)``.
    learning_rate:
        Adam optimiser initial learning rate.
    trainable_backbone:
        If ``True``, the backbone is left trainable (fine-tuning mode).
        If ``False`` (default), the backbone is frozen.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple[keras.Model, keras.Model]
        ``(model, backbone)`` where ``model`` is the compiled Keras model
        and ``backbone`` is the EfficientNetB0 feature extractor.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # ------------------------------------------------------------------
    # 1. Backbone – EfficientNetB0 (ImageNet weights, no top)
    # ------------------------------------------------------------------
    # EfficientNetB0 has built-in preprocessing: it expects input in
    # [0, 255] and normalises internally. We pass include_preprocessing=True
    # (the default) and rescale from [0,1] → [0,255] before the backbone.
    backbone = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,          # remove ImageNet classifier
        weights="imagenet",         # pre-trained ImageNet weights
        pooling=None,               # we add our own pooling below
    )
    backbone.trainable = trainable_backbone

    logging.info("Backbone Name   : %s", backbone.name)
    logging.info("Backbone Layers : %d", len(backbone.layers))
    logging.info("Backbone Frozen : %s", not backbone.trainable)

    # ------------------------------------------------------------------
    # 2. Build model using Functional API
    # ------------------------------------------------------------------
    inputs = keras.Input(shape=input_shape, name="input_image")

    # EfficientNetB0 expects [0, 255]; our pipeline produces [0, 1].
    # Rescale before passing to the backbone.
    x = layers.Rescaling(scale=255.0, name="rescale_to_255")(inputs)

    # Pass through the backbone.
    # training=False keeps BatchNorm layers in inference mode while frozen.
    x = backbone(x, training=False)

    # --- Custom Classifier Head ---
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)

    # Dense block 1
    x = layers.Dense(
        DENSE_UNITS_1,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
        name="dense_256",
    )(x)
    x = layers.Dropout(DROPOUT_RATE_1, seed=random_seed, name="dropout_1")(x)

    # Dense block 2
    x = layers.Dense(
        DENSE_UNITS_2,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
        name="dense_128",
    )(x)
    x = layers.Dropout(DROPOUT_RATE_2, seed=random_seed, name="dropout_2")(x)

    # Binary output: sigmoid → probability of No_Mine class
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="sigmoid",
        name="output",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetB0_Baseline",
    )

    # ------------------------------------------------------------------
    # 3. Compile
    # ------------------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model, backbone


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_model_summary(model: keras.Model) -> None:
    """Print a structured summary of the model layers and parameters."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "efficientnetb0_summary.txt"

    lines: list[str] = []
    def _collect(line: str) -> None:
        lines.append(line)

    model.summary(print_fn=_collect, expand_nested=False, show_trainable=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    logging.info("MODEL SUMMARY – %s", model.name)
    logging.info("%s", "=" * 60)
    for line in lines:
        logging.info(line)
    logging.info("%s", "-" * 60)
    logging.info("Total Parameters    : %s", f"{total_params:,}")
    logging.info("Trainable Params    : %s", f"{trainable_params:,}")
    logging.info("Non-Trainable Params: %s", f"{non_trainable_params:,}")
    logging.info("%s", "=" * 60)


def freeze_backbone(backbone: keras.Model) -> None:
    """Freeze all layers of the backbone (make them non-trainable)."""
    backbone.trainable = False
    for layer in backbone.layers:
        layer.trainable = False
    logging.info(
        "Backbone frozen – %d layers set to non-trainable.", len(backbone.layers)
    )


def unfreeze_backbone(backbone: keras.Model) -> None:
    """Unfreeze the entire backbone (make all layers trainable)."""
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = True
    logging.info(
        "Backbone unfrozen – %d layers set to trainable.", len(backbone.layers)
    )


def unfreeze_top_layers(backbone: keras.Model, num_layers: int = 20) -> None:
    """Unfreeze the top *num_layers* of the backbone while keeping the rest frozen.

    Parameters
    ----------
    num_layers: int
        Number of layers from the end of the backbone to set trainable.
    """
    freeze_backbone(backbone)
    for layer in backbone.layers[-num_layers:]:
        layer.trainable = True
    logging.info("Top %d backbone layers unfrozen.", num_layers)


def verify_output_shape(model: keras.Model) -> None:
    """Run a single random forward-pass to confirm the output shape."""
    dummy = tf.random.uniform((1, *INPUT_SHAPE))
    out = model(dummy, training=False)
    assert out.shape == (1, 1), f"Unexpected output shape: {out.shape}"
    logging.info("[PASS] Forward-pass OK → output shape : %s", tuple(out.shape))
    logging.info(
        "Output Probability : %.6f",
        float(out.numpy()[0, 0])
    )

    trainable_weights = len(model.trainable_weights)
    frozen_weights = len(model.weights) - trainable_weights
    logging.info("Trainable Weights  : %d", trainable_weights)
    logging.info("Frozen Weights     : %d", frozen_weights)


def save_model_artifacts(
    model: keras.Model,
    backbone: keras.Model,
    out_dir: Path = RESULTS_DIR,
) -> None:
    """Save model architecture JSON and initial weights."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "efficientnetb0.json"
    with open(json_path, "w", encoding="utf-8") as f_json:
        f_json.write(model.to_json())
    logging.info("[INFO] Model architecture JSON saved → %s", json_path)

    keras_path = out_dir / "efficientnetb0_initial.keras"
    model.save(keras_path)
    logging.info("[INFO] Initial untrained model saved → %s", keras_path)

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    logging.info("--- Model Diagnostics ---")
    logging.info("Input Shape          : %s", model.input_shape[1:])
    logging.info("Output Shape         : %s", model.output_shape[1:])
    logging.info("Total Parameters     : %s", f"{total_params:,}")
    logging.info("Trainable Parameters : %s", f"{trainable_params:,}")
    logging.info("Non-Trainable Params : %s", f"{non_trainable_params:,}")
    logging.info("Backbone Frozen?     : %s", not backbone.trainable)
    logging.info("Backbone Layers      : %d", len(backbone.layers))


def save_model_diagram(
    model: keras.Model,
    out_dir: Path = FIGURES_DIR,
) -> None:
    """Save the model architecture diagram (requires pydot + graphviz)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "efficientnetb0_architecture.png"

    try:
        keras.utils.plot_model(
            model,
            to_file=str(output_path),
            show_shapes=True,
            show_layer_names=True,
            show_dtype=False,
            expand_nested=False,
            dpi=120,
        )
        logging.info("Model architecture diagram saved → %s", output_path)
    except Exception as exc:
        logging.warning("Could not generate architecture diagram: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Build the EfficientNetB0 model, output diagnostics, and save artifacts."""
    logging.info("%s", "=" * 60)
    logging.info("PHASE 10b - BASELINE MODEL : EfficientNetB0")
    logging.info("%s", "=" * 60)

    logging.info("[1/5] Building model ...")
    model, backbone = build_model(
        input_shape=INPUT_SHAPE,
        learning_rate=LEARNING_RATE,
        trainable_backbone=False,
    )

    logging.info("[2/5] Model summary ...")
    get_model_summary(model)

    logging.info("[3/5] Verifying forward-pass ...")
    verify_output_shape(model)

    logging.info("[4/5] Saving architecture diagram ...")
    save_model_diagram(model)

    logging.info("[5/5] Saving model artifacts ...")
    save_model_artifacts(model, backbone)

    logging.info("%s", "=" * 60)
    logging.info("DONE – EfficientNetB0 model ready for training.")
    logging.info("%s", "=" * 60)


if __name__ == "__main__":
    main()

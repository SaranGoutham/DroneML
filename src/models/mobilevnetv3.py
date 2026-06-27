# -*- coding: utf-8 -*-
"""mobilevnetv3.py – DroneML
==============================

Phase 8 – Baseline Model (MobileNetV3-Large)
----------------------------------------------
Builds and compiles a **binary classification** model using MobileNetV3-Large
as a frozen feature-extractor (ImageNet weights) with a custom classifier head
appended on top.

Architecture Overview
---------------------
  Input  →  MobileNetV3-Large (frozen backbone, ImageNet weights)
         →  GlobalAveragePooling2D
         →  BatchNormalization
         →  Dense(256, activation='relu') + L2 + Dropout(0.4)
         →  Dense(128, activation='relu') + L2 + Dropout(0.3)
         →  Dense(1,  activation='sigmoid')   ← Binary output

Label convention (inherited from dataset_loader / create_tf_dataset):
    0 → No_Mine
    1 → Mine

Public API
----------
- ``build_model``         – construct and return the compiled Keras model.
- ``get_model_summary``   – print the model summary to stdout.
- ``verify_output_shape`` – run a dummy forward-pass for shape verification.
- ``save_model_diagram``  – save an architecture diagram PNG (requires pydot).
- ``main``                – CLI entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import IMAGE_SIZE, RESULTS_DIR, RANDOM_STATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SHAPE: tuple = (*IMAGE_SIZE, 3)        # (224, 224, 3)
NUM_CLASSES: int = 1                          # Binary → single sigmoid unit

# Classifier head hyper-parameters
DENSE_UNITS_1: int    = 256
DENSE_UNITS_2: int    = 128
DROPOUT_RATE_1: float = 0.40
DROPOUT_RATE_2: float = 0.30
L2_WEIGHT: float      = 1e-4

# Optimiser
LEARNING_RATE: float = 1e-3    # frozen-backbone phase

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    input_shape: tuple = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    trainable_backbone: bool = False,
    random_seed: int = RANDOM_STATE,
) -> keras.Model:
    """Construct and compile the MobileNetV3-Large baseline model.

    Parameters
    ----------
    input_shape:
        HWC tuple for a single image. Default ``(224, 224, 3)``.
    learning_rate:
        Initial learning rate for the Adam optimiser.
    trainable_backbone:
        ``False`` (default) → backbone weights are frozen (baseline phase).
        ``True`` → all backbone weights are trainable (fine-tuning phase).
    random_seed:
        Seed passed to Dropout layers for reproducibility.

    Returns
    -------
    keras.Model
        A compiled Keras model ready for ``.fit()``.

    Notes
    -----
    * MobileNetV3-Large expects float32 pixel values **already normalised**
      to ``[0, 1]``.  The data pipeline in ``create_tf_dataset.py`` performs
      this division by 255.
    * ``include_preprocessing=False`` because our pipeline already normalises
      the images.
    * ``include_top=False`` removes the original ImageNet classifier so we
      can attach a custom binary head.
    """
    tf.random.set_seed(random_seed)

    # ------------------------------------------------------------------
    # 1. Backbone – MobileNetV3-Large (ImageNet weights, no top)
    # ------------------------------------------------------------------
    backbone = keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,              # remove ImageNet classifier
        weights="imagenet",             # pre-trained ImageNet weights
        include_preprocessing=False,    # images are already in [0, 1]
        pooling=None,                   # we add our own pooling below
    )
    backbone.trainable = trainable_backbone  # freeze for baseline phase

    # ------------------------------------------------------------------
    # 2. Build model using Functional API
    # ------------------------------------------------------------------
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Pass through the frozen backbone.
    # training=False keeps BatchNorm layers in inference mode while frozen.
    x = backbone(inputs, training=False)

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

    # Binary output: sigmoid → probability of Mine class
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="sigmoid",
        name="output",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV3Large_Baseline",
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

    return model


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_model_summary(model: keras.Model) -> None:
    """Print a structured summary of the model layers and parameters."""
    print("\n" + "=" * 60)
    print(f"  MODEL : {model.name}")
    print("=" * 60)
    model.summary(expand_nested=False, show_trainable=True)

    total     = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    frozen    = total - trainable

    print("\n" + "-" * 60)
    print(f"  Total Parameters    : {total:,}")
    print(f"  Trainable           : {trainable:,}")
    print(f"  Non-Trainable       : {frozen:,}")
    print("-" * 60 + "\n")


def verify_output_shape(model: keras.Model) -> None:
    """Run a single random forward-pass to confirm the output shape is (1, 1)."""
    dummy = tf.random.uniform((1, *INPUT_SHAPE))
    out   = model(dummy, training=False)
    assert out.shape == (1, 1), f"Unexpected output shape: {out.shape}"
    print(f"[PASS] Forward-pass OK  ->  output shape : {tuple(out.shape)}")
    print(f"       Raw sigmoid value : {out.numpy()[0, 0]:.6f}")
    prediction = "Mine" if out.numpy()[0, 0] >= 0.5 else "No_Mine"
    print(f"       Prediction (random input) : {prediction}")


def save_model_diagram(model: keras.Model, out_dir: Path = RESULTS_DIR) -> None:
    """Save a model architecture diagram to *out_dir*/model_architecture.png.

    Requires the ``pydot`` and ``graphviz`` packages.  Skips silently if they
    are not available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_architecture.png"
    try:
        keras.utils.plot_model(
            model,
            to_file=str(out_path),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=False,
            dpi=96,
        )
        print(f"[INFO] Model diagram saved → {out_path}")
    except Exception as exc:   # pydot / graphviz not installed
        print(f"[WARN] Could not save model diagram: {exc}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Build the baseline model, print summary, and verify a forward-pass."""
    print("\n" + "=" * 60)
    print("  PHASE 8 - BASELINE MODEL : MobileNetV3-Large")
    print("=" * 60)

    print("\n[1/4] Building model ...")
    model = build_model(
        input_shape=INPUT_SHAPE,
        learning_rate=LEARNING_RATE,
        trainable_backbone=False,   # backbone frozen for baseline phase
    )

    print("[2/4] Model summary ...")
    get_model_summary(model)

    print("[3/4] Verifying forward-pass ...")
    verify_output_shape(model)

    print("[4/4] Saving architecture diagram ...")
    save_model_diagram(model)

    print("\n[DONE] Baseline model ready for training (Phase 9).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

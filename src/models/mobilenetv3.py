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
    1 → No_Mine
    0 → Mine

Public API
----------
- ``build_model``         – construct and return the compiled Keras model.
- ``get_model_summary``   – print the model summary to stdout.
- ``verify_output_shape`` – run a dummy forward-pass for shape verification.
- ``save_model_diagram``  – save an architecture diagram PNG (requires pydot).
- ``main``                – CLI entry point.
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
    MOBILENETV3_LEARNING_RATE,
    MOBILENETV3_DENSE_UNITS_1,
    MOBILENETV3_DENSE_UNITS_2,
    MOBILENETV3_DROPOUT_RATE_1,
    MOBILENETV3_DROPOUT_RATE_2,
    MOBILENETV3_L2_WEIGHT,
    FIGURES_DIR,
)



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SHAPE: tuple[int, int, int] = (*IMAGE_SIZE, 3)        # (224, 224, 3)
NUM_CLASSES: int = 1                          # Binary → single sigmoid unit

# Classifier head hyper‑parameters (imported from config)
DENSE_UNITS_1: int    = MOBILENETV3_DENSE_UNITS_1
DENSE_UNITS_2: int    = MOBILENETV3_DENSE_UNITS_2
DROPOUT_RATE_1: float = MOBILENETV3_DROPOUT_RATE_1
DROPOUT_RATE_2: float = MOBILENETV3_DROPOUT_RATE_2
L2_WEIGHT: float      = MOBILENETV3_L2_WEIGHT

# Optimiser
LEARNING_RATE: float = MOBILENETV3_LEARNING_RATE  # frozen‑backbone phase


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    trainable_backbone: bool = False,
    random_seed: int = RANDOM_STATE,
) -> tuple[keras.Model, keras.Model]:
    """Construct and compile the MobileNetV3-Large baseline model.

    Returns
    -------
    tuple[keras.Model, keras.Model]
        (model, backbone) where ``model`` is the compiled Keras model and ``backbone``
        is the MobileNetV3Large feature extractor.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    # ------------------------------------------------------------------
    # 1. Backbone – MobileNetV3-Large (ImageNet weights, no top)
    # ------------------------------------------------------------------
    backbone = keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,              # remove ImageNet classifier
        weights="imagenet",             # pre‑trained ImageNet weights
        include_preprocessing=False,    # images are already in [0, 1]
        pooling=None,                   # we add our own pooling below
    )
    backbone.trainable = trainable_backbone  # freeze for baseline phase
    logging.info("Backbone Name   : %s", backbone.name)
    logging.info("Backbone Layers : %d", len(backbone.layers))
    logging.info("Backbone Frozen : %s", not backbone.trainable)
       # ------------------------------------------------------------------
    # 2. Build model using Functional API
    # ------------------------------------------------------------------
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Pass through the backbone.
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

    return model, backbone


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_model_summary(model: keras.Model) -> None:
    """Print a structured summary of the model layers and parameters.

    The summary is also written to ``results/model_summary.txt`` for
    reproducibility.
    """
    # Ensure the results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "model_summary.txt"

    # Capture Keras summary output
    lines: list[str] = []
    def _collect(line: str) -> None:
        lines.append(line)

    model.summary(print_fn=_collect, expand_nested=False, show_trainable=True)

    # Write to file
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )

    non_trainable_params = (
        total_params - trainable_params
    )
    # Logging output
    logging.info("MODEL SUMMARY – %s", model.name)
    logging.info("%s", "=" * 60)
    for line in lines:
        logging.info(line)
    logging.info("%s", "-" * 60)
    logging.info("Total Parameters    : %s", f"{total_params:,}")
    logging.info("Trainable Params    : %s", f"{trainable_params:,}")
    logging.info("Non‑Trainable Params: %s", f"{non_trainable_params:,}")
    logging.info("%s", "=" * 60)

def freeze_backbone(backbone: keras.Model) -> None:
    """Freeze all layers of the backbone (make them non‑trainable)."""
    backbone.trainable = False
    for layer in backbone.layers:
        layer.trainable = False
    logging.info("Backbone frozen – %d layers set to non‑trainable.", len(backbone.layers))

def unfreeze_backbone(backbone: keras.Model) -> None:
    """Unfreeze the entire backbone (make all layers trainable)."""
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = True
    logging.info("Backbone unfrozen – %d layers set to trainable.", len(backbone.layers))

def unfreeze_top_layers(backbone: keras.Model, num_layers: int = 10) -> None:
    """Unfreeze the top *num_layers* of the backbone while keeping the rest frozen.

    Parameters
    ----------
    num_layers: int
        Number of layers from the end of the backbone to set trainable.
    """
    # Ensure backbone is initially frozen
    freeze_backbone(backbone)
    # Unfreeze the specified top layers
    for layer in backbone.layers[-num_layers:]:
        layer.trainable = True
    logging.info("Top %d backbone layers unfrozen.", num_layers)



def verify_output_shape(model: keras.Model) -> None:
    """Run a single random forward‑pass to confirm the output shape.

    Also prints the number of trainable and frozen weights.
    """
    dummy = tf.random.uniform((1, *INPUT_SHAPE))
    out = model(dummy, training=False)
    assert out.shape == (1, 1), f"Unexpected output shape: {out.shape}"
    logging.info("[PASS] Forward‑pass OK → output shape : %s", tuple(out.shape))
    logging.info(
        "Output Probability : %.6f",
        float(out.numpy()[0, 0])
    )   

    # Additional diagnostics
    trainable_weights = len(model.trainable_weights)
    frozen_weights = len(model.weights) - trainable_weights
    logging.info("Trainable Weights  : %d", trainable_weights)
    logging.info("Frozen Weights     : %d", frozen_weights)



def save_model_artifacts(model: keras.Model, backbone: keras.Model, out_dir: Path = RESULTS_DIR) -> None:
    """Save model summary, architecture JSON, and initial weights.

    This function is intended for the *untrained* baseline model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model summary (already written by get_model_summary)
    # Ensure summary file exists – get_model_summary writes it.
    # 2. Architecture JSON
    json_path = out_dir / "mobilenetv3.json"
    with open(json_path, "w", encoding="utf-8") as f_json:
        f_json.write(model.to_json())
    logging.info("[INFO] Model architecture JSON saved → %s", json_path)

    # 3. Initial untrained model (weights only)
    keras_path = out_dir / "mobilenetv3_initial.keras"
    model.save(keras_path)
    logging.info("[INFO] Initial untrained model saved → %s", keras_path)

    # 4. Additional diagnostics
    input_shape = model.input_shape[1:]
    output_shape = model.output_shape[1:]
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params
    backbone_layers = len(backbone.layers)
    backbone_frozen = not backbone.trainable

    logging.info("--- Model Diagnostics ---")
    logging.info("Input Shape          : %s", input_shape)
    logging.info("Output Shape         : %s", output_shape)
    logging.info("Total Parameters     : %s", f"{total_params:,}")
    logging.info("Trainable Parameters : %s", f"{trainable_params:,}")
    logging.info("Non‑Trainable Params : %s", f"{non_trainable_params:,}")
    logging.info("Trainable Weights    : %d", len(model.trainable_weights))
    logging.info("Frozen Weights       : %d", len(model.weights) - len(model.trainable_weights))
    logging.info("Backbone Frozen?    : %s", backbone_frozen)
    logging.info("Backbone Layers Count: %d", backbone_layers)


def save_model_diagram(
    model: keras.Model,
    out_dir: Path = FIGURES_DIR,
) -> None:
    """
    Save the model architecture diagram.

    Requires:
        pip install pydot graphviz

    Also requires Graphviz to be installed on the system.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "model_architecture.png"

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
        logging.warning(
            "Could not generate architecture diagram: %s",
            exc,
        )
# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Build the baseline model, output diagnostics, and save artifacts."""
    logging.info("%s", "=" * 60)
    logging.info("PHASE 8 - BASELINE MODEL : MobileNetV3-Large")
    logging.info("%s", "=" * 60)

    logging.info("[1/6] Building model ...")
    model, backbone = build_model(
        input_shape=INPUT_SHAPE,
        learning_rate=LEARNING_RATE,
        trainable_backbone=False,   # backbone frozen for baseline phase
    )

    logging.info("[2/6] Model summary ...")
    get_model_summary(model)

    logging.info("[3/6] Verifying forward‑pass ...")
    verify_output_shape(model)

    logging.info("[4/6] Saving architecture diagram ...")
    save_model_diagram(model)

    logging.info("[5/6] Saving model artifacts ...")
    save_model_artifacts(model, backbone)

    logging.info("[6/6] Baseline model completed.")


    logging.info("%s", "=" * 60)
    logging.info("DONE – Baseline model ready for training (Phase 9).")
    logging.info("%s", "=" * 60)



if __name__ == "__main__":
    main()

"""
MobileNetV3-Large 3-Class Architecture for DroneML classification_v2
===================================================================

Builds and compiles the baseline MobileNetV3-Large model with pre-trained
ImageNet weights and a custom classification head for 3-class thermal detection
(Mine, Known, Background).
"""

import random
import logging
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.training.config import (
    INPUT_SHAPE,
    NUM_CLASSES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_L2_WEIGHT,
    DEFAULT_DROPOUT_1,
    DEFAULT_DROPOUT_2,
    DEFAULT_DENSE_UNITS_1,
    DEFAULT_DENSE_UNITS_2,
    DEFAULT_SEED,
)

logger = logging.getLogger(__name__)


def build_mobilenetv3_large(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    weights: str = "imagenet",
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2_weight: float = DEFAULT_L2_WEIGHT,
    dropout_1: float = DEFAULT_DROPOUT_1,
    dropout_2: float = DEFAULT_DROPOUT_2,
    dense_1: int = DEFAULT_DENSE_UNITS_1,
    dense_2: int = DEFAULT_DENSE_UNITS_2,
    trainable_backbone: bool = False,
    random_seed: int = DEFAULT_SEED,
    optimizer_name: str = "adam",
) -> keras.Model:
    """
    Construct and compile the MobileNetV3-Large baseline model for 3-class classification.
    
    Architecture:
      Input (224, 224, 3)
        -> MobileNetV3-Large Backbone (ImageNet weights, include_preprocessing=True)
        -> GlobalAveragePooling2D
        -> BatchNormalization
        -> Dense(256, relu, L2) + Dropout(0.3)
        -> Dense(128, relu, L2) + Dropout(0.2)
        -> Dense(3, softmax, float32, name='predictions')
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # 1. Backbone: MobileNetV3-Large
    backbone = keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        include_preprocessing=True,
        pooling=None,
    )
    backbone.trainable = trainable_backbone

    logger.info(
        f"Initialized MobileNetV3-Large backbone ({len(backbone.layers)} layers, weights='{weights}', trainable={trainable_backbone})"
    )

    # 2. Classifier Head
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = backbone(inputs, training=False)  # keep BatchNorm frozen during baseline phase
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)

    x = layers.Dense(
        dense_1,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_weight),
        name="dense_head_1",
    )(x)
    x = layers.Dropout(dropout_1, seed=random_seed, name="dropout_1")(x)

    x = layers.Dense(
        dense_2,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_weight),
        name="dense_head_2",
    )(x)
    x = layers.Dropout(dropout_2, seed=random_seed, name="dropout_2")(x)

    # Output probabilities (explicit float32 for mixed precision safety)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV3Large_Thermal_Baseline",
    )

    # 3. Optimizer selection
    opt_lower = optimizer_name.lower()
    if opt_lower == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_lower == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=l2_weight)
    elif opt_lower == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        logger.warning(f"Unrecognized optimizer '{optimizer_name}', falling back to Adam.")
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # 4. Compilation
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    trainable_params = int(np.sum([keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable_params = int(np.sum([keras.backend.count_params(w) for w in model.non_trainable_weights]))
    logger.info(
        f"Compiled {model.name}: {trainable_params:,} trainable parameters, {non_trainable_params:,} frozen parameters"
    )

    return model


def unfreeze_backbone_top_layers(
    model: keras.Model,
    unfreeze_layers: int = 30,
    learning_rate: float = 1e-4,
) -> None:
    """
    Unfreeze the top N layers of the MobileNetV3 backbone for fine-tuning,
    keeping BatchNormalization layers in inference mode to preserve statistics.
    """
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone = layer
            break

    if backbone is None:
        logger.warning("Could not find nested backbone sub-model to unfreeze.")
        return

    backbone.trainable = True
    total_layers = len(backbone.layers)
    freeze_up_to = max(0, total_layers - unfreeze_layers)

    for i, layer in enumerate(backbone.layers):
        if i < freeze_up_to or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    trainable_params = int(np.sum([keras.backend.count_params(w) for w in model.trainable_weights]))
    logger.info(
        f"Fine-tune setup complete: Unfroze last {unfreeze_layers} backbone layers. Trainable params: {trainable_params:,}"
    )

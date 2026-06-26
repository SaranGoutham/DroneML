# augment.py – DroneML Specification
"""Utility module to build a TensorFlow/Keras augmentation pipeline for training.

This module provides functions to create a `keras.Sequential` model that applies realistic
augmentations for UAV thermal imagery. The pipeline is intended for **training** data
only; validation and test data are left unchanged.

The pipeline includes horizontal flip, rotation (±10%), zoom (15%), translation (10%) and contrast adjustment (10%).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)

def build_train_augmentation() -> keras.Sequential:
    """Construct a Keras `Sequential` model containing training augmentations.

    Returns
    -------
    keras.Sequential
        A sequential model that applies the configured augmentations when called.
    """
    augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.10),
    ]
    return keras.Sequential(augmentation_layers, name="train_augmentation")

AUGMENTATION_PIPELINE = build_train_augmentation()

def augment_image(image: tf.Tensor, training: bool = True) -> tf.Tensor:
    """Apply augmentation pipeline to a single image tensor.

    Parameters
    ----------
    image : tf.Tensor
        Image tensor of shape `(height, width, channels)`.
    training : bool, optional
        If ``True`` the image is passed through the augmentation pipeline; otherwise the original image is returned unchanged.

    Returns
    -------
    tf.Tensor
        The (potentially) augmented image tensor.
    """
    if not training:
        return image
    image_batch = tf.expand_dims(image, axis=0)  # shape: (1, H, W, C)
    image = AUGMENTATION_PIPELINE(image_batch, training=True)
    return tf.squeeze(image, axis=0)

def show_pipeline() -> None:
    """Print a human‑readable description of the augmentation pipeline.
    """
    lines = [
        "========================================",
        "",
        "TRAINING AUGMENTATION",
        "",
        "========================================",
        "",
        "✓ Horizontal Flip",
        "✓ Rotation (±10%)",
        "✓ Zoom (15%)",
        "✓ Translation (10%)",
        "✓ Contrast (10%)",
        "",
        "Pipeline Ready",
    ]
    print("\n".join(lines))

def main() -> None:
    """Entry point for module testing.
    """
    pipeline = build_train_augmentation()
    show_pipeline()
    assert isinstance(pipeline, keras.Sequential), "Pipeline is not a Sequential model"
    print("Augmentation Model Created Successfully")

if __name__ == "__main__":
    main()

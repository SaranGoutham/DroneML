"""class_weights.py

Utility module for computing class weights for the binary Mine vs No_Mine
classification problem used during model training.

The module is deliberately lightweight: it receives a ``tf.data.Dataset``
containing the training samples, extracts the labels, and computes balanced
class weights using ``sklearn.utils.class_weight.compute_class_weight``.
It does **not** load data from disk, build models, or perform any training.

Public API
----------
``compute_class_weights(train_dataset)``
    Returns a ``dict[int, float]`` suitable for the ``class_weight`` argument
    of ``model.fit``.
"""

import logging
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


def _extract_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """Extract label values from a ``tf.data.Dataset``.

    The dataset is expected to yield either ``(features, labels)`` tuples or
    ``labels`` directly. All label tensors are converted to NumPy scalars and
    concatenated into a one‑dimensional ``np.ndarray``.

    Parameters
    ----------
    dataset: tf.data.Dataset
        The training dataset produced by ``create_datasets``.

    Returns
    -------
    np.ndarray
        Array of integer label values.
    """
    labels = []
    for element in dataset:
        # ``element`` may be a tuple (features, labels) or just labels.
        if isinstance(element, tuple) and len(element) == 2:
            _, y = element
        else:
            y = element
        # Convert tf.Tensor or eager value to NumPy array.
        y_np = y.numpy() if isinstance(y, tf.Tensor) else np.array(y)
        # Flatten in case of batch dimension.
        labels.extend(y_np.ravel().tolist())
    if not labels:
        raise ValueError("The provided training dataset is empty.")
    return np.array(labels, dtype=int)


def compute_class_weights(
    train_dataset: tf.data.Dataset,
) -> Dict[int, float]:
    """Compute balanced class weights for a binary classification dataset.

    The function iterates over ``train_dataset`` to count occurrences of each
    class (0 – No_Mine, 1 – Mine) and then uses ``sklearn.utils.
    class_weight.compute_class_weight`` with ``class_weight='balanced'`` to
    produce weight values. The result is a dictionary mapping class indices to
    their corresponding weight, compatible with ``model.fit(...,
    class_weight=weights)``.

    Parameters
    ----------
    train_dataset: tf.data.Dataset
        Training dataset containing the samples. Labels must be integer values
        0 or 1.

    Returns
    -------
    dict[int, float]
        Mapping ``{0: weight_for_no_mine, 1: weight_for_mine}``.

    Raises
    ------
    ValueError
        If the dataset is empty, labels are missing, or label values are not
        0 or 1.
    """
    logger.info("Computing class weights...")

    # Extract label array from the dataset.
    labels = _extract_labels(train_dataset)

    # Validate label values.
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(
            f"Invalid label values found: {unique_labels}. Expected only 0 and 1."
        )

    # Count samples per class for logging.
    count_no_mine = int((labels == 0).sum())
    count_mine = int((labels == 1).sum())
    logger.info("Mine Images      : %d", count_mine)
    logger.info("No_Mine Images   : %d", count_no_mine)

    # Compute balanced weights.
    class_weights_array = compute_class_weight(
        class_weight="balanced", classes=[0, 1], y=labels
    )
    class_weights = {int(cls): float(wt) for cls, wt in zip([0, 1], class_weights_array)}

    logger.info("Class Weights")
    for cls, wt in class_weights.items():
        logger.info("Class %d : %.4f", cls, wt)
    logger.info("Class weights ready.")
    return class_weights

__all__ = ["compute_class_weights"]

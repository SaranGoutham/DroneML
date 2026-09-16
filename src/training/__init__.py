"""
Training Package for DroneML classification_v2
==============================================

Provides remote-ready training and evaluation pipelines for thermal landmine
detection models, supporting dual evaluation protocols, manifest-driven dataset
streaming, and comprehensive multi-class metrics.
"""

from src.training.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    IMAGE_SIZE,
    INPUT_SHAPE,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "NUM_CLASSES",
    "IMAGE_SIZE",
    "INPUT_SHAPE",
]

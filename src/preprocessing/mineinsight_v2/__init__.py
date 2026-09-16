"""
MineInsight classification_v2 Preprocessing Package
===================================================

Provides a reproducible, leakage-safe, and metadata-preserving pipeline for
converting MineInsight raw LWIR frames into a 3-category classification dataset:
Mine, Known, and Background.
"""

from .config import (
    CATEGORY_MINE,
    CATEGORY_KNOWN,
    CATEGORY_BACKGROUND,
    CLASSIFICATION_V2_DIR,
    REPORTS_V2_DIR,
)

__all__ = [
    "CATEGORY_MINE",
    "CATEGORY_KNOWN",
    "CATEGORY_BACKGROUND",
    "CLASSIFICATION_V2_DIR",
    "REPORTS_V2_DIR",
]

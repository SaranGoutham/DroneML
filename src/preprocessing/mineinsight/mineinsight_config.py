"""
MineInsight Configuration
=========================

Configuration module for the MineInsight dataset preprocessing pipeline.
Contains only dataset paths, image size, padding amount, supported extensions,
report locations, and constants.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MINEINSIGHT_DATA_DIR = PROJECT_ROOT / "data" / "mineinsight"
RAW_DIR = MINEINSIGHT_DATA_DIR / "raw"
PROCESSED_DIR = MINEINSIGHT_DATA_DIR / "processed"
CLASSIFICATION_DIR = MINEINSIGHT_DATA_DIR / "classification"
REPORTS_DIR = MINEINSIGHT_DATA_DIR / "reports"
RESULTS_DIR = PROJECT_ROOT / "results" / "mineinsight"

# Image constants
IMAGE_SIZE = (224, 224)
PADDING_AMOUNT = 0

# Supported Extensions
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]

# Constants
MINE_CLASS = "Mine"
NO_MINE_CLASS = "No_Mine"

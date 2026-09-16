"""
Configuration Module for MineInsight classification_v2 Pipeline
================================================================

Defines canonical paths, target taxonomy constants, crop geometry parameters,
quality thresholds, and split protocol assignments.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

# Project Roots & Directories
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MINEINSIGHT_DATA_DIR = PROJECT_ROOT / "data" / "mineinsight"
RAW_DIR = MINEINSIGHT_DATA_DIR / "raw"
TARGETS_YAML_PATH = RAW_DIR / "targets_list.yaml"

# classification_v2 Single Canonical Storage Tree
CLASSIFICATION_V2_DIR = MINEINSIGHT_DATA_DIR / "classification_v2"
MINE_DIR = CLASSIFICATION_V2_DIR / "Mine"
KNOWN_DIR = CLASSIFICATION_V2_DIR / "Known"
BACKGROUND_DIR = CLASSIFICATION_V2_DIR / "Background"

# Reports and Manifests
REPORTS_V2_DIR = MINEINSIGHT_DATA_DIR / "reports_v2"
MASTER_MANIFEST_CSV = REPORTS_V2_DIR / "master_crop_manifest.csv"
PROTOCOL1_SPLIT_CSV = REPORTS_V2_DIR / "protocol1_split.csv"
PROTOCOL2_SPLIT_CSV = REPORTS_V2_DIR / "protocol2_split.csv"
DRY_RUN_SUMMARY_JSON = REPORTS_V2_DIR / "dry_run_summary.json"
ORPHAN_QUARANTINE_JSON = REPORTS_V2_DIR / "orphan_quarantine_manifest.json"

# Image & Crop Parameters
RAW_IMAGE_RESOLUTION: Tuple[int, int] = (640, 512)  # (width, height)
CROP_TARGET_SIZE: Tuple[int, int] = (224, 224)
CONTEXT_PADDING_RATIO: float = 0.20                 # 20% contextual boundary padding
JPEG_QUALITY: int = 95

# Background Sampling Configuration
BG_TEMPORAL_STRIDE_DEFAULT: int = 1                 # k=1 initially as specified
BG_MIN_VARIANCE_STD_DEV: float = 2.0                # Thermal entropy threshold

# Sequences Registry (6 official sequences)
VALID_SEQUENCES: List[str] = [
    "track_1_s1",
    "track_1_s2",
    "track_2_s1",
    "track_2_s2",
    "track_3_s1",
    "track_3_s2",
]

# Supported Image Extensions
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Category Classification Rules
KNOWN_CLASS_IDS: Set[int] = set(range(1, 21))      # IDs 1 to 20
MINE_CLASS_IDS: Set[int] = {
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 53, 55, 56, 57,
}
BACKGROUND_CLASS_ID: int = -1

# Category Enum Strings
CATEGORY_MINE = "Mine"
CATEGORY_KNOWN = "Known"
CATEGORY_BACKGROUND = "Background"

# Protocol 1: Physical Target Instance Allocations
# Multi-instance target allocations:
P1_INSTANCE_ALLOCATION: Dict[str, Dict[str, List[int]]] = {
    "PFM-1": {
        "train": [22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        "validation": [32, 33],
        "test": [34, 35, 36],
    },
    "PMN": {
        "train": [21],
        "validation": [],
        "test": [43],
    },
    "M6": {
        "train": [39],
        "validation": [],
        "test": [40],
    },
    "C-3": {
        "train": [55],
        "validation": [57],
        "test": [],
    },
    "Plastic_Water_Bottle": {
        "train": [13],
        "validation": [14],
        "test": [15],
    },
    "Plastic_Chips_Bag": {
        "train": [2],
        "validation": [],
        "test": [5],
    },
    "Glass_Jar": {
        "train": [6],
        "validation": [4],
        "test": [],
    },
}

# Singleton instances partition for Protocol 1
P1_SINGLETON_MINE_TRAIN: List[int] = [41, 42, 44, 45, 47, 48, 52, 53, 56]
P1_SINGLETON_MINE_VAL: List[int] = [46]
P1_SINGLETON_MINE_TEST: List[int] = []

P1_SINGLETON_KNOWN_TRAIN: List[int] = [1, 3, 7, 9, 10, 11, 12, 16, 19, 20]
P1_SINGLETON_KNOWN_VAL: List[int] = [18]
P1_SINGLETON_KNOWN_TEST: List[int] = [8, 17]

# Protocol 2: Flight-Holdout Sequences
P2_TRAIN_SEQUENCES: List[str] = ["track_1_s1", "track_2_s1", "track_3_s1"]
P2_TEST_SEQUENCES: List[str] = ["track_1_s2", "track_2_s2", "track_3_s2"]

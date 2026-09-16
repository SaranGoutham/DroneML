"""
Training Configuration Module for DroneML classification_v2
============================================================

Defines paths, class index mappings, default hyperparameters, and evaluation
constants for thermal model training.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Project & Dataset Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mineinsight"
CLASSIFICATION_V2_DIR = DATA_DIR / "classification_v2"
REPORTS_V2_DIR = DATA_DIR / "reports_v2"

PROTOCOL1_SPLIT_CSV = REPORTS_V2_DIR / "protocol1_split.csv"
PROTOCOL2_SPLIT_CSV = REPORTS_V2_DIR / "protocol2_split.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "thermal_v2"

# 3-Class Target Taxonomy
CLASS_NAMES: List[str] = ["Mine", "Known", "Background"]
CLASS_TO_IDX: Dict[str, int] = {
    "Mine": 0,
    "Known": 1,
    "Background": 2,
}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES: int = len(CLASS_NAMES)

# Input Dimensions
IMAGE_SIZE: Tuple[int, int] = (224, 224)
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)

# Default Training Hyperparameters
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_EPOCHS: int = 30
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_FINETUNE_LR: float = 1e-4
DEFAULT_OPTIMIZER: str = "adam"
DEFAULT_SEED: int = 42

# Callbacks & Regularization
DEFAULT_EARLY_STOPPING_PATIENCE: int = 7
DEFAULT_REDUCE_LR_PATIENCE: int = 3
DEFAULT_REDUCE_LR_FACTOR: float = 0.2
DEFAULT_MIN_LR: float = 1e-6
DEFAULT_L2_WEIGHT: float = 1e-4
DEFAULT_DROPOUT_1: float = 0.3
DEFAULT_DROPOUT_2: float = 0.2
DEFAULT_DENSE_UNITS_1: int = 256
DEFAULT_DENSE_UNITS_2: int = 128

# Remote Smoke-Test Parameters (Minimal resource verification)
SMOKE_TEST_BATCHES_TRAIN: int = 5
SMOKE_TEST_BATCHES_VAL: int = 3
SMOKE_TEST_BATCHES_TEST: int = 3
SMOKE_TEST_EPOCHS: int = 2

"""
Project Configuration
DroneML
"""

from pathlib import Path

# =====================================================
# PROJECT ROOT
# =====================================================

PROJECT_ROOT = Path(r"D:\DroneML")

# =====================================================
# DATASET
# =====================================================

DATASET_DIR = PROJECT_ROOT / "dataset" / "DatainBrief_dataset_1"

# =====================================================
# RESULTS
# =====================================================

RESULTS_DIR = PROJECT_ROOT / "results"

# =====================================================
# PROCESSED DATASET
# =====================================================

PROCESSED_DIR = PROJECT_ROOT / "processed_dataset"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "validation"
TEST_DIR = PROCESSED_DIR / "test"

MINE_DIR = PROCESSED_DIR / "Mine"
NO_MINE_DIR = PROCESSED_DIR / "No_Mine"

# =====================================================
# METADATA
# =====================================================

METADATA_FILE = RESULTS_DIR / "metadata_checked.csv"

# =====================================================
# IMAGE SETTINGS
# =====================================================

IMAGE_SIZE = (224, 224)

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

GAMMA = 1.2

# =====================================================
# RANDOM SEED
# =====================================================

RANDOM_STATE = 42

# =====================================================
# CREATE DIRECTORIES
# =====================================================

RESULTS_DIR.mkdir(exist_ok=True)

PROCESSED_DIR.mkdir(exist_ok=True)

MINE_DIR.mkdir(parents=True, exist_ok=True)
NO_MINE_DIR.mkdir(parents=True, exist_ok=True)
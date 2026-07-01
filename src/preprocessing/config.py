"""
Project Configuration
DroneML
"""

from pathlib import Path

# =====================================================
# PROJECT ROOT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# =====================================================
# DATASET
# =====================================================

DATASET_DIR = PROJECT_ROOT / "dataset" / "DatainBrief_dataset_1"

# =====================================================
# RESULTS
# =====================================================

RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"

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

RAW_METADATA_FILE = RESULTS_DIR / "metadata.csv"
CHECKED_METADATA_FILE = RESULTS_DIR / "metadata_checked.csv"
METADATA_FILE = CHECKED_METADATA_FILE

DUPLICATE_IMAGES_FILE = RESULTS_DIR / "duplicate_images.csv"

# =====================================================
# REPORTS
# =====================================================

PREPROCESSING_REPORT_FILE = RESULTS_DIR / "preprocessing_report.txt"
SPLIT_REPORT_FILE = RESULTS_DIR / "split_report.txt"
VERIFICATION_REPORT_FILE = RESULTS_DIR / "verification_report.txt"
DATASET_SUMMARY_FILE = GRAPHS_DIR / "dataset_summary.csv"

FLIGHT_DISTRIBUTION_FILE = RESULTS_DIR / "flight_distribution.csv"
DEPTH_DISTRIBUTION_FILE = RESULTS_DIR / "depth_distribution.csv"
CLASS_DISTRIBUTION_FILE = RESULTS_DIR / "class_distribution.csv"

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
# OFFLINE DATA AUGMENTATION
# =====================================================

OFFLINE_AUG_REPORT_FILE = RESULTS_DIR / "augmentation_report.txt"

# Class-specific multiplication factors to balance the training split
AUGMENTATION_FACTORS = {
    "Mine": 0,      # 220 -> 660 total (original + 440 augmented)
    "No_Mine": 9,   # 30 -> 630 total (original + 600 augmented)
}

# Ensure directories exist (including figures)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MINE_DIR.mkdir(parents=True, exist_ok=True)
NO_MINE_DIR.mkdir(parents=True, exist_ok=True)

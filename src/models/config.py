"""
Model Configuration
DroneML

Central configuration for model architecture, training,
checkpoints, TensorBoard, and model artifacts.
"""

from pathlib import Path

from src.preprocessing.config import (
    IMAGE_SIZE,
    RANDOM_STATE,
    RESULTS_DIR,
)

# =====================================================
# MOBILE NET V3 CONFIGURATION
# =====================================================

MOBILENETV3_LEARNING_RATE: float = 1e-3

MOBILENETV3_DENSE_UNITS_1: int = 256
MOBILENETV3_DENSE_UNITS_2: int = 128

MOBILENETV3_DROPOUT_RATE_1: float = 0.40
MOBILENETV3_DROPOUT_RATE_2: float = 0.30

MOBILENETV3_L2_WEIGHT: float = 1e-4

# =====================================================
# TRAINING CONFIGURATION
# =====================================================

BATCH_SIZE: int = 32

EPOCHS: int = 50

PATIENCE: int = 8

# =====================================================
# FINE TUNING
# =====================================================

FINE_TUNE_EPOCHS: int = 20

FINE_TUNE_LEARNING_RATE: float = 1e-5

# =====================================================
# CALLBACK CONFIGURATION
# =====================================================

REDUCE_LR_FACTOR: float = 0.2

MIN_LEARNING_RATE: float = 1e-7

TENSORBOARD_UPDATE_FREQ: str = "epoch"
# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

FIGURES_DIR = RESULTS_DIR / "figures"

MODELS_DIR = RESULTS_DIR / "models"

CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

HISTORY_DIR = RESULTS_DIR / "history"

LOGS_DIR = RESULTS_DIR / "logs"

# =====================================================
# OUTPUT FILES
# =====================================================

BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.keras"

FINE_TUNED_MODEL_PATH = (
    CHECKPOINTS_DIR / "best_model_finetuned.keras"
)

TRAINING_HISTORY_JSON = (
    HISTORY_DIR / "training_history.json"
)

TRAINING_HISTORY_PKL = (
    HISTORY_DIR / "training_history.pkl"
)

TRAINING_LOG_CSV = (
    LOGS_DIR / "training_log.csv"
)

TENSORBOARD_LOG_DIR = (
    LOGS_DIR / "tensorboard"
)

# =====================================================
# CREATE DIRECTORIES
# =====================================================

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# EXPORTS
# =====================================================

__all__ = [
    "IMAGE_SIZE",
    "RANDOM_STATE",
    "RESULTS_DIR",
    "MOBILENETV3_LEARNING_RATE",
    "MOBILENETV3_DENSE_UNITS_1",
    "MOBILENETV3_DENSE_UNITS_2",
    "MOBILENETV3_DROPOUT_RATE_1",
    "MOBILENETV3_DROPOUT_RATE_2",
    "MOBILENETV3_L2_WEIGHT",
    "BATCH_SIZE",
    "EPOCHS",
    "PATIENCE",
    "FINE_TUNE_EPOCHS",
    "FINE_TUNE_LEARNING_RATE",
    "FIGURES_DIR",
    "MODELS_DIR",
    "CHECKPOINTS_DIR",
    "HISTORY_DIR",
    "LOGS_DIR",
    "BEST_MODEL_PATH",
    "FINE_TUNED_MODEL_PATH",
    "TRAINING_HISTORY_JSON",
    "TRAINING_HISTORY_PKL",
    "TRAINING_LOG_CSV",
    "TENSORBOARD_LOG_DIR",
    "REDUCE_LR_FACTOR",
    "MIN_LEARNING_RATE",
    "TENSORBOARD_UPDATE_FREQ",
]
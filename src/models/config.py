"""
Model Configuration
DroneML

Central configuration for model architecture, training,
checkpoints, TensorBoard, and model artifacts.
Supports MobileNetV3-Large, ResNet50, and EfficientNetB0.
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
# RESNET50 CONFIGURATION
# =====================================================

RESNET50_LEARNING_RATE: float = 1e-3

RESNET50_DENSE_UNITS_1: int = 256
RESNET50_DENSE_UNITS_2: int = 128

RESNET50_DROPOUT_RATE_1: float = 0.40
RESNET50_DROPOUT_RATE_2: float = 0.30

RESNET50_L2_WEIGHT: float = 1e-4

# =====================================================
# EFFICIENTNETB0 CONFIGURATION
# =====================================================

EFFICIENTNETB0_LEARNING_RATE: float = 1e-3

EFFICIENTNETB0_DENSE_UNITS_1: int = 256
EFFICIENTNETB0_DENSE_UNITS_2: int = 128

EFFICIENTNETB0_DROPOUT_RATE_1: float = 0.40
EFFICIENTNETB0_DROPOUT_RATE_2: float = 0.30

EFFICIENTNETB0_L2_WEIGHT: float = 1e-4

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

FINE_TUNE_UNFREEZE_LAYERS: int = 20

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

COMPARISON_DIR = RESULTS_DIR / "comparison"

# =====================================================
# OUTPUT FILES – MobileNetV3
# =====================================================

MOBILENETV3_BEST_MODEL_PATH = (
    CHECKPOINTS_DIR / "mobilenetv3_best.keras"
)

MOBILENETV3_FINETUNED_MODEL_PATH = (
    CHECKPOINTS_DIR / "mobilenetv3_finetuned.keras"
)

MOBILENETV3_FINAL_MODEL_PATH = (
    MODELS_DIR / "mobilenetv3_final.keras"
)

MOBILENETV3_HISTORY_JSON = (
    HISTORY_DIR / "mobilenetv3_history.json"
)

MOBILENETV3_HISTORY_PKL = (
    HISTORY_DIR / "mobilenetv3_history.pkl"
)

# =====================================================
# OUTPUT FILES – ResNet50
# =====================================================

RESNET50_BEST_MODEL_PATH = (
    CHECKPOINTS_DIR / "resnet50_best.keras"
)

RESNET50_FINETUNED_MODEL_PATH = (
    CHECKPOINTS_DIR / "resnet50_finetuned.keras"
)

RESNET50_FINAL_MODEL_PATH = (
    MODELS_DIR / "resnet50_final.keras"
)

RESNET50_HISTORY_JSON = (
    HISTORY_DIR / "resnet50_history.json"
)

RESNET50_HISTORY_PKL = (
    HISTORY_DIR / "resnet50_history.pkl"
)

# =====================================================
# OUTPUT FILES – EfficientNetB0
# =====================================================

EFFICIENTNETB0_BEST_MODEL_PATH = (
    CHECKPOINTS_DIR / "efficientnetb0_best.keras"
)

EFFICIENTNETB0_FINETUNED_MODEL_PATH = (
    CHECKPOINTS_DIR / "efficientnetb0_finetuned.keras"
)

EFFICIENTNETB0_FINAL_MODEL_PATH = (
    MODELS_DIR / "efficientnetb0_final.keras"
)

EFFICIENTNETB0_HISTORY_JSON = (
    HISTORY_DIR / "efficientnetb0_history.json"
)

EFFICIENTNETB0_HISTORY_PKL = (
    HISTORY_DIR / "efficientnetb0_history.pkl"
)

# =====================================================
# OUTPUT FILES – Legacy (kept for backward compat.)
# =====================================================

BEST_MODEL_PATH = MOBILENETV3_BEST_MODEL_PATH

FINE_TUNED_MODEL_PATH = MOBILENETV3_FINETUNED_MODEL_PATH

TRAINING_HISTORY_JSON = MOBILENETV3_HISTORY_JSON

TRAINING_HISTORY_PKL = MOBILENETV3_HISTORY_PKL

TRAINING_LOG_CSV = LOGS_DIR / "training_log.csv"

TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"

# =====================================================
# COMPARISON FILES
# =====================================================

COMPARISON_RESULTS_JSON = COMPARISON_DIR / "comparison_results.json"

COMPARISON_REPORT_MD = COMPARISON_DIR / "comparison_report.md"

# =====================================================
# CREATE DIRECTORIES
# =====================================================

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# EXPORTS
# =====================================================

__all__ = [
    # Inherited from preprocessing config
    "IMAGE_SIZE",
    "RANDOM_STATE",
    "RESULTS_DIR",
    # MobileNetV3 hyperparams
    "MOBILENETV3_LEARNING_RATE",
    "MOBILENETV3_DENSE_UNITS_1",
    "MOBILENETV3_DENSE_UNITS_2",
    "MOBILENETV3_DROPOUT_RATE_1",
    "MOBILENETV3_DROPOUT_RATE_2",
    "MOBILENETV3_L2_WEIGHT",
    # ResNet50 hyperparams
    "RESNET50_LEARNING_RATE",
    "RESNET50_DENSE_UNITS_1",
    "RESNET50_DENSE_UNITS_2",
    "RESNET50_DROPOUT_RATE_1",
    "RESNET50_DROPOUT_RATE_2",
    "RESNET50_L2_WEIGHT",
    # EfficientNetB0 hyperparams
    "EFFICIENTNETB0_LEARNING_RATE",
    "EFFICIENTNETB0_DENSE_UNITS_1",
    "EFFICIENTNETB0_DENSE_UNITS_2",
    "EFFICIENTNETB0_DROPOUT_RATE_1",
    "EFFICIENTNETB0_DROPOUT_RATE_2",
    "EFFICIENTNETB0_L2_WEIGHT",
    # Training
    "BATCH_SIZE",
    "EPOCHS",
    "PATIENCE",
    "FINE_TUNE_EPOCHS",
    "FINE_TUNE_LEARNING_RATE",
    "FINE_TUNE_UNFREEZE_LAYERS",
    # Callbacks
    "REDUCE_LR_FACTOR",
    "MIN_LEARNING_RATE",
    "TENSORBOARD_UPDATE_FREQ",
    # Directories
    "FIGURES_DIR",
    "MODELS_DIR",
    "CHECKPOINTS_DIR",
    "HISTORY_DIR",
    "LOGS_DIR",
    "COMPARISON_DIR",
    # MobileNetV3 paths
    "MOBILENETV3_BEST_MODEL_PATH",
    "MOBILENETV3_FINETUNED_MODEL_PATH",
    "MOBILENETV3_FINAL_MODEL_PATH",
    "MOBILENETV3_HISTORY_JSON",
    "MOBILENETV3_HISTORY_PKL",
    # ResNet50 paths
    "RESNET50_BEST_MODEL_PATH",
    "RESNET50_FINETUNED_MODEL_PATH",
    "RESNET50_FINAL_MODEL_PATH",
    "RESNET50_HISTORY_JSON",
    "RESNET50_HISTORY_PKL",
    # EfficientNetB0 paths
    "EFFICIENTNETB0_BEST_MODEL_PATH",
    "EFFICIENTNETB0_FINETUNED_MODEL_PATH",
    "EFFICIENTNETB0_FINAL_MODEL_PATH",
    "EFFICIENTNETB0_HISTORY_JSON",
    "EFFICIENTNETB0_HISTORY_PKL",
    # Legacy aliases
    "BEST_MODEL_PATH",
    "FINE_TUNED_MODEL_PATH",
    "TRAINING_HISTORY_JSON",
    "TRAINING_HISTORY_PKL",
    "TRAINING_LOG_CSV",
    "TENSORBOARD_LOG_DIR",
    # Comparison
    "COMPARISON_RESULTS_JSON",
    "COMPARISON_REPORT_MD",
]
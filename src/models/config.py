"""
Model Configuration
DroneML

Central configuration for model architecture, training,
checkpoints, TensorBoard, and model artifacts.
Supports MobileNetV3-Large, ResNet50, and EfficientNetB0.

Phase directory layout
-----------------------
  results/
    figures/            – training curve PNGs
    models/             – final saved .keras models
    checkpoints/        – best / fine-tuned .keras checkpoints
    history/            – per-model training history JSON/pkl
    logs/               – TensorBoard + CSV logs
    comparison/         – multi-model comparison charts & report
    evaluation/         – Phase 10 evaluation outputs
    gradcam/            – Phase 11 Grad-CAM heatmaps
    error_analysis/     – Phase 12 error analysis outputs
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

REDUCE_LR_PATIENCE = PATIENCE

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

# Phase 10 – Evaluation
EVALUATION_DIR = RESULTS_DIR / "evaluation"

# Phase 11 – Explainability (Grad-CAM)
GRADCAM_DIR = RESULTS_DIR / "gradcam"

# Phase 12 – Error Analysis
ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"

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
# PHASE 10 – EVALUATION FILES
# =====================================================

# Per-model confusion matrices
MOBILENETEV3_CONFUSION_PNG = (
    EVALUATION_DIR / "confusion_matrix_mobilenetv3.png"
)
RESNET50_CONFUSION_PNG = (
    EVALUATION_DIR / "confusion_matrix_resnet50.png"
)
EFFICIENTNETB0_CONFUSION_PNG = (
    EVALUATION_DIR / "confusion_matrix_efficientnetb0.png"
)

# Shared ROC curve (all models on one plot)
ROC_CURVE_PNG = EVALUATION_DIR / "roc_curve.png"

# Per-model metrics JSON
MOBILENETV3_METRICS_JSON = (
    EVALUATION_DIR / "metrics_mobilenetv3.json"
)
RESNET50_METRICS_JSON = (
    EVALUATION_DIR / "metrics_resnet50.json"
)
EFFICIENTNETB0_METRICS_JSON = (
    EVALUATION_DIR / "metrics_efficientnetb0.json"
)

# Combined metrics (all models)
ALL_METRICS_JSON = EVALUATION_DIR / "metrics.json"

# =====================================================
# PHASE 11 – GRADCAM FILES
# =====================================================

# Per-model Grad-CAM sub-directories (created at runtime)
GRADCAM_MOBILENETV3_DIR = GRADCAM_DIR / "mobilenetv3"
GRADCAM_RESNET50_DIR    = GRADCAM_DIR / "resnet50"
GRADCAM_EFFICIENTNETB0_DIR = GRADCAM_DIR / "efficientnetb0"

# =====================================================
# PHASE 12 – ERROR ANALYSIS FILES
# =====================================================

ERROR_ANALYSIS_FP_DIR = ERROR_ANALYSIS_DIR / "false_positives"
ERROR_ANALYSIS_FN_DIR = ERROR_ANALYSIS_DIR / "false_negatives"
ERROR_ANALYSIS_REPORT_MD = ERROR_ANALYSIS_DIR / "error_report.md"
ERROR_ANALYSIS_FLIGHT_PNG = ERROR_ANALYSIS_DIR / "flight_wise_accuracy.png"
ERROR_ANALYSIS_ALTITUDE_PNG = ERROR_ANALYSIS_DIR / "altitude_wise_accuracy.png"
ERROR_ANALYSIS_DEPTH_PNG = ERROR_ANALYSIS_DIR / "depth_wise_accuracy.png"
ERROR_ANALYSIS_ZONE_PNG = ERROR_ANALYSIS_DIR / "zone_wise_accuracy.png"

# =====================================================
# CREATE DIRECTORIES
# =====================================================

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
ERROR_ANALYSIS_FP_DIR.mkdir(parents=True, exist_ok=True)
ERROR_ANALYSIS_FN_DIR.mkdir(parents=True, exist_ok=True)

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
    # Phase 10 – Evaluation
    "EVALUATION_DIR",
    "ROC_CURVE_PNG",
    "ALL_METRICS_JSON",
    "MOBILENETEV3_CONFUSION_PNG",
    "RESNET50_CONFUSION_PNG",
    "EFFICIENTNETB0_CONFUSION_PNG",
    "MOBILENETV3_METRICS_JSON",
    "RESNET50_METRICS_JSON",
    "EFFICIENTNETB0_METRICS_JSON",
    # Phase 11 – Grad-CAM
    "GRADCAM_DIR",
    "GRADCAM_MOBILENETV3_DIR",
    "GRADCAM_RESNET50_DIR",
    "GRADCAM_EFFICIENTNETB0_DIR",
    # Phase 12 – Error Analysis
    "ERROR_ANALYSIS_DIR",
    "ERROR_ANALYSIS_FP_DIR",
    "ERROR_ANALYSIS_FN_DIR",
    "ERROR_ANALYSIS_REPORT_MD",
    "ERROR_ANALYSIS_FLIGHT_PNG",
    "ERROR_ANALYSIS_ALTITUDE_PNG",
    "ERROR_ANALYSIS_DEPTH_PNG",
    "ERROR_ANALYSIS_ZONE_PNG",
]
"""Model-specific configuration for DroneML."""

from src.preprocessing.config import IMAGE_SIZE, RANDOM_STATE, RESULTS_DIR

# MobileNetV3 hyper-parameters
MOBILENETV3_LEARNING_RATE: float = 1e-3  # frozen-backbone phase
MOBILENETV3_DENSE_UNITS_1: int = 256
MOBILENETV3_DENSE_UNITS_2: int = 128
MOBILENETV3_DROPOUT_RATE_1: float = 0.40
MOBILENETV3_DROPOUT_RATE_2: float = 0.30
MOBILENETV3_L2_WEIGHT: float = 1e-4

# Directory for saving model architecture diagrams
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
    "FIGURES_DIR",
]

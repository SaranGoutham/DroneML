from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from src.models.config import (
    TRAINING_HISTORY_PKL,
    FIGURES_DIR,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(TRAINING_HISTORY_PKL, "rb") as f:
    history = pickle.load(f)

# -----------------------
# Loss Curve
# -----------------------

plt.figure(figsize=(8, 5))

plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()

plt.tight_layout()

plt.savefig(FIGURES_DIR / "loss_curve.png")

plt.close()

# -----------------------
# Accuracy Curve
# -----------------------

plt.figure(figsize=(8, 5))

plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(FIGURES_DIR / "accuracy_curve.png")

plt.close()

print("Training curves saved successfully.")
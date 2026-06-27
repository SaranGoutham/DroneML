import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import CHECKED_METADATA_FILE

# ======================================
# CONFIGURATION
# ======================================

METADATA_FILE = CHECKED_METADATA_FILE

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv(METADATA_FILE)

# Select a flight and zone
flight = "17_02_2020"
zone = "Zone 7 Mine 5cm depth"

subset = df[
    (df["flight_date"] == flight) &
    (df["zone"] == zone)
]

jpg = subset[subset["format_folder"] == "JPG"]
rjpg = subset[subset["format_folder"] == "R_JPG"]
tiff = subset[subset["actual_format"] == "TIFF"]

print("="*60)
print("COMPARE FORMATS")
print("="*60)

print(f"Flight : {flight}")
print(f"Zone   : {zone}")

print("\nImages Found")

print("JPG   :", len(jpg))
print("R_JPG :", len(rjpg))
print("TIFF  :", len(tiff))

# ======================================
# DISPLAY
# ======================================

fig, axes = plt.subplots(3, 3, figsize=(15,15))

titles = [
    "JPG",
    "R_JPG",
    "TIFF"
]

datasets = [
    jpg,
    rjpg,
    tiff
]

for row in range(3):

    data = datasets[row].head(3)

    for col in range(3):

        ax = axes[row][col]

        if col >= len(data):

            ax.axis("off")
            continue

        img_path = data.iloc[col]["image_path"]

        img = Image.open(img_path)

        img = np.array(img)

        if data.iloc[col]["actual_format"] == "TIFF":

            img = img.astype(np.float32)

            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            ax.imshow(img, cmap="inferno")

        else:

            ax.imshow(img)

        if col == 0:
            ax.set_ylabel(titles[row], fontsize=12)

        ax.set_title(data.iloc[col]["filename"][:18], fontsize=8)

        ax.axis("off")

plt.suptitle(
    "Comparison of JPG vs R_JPG vs TIFF",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.show()

print("\nDone.")

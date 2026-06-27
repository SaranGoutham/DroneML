import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import CHECKED_METADATA_FILE

# ==========================
# CONFIGURATION
# ==========================

METADATA_FILE = CHECKED_METADATA_FILE

NUM_IMAGES = 20

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(METADATA_FILE)

# Filter No Mine images
no_mine = df[df["label"] == "No_Mine"]

print("=" * 60)
print("NO MINE IMAGE VIEWER")
print("=" * 60)
print(f"Total No Mine Images : {len(no_mine)}")

# Random sample
sample = no_mine.sample(
    n=min(NUM_IMAGES, len(no_mine)),
    random_state=random.randint(1, 10000)
)

# ==========================
# DISPLAY IMAGES
# ==========================

fig, axes = plt.subplots(4, 5, figsize=(18, 12))
axes = axes.flatten()

for ax, (_, row) in zip(axes, sample.iterrows()):

    img = Image.open(row["image_path"])

    # Display grayscale correctly
    if img.mode in ["L", "I;16", "I"]:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)

    title = (
        f"{row['flight_date']}\n"
        f"{row['format_folder']}\n"
        f"{row['zone']}"
    )

    ax.set_title(title, fontsize=8)
    ax.axis("off")

# Hide unused axes
for ax in axes[len(sample):]:
    ax.axis("off")

plt.suptitle(
    "Random No Mine Samples",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

# ==========================
# PRINT SAMPLE DETAILS
# ==========================

print("\nSample Image Details\n")

print(
    sample[
        [
            "flight_date",
            "format_folder",
            "zone",
            "filename",
            "width",
            "height",
            "actual_format"
        ]
    ]
)

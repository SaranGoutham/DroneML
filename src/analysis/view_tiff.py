import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import CHECKED_METADATA_FILE

# =====================================
# CONFIGURATION
# =====================================

METADATA_FILE = CHECKED_METADATA_FILE

NUM_IMAGES = 12

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(METADATA_FILE)

# Select only TIFF images
tiff_df = df[df["actual_format"] == "TIFF"]

print("=" * 60)
print("TIFF IMAGE VIEWER")
print("=" * 60)

print(f"Total TIFF Images : {len(tiff_df)}")

sample = tiff_df.sample(
    n=min(NUM_IMAGES, len(tiff_df)),
    random_state=random.randint(1,10000)
)

# =====================================
# DISPLAY
# =====================================

fig, axes = plt.subplots(3,4, figsize=(16,12))
axes = axes.flatten()

for ax, (_, row) in zip(axes, sample.iterrows()):

    img = Image.open(row["image_path"])

    img = np.array(img).astype(np.float32)

    # Normalize 16-bit image for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    ax.imshow(img, cmap="inferno")

    title = (
        f"{row['flight_date']}\n"
        f"{row['depth']}\n"
        f"{row['zone']}"
    )

    ax.set_title(title, fontsize=8)
    ax.axis("off")

for ax in axes[len(sample):]:
    ax.axis("off")

plt.suptitle(
    "16-bit Thermal TIFF Images",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

# =====================================
# INFORMATION
# =====================================

print("\nSample Information\n")

print(sample[
[
"flight_date",
"zone",
"depth",
"width",
"height",
"actual_format",
"mode",
"image_path"
]
])

print("\nImage Statistics\n")

print("Resolution Counts")

print(
    tiff_df.groupby(["width","height"]).size()
)

print("\nImage Modes")

print(
    tiff_df["mode"].value_counts()
)

import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import (
    CHECKED_METADATA_FILE,
    DUPLICATE_IMAGES_FILE,
    RAW_METADATA_FILE,
)

METADATA_FILE = RAW_METADATA_FILE
OUTPUT_FILE = CHECKED_METADATA_FILE

df = pd.read_csv(METADATA_FILE)

df["width"] = 0
df["height"] = 0
df["channels"] = 0
df["mode"] = ""
df["file_size_kb"] = 0.0
df["actual_format"] = ""
df["corrupted"] = False
df["folder_extension_match"] = True
df["image_hash"] = ""

corrupted = 0
mismatch = 0

for i, row in df.iterrows():

    path = Path(row["image_path"])

    try:

        with Image.open(path) as img:

            img.verify()

        with Image.open(path) as img:

            width, height = img.size

            mode = img.mode

            fmt = img.format

            size = os.path.getsize(path) / 1024

            if mode == "RGB":
                channels = 3
            elif mode == "RGBA":
                channels = 4
            elif mode == "L":
                channels = 1
            else:
                channels = len(img.getbands())

        with open(path, "rb") as f:
            image_hash = hashlib.md5(f.read()).hexdigest()

        df.at[i, "width"] = width
        df.at[i, "height"] = height
        df.at[i, "channels"] = channels
        df.at[i, "mode"] = mode
        df.at[i, "file_size_kb"] = round(size, 2)
        df.at[i, "actual_format"] = fmt
        df.at[i, "image_hash"] = image_hash

        folder = row["format_folder"].upper()
        ext = path.suffix.lower()

        match = True

        if folder == "TIFF" and ext not in [".tif", ".tiff"]:
            match = False

        if folder == "JPG" and ext not in [".jpg", ".jpeg"]:
            match = False

        if folder == "R_JPG" and ext not in [".jpg", ".jpeg"]:
            match = False

        df.at[i, "folder_extension_match"] = match

        if not match:
            mismatch += 1

    except Exception:

        corrupted += 1
        df.at[i, "corrupted"] = True

duplicates = df[df.duplicated("image_hash", keep=False)]

duplicates.to_csv(
    DUPLICATE_IMAGES_FILE,
    index=False
)

df.to_csv(OUTPUT_FILE, index=False)

print("="*60)
print("IMAGE AUDIT COMPLETE")
print("="*60)
print("Images :", len(df))
print("Corrupted :", corrupted)
print("Wrong Extension :", mismatch)
print("Duplicates :", len(duplicates))
print("="*60)

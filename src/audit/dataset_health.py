import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import (
    CHECKED_METADATA_FILE,
    CLASS_DISTRIBUTION_FILE,
    DEPTH_DISTRIBUTION_FILE,
    FLIGHT_DISTRIBUTION_FILE,
)

df = pd.read_csv(CHECKED_METADATA_FILE)

print("="*60)
print("DATASET HEALTH REPORT")
print("="*60)

print(f"Total Images            : {len(df)}")
print(f"Mine Images             : {(df['label']=='Mine').sum()}")
print(f"No Mine Images          : {(df['label']=='No_Mine').sum()}")

print("\nFlights")
print(df["flight_date"].value_counts())

print("\nFormats")
print(df["format_folder"].value_counts())

print("\nDepths")
print(df["depth"].value_counts())

print("\nImage Extensions")
print(df["extension"].value_counts())

print("\nImage Formats")
print(df["actual_format"].value_counts())

print("\nImage Modes")
print(df["mode"].value_counts())

print("\nImage Sizes")
print(df.groupby(["width","height"]).size())

print("\nFolder Extension Mismatch")
print((~df["folder_extension_match"]).sum())

print("\nCorrupted Images")
print(df["corrupted"].sum())

print("="*60)

df.groupby("flight_date").size().to_csv(
    FLIGHT_DISTRIBUTION_FILE
)

df.groupby("depth").size().to_csv(
    DEPTH_DISTRIBUTION_FILE
)

df.groupby("label").size().to_csv(
    CLASS_DISTRIBUTION_FILE
)

print("CSV Reports Saved.")

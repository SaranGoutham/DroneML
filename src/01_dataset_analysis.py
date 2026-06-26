import os
from pathlib import Path

# ==========================================================
# LANDMINE DATASET ANALYSIS - VERSION 1
# Purpose:
#   Scan dataset structure and verify folders
# ==========================================================

# -----------------------------
# CHANGE THIS PATH IF REQUIRED
# -----------------------------
DATASET_PATH = r"D:\DroneML\dataset\DatainBrief_dataset_1"

print("=" * 60)
print("LANDMINE DATASET ANALYSIS")
print("=" * 60)
print(f"Dataset Path : {DATASET_PATH}\n")

dataset = Path(DATASET_PATH)

if not dataset.exists():
    print("❌ Dataset path not found!")
    exit()

# Find flight folders
flight_folders = sorted([f for f in dataset.iterdir() if f.is_dir()])

print(f"Total Flight Dates Found : {len(flight_folders)}")
print("=" * 60)

total_formats = 0
total_zone_folders = 0

for flight in flight_folders:

    print(f"\n📅 Flight Date : {flight.name}")

    format_folders = sorted([f for f in flight.iterdir() if f.is_dir()])

    print(f"Image Formats : {len(format_folders)}")

    total_formats += len(format_folders)

    for fmt in format_folders:

        zone_folders = sorted([z for z in fmt.iterdir() if z.is_dir()])

        total_zone_folders += len(zone_folders)

        print(f"   ├── {fmt.name:<6} : {len(zone_folders)} Zones")

        for zone in zone_folders:
            print(f"   │      └── {zone.name}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Flight Dates      : {len(flight_folders)}")
print(f"Image Formats     : {total_formats}")
print(f"Zone Folders      : {total_zone_folders}")

print("=" * 60)
print("Dataset structure verified successfully.")
print("=" * 60)
import re
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import DATASET_DIR, RAW_METADATA_FILE, RESULTS_DIR

# ==========================
# CONFIGURATION
# ==========================

DATASET_PATH = DATASET_DIR
RESULTS_PATH = RESULTS_DIR

VALID_FORMATS = ["JPG", "R_JPG", "TIFF"]

VALID_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
]

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# ==========================
# STORAGE
# ==========================

metadata = []

unexpected_folders = []

empty_folders = []

total_images = 0

# ==========================
# HELPER FUNCTIONS
# ==========================

def extract_altitude(filename):

    m = re.search(r'FlightAltitude[_ ]*([0-9.]+)m', filename)

    if m:
        return float(m.group(1))

    return None


def extract_temperature(filename):

    m = re.search(r'([0-9]+)\s*°?C', filename)

    if m:
        return int(m.group(1))

    return None


def get_depth(zone_name):

    zone = zone_name.lower()

    if "free" in zone:
        return "Free"

    m = re.search(r'(\d+)\s*cm', zone)

    if m:
        return m.group(1) + " cm"

    return "Unknown"


def get_label(zone_name):

    if "free" in zone_name.lower():
        return "No_Mine"

    return "Mine"

# ==========================
# START
# ==========================

print("=" * 70)
print("LANDMINE DATASET AUDITOR")
print("=" * 70)

dataset = Path(DATASET_PATH)

flight_folders = sorted(
    [x for x in dataset.iterdir() if x.is_dir()]
)

print(f"\nFlights Found : {len(flight_folders)}")

# ==========================
# SCAN
# ==========================

for flight in flight_folders:

    print(f"\nFlight : {flight.name}")

    for fmt in sorted(flight.iterdir()):

        if not fmt.is_dir():
            continue

        print(f"   Format : {fmt.name}")

        if fmt.name not in VALID_FORMATS:
            unexpected_folders.append(str(fmt))
            continue

        for zone in sorted(fmt.iterdir()):

            if not zone.is_dir():
                continue

            zone_name = zone.name

            image_files = []

            for ext in VALID_EXTENSIONS:

                image_files.extend(zone.glob(f"*{ext}"))
                image_files.extend(zone.glob(f"*{ext.upper()}"))

            if len(image_files) == 0:
                empty_folders.append(str(zone))
                continue

            for img in image_files:

                total_images += 1

                metadata.append({

                    "flight_date": flight.name,

                    "format_folder": fmt.name,

                    "zone": zone_name,

                    "depth": get_depth(zone_name),

                    "label": get_label(zone_name),

                    "filename": img.name,

                    "extension": img.suffix,

                    "altitude": extract_altitude(img.name),

                    "temperature": extract_temperature(img.name),

                    "image_path": str(img)

                })

# ==========================
# SAVE CSV
# ==========================

df = pd.DataFrame(metadata)

csv_path = RAW_METADATA_FILE

df.to_csv(csv_path, index=False)

# ==========================
# REPORT
# ==========================

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Flights            : {len(flight_folders)}")
print(f"Images             : {total_images}")
print(f"Metadata Rows      : {len(df)}")
print(f"Empty Folders      : {len(empty_folders)}")
print(f"Unexpected Folders : {len(unexpected_folders)}")

print("\nMetadata saved to")

print(csv_path)

print("=" * 70)

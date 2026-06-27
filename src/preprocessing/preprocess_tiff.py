"""
preprocess_tiff.py
==================
First preprocessing step of the DroneML pipeline.

Converts raw 16-bit thermal TIFF images from the Mendeley landmine dataset
into clean, standardized 224×224 RGB PNG images suitable for training
MobileNetV3-Large.

Processing Pipeline
-------------------
1.  Read TIFF image (Pillow → NumPy array)
2.  Verify image is a valid TIFF (magic-byte check)
3.  Convert uint16 → float32 NumPy array
4.  Min-Max normalize pixel values: 0–65535 → 0–255 (uint8)
5.  Apply CLAHE  (clip_limit=2.0, grid_size=8×8)
6.  Apply Gamma Correction (gamma=1.2)
7.  Resize to 224×224 (LANCZOS)
8.  Convert grayscale → RGB (channel duplication)
9.  Save as PNG
10. Preserve class label (Mine / No_Mine) from zone folder name

Dataset location : data/mendeley/
Output directory : processed_dataset/Mine/  and  processed_dataset/No_Mine/
Report           : results/preprocessing_report.txt

Label derivation
----------------
Zone folders whose name contains "free zone" (case-insensitive) -> No_Mine
All other zone folders                                           -> Mine

Note: In this dataset TIFF files are stored with a .jpg extension
      (confirmed via magic-byte analysis). Pillow handles them correctly.

Pipeline position
-----------------
Dataset Audit -> EDA -> preprocess_tiff.py -> split_by_flight.py
    -> dataset_loader.py -> augment.py -> MobileNetV3-Large
    -> Training -> Evaluation
"""

import time
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Paths (relative to the project root, i.e. the directory where this script lives)
from src.preprocessing.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    DATASET_DIR,
    GAMMA as CONFIG_GAMMA,
    IMAGE_SIZE,
    RESULTS_DIR,
    PROCESSED_DIR,
    METADATA_FILE,
    PREPROCESSING_REPORT_FILE,
)

DATASET_ROOT = DATASET_DIR
if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_ROOT}"
    )
OUTPUT_DIR = PROCESSED_DIR
METADATA_CSV = METADATA_FILE
REPORT_FILE = PREPROCESSING_REPORT_FILE
# Image processing parameters
TARGET_SIZE    = IMAGE_SIZE
CLAHE_CLIP     = CLAHE_CLIP_LIMIT
CLAHE_GRID     = CLAHE_GRID_SIZE
GAMMA          = CONFIG_GAMMA

# TIFF magic bytes (little-endian "II" and big-endian "MM" Intel/Motorola)
TIFF_MAGIC     = {b"II", b"MM"}

# Label mapping: any zone whose lowercase name contains this keyword -> No_Mine
NO_MINE_KEYWORD = "free zone"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def is_tiff_magic(path: Path) -> bool:
    """Return True if *path* starts with a TIFF magic byte sequence."""
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
        return magic in TIFF_MAGIC
    except OSError:
        return False


def derive_label(zone_folder_name: str) -> str:
    """
    Infer Mine/No_Mine label from the zone folder name.

    "free zone"  -> No_Mine
    Anything else -> Mine
    """
    return "No_Mine" if NO_MINE_KEYWORD in zone_folder_name.lower() else "Mine"


def normalize_uint16(array: np.ndarray) -> np.ndarray:
    """
    Min-Max normalize a uint16 NumPy array to uint8 (0-255).

    Parameters
    ----------
    array : np.ndarray
        Input array with dtype uint16.

    Returns
    -------
    np.ndarray
        Normalized array with dtype uint8.
    """
    return cv2.normalize(
        array,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters
    ----------
    image : np.ndarray
        Single-channel uint8 image.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(image)


def apply_gamma(image: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """
    Apply gamma correction to a uint8 image using a lookup table.

    Parameters
    ----------
    image : np.ndarray
        Single-channel uint8 image.
    gamma : float
        Gamma value. >1 brightens dark regions; <1 darkens them.

    Returns
    -------
    np.ndarray
        Gamma-corrected uint8 image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def process_image(src_path: Path):
    """
    Run the full preprocessing pipeline on a single TIFF image.

    Steps
    -----
    1. Validate TIFF magic bytes.
    2. Load with Pillow (handles I;16 mode correctly).
    3. Convert to uint16 NumPy array.
    4. Min-Max normalize -> uint8.
    5. CLAHE enhancement.
    6. Gamma correction.
    7. Resize to 224x224.
    8. Grayscale -> RGB.

    Parameters
    ----------
    src_path : Path
        Absolute or relative path to the source TIFF file.

    Returns
    -------
    np.ndarray or None
        Processed RGB image array (H x W x 3, uint8), or None on failure.
    """
    # ── 1. TIFF magic-byte check ──────────────────────────────────────────────
    if not is_tiff_magic(src_path):
        raise ValueError(f"Not a TIFF (bad magic bytes): {src_path.name}")

    # ── 2. Load with Pillow ───────────────────────────────────────────────────
    try:
        pil_img = Image.open(src_path)
    except (UnidentifiedImageError, OSError) as exc:
        raise IOError(f"Cannot open image: {exc}") from exc

    # ── 3. Convert to uint16 NumPy array ─────────────────────────────────────
    # Pillow returns mode "I;16" (16-bit little-endian) for these files.
    # np.array() on an I;16 image gives int32 by default; cast explicitly.
    arr = np.array(pil_img, dtype=np.uint16)

    # ── 4. Min-Max normalization  0-65535 -> 0-255 ───────────────────────────
    arr_u8 = normalize_uint16(arr)

    # ── 5. Median Filter (Noise Reduction) ────────────────────────────────────
    arr_filtered = cv2.medianBlur(arr_u8, 3)

    # ── 6. CLAHE ──────────────────────────────────────────────────────────────
    arr_clahe = apply_clahe(arr_filtered)

    # ── 7. Gamma Correction ───────────────────────────────────────────────────
    arr_gamma = apply_gamma(arr_clahe)

    # ── 8. Resize to 224x224 (LANCZOS = high-quality downsampling) ────────────
    arr_resized = cv2.resize(
        arr_gamma,
        TARGET_SIZE,
        interpolation=cv2.INTER_LANCZOS4,
    )

    # ── 9. Grayscale -> RGB ───────────────────────────────────────────────────
    arr_rgb = cv2.cvtColor(arr_resized, cv2.COLOR_GRAY2RGB)

    return arr_rgb


def save_image(arr: np.ndarray, dst_path: Path) -> None:
    """
    Save a uint8 RGB NumPy array as a PNG file.

    Parameters
    ----------
    arr : np.ndarray
        RGB image array (H x W x 3, uint8).
    dst_path : Path
        Destination file path (will be created, parents must exist).
    """
    # OpenCV expects BGR; convert before saving
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(dst_path), bgr)
    if not ok:
        raise IOError(f"cv2.imwrite failed for: {dst_path}")


# ──────────────────────────────────────────────────────────────────────────────
# DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────

def discover_tiff_images(dataset_root: Path) -> list:
    """
    Walk *dataset_root* and collect every file inside a 'TIFF' subfolder.

    Returns
    -------
    list of dict with keys:
        path  : Path  – absolute path to the file
        label : str   – "Mine" or "No_Mine"
        flight: str   – flight date folder name (e.g. "13_02_2020")
        zone  : str   – zone folder name
    """
    records = []
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}

    for tiff_dir in sorted(dataset_root.rglob("TIFF")):
        if not tiff_dir.is_dir():
            continue
        flight_name = tiff_dir.parent.name  # e.g. "13_02_2020"
        for zone_dir in sorted(tiff_dir.iterdir()):
            if not zone_dir.is_dir():
                continue
            label = derive_label(zone_dir.name)
            for img_path in sorted(zone_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                    records.append(
                        {
                            "path": img_path,
                            "label": label,
                            "flight": flight_name,
                            "zone": zone_dir.name,
                        }
                    )
    return records


def load_metadata(csv_path: Path):
    """
    Load optional metadata CSV (results/metadata_checked.csv) if it exists.

    The CSV is expected to contain at minimum a 'filepath' column and a
    'label' column. When present its label column overrides folder-derived
    labels; when absent folder-name heuristics are used exclusively.

    Returns None if the file does not exist or cannot be parsed.
    """
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        required = {"image_path", "label"}
        if not required.issubset(set(df.columns)):
            print(
                f"  [WARNING] metadata CSV missing required columns "
                f"({required}). Falling back to folder-name labels."
            )
            return None
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARNING] Could not read metadata CSV: {exc}")
        return None


def apply_metadata_labels(records: list, meta_df) -> list:
    """
    Override folder-derived labels with values from the metadata CSV.

    The CSV's 'image_path' values are matched against the string representation
    of each record's path (partial suffix match, OS-agnostic).
    """
    if meta_df is None:
        return records

    # Build a lookup: normalised partial path -> label
    lookup = {}
    for _, row in meta_df.iterrows():
        key = Path(row["image_path"]).as_posix().lower()
        lookup[key] = str(row["label"])

    updated = 0
    for rec in records:
        posix_path = rec["path"].as_posix().lower()
        for key, lbl in lookup.items():
            if posix_path.endswith(key) or key.endswith(
                Path(rec["path"]).name.lower()
            ):
                rec["label"] = lbl
                updated += 1
                break

    if updated:
        print(f"  Overrode {updated} labels from metadata CSV.")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────

def write_report(
    report_path: Path,
    *,
    total: int,
    success: int,
    failed: int,
    mine_count: int,
    no_mine_count: int,
    elapsed: float,
    output_dir: Path,
) -> None:
    """Write a plain-text preprocessing report to *report_path*."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 45,
        "PREPROCESSING REPORT",
        "=" * 45,
        f"Generated        : {timestamp}",
        "",
        "── Dataset ──────────────────────────────────",
        f"Total TIFF found : {total}",
        f"Successfully proc: {success}",
        f"Failed           : {failed}",
        "",
        "── Class Distribution ───────────────────────",
        f"Mine images      : {mine_count}",
        f"No Mine images   : {no_mine_count}",
        "",
        "── Output ───────────────────────────────────",
        f"Output directory : {output_dir.resolve()}",
        f"Image size       : {TARGET_SIZE[0]} x {TARGET_SIZE[1]} px",
        f"Format           : PNG  (RGB)",
        "",
        "── Processing Parameters ────────────────────",
        f"CLAHE clip limit : {CLAHE_CLIP}",
        f"CLAHE grid size  : {CLAHE_GRID[0]} x {CLAHE_GRID[1]}",
        f"Gamma value      : {GAMMA}",
        "",
        "── Timing ───────────────────────────────────",
        f"Processing time  : {elapsed:.2f} s",
        f"Throughput       : {success / elapsed:.1f} img/s" if elapsed > 0 else "",
        "=" * 45,
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 45)
    print("TIFF PREPROCESSING")
    print("=" * 45)
    print()

    start_time = time.perf_counter()

    # ── 1. Setup output directories ───────────────────────────────────────────
    mine_dir    = OUTPUT_DIR / "Mine"
    no_mine_dir = OUTPUT_DIR / "No_Mine"
    mine_dir.mkdir(parents=True, exist_ok=True)
    no_mine_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Read metadata (optional) ───────────────────────────────────────────
    print("Reading Metadata...")
    meta_df = load_metadata(METADATA_CSV)
    if meta_df is not None:
        print(f"  Loaded {len(meta_df)} rows from {METADATA_CSV}.")
    else:
        print("  No metadata CSV found – deriving labels from folder names.")
    print()

    # ── 3. Discover TIFF images ───────────────────────────────────────────────
    records = discover_tiff_images(DATASET_ROOT)
    if meta_df is not None:
        records = apply_metadata_labels(records, meta_df)

    total = len(records)
    print(f"Found TIFF Images : {total}")
    print()

    if total == 0:
        print("[ERROR] No images found. Check DATASET_ROOT path.")
        return

    # ── 4. Per-class counters for naming ──────────────────────────────────────
    name_counters = {"Mine": 0, "No_Mine": 0}
    success_count = 0
    failed_count  = 0
    failed_files  = []

    # ── 5. Processing loop ────────────────────────────────────────────────────
    print("Processing...")
    bar = tqdm(
        records,
        total=total,
        unit="img",
        bar_format="{l_bar}{bar:30}{r_bar}",
        colour="cyan",
    )

    for rec in bar:
        src_path = rec["path"]
        label    = rec["label"]

        name_counters[label] += 1
        flight = rec["flight"].replace(" ", "_")
        zone = (
            rec["zone"]
            .replace(" ", "_")
            .replace("/", "_")
        )
        prefix = "Mine" if label == "Mine" else "NoMine"
        out_filename = (
            f"{flight}_{zone}_{prefix}_{name_counters[label]:06d}.png"
        )

        out_dir = mine_dir if label == "Mine" else no_mine_dir
        dst_path = out_dir / out_filename
        
        bar.set_postfix_str(f"-> {out_filename}", refresh=False)

        try:
            arr_rgb = process_image(src_path)
            if arr_rgb is None:
                raise RuntimeError("process_image returned None")
            save_image(arr_rgb, dst_path)
            success_count += 1

        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            failed_files.append(f"{src_path.name}: {exc}")
            bar.write(f"  [FAIL] {src_path.name} – {exc}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start_time

    mine_count    = name_counters["Mine"]
    no_mine_count = name_counters["No_Mine"]

    print()
    print(f"Successfully Processed : {success_count}")
    print(f"Failed                 : {failed_count}")
    print()
    print(f"Mine Images    : {mine_count}")
    print(f"No Mine Images : {no_mine_count}")
    print()
    print("Processed Dataset Saved To")
    print(f"  {OUTPUT_DIR}/")
    print()

    if failed_files:
        print("-" * 45)
        print("Failed Files:")
        for entry in failed_files:
            print(f"  - {entry}")
        print()

    # ── 7. Write report ───────────────────────────────────────────────────────
    write_report(
        REPORT_FILE,
        total=total,
        success=success_count,
        failed=failed_count,
        mine_count=mine_count,
        no_mine_count=no_mine_count,
        elapsed=elapsed,
        output_dir=OUTPUT_DIR,
    )
    print(f"Report saved to: {REPORT_FILE}")
    print()
    if elapsed > 0:
        print(
            f"Processing time: {elapsed:.2f}s  "
            f"({success_count / elapsed:.1f} img/s)"
        )

    print()
    print("=" * 45)
    print("Finished")
    print("=" * 45)
    print()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""dataset_loader.py
=====================

Loads the flight‑split processed dataset and prepares plain Python records for
subsequent stages (augmentation, TF conversion, training). The module performs
only validation and record creation – no image preprocessing, augmentation, or
TensorFlow operations are performed here.

Expected input layout (produced by ``split_by_flight.py``)::

    processed_dataset/
        train/
            Mine/
            No_Mine/
        validation/
            Mine/
            No_Mine/
        test/
            Mine/
            No_Mine/

All images are PNG, RGB, 224×224.

The public API consists of the following helper functions:

- ``validate_dataset`` – sanity‑check the directory structure and image
  integrity.
- ``collect_records`` – walk a split directory and build a list of dictionaries
  describing each sample.
- ``load_split`` – thin wrapper that validates and returns records for a single
  split.
- ``print_statistics`` – pretty‑print per‑split counts.
- ``load_dataset`` – orchestrates validation, loading of all three splits and
  shuffles the training set.
- ``main`` – CLI entry point that mirrors the console output style defined in
  the specification.

Only the standard library and Pillow are used.
"""

from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import IMAGE_SIZE, PROCESSED_DIR, RANDOM_STATE

# ---------------------------------------------------------------------------
# CONFIGURATION / CONSTANTS
# ---------------------------------------------------------------------------

# Root directory containing the split dataset (produced by ``split_by_flight.py``).
DATASET_ROOT: Path = PROCESSED_DIR

# Expected class folder names.
CLASSES: List[str] = ["Mine", "No_Mine"]

# Label mapping – matches the specification.
CLASS_TO_INDEX: Dict[str, int] = {"No_Mine": 0, "Mine": 1}
INDEX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_INDEX.items()}

# Expected image dimensions (width, height).
EXPECTED_SIZE: Tuple[int, int] = IMAGE_SIZE

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _fail(msg: str) -> None:
    """Print an error message and exit the program.

    All validation failures funnel through this function for a consistent user
    experience.
    """
    print(f"[ERROR] {msg}")
    sys.exit(1)


def _verify_image(path: Path) -> None:
    """Validate that *path* points to a 224×224 RGB PNG image.

    Raises ``ValueError`` with a descriptive message if any check fails.
    """
    if path.suffix.lower() != ".png":
        raise ValueError(f"File is not a PNG: {path}")
    try:
        with Image.open(path) as img:
            if img.format != "PNG":
                raise ValueError(f"Image format is not PNG (found {img.format}): {path}")
            if img.mode != "RGB":
                raise ValueError(f"Image mode is not RGB (found {img.mode}): {path}")
            if img.size != EXPECTED_SIZE:
                raise ValueError(f"Image size {img.size} != {EXPECTED_SIZE}: {path}")
    except OSError as exc:
        raise ValueError(f"Unable to open image {path}: {exc}")

# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def validate_dataset(root: Path = DATASET_ROOT) -> None:
    """Validate the overall dataset layout and image integrity.

    Checks performed:
    - ``train``, ``validation`` and ``test`` directories exist.
    - Within each split, ``Mine`` and ``No_Mine`` folders exist.
    - Each class folder contains at least one PNG file.
    - Every PNG file passes ``_verify_image`` (RGB, 224×224).
    """
    for split in ("train", "validation", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            _fail(f"Missing split folder: {split_dir}")
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                _fail(f"Missing class folder '{cls}' in split '{split}': {cls_dir}")
            png_files = list(cls_dir.glob("*.png"))
            if not png_files:
                _fail(f"No PNG images found in {cls_dir}")
            for img_path in png_files:
                try:
                    _verify_image(img_path)
                except ValueError as exc:
                    _fail(str(exc))
    print("[PASS] Dataset validation succeeded.")


def collect_records(split: str, root: Path = DATASET_ROOT) -> List[Dict[str, object]]:
    """Collect records for *split* (``train``, ``validation`` or ``test``).

    Returns a list of dictionaries with keys:
    ``image_path`` (Path), ``label`` (int), ``class_name`` (str), ``split`` (str).
    """
    split_dir = root / split
    records: List[Dict[str, object]] = []
    for cls in CLASSES:
        cls_dir = split_dir / cls
        for img_path in sorted(cls_dir.glob("*.png")):
            # Image validity has already been checked by ``validate_dataset``.
            records.append({
                "image_path": img_path,
                "label": CLASS_TO_INDEX[cls],
                "class_name": cls,
                "split": split,
            })
    return records


def load_split(split_name: str, root: Path = DATASET_ROOT) -> List[Dict[str, object]]:
    """Load a single split after ensuring the directory exists.

    This thin wrapper exists for API symmetry; the heavy validation is performed
    by ``validate_dataset``.
    """
    split_dir = root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return collect_records(split_name, root)


def print_statistics(train: List[Dict], val: List[Dict], test: List[Dict]) -> None:
    """Print statistics matching the console format from the specification.
    """
    def count(split_records: List[Dict]) -> Tuple[int, int]:
        mine = sum(1 for r in split_records if r["class_name"] == "Mine")
        no_mine = sum(1 for r in split_records if r["class_name"] == "No_Mine")
        return mine, no_mine

    print("\nDataset Statistics\n")
    for name, records in (("Train", train), ("Validation", val), ("Test", test)):
        mine_cnt, no_mine_cnt = count(records)
        total = len(records)
        print(f"{name}\n  Mine    : {mine_cnt}\n  No Mine : {no_mine_cnt}\n  Total   : {total}\n")
    total_all = len(train) + len(val) + len(test)
    print(f"Total Images : {total_all}\n")


def load_dataset(root: Path = DATASET_ROOT) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Validate the dataset and return record lists for all three splits.

    The training records are shuffled in‑place to provide randomness for later
    stages.
    """
    # 1️⃣ Validate overall structure and image integrity.
    validate_dataset(root)

    # 2️⃣ Load each split.
    train_records = load_split("train", root)
    val_records = load_split("validation", root)
    test_records = load_split("test", root)

    # 3️⃣ Shuffle training set only.
    random.Random(RANDOM_STATE).shuffle(train_records)

    # 4️⃣ Print a concise summary.
    print_statistics(train_records, val_records, test_records)

    return train_records, val_records, test_records


def main() -> None:
    """CLI entry point that mirrors the specification's console output."""
    print("\n" + "=" * 41)
    print("DATASET LOADER")
    print("=" * 41 + "\n")
    print("Validating Dataset…")
    # ``load_dataset`` performs validation and prints a PASS line.
    train, val, test = load_dataset()
    # The statistics are already printed by ``load_dataset``; we add a final line.
    print("Finished")
    print("=" * 41)

if __name__ == "__main__":
    main()

# verify_dataset.py – DroneML Specification
"""Utility module to verify the integrity of the processed DroneML dataset.

The verification performs a series of checks on the folder structure, image
properties, corruption, duplicates, and data leakage across splits. A concise
report is written to ``results/verification_report.txt``.

Only the standard library and Pillow are used – no TensorFlow, OpenCV or NumPy.
"""

from __future__ import annotations

import collections
import datetime
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import PROCESSED_DIR, VERIFICATION_REPORT_FILE

# ---------------------------------------------------------------------------
# Helper Types
# ---------------------------------------------------------------------------

SplitCounts = Dict[str, Dict[str, int]]  # split -> class -> count

# ---------------------------------------------------------------------------
# Verification Functions
# ---------------------------------------------------------------------------

def verify_structure(base_path: Path) -> Tuple[bool, List[str]]:
    """Verify that the required folder hierarchy exists.

    Expected layout::

        base_path/
            train/{Mine,No_Mine}/
            validation/{Mine,No_Mine}/
            test/{Mine,No_Mine}/

    Parameters
    ----------
    base_path:
        Root directory that contains the three splits.

    Returns
    -------
    Tuple[bool, List[str]]
        ``True`` if the structure is correct, otherwise ``False``. The list
        contains human‑readable messages describing missing items.
    """
    required_splits = {"train", "validation", "test"}
    required_classes = {"Mine", "No_Mine"}
    missing: List[str] = []
    ok = True

    for split in required_splits:
        split_dir = base_path / split
        if not split_dir.is_dir():
            ok = False
            missing.append(f"Missing split directory: {split_dir}")
            continue
        for cls in required_classes:
            class_dir = split_dir / cls
            if not class_dir.is_dir():
                ok = False
                missing.append(f"Missing class folder: {class_dir}")
    return ok, missing


def _check_image(image_path: Path) -> Tuple[bool, str | None]:
    """Validate a single image.

    Returns a tuple *(valid, error_message)*. ``error_message`` is ``None`` when
    the image passes all checks.
    """
    if image_path.suffix.lower() != ".png":
        return False, "Not a PNG file"
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                return False, f"Mode {img.mode} (expected RGB)"
            if img.size != (224, 224):
                return False, f"Size {img.size} (expected 224x224)"
    except (UnidentifiedImageError, OSError) as exc:
        return False, f"Corrupted image ({exc})"
    return True, None


def verify_images(base_path: Path) -> Tuple[bool, List[str]]:
    """Validate every image inside the dataset.

    Checks that each file is a PNG, RGB, and 224×224. Corrupted files are also
    reported.
    """
    errors: List[str] = []
    for img_path in base_path.rglob("*.png"):
        valid, err = _check_image(img_path)
        if not valid:
            errors.append(f"{img_path}: {err}")
    return len(errors) == 0, errors


def verify_duplicates(base_path: Path) -> Tuple[bool, List[str]]:
    """Detect duplicate filenames within each split.

    Returns ``True`` when no duplicates are found.
    """
    duplicate_reports: List[str] = []
    for split in ("train", "validation", "test"):
        split_dir = base_path / split
        if not split_dir.is_dir():
            continue  # Already reported by ``verify_structure``
        name_counter: collections.Counter = collections.Counter()
        for file_path in split_dir.rglob("*.png"):
            name_counter[file_path.name] += 1
        duplicates = [name for name, cnt in name_counter.items() if cnt > 1]
        if duplicates:
            duplicate_reports.append(
                f"Duplicate filenames in {split}: {', '.join(duplicates)}"
            )
    return len(duplicate_reports) == 0, duplicate_reports


def verify_leakage(base_path: Path) -> Tuple[bool, List[str]]:
    """Ensure that no image filename appears in more than one split.

    Returns ``True`` when the dataset is leakage‑free.
    """
    filename_to_splits: Dict[str, Set[str]] = {}
    for split in ("train", "validation", "test"):
        split_dir = base_path / split
        if not split_dir.is_dir():
            continue
        for file_path in split_dir.rglob("*.png"):
            filename = file_path.name
            filename_to_splits.setdefault(filename, set()).add(split)
    leakage: List[str] = []
    for filename, splits in filename_to_splits.items():
        if len(splits) > 1:
            leakage.append(f"{filename} appears in splits: {', '.join(sorted(splits))}")
    return len(leakage) == 0, leakage


def _count_classes(base_path: Path) -> Tuple[SplitCounts, int]:
    """Count images per class for each split and return the grand total.
    """
    counts: SplitCounts = {
        "train": {"Mine": 0, "No_Mine": 0},
        "validation": {"Mine": 0, "No_Mine": 0},
        "test": {"Mine": 0, "No_Mine": 0},
    }
    total = 0
    for split in counts:
        for cls in counts[split]:
            class_dir = base_path / split / cls
            if class_dir.is_dir():
                num = len(list(class_dir.rglob("*.png")))
                counts[split][cls] = num
                total += num
    return counts, total


def print_summary(base_path: Path) -> None:
    """Print a formatted verification summary to STDOUT.
    """
    sep = "=" * 39
    print(sep)
    print("DATASET VERIFICATION")
    print(sep)

    # Folder structure
    struct_ok, struct_msgs = verify_structure(base_path)
    print("Checking Folder Structure")
    print("PASS" if struct_ok else "FAIL")
    if not struct_ok:
        for line in struct_msgs:
            print(line)

    # Images
    image_ok, image_errs = verify_images(base_path)
    print("Checking Images")
    print("PASS" if image_ok else "FAIL")

    # Corrupted Images
    print("Checking Corrupted Images")
    print("PASS" if image_ok else "FAIL")
    if not image_ok:
        for line in image_errs:
            print(line)

    # Duplicates
    dup_ok, dup_msgs = verify_duplicates(base_path)
    print("Checking Duplicate Images")
    print("PASS" if dup_ok else "FAIL")
    if not dup_ok:
        for line in dup_msgs:
            print(line)

    # Leakage
    leak_ok, leak_msgs = verify_leakage(base_path)
    print("Checking Leakage")
    print("PASS" if leak_ok else "FAIL")
    if not leak_ok:
        for line in leak_msgs:
            print(line)

    # Summary
    counts, total_images = _count_classes(base_path)
    print("\nSummary")
    for split in ("train", "validation", "test"):
        print(f"{split.capitalize()}")
        for cls in ("Mine", "No_Mine"):
            print(f"  {cls}: {counts[split][cls]}")
    print(f"Total Images: {total_images}")
    if struct_msgs or image_errs  or dup_msgs or leak_msgs:
        print("\nDataset Verification Failed")
    else:
        print("\nDataset Verification Successful")


def _write_report(base_path: Path, report_path: Path) -> None:
    """Create ``verification_report.txt`` with detailed statistics.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")   
    counts, total_images = _count_classes(base_path)
    _, struct_msgs = verify_structure(base_path)
    _, image_errs = verify_images(base_path)
    _, dup_msgs = verify_duplicates(base_path)
    _, leak_msgs = verify_leakage(base_path)
    report_lines = [
        f"Timestamp: {timestamp}",
        f"Total Images: {total_images}",
    ]
    for split in ("train", "validation", "test"):
        report_lines.append(f"{split.capitalize()} Images: {sum(counts[split].values())}")
        for cls in ("Mine", "No_Mine"):
            report_lines.append(f"{split.capitalize()} {cls} Count: {counts[split][cls]}")
    report_lines.append(f"Corrupted Images: {len(image_errs)}")
    report_lines.append(f"Duplicate Images: {len(dup_msgs)}")
    report_lines.append(f"Leakage Result: {'PASS' if not leak_msgs else 'FAIL'}")
    report_lines.append(
        "Final Status: " + ("Successful" if not (struct_msgs or image_errs or dup_msgs or leak_msgs) else "Failed")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    """Run the full verification pipeline and generate the report.

    The default ``base_path`` is the project-rooted ``processed_dataset``.
    """
    base_path = PROCESSED_DIR
    print_summary(base_path)
    report_path = VERIFICATION_REPORT_FILE
    _write_report(base_path, report_path)
    print(f"Verification report written to {report_path}")


if __name__ == "__main__":
    main()

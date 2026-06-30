# -*- coding: utf-8 -*-
"""augment_offline.py
======================

Third step in the DroneML preprocessing pipeline.

Applies offline data augmentation to the training dataset inside
`processed_dataset/train/` to increase dataset size and resolve the
class imbalance (220 Mine vs. 30 No_Mine).

It operates purely on the preprocessed PNG files produced by
`preprocess_tiff.py` and split by `split_by_flight.py`.

The script runs in an idempotent fashion: any existing augmented files
containing the suffix `_aug_` will be deleted before starting a new run.
"""

from __future__ import annotations

import datetime
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import (
    AUGMENTATION_FACTORS,
    OFFLINE_AUG_REPORT_FILE,
    PROCESSED_DIR,
    RANDOM_STATE,
)

# Set random seeds for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Class sub-folders inside processed_dataset/train
CLASSES = ["Mine", "No_Mine"]


def clean_existing_augmentations(directory: Path) -> int:
    """Scan directory and remove any previously generated augmented files.

    Augmented files are identified by the presence of '_aug_' in their names.

    Returns the count of deleted files.
    """
    deleted_count = 0
    if not directory.exists():
        return deleted_count

    for file_path in directory.rglob("*.png"):
        if "_aug_" in file_path.name:
            try:
                file_path.unlink()
                deleted_count += 1
            except OSError as exc:
                print(f"[WARNING] Could not delete {file_path}: {exc}")
    return deleted_count


def apply_augmentations(image: np.ndarray) -> np.ndarray:
    """Apply a sequence of random, physically realistic transformations to an image.

    Transformations:
    1. Random Horizontal/Vertical Flip (50% probability each)
    2. Random Rotation (±15 degrees with border replication)
    3. Random Translation (±10% width/height shifts)
    4. Random Scale/Zoom (90% to 110% zoom factor)
    5. Random Contrast and Brightness scaling
    6. Gaussian Blur (30% probability)
    """
    augmented = image.copy()
    h, w = augmented.shape[:2]

    # 1. Random Flips
    if random.random() < 0.5:
        augmented = cv2.flip(augmented, 1)  # Horizontal flip
    if random.random() < 0.5:
        augmented = cv2.flip(augmented, 0)  # Vertical flip

    # 2. Random Rotation
    angle = random.uniform(-15, 15)
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented = cv2.warpAffine(
        augmented,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # 3. Random Translation
    tx = random.uniform(-0.10, 0.10) * w
    ty = random.uniform(-0.10, 0.10) * h
    trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented = cv2.warpAffine(
        augmented,
        trans_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # 4. Random Zoom (Scale)
    zoom_factor = random.uniform(0.9, 1.10)
    if zoom_factor != 1.0:
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized = cv2.resize(augmented, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if zoom_factor < 1.0:
            # Zoom out: pad image borders back to original dimensions
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            augmented = cv2.copyMakeBorder(
                resized,
                pad_y,
                h - new_h - pad_y,
                pad_x,
                w - new_w - pad_x,
                borderType=cv2.BORDER_REPLICATE,
            )
        else:
            # Zoom in: crop central section back to original dimensions
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            augmented = resized[start_y : start_y + h, start_x : start_x + w]

    # 5. Random Contrast and Brightness
    alpha = random.uniform(0.85, 1.15)  # Contrast scale
    beta = random.uniform(-15, 15)      # Brightness offset
    augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

    # 6. Random Gaussian Blur
    if random.random() < 0.3:
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    return augmented


def write_report(
    report_path: Path,
    *,
    original_counts: Dict[str, int],
    augmented_counts: Dict[str, int],
    deleted_count: int,
    elapsed: float,
) -> None:
    """Save a plain-text report of the offline augmentation run."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_orig = sum(original_counts.values())
    total_aug = sum(augmented_counts.values())

    lines = [
        "=" * 45,
        "OFFLINE DATA AUGMENTATION REPORT",
        "=" * 45,
        f"Generated        : {timestamp}",
        f"Random Seed      : {RANDOM_STATE}",
        "",
        "── Cleanup ──────────────────────────────────",
        f"Pre-existing files deleted : {deleted_count}",
        "",
        "── Original Training Set ────────────────────",
        f"Mine               : {original_counts['Mine']}",
        f"No_Mine            : {original_counts['No_Mine']}",
        f"Total Original     : {total_orig}",
        "",
        "── Augmentation Multipliers ──────────────────",
        f"Mine Factor        : {AUGMENTATION_FACTORS['Mine']}x",
        f"No_Mine Factor     : {AUGMENTATION_FACTORS['No_Mine']}x",
        "",
        "── Generated Augmented Files ────────────────",
        f"Mine Augmented     : {augmented_counts['Mine']}",
        f"No_Mine Augmented  : {augmented_counts['No_Mine']}",
        f"Total Augmented    : {total_aug}",
        "",
        "── Final Training Set Status ────────────────",
        f"Mine (Total)       : {original_counts['Mine'] + augmented_counts['Mine']}",
        f"No_Mine (Total)    : {original_counts['No_Mine'] + augmented_counts['No_Mine']}",
        f"Total Images       : {total_orig + total_aug}",
        f"Class Balance Ratio: {(original_counts['Mine'] + augmented_counts['Mine']) / (original_counts['No_Mine'] + augmented_counts['No_Mine']):.2f} : 1",
        "",
        "── Processing Parameters ────────────────────",
        "✓ Idempotent run (Auto clean-up)",
        "✓ Random Flips (Horizontal / Vertical)",
        "✓ Random Yaw Rotation (±15° with border replication)",
        "✓ Random Translation (±10% width/height)",
        "✓ Random Scale / Zoom (90% - 110%)",
        "✓ Contrast Adjustment (alpha: 0.85 - 1.15)",
        "✓ Brightness Offset (beta: -15 to +15)",
        "✓ Gaussian Blur (30% probability, kernel 3x3)",
        "",
        "── Performance ──────────────────────────────",
        f"Execution Time     : {elapsed:.2f} s",
        f"Augmentation Speed : {total_aug / elapsed:.1f} img/s" if elapsed > 0 else "",
        "=" * 45,
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print()
    print("=" * 45)
    print("OFFLINE DATA AUGMENTATION")
    print("=" * 45)
    print()

    start_time = time.perf_counter()

    train_dir = PROCESSED_DIR / "train"
    if not train_dir.exists():
        print(f"[ERROR] Training directory not found at: {train_dir.resolve()}")
        print("Please run `preprocess_tiff` and `split_by_flight` first.")
        sys.exit(1)

    # 1. Clean existing augmented images (idempotence check)
    print("Cleaning up old augmented images...")
    deleted_count = clean_existing_augmentations(train_dir)
    if deleted_count > 0:
        print(f"  Removed {deleted_count} previous augmented PNG files.")
    else:
        print("  No previous augmented files found. Safe to proceed.")
    print()

    # 2. Gather original files
    original_files: Dict[str, List[Path]] = {cls: [] for cls in CLASSES}
    for cls in CLASSES:
        cls_dir = train_dir / cls
        if cls_dir.is_dir():
            # Filter out any files containing _aug_ just in case (should be empty though)
            original_files[cls] = sorted(
                [p for p in cls_dir.glob("*.png") if "_aug_" not in p.name]
            )

    original_counts = {cls: len(files) for cls, files in original_files.items()}
    print("Original training image counts:")
    print(f"  Mine    : {original_counts['Mine']}")
    print(f"  No_Mine : {original_counts['No_Mine']}")
    print()

    if sum(original_counts.values()) == 0:
        print("[ERROR] No original images found inside train folders.")
        sys.exit(1)

    # 3. Augment images
    augmented_counts = {"Mine": 0, "No_Mine": 0}

    print("Generating offline augmentations...")
    for cls in CLASSES:
        files = original_files[cls]
        factor = AUGMENTATION_FACTORS.get(cls, 0)
        if factor <= 0:
            continue

        print(f"  Augmenting {cls} split ({factor}x scale)...")
        bar = tqdm(
            files,
            total=len(files),
            unit="img",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour="cyan",
        )

        for src_path in bar:
            # Read source image using OpenCV
            img = cv2.imread(str(src_path))
            if img is None:
                print(f"[WARNING] Could not read image: {src_path.name}")
                continue

            stem = src_path.stem
            ext = src_path.suffix

            for idx in range(1, factor + 1):
                aug_img = apply_augmentations(img)
                # Build destination filename
                out_filename = f"{stem}_aug_{idx:02d}{ext}"
                dst_path = src_path.parent / out_filename

                cv2.imwrite(str(dst_path), aug_img)
                augmented_counts[cls] += 1

    elapsed = time.perf_counter() - start_time

    # 4. Save and print report
    write_report(
        OFFLINE_AUG_REPORT_FILE,
        original_counts=original_counts,
        augmented_counts=augmented_counts,
        deleted_count=deleted_count,
        elapsed=elapsed,
    )

    print()
    print("Augmentation complete!")
    print(f"  Added Mine    : {augmented_counts['Mine']} images")
    print(f"  Added No_Mine : {augmented_counts['No_Mine']} images")
    print(f"  Total Added   : {sum(augmented_counts.values())}")
    print()
    print(f"New Training split size:")
    print(f"  Mine (Total)  : {original_counts['Mine'] + augmented_counts['Mine']}")
    print(f"  No_Mine (Total): {original_counts['No_Mine'] + augmented_counts['No_Mine']}")
    print(f"  Combined Total: {sum(original_counts.values()) + sum(augmented_counts.values())}")
    print()
    print(f"Report saved to: {OFFLINE_AUG_REPORT_FILE}")
    print(f"Execution took : {elapsed:.2f} seconds")
    print("=" * 45)


if __name__ == "__main__":
    main()

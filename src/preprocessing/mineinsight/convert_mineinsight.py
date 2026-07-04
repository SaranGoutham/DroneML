"""
MineInsight Conversion
======================

Convert the MineInsight object detection dataset into a binary classification dataset.
Generates crops using crop_utils.py based on targets.py configuration.
"""

import os
import json
import csv
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional
from PIL import Image

# Import configs and utilities
from src.preprocessing.mineinsight.mineinsight_config import (
    RAW_DIR,
    CLASSIFICATION_DIR,
    REPORTS_DIR,
    IMAGE_SIZE,
    PADDING_AMOUNT,
    SUPPORTED_EXTENSIONS,
    MINE_CLASS,
    NO_MINE_CLASS
)
from src.preprocessing.mineinsight.targets import load_targets, is_mine, is_non_mine
from src.preprocessing.mineinsight.crop_utils import process_crop

logger = logging.getLogger(__name__)

@dataclass
class ConversionStatistics:
    images_processed: int = 0
    images_skipped: int = 0
    annotations_processed: int = 0
    mine_crops: int = 0
    non_mine_crops: int = 0
    crop_failures: int = 0
    invalid_annotations: int = 0


def generate_filename(
    prefix: str, img_stem: str, obj_idx: int, class_type: str
) -> str:
    """Generate a unique filename for the cropped image."""
    return f"{prefix}_{img_stem}_obj{obj_idx:02d}_{class_type.lower()}.jpg"


def process_sequence(
    prefix: str,
    img_dir: Path,
    lbl_dir: Path,
    stats: ConversionStatistics,
    mine_dir: Path,
    no_mine_dir: Path
) -> None:
    """Process a single sequence (track/sequence) directory."""
    if not img_dir.exists() or not lbl_dir.exists():
        logger.warning(f"Missing image or label directory for prefix {prefix}. Skipping.")
        return

    logger.info(f"Processing sequence: {prefix}")
    
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
            
        stats.images_processed += 1
        img_stem = img_path.stem
        lbl_path = lbl_dir / f"{img_stem}.txt"
        
        if not lbl_path.exists():
            logger.warning(f"Label missing for image {img_path.name}. Skipping.")
            stats.images_skipped += 1
            continue
            
        try:
            with open(lbl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Failed to read label {lbl_path}: {e}")
            stats.images_skipped += 1
            continue
            
        if not lines:
            logger.debug(f"Empty label {lbl_path.name}. Skipping image.")
            stats.images_skipped += 1
            continue
            
        try:
            with Image.open(img_path) as img:
                # Load image fully to ensure it's valid
                img.load()
                
                for obj_idx, line in enumerate(lines, start=1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        stats.invalid_annotations += 1
                        logger.warning(f"Invalid annotation in {lbl_path.name} line {obj_idx}")
                        continue
                        
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except ValueError:
                        stats.invalid_annotations += 1
                        logger.warning(f"Invalid numeric value in {lbl_path.name} line {obj_idx}")
                        continue
                        
                    stats.annotations_processed += 1
                    
                    if is_mine(class_id):
                        class_type = MINE_CLASS
                        out_dir = mine_dir
                    elif is_non_mine(class_id):
                        class_type = NO_MINE_CLASS
                        out_dir = no_mine_dir
                    else:
                        stats.invalid_annotations += 1
                        logger.warning(f"Unknown class ID {class_id} in {lbl_path.name}")
                        continue
                        
                    filename = generate_filename(prefix, img_stem, obj_idx, class_type)
                    out_path = out_dir / filename
                    
                    if out_path.exists():
                        logger.warning(f"File {out_path.name} already exists. Skipping.")
                        continue
                        
                    try:
                        crop = process_crop(
                            image=img,
                            x_center=x_center,
                            y_center=y_center,
                            w=w,
                            h=h,
                            padding_ratio=PADDING_AMOUNT,
                            target_size=IMAGE_SIZE
                        )
                        # Save image
                        crop.save(out_path, format="JPEG", quality=95)
                        
                        if class_type == MINE_CLASS:
                            stats.mine_crops += 1
                        else:
                            stats.non_mine_crops += 1
                            
                    except Exception as e:
                        stats.crop_failures += 1
                        logger.warning(f"Crop failure for {lbl_path.name} obj {obj_idx}: {e}")
                        
        except Exception as e:
            logger.warning(f"Failed to process image {img_path.name}: {e}")
            stats.images_skipped += 1
            continue


def export_reports(stats: ConversionStatistics) -> None:
    """Export conversion reports in JSON and CSV formats."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    json_path = REPORTS_DIR / "conversion_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(stats), f, indent=4)
    logger.info(f"Exported JSON report to {json_path}")
    
    csv_path = REPORTS_DIR / "conversion_report.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Count"])
        for key, value in asdict(stats).items():
            writer.writerow([key, value])
    logger.info(f"Exported CSV report to {csv_path}")


def main() -> None:
    """Run the complete dataset conversion."""
    print("Initializing MineInsight conversion...")
    
    # Load targets config
    load_targets()
    
    # Setup directories
    mine_dir = CLASSIFICATION_DIR / MINE_CLASS
    no_mine_dir = CLASSIFICATION_DIR / NO_MINE_CLASS
    mine_dir.mkdir(parents=True, exist_ok=True)
    no_mine_dir.mkdir(parents=True, exist_ok=True)
    
    stats = ConversionStatistics()
    
    # Discover sequences
    img_dirs = list(RAW_DIR.glob("track_*_lwir_images"))
    if not img_dirs:
        logger.warning(f"No image directories found in {RAW_DIR}")
        
    for img_dir in img_dirs:
        prefix = img_dir.name.replace("_lwir_images", "")
        lbl_dir = RAW_DIR / f"{prefix}_lwir_labels_reproj"
        process_sequence(prefix, img_dir, lbl_dir, stats, mine_dir, no_mine_dir)
        
    export_reports(stats)
    
    print("\n====================================")
    print("MineInsight Conversion")
    print("====================================")
    print(f"Images processed    : {stats.images_processed}")
    print(f"Images skipped      : {stats.images_skipped}")
    print(f"Mine crops          : {stats.mine_crops}")
    print(f"Non_Mine crops      : {stats.non_mine_crops}")
    print(f"Crop failures       : {stats.crop_failures}")
    print("====================================\n")
    
    # Verify outputs
    print("Verifying outputs...")
    if mine_dir.exists() and no_mine_dir.exists():
        mine_count = len(list(mine_dir.glob("*.jpg")))
        non_mine_count = len(list(no_mine_dir.glob("*.jpg")))
        print(f"Mine folder exists. File count: {mine_count}")
        print(f"No_Mine folder exists. File count: {non_mine_count}")
    else:
        print("ERROR: Output folders are missing.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except Exception as e:
        logger.critical(f"Conversion failed: {e}")

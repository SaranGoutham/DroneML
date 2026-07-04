"""
Dataset Scanner Module
======================

Responsible for scanning the MineInsight dataset, verifying images and labels,
calculating statistics, and detecting missing pairs. Performs inspection only,
without modifying any files, cropping, splitting, or classifying.
"""

import os
import json
import csv
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional
from PIL import Image, UnidentifiedImageError

# Configurations and Targets
from src.preprocessing.mineinsight.mineinsight_config import (
    RAW_DIR,
    REPORTS_DIR,
    SUPPORTED_EXTENSIONS
)
from src.preprocessing.mineinsight.targets import (
    load_targets,
    is_mine,
    is_non_mine,
    get_target_name
)

logger = logging.getLogger(__name__)

# ========================================================
# DATACLASSES
# ========================================================

@dataclass
class ImageStatistics:
    total_images: int = 0
    corrupted_images: int = 0
    unreadable_images: int = 0
    duplicate_filenames: int = 0
    resolutions: Set[Tuple[int, int]] = field(default_factory=set)

@dataclass
class LabelStatistics:
    total_labels: int = 0
    empty_labels: int = 0
    malformed_rows: int = 0
    invalid_numeric_values: int = 0
    invalid_class_ids: int = 0
    invalid_coordinates: int = 0
    invalid_dimensions: int = 0

@dataclass
class AnnotationStatistics:
    total_annotations: int = 0
    mine_annotations: int = 0
    non_mine_annotations: int = 0
    images_zero_objects: int = 0
    images_one_object: int = 0
    images_multiple_objects: int = 0
    max_objects: int = 0
    avg_objects_per_image: float = 0.0

@dataclass
class TrackStatistics:
    track_name: str
    images: ImageStatistics = field(default_factory=ImageStatistics)
    labels: LabelStatistics = field(default_factory=LabelStatistics)
    annotations: AnnotationStatistics = field(default_factory=AnnotationStatistics)
    missing_labels: int = 0
    orphan_labels: int = 0
    matching_pairs: int = 0
    status: str = "PASS"

@dataclass
class DatasetStatistics:
    total_images: int = 0
    total_labels: int = 0
    total_annotations: int = 0
    total_mine: int = 0
    total_non_mine: int = 0
    status: str = "PASS"
    tracks: Dict[str, TrackStatistics] = field(default_factory=dict)

# ========================================================
# IMPLEMENTATION
# ========================================================

def validate_label_file(label_path: Path, track_stats: TrackStatistics) -> int:
    """
    Validate a single YOLO label file.
    Updates the track_stats with any errors found and returns the number of valid objects.
    """
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning(f"Could not read label file {label_path}: {e}")
        return 0

    if not lines:
        track_stats.labels.empty_labels += 1
        return 0

    valid_objects = 0
    for line_idx, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
            
        if len(parts) != 5:
            track_stats.labels.malformed_rows += 1
            logger.warning(f"Malformed row in {label_path} at line {line_idx+1}")
            continue
            
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            track_stats.labels.invalid_numeric_values += 1
            logger.warning(f"Invalid numeric values in {label_path} at line {line_idx+1}")
            continue
            
        try:
            get_target_name(class_id)
        except KeyError:
            track_stats.labels.invalid_class_ids += 1
            logger.warning(f"Invalid class ID {class_id} in {label_path} at line {line_idx+1}")
            continue

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            track_stats.labels.invalid_coordinates += 1
            logger.warning(f"Invalid coordinates in {label_path} at line {line_idx+1}")
            continue
            
        if width <= 0 or height <= 0:
            track_stats.labels.invalid_dimensions += 1
            logger.warning(f"Invalid dimensions (<=0) in {label_path} at line {line_idx+1}")
            continue

        valid_objects += 1
        track_stats.annotations.total_annotations += 1
        
        if is_mine(class_id):
            track_stats.annotations.mine_annotations += 1
        elif is_non_mine(class_id):
            track_stats.annotations.non_mine_annotations += 1
            
    return valid_objects

def scan_image_folder(img_dir: Path, track_stats: TrackStatistics) -> Set[str]:
    """
    Scan and verify an image directory.
    Returns a set of image filename stems.
    """
    if not img_dir.exists():
        logger.warning(f"Image directory not found: {img_dir}")
        return set()
        
    image_stems = set()
    seen_filenames = set()
    
    for file_path in img_dir.iterdir():
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
            
        if file_path.name in seen_filenames:
            track_stats.images.duplicate_filenames += 1
            logger.warning(f"Duplicate filename found: {file_path}")
            continue
        seen_filenames.add(file_path.name)
        
        track_stats.images.total_images += 1
        image_stems.add(file_path.stem)
        
        try:
            with Image.open(file_path) as img:
                img.verify()
                track_stats.images.resolutions.add(img.size)
        except UnidentifiedImageError:
            track_stats.images.unreadable_images += 1
            logger.warning(f"Unreadable image: {file_path}")
        except Exception as e:
            track_stats.images.corrupted_images += 1
            logger.warning(f"Corrupted image {file_path}: {e}")
            
    return image_stems

def scan_label_folder(lbl_dir: Path, track_stats: TrackStatistics) -> Dict[str, int]:
    """
    Scan and verify a label directory.
    Returns a dictionary mapping label filename stems to object counts.
    """
    if not lbl_dir.exists():
        logger.warning(f"Label directory not found: {lbl_dir}")
        return {}
        
    label_counts = {}
    
    for file_path in lbl_dir.iterdir():
        if file_path.suffix.lower() != ".txt":
            continue
            
        track_stats.labels.total_labels += 1
        valid_objs = validate_label_file(file_path, track_stats)
        label_counts[file_path.stem] = valid_objs
        
    return label_counts

def scan_track(track_name: str, img_dir: Path, lbl_dir: Path) -> Optional[TrackStatistics]:
    """
    Scan a specific track/sequence folder.
    Returns TrackStatistics if the folder exists, otherwise None.
    """
    if not img_dir.exists() and not lbl_dir.exists():
        return None
        
    logger.info(f"Scanning track: {track_name}")
    stats = TrackStatistics(track_name=track_name)
    
    if not img_dir.exists():
        logger.warning(f"[{track_name}] Image directory missing.")
        stats.status = "FAIL"
    if not lbl_dir.exists():
        logger.warning(f"[{track_name}] Label directory missing.")
        stats.status = "FAIL"
        
    image_stems = scan_image_folder(img_dir, stats)
    label_counts = scan_label_folder(lbl_dir, stats)
    
    label_stems = set(label_counts.keys())
    
    # Pair checks
    stats.missing_labels = len(image_stems - label_stems)
    stats.orphan_labels = len(label_stems - image_stems)
    stats.matching_pairs = len(image_stems.intersection(label_stems))
    
    if stats.missing_labels > 0 or stats.orphan_labels > 0:
        logger.warning(f"[{track_name}] Found {stats.missing_labels} missing labels and {stats.orphan_labels} orphan labels.")
    if stats.missing_labels > 0:
        stats.status = "FAIL"
        
    # Object statistics based on all parsed labels
    for stem, obj_count in label_counts.items():
        if obj_count == 0:
            stats.annotations.images_zero_objects += 1
        elif obj_count == 1:
            stats.annotations.images_one_object += 1
        else:
            stats.annotations.images_multiple_objects += 1
            
        if obj_count > stats.annotations.max_objects:
            stats.annotations.max_objects = obj_count
            
    if stats.labels.total_labels > 0:
        stats.annotations.avg_objects_per_image = stats.annotations.total_annotations / stats.labels.total_labels
        
    return stats

def compute_statistics(tracks: List[TrackStatistics]) -> DatasetStatistics:
    """Compute overall dataset statistics from individual track statistics."""
    overall = DatasetStatistics()
    
    for t in tracks:
        overall.tracks[t.track_name] = t
        overall.total_images += t.images.total_images
        overall.total_labels += t.labels.total_labels
        overall.total_annotations += t.annotations.total_annotations
        overall.total_mine += t.annotations.mine_annotations
        overall.total_non_mine += t.annotations.non_mine_annotations
        if t.status == "FAIL":
            overall.status = "FAIL"
            
    return overall

def scan_dataset() -> DatasetStatistics:
    """
    Scan all predefined tracks in the MineInsight raw dataset.
    Automatically skips missing tracks, dynamically discovers track directories,
    and returns aggregated statistics.
    """
    logger.info("Starting dataset scan...")
    load_targets()
    
    possible_tracks = [
        "Track1 Seq1", "Track1 Seq2", 
        "Track2 Seq1", "Track2 Seq2", 
        "Track3 Seq1", "Track3 Seq2"
    ]
    
    expected_prefixes = {}
    for pt in possible_tracks:
        parts = pt.split()
        if len(parts) == 2:
            tr = parts[0].replace("Track", "")
            sq = parts[1].replace("Seq", "")
            prefix = f"track_{tr}_s{sq}"
            expected_prefixes[prefix] = pt

    # Dynamically find directories
    found_img_dirs = list(RAW_DIR.glob("track_*_lwir_images"))
    found_lbl_dirs = list(RAW_DIR.glob("track_*_lwir_labels_reproj"))
    
    for d in found_img_dirs:
        logger.info(f"Found image directory: {d}")
    for d in found_lbl_dirs:
        logger.info(f"Found label directory: {d}")
        
    detected_prefixes = set()
    for d in found_img_dirs:
        detected_prefixes.add(d.name.replace("_lwir_images", ""))
    for d in found_lbl_dirs:
        detected_prefixes.add(d.name.replace("_lwir_labels_reproj", ""))
        
    for prefix, pt in expected_prefixes.items():
        img_dir = RAW_DIR / f"{prefix}_lwir_images"
        lbl_dir = RAW_DIR / f"{prefix}_lwir_labels_reproj"
        logger.info(f"Searching for sequence {pt} directories: {img_dir} and {lbl_dir}")
        if prefix not in detected_prefixes:
            logger.warning(f"Sequence {pt} not found.")

    parsed_tracks = []
    
    for prefix in sorted(detected_prefixes):
        pt = expected_prefixes.get(prefix, f"Unknown Sequence ({prefix})")
        logger.info(f"Found sequence {pt}")
        
        img_dir = RAW_DIR / f"{prefix}_lwir_images"
        lbl_dir = RAW_DIR / f"{prefix}_lwir_labels_reproj"
        
        t_stat = scan_track(pt, img_dir, lbl_dir)
        if t_stat is not None:
            parsed_tracks.append(t_stat)
            
    overall = compute_statistics(parsed_tracks)
    
    if len(parsed_tracks) > 0 and overall.total_images == 0:
        raise ValueError("At least one sequence exists, but no images were scanned.")
        
    return overall

def print_summary(stats: DatasetStatistics) -> None:
    """Print the final dataset scanner summary to the console."""
    print("========================================================")
    print("MineInsight Dataset Scanner")
    print("========================================================")
    
    for track_name, t in stats.tracks.items():
        print(f"\n{track_name}")
        print(f"Images                : {t.images.total_images}")
        print(f"Labels                : {t.labels.total_labels}")
        print(f"Matching pairs        : {t.matching_pairs}")
        print(f"Missing labels        : {t.missing_labels}")
        print(f"Orphan labels         : {t.orphan_labels}")
        print(f"Corrupted images      : {t.images.corrupted_images}")
        print(f"Empty labels          : {t.labels.empty_labels}")
        print(f"Mine annotations      : {t.annotations.mine_annotations}")
        print(f"Non-Mine annotations  : {t.annotations.non_mine_annotations}")
        print(f"Average objects/image : {t.annotations.avg_objects_per_image:.2f}")
        print(f"Status                : {t.status}")
        print("========================================================")

    print("\nOverall dataset")
    print(f"Images       : {stats.total_images}")
    print(f"Labels       : {stats.total_labels}")
    print(f"Annotations  : {stats.total_annotations}")
    print(f"Mine         : {stats.total_mine}")
    print(f"Non_Mine     : {stats.total_non_mine}")
    print(f"Status       : {stats.status}")
    print("========================================================")

def export_json(stats: DatasetStatistics, output_dir: Path) -> None:
    """Export the dataset statistics to a JSON report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dataset_scan.json"
    
    class StatsEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return super().default(obj)
            
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(stats), f, indent=4, cls=StatsEncoder)
    logger.info(f"Exported JSON report to {out_path}")

def export_csv(stats: DatasetStatistics, output_dir: Path) -> None:
    """Export the dataset statistics to a CSV report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dataset_scan.csv"
    
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Track", "Images", "Labels", "Matching Pairs", "Missing Labels", 
                         "Orphan Labels", "Corrupted Images", "Empty Labels", 
                         "Mine Annotations", "Non-Mine Annotations", "Status"])
                         
        for track_name, t in stats.tracks.items():
            writer.writerow([
                track_name, t.images.total_images, t.labels.total_labels,
                t.matching_pairs, t.missing_labels, t.orphan_labels,
                t.images.corrupted_images, t.labels.empty_labels,
                t.annotations.mine_annotations, t.annotations.non_mine_annotations,
                t.status
            ])
            
        writer.writerow([])
        writer.writerow(["Overall", stats.total_images, stats.total_labels, 
                         sum(t.matching_pairs for t in stats.tracks.values()),
                         sum(t.missing_labels for t in stats.tracks.values()),
                         sum(t.orphan_labels for t in stats.tracks.values()),
                         sum(t.images.corrupted_images for t in stats.tracks.values()),
                         sum(t.labels.empty_labels for t in stats.tracks.values()),
                         stats.total_mine, stats.total_non_mine, stats.status])
                         
    logger.info(f"Exported CSV report to {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    try:
        stats = scan_dataset()
        print_summary(stats)
        export_json(stats, REPORTS_DIR)
        export_csv(stats, REPORTS_DIR)
    except Exception as e:
        logger.critical(f"Dataset scan failed critically: {e}")

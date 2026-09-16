"""
Master Manifest Module for MineInsight v2
=========================================

Defines the complete metadata record schema for every crop and provides
deterministic serialization to CSV and JSON formats.
"""

import csv
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.preprocessing.mineinsight_v2.config import (
    MASTER_MANIFEST_CSV,
    PROTOCOL1_SPLIT_CSV,
    PROTOCOL2_SPLIT_CSV,
)


@dataclass
class CropManifestRecord:
    crop_id: str
    crop_rel_path: str
    category: str
    target_instance_id: int
    target_name: str
    semantic_subgroup: str
    source_track: str
    source_sequence: str
    source_image_name: str
    source_label_name: str
    annotation_line_idx: int
    orig_x_center: float
    orig_y_center: float
    orig_width: float
    orig_height: float
    crop_xmin_px: int
    crop_ymin_px: int
    crop_xmax_px: int
    crop_ymax_px: int
    crop_width: int
    crop_height: int
    flight_timestamp_epoch: float
    protocol1_split: str = "unassigned"
    protocol2_split: str = "unassigned"
    file_sha256: str = ""

    @staticmethod
    def generate_crop_id(crop_rel_path: str) -> str:
        """Deterministic 16-character hex hash from relative path."""
        return hashlib.sha256(crop_rel_path.encode("utf-8")).hexdigest()[:16]


def parse_timestamp_from_stem(stem: str) -> float:
    """
    Extract epoch timestamp from filename stem like:
    track_1_s1_lwir_1730290698_207892742 -> 1730290698.207892742
    """
    parts = stem.split("_")
    for i in range(len(parts) - 1):
        p1, p2 = parts[i], parts[i + 1]
        if p1.isdigit() and len(p1) == 10 and p2.isdigit() and len(p2) == 9:
            return float(f"{p1}.{p2}")
    return 0.0


def export_manifest_csv(
    records: List[CropManifestRecord],
    dest_csv: Path,
) -> None:
    """Write list of CropManifestRecord to CSV."""
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    fieldnames = list(asdict(records[0]).keys())
    with open(dest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def export_split_manifest_csv(
    records: List[CropManifestRecord],
    split_column: str,
    dest_csv: Path,
) -> None:
    """Export a lightweight training split manifest: crop_rel_path, category, split."""
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_rel_path", "category", "target_instance_id", "target_name", "split"])
        for r in records:
            split_val = getattr(r, split_column, "unassigned")
            writer.writerow([r.crop_rel_path, r.category, r.target_instance_id, r.target_name, split_val])

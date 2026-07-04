"""Verify MineInsight classification splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.preprocessing.mineinsight.mineinsight_config import (
    MINE_CLASS,
    NO_MINE_CLASS,
    PROCESSED_DIR,
    REPORTS_DIR,
    SUPPORTED_EXTENSIONS,
)
from src.preprocessing.mineinsight.split_mineinsight import source_group_id

logger = logging.getLogger(__name__)


def iter_images(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for split in ("train", "validation", "test"):
        for class_name in (MINE_CLASS, NO_MINE_CLASS):
            class_dir = dataset_dir / split / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.iterdir():
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    rows.append((split, class_name, path))
    return rows


def generate_statistics(dataset_dir: Path | str = PROCESSED_DIR) -> dict[str, Any]:
    """Generate class and split counts for MineInsight processed data."""
    dataset_dir = Path(dataset_dir)
    stats: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "splits": {},
        "total_images": 0,
    }
    for split in ("train", "validation", "test"):
        stats["splits"][split] = {}
        split_total = 0
        for class_name in (MINE_CLASS, NO_MINE_CLASS):
            count = len(
                [
                    path
                    for path in (dataset_dir / split / class_name).glob("*")
                    if path.suffix.lower() in SUPPORTED_EXTENSIONS
                ]
            )
            stats["splits"][split][class_name] = count
            split_total += count
        stats["splits"][split]["total"] = split_total
        stats["total_images"] += split_total
    return stats


def check_corrupted_images(dataset_dir: Path | str = PROCESSED_DIR) -> list[str]:
    """Return paths for images PIL cannot open/verify."""
    corrupted: list[str] = []
    for _, _, path in iter_images(Path(dataset_dir)):
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning("Unreadable image %s: %s", path, exc)
            corrupted.append(str(path))
    return corrupted


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicates(dataset_dir: Path | str = PROCESSED_DIR) -> list[dict[str, Any]]:
    """Find byte-identical duplicate files across the processed dataset."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for _, _, path in iter_images(Path(dataset_dir)):
        by_hash[file_sha256(path)].append(str(path))
    return [
        {"sha256": digest, "files": paths}
        for digest, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]


def find_leakage(dataset_dir: Path | str = PROCESSED_DIR) -> list[dict[str, Any]]:
    """Find source-image crop groups that appear in multiple splits."""
    by_group: dict[str, set[str]] = defaultdict(set)
    for split, _, path in iter_images(Path(dataset_dir)):
        by_group[source_group_id(path)].add(split)
    return [
        {"source_group": group, "splits": sorted(splits)}
        for group, splits in sorted(by_group.items())
        if len(splits) > 1
    ]


def verify_dataset(dataset_dir: Path | str = PROCESSED_DIR) -> dict[str, Any]:
    """Run MineInsight verification and write a JSON report."""
    dataset_dir = Path(dataset_dir)
    report = {
        "statistics": generate_statistics(dataset_dir),
        "corrupted_images": check_corrupted_images(dataset_dir),
        "duplicates": find_duplicates(dataset_dir),
        "leakage": find_leakage(dataset_dir),
    }
    report["status"] = "PASS"
    if report["corrupted_images"] or report["leakage"]:
        report["status"] = "FAIL"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    logger.info("MineInsight verification report written to %s", report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify MineInsight processed splits.")
    parser.add_argument("--dataset-dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    report = verify_dataset(parse_args().dataset_dir)
    stats = report["statistics"]
    print("MineInsight verification")
    print(f"Status: {report['status']}")
    print(f"Total images: {stats['total_images']}")
    print(f"Corrupted images: {len(report['corrupted_images'])}")
    print(f"Duplicate groups: {len(report['duplicates'])}")
    print(f"Leakage groups: {len(report['leakage'])}")


if __name__ == "__main__":
    main()

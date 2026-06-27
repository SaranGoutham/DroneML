# -*- coding: utf-8 -*-
"""split_by_flight.py
=====================

Second step in the DroneML preprocessing pipeline.

This module **splits the processed PNG dataset** into train / validation / test
sub‑folders **based only on the flight date encoded in each filename**.
It operates solely on the processed PNG files produced by ``preprocess_tiff.py``
and does **not** require any raw TIFF data or external metadata.

Expected input layout::

    processed_dataset/
        Mine/
            13_02_2020_Zone_1_Mine_1cm_depth_Mine_000001.png
            ...
        No_Mine/
            13_02_2020_Zone_1_free_zone_NoMine_000001.png
            ...

Output layout (created inside the same ``processed_dataset`` root)::

    processed_dataset/
        train/      (flights assigned to "train")
            Mine/
            No_Mine/
        validation/ (flights assigned to "validation")
            Mine/
            No_Mine/
        test/       (flights assigned to "test")
            Mine/
            No_Mine/

A plain‑text report is written to ``results/split_report.txt``.
"""

from __future__ import annotations

import datetime
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import (
    PROCESSED_DIR as CONFIG_PROCESSED_DIR,
    RESULTS_DIR as CONFIG_RESULTS_DIR,
    SPLIT_REPORT_FILE,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Root of the processed PNG dataset created by ``preprocess_tiff.py``.
PROCESSED_DIR: Path = CONFIG_PROCESSED_DIR

# Output directory – split folders are created *inside* the processed directory.
OUTPUT_DIR: Path = PROCESSED_DIR

# Report location.
RESULTS_DIR: Path = CONFIG_RESULTS_DIR
REPORT_FILE: Path = SPLIT_REPORT_FILE

# Class sub‑folders – must match what ``preprocess_tiff.py`` created.
CLASSES: List[str] = ["Mine", "No_Mine"]

# ---------------------------------------------------------------------------
# FLIGHT → SPLIT ASSIGNMENT (edit to control splits)
# ---------------------------------------------------------------------------
FLIGHT_SPLIT_ASSIGNMENT: Dict[str, str] = {
    "13_02_2020": "train",
    "03_03_2020": "train",
    "17_02_2020": "validation",
    "24_02_2020": "test",
}

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _extract_flight_from_filename(filename: str) -> str:
    """
    Extract the flight date (DD_MM_YYYY) from a processed image filename.

    Example
    -------
    13_02_2020_Zone_1_Mine_1cm_depth_Mine_000001.png
    ↓
    13_02_2020
    """
    parts = Path(filename).stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid filename: {filename}")

    flight = "_".join(parts[:3])

    # Validate DD_MM_YYYY format
    if len(flight) != 10:
        raise ValueError(f"Invalid flight identifier: {flight}")

    return flight


def _validate_processed_dir(root: Path) -> Tuple[int, int]:
    """Ensure ``root`` contains ``Mine`` and ``No_Mine`` sub‑folders with PNGs.

    Returns ``(mine_count, no_mine_count)``.
    """
    if not root.is_dir():
        print(f"[ERROR] Processed dataset directory not found: {root.resolve()}")
        sys.exit(1)
    counts: List[int] = []
    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            print(f"[ERROR] Expected class folder missing: {cls_dir.resolve()}")
            sys.exit(1)
        files = list(cls_dir.glob("*.png"))
        if not files:
            print(f"[ERROR] No PNG files found in {cls_dir.resolve()}")
            sys.exit(1)
        counts.append(len(files))
    return counts[0], counts[1]  # type: ignore[return-value]


def _collect_image_records(root: Path) -> List[Dict[str, Path | str]]:
    """Walk ``root`` and return a list of image records.

    Each record holds ``path`` (Path), ``flight`` (str) and ``label`` (class name).
    """
    records: List[Dict[str, Path | str]] = []
    for label in CLASSES:
        for img_path in sorted((root / label).glob("*.png")):
            flight = _extract_flight_from_filename(img_path.name)
            records.append({"path": img_path, "flight": flight, "label": label})
    return records


def _assign_splits(records: List[Dict[str, Path | str]]) -> List[Dict[str, str]]:
    """Add a ``split`` entry to each record based on ``FLIGHT_SPLIT_ASSIGNMENT``.

    Records with flights missing from the mapping are skipped with a warning.
    """
    assigned: List[Dict[str, str]] = []
    missing_flights = set()
    for rec in records:
        split = FLIGHT_SPLIT_ASSIGNMENT.get(rec["flight"])
        if split is None:
            missing_flights.add(rec["flight"])  # type: ignore[arg-type]
            continue
        assigned.append({
            "path": rec["path"],
            "flight": rec["flight"],
            "label": rec["label"],
            "split": split,
        })
    if missing_flights:
        print(f"[WARNING] {len(missing_flights)} flight(s) not in FLIGHT_SPLIT_ASSIGNMENT and will be skipped:")
        for f in sorted(missing_flights):
            print(f"  {f}")
    return assigned


def _prepare_output_dirs(root: Path) -> None:
    """Create ``train``, ``validation`` and ``test`` sub‑folders for each class."""
    for split in ("train", "validation", "test"):
        for cls in CLASSES:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def _copy_images(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Copy images to their split destination.

    Returns a new list with an added ``dst`` key (string). Existing files are
    left untouched (idempotent).
    """
    _prepare_output_dirs(OUTPUT_DIR)
    result: List[Dict[str, str]] = []
    bar = tqdm(
        records,
        total=len(records),
        unit="img",
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour="cyan",
    )
    for rec in bar:
        src: Path = rec["path"]  # type: ignore[assignment]
        dst: Path = OUTPUT_DIR / rec["split"] / rec["label"] / src.name
        bar.set_postfix_str(f"{rec['split']}/{rec['label']}/{src.name}", refresh=False)
        if dst.exists():
            rec["dst"] = str(dst)
            result.append(rec)
            continue
        try:
            shutil.copy2(src, dst)
            rec["dst"] = str(dst)
        except OSError as exc:
            print(f"[ERROR] Could not copy {src} → {dst}: {exc}")
            rec["dst"] = "ERROR"
        result.append(rec)
    return result


def _verify_split(records: List[Dict[str, str]]) -> bool:
    """Post‑copy verification.

    Checks for missing copies, flight leakage and folder presence.
    Returns ``True`` if everything passes.
    """
    all_ok = True
    # Missing or error copies
    bad = [r for r in records if r.get("dst") in {"MISSING", "ERROR"}]
    if bad:
        print(f"[FAIL] {len(bad)} file(s) could not be copied.")
        all_ok = False
    else:
        print("[PASS] All images copied successfully.")
    # Leakage check
    flight_splits: Dict[str, set] = defaultdict(set)
    for r in records:
        flight_splits[r["flight"]].add(r["split"])
    leakage = False
    for flight, splits in flight_splits.items():
        if len(splits) > 1:
            print(f"[FAIL] Flight {flight} appears in multiple splits: {splits}")
            leakage = True
    if not leakage:
        print("[PASS] No flight leakage detected.")
    else:
        all_ok = False
    # Folder structure check
    for split in ("train", "validation", "test"):
        for cls in CLASSES:
            if not (OUTPUT_DIR / split / cls).is_dir():
                print(f"[FAIL] Missing folder: {OUTPUT_DIR / split / cls}")
                all_ok = False
    if all_ok:
        print("[PASS] Split directory structure verified.")
    return all_ok


def _write_report(records: List[Dict[str, str]], leakage_free: bool, path: Path) -> None:
    """Write a human‑readable split report.

    Includes timestamps, flight‑to‑split mapping, per‑split counts and leakage
    result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [
        "=" * 45,
        "FLIGHT-BASED SPLIT REPORT",
        "=" * 45,
        f"Generated : {timestamp}",
        "",
    ]
    # Flight assignments
    lines.append("-- Flight Assignments --------------------------------------------------")
    flight_map = {r["flight"]: r["split"] for r in records}
    for flight in sorted(flight_map):
        lines.append(f"  {flight:20s} -> {flight_map[flight]}")
    lines.append("")
    # Per‑split statistics
    lines.append("-- Images per Split ---------------------------------------------------")
    total = len(records)
    for split in ("train", "validation", "test"):
        sub = [r for r in records if r["split"] == split]
        cnt = len(sub)
        pct = 100.0 * cnt / total if total else 0.0
        mine = sum(1 for r in sub if r["label"] == "Mine")
        nomine = sum(1 for r in sub if r["label"] == "No_Mine")
        lines.extend([
            f"  {split.upper():10s} total={cnt:4d} ({pct:5.1f}%)",
            f"                Mine    = {mine}",
            f"                No_Mine = {nomine}",
        ])
    lines.append("")
    # Overall totals
    lines.append("-- Overall Totals -----------------------------------------------------")
    mine_total = sum(1 for r in records if r["label"] == "Mine")
    nomine_total = sum(1 for r in records if r["label"] == "No_Mine")
    lines.extend([
        f"  Total images : {total}",
        f"  Mine         : {mine_total}",
        f"  No_Mine      : {nomine_total}",
        "",
    ])
    # Leakage result
    lines.append("-- Leakage Check ------------------------------------------------------")
    lines.append(f"  Result : {'PASS' if leakage_free else 'FAIL'}")
    lines.append("")
    lines.append("=" * 45)
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_header() -> None:
    print("\n" + "=" * 41)
    print("FLIGHT-BASED DATASET SPLIT")
    print("=" * 41 + "\n")


def _print_flight_assignments() -> None:
    print("Flight Assignments (configured):")
    for flight, split in sorted(FLIGHT_SPLIT_ASSIGNMENT.items()):
        print(f"  {flight} -> {split}")
    print()


def _print_statistics(records: List[Dict[str, str]]) -> None:
    print("Split statistics:")
    for split in ("train", "validation", "test"):
        sub = [r for r in records if r["split"] == split]
        total = len(sub)
        mine = sum(1 for r in sub if r["label"] == "Mine")
        nomine = sum(1 for r in sub if r["label"] == "No_Mine")
        print(f"  {split.capitalize():10s}: {total} total (Mine={mine}, No_Mine={nomine})")
    print()


def main() -> None:
    """Run the flight‑based split.

    Steps:
    1️⃣ Validate processed dataset.
    2️⃣ Gather image records and extract flights.
    3️⃣ Assign splits.
    4️⃣ Copy images.
    5️⃣ Verify no leakage and correct folder structure.
    6️⃣ Write a summary report.
    """
    _print_header()

    # 1️⃣ Validation
    print("Validating processed dataset …")
    mine_cnt, nomine_cnt = _validate_processed_dir(PROCESSED_DIR)
    print(f"  Mine    : {mine_cnt} images")
    print(f"  No_Mine : {nomine_cnt} images\n")

    # 2️⃣ Collect records
    records = _collect_image_records(PROCESSED_DIR)
    print(f"Found {len(records)} PNG images across classes.\n")

    # 3️⃣ Assign splits
    _print_flight_assignments()
    records = _assign_splits(records)
    if not records:
        print("[ERROR] No images matched any configured flight split. Exiting.")
        sys.exit(1)

    # 4️⃣ Copy images
    print("Copying images …")
    records = _copy_images(records)
    print()

    # 5️⃣ Stats & verification
    _print_statistics(records)
    print("Running verification …")
    leakage_ok = _verify_split(records)
    print()

    # 6️⃣ Report
    print("Writing report …")
    _write_report(records, leakage_ok, REPORT_FILE)
    print(f"Report saved to {REPORT_FILE}\n")

    print("=" * 41)
    print("Finished")
    print("=" * 41)


if __name__ == "__main__":
    main()

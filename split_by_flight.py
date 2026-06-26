"""
split_by_flight.py
==================
Second step in the DroneML preprocessing pipeline.

Splits the processed PNG dataset into train / validation / test subsets
using **flight-based** partitioning to prevent data leakage.

Images captured during the same UAV flight are highly spatially correlated.
Splitting by individual image (random split) would cause images from the same
flight to appear in both training and test sets, inflating evaluation metrics.

This module reconstructs the flight → image mapping by replaying the same
sorted discovery pass used by preprocess_tiff.py, then assigns flights to
splits and copies files with shutil.copy2().

Pipeline position
-----------------
Dataset Audit → EDA → preprocess_tiff.py → split_by_flight.py
    → dataset_loader.py → augment.py → verify_dataset.py
    → create_tf_dataset.py → MobileNetV3-Large → Training → Evaluation

Input
-----
processed_dataset/
    Mine/       (Mine_000001.png ... Mine_NNNNNN.png)
    No_Mine/    (NoMine_000001.png ... NoMine_NNNNNN.png)

Output
------
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

Report
------
results/split_report.txt
"""

from __future__ import annotations

import datetime
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Raw Mendeley dataset (needed only to rebuild the flight → image index).
DATASET_ROOT = Path("data/mendeley")

# Where preprocess_tiff.py wrote the processed PNGs.
PROCESSED_DIR = Path("processed_dataset")

# Output parent (same as PROCESSED_DIR; splits are sub-folders).
OUTPUT_DIR = Path("processed_dataset")

# Where the split report is written.
RESULTS_DIR = Path("results")
REPORT_FILE = RESULTS_DIR / "split_report.txt"

# Class folder names (must match preprocess_tiff.py output).
CLASSES: List[str] = ["Mine", "No_Mine"]

# Keyword used in zone-folder names to derive the No_Mine label.
NO_MINE_KEYWORD = "free zone"

# ── Flight assignment ─────────────────────────────────────────────────────────
# Assign each flight date to a split.
# Edit this dict to change the split strategy.
# Keys   : flight-date strings (directory names inside DATASET_ROOT)
# Values : "train" | "validation" | "test"
FLIGHT_SPLIT_ASSIGNMENT: Dict[str, str] = {
    "13_02_2020": "train",
    "03_03_2020": "train",
    "17_02_2020": "validation",
    "24_02_2020": "test",
}

# ──────────────────────────────────────────────────────────────────────────────
# LABEL / DISCOVERY HELPERS  (mirrors preprocess_tiff.py logic exactly)
# ──────────────────────────────────────────────────────────────────────────────

def _derive_label(zone_folder_name: str) -> str:
    """
    Infer Mine / No_Mine label from a zone folder name.

    Mirrors the derive_label function in preprocess_tiff.py so that the
    sequential counters stay in sync.
    """
    return "No_Mine" if NO_MINE_KEYWORD in zone_folder_name.lower() else "Mine"


def _discover_records(dataset_root: Path) -> List[dict]:
    """
    Walk *dataset_root* in the same sorted order as preprocess_tiff.py and
    return one record per source image.

    Each record contains:
        flight : str  - flight-date folder name (e.g. "13_02_2020")
        label  : str  - "Mine" or "No_Mine"
        zone   : str  - zone folder name
        src    : Path - path to the original (raw) source file

    Records are returned in the exact order that preprocess_tiff.py processed
    them, which is essential for mapping sequential output filenames back to
    their originating flights.
    """
    records: List[dict] = []
    for tiff_dir in sorted(dataset_root.rglob("TIFF")):
        if not tiff_dir.is_dir():
            continue
        flight_name = tiff_dir.parent.name  # e.g. "13_02_2020"
        for zone_dir in sorted(tiff_dir.iterdir()):
            if not zone_dir.is_dir():
                continue
            label = _derive_label(zone_dir.name)
            for img_path in sorted(zone_dir.iterdir()):
                if img_path.is_file():
                    records.append(
                        {
                            "flight": flight_name,
                            "label": label,
                            "zone": zone_dir.name,
                            "src": img_path,
                        }
                    )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# INDEX BUILDING
# ──────────────────────────────────────────────────────────────────────────────

def build_flight_index(
    dataset_root: Path,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Reconstruct the mapping between processed PNG filenames and their source
    flight dates.

    preprocess_tiff.py assigned sequential per-class numbers in sorted
    discovery order::

        Mine_000001.png, Mine_000002.png, ...
        NoMine_000001.png, NoMine_000002.png, ...

    This function replays the same walk and assigns the same sequential
    numbers, then matches them to the actual files in *processed_dir*.

    Parameters
    ----------
    dataset_root : Path
        Root of the raw Mendeley dataset (contains flight-date sub-folders).
    processed_dir : Path
        Root of the processed PNG output from preprocess_tiff.py.

    Returns
    -------
    pd.DataFrame with columns:
        filename : str  - basename of the processed PNG (e.g. "Mine_000045.png")
        label    : str  - "Mine" or "No_Mine"
        flight   : str  - flight-date string (e.g. "13_02_2020")
        zone     : str  - zone folder name
        path     : Path - full path to the processed PNG
    """
    records = _discover_records(dataset_root)

    # Per-class sequential counters (same as preprocess_tiff.py).
    counters: Dict[str, int] = {"Mine": 0, "No_Mine": 0}

    rows: List[dict] = []
    for rec in records:
        label = rec["label"]
        counters[label] += 1
        prefix = "Mine" if label == "Mine" else "NoMine"
        filename = f"{prefix}_{counters[label]:06d}.png"
        full_path = processed_dir / label / filename

        rows.append(
            {
                "filename": filename,
                "label": label,
                "flight": rec["flight"],
                "zone": rec["zone"],
                "path": full_path,
            }
        )

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def validate_processed_dir(processed_dir: Path) -> Tuple[int, int]:
    """
    Check that the processed dataset directory exists and contains images.

    Parameters
    ----------
    processed_dir : Path
        Root directory of the processed dataset.

    Returns
    -------
    Tuple[int, int]
        (mine_count, no_mine_count)

    Raises
    ------
    SystemExit
        If the directory or required sub-folders are missing / empty.
    """
    if not processed_dir.exists():
        print(f"[ERROR] processed_dataset directory not found: {processed_dir.resolve()}")
        sys.exit(1)

    mine_dir    = processed_dir / "Mine"
    no_mine_dir = processed_dir / "No_Mine"

    for d in (mine_dir, no_mine_dir):
        if not d.exists():
            print(f"[ERROR] Expected class folder not found: {d.resolve()}")
            sys.exit(1)

    mine_files    = sorted(mine_dir.glob("*.png"))
    no_mine_files = sorted(no_mine_dir.glob("*.png"))

    if not mine_files:
        print(f"[ERROR] Mine folder is empty: {mine_dir.resolve()}")
        sys.exit(1)

    if not no_mine_files:
        print(f"[ERROR] No_Mine folder is empty: {no_mine_dir.resolve()}")
        sys.exit(1)

    return len(mine_files), len(no_mine_files)


def validate_index(df: pd.DataFrame, processed_dir: Path) -> None:
    """
    Cross-check the reconstructed index against the files on disk.

    Prints warnings for any files that are in the index but not on disk,
    or on disk but not in the index. Does not raise; the split proceeds
    with whatever overlap exists.

    Parameters
    ----------
    df : pd.DataFrame
        The index DataFrame returned by build_flight_index().
    processed_dir : Path
        Root of the processed PNG output directory.
    """
    for cls in CLASSES:
        disk_files = {p.name for p in (processed_dir / cls).glob("*.png")}
        idx_files  = set(df.loc[df["label"] == cls, "filename"])

        missing = idx_files  - disk_files
        extra   = disk_files - idx_files

        if missing:
            sample = sorted(missing)[:5]
            print(
                f"[WARNING] {len(missing)} file(s) in index but NOT on disk "
                f"({cls}): {sample} ..."
            )
        if extra:
            sample = sorted(extra)[:5]
            print(
                f"[WARNING] {len(extra)} file(s) on disk but NOT in index "
                f"({cls}): {sample} ..."
            )


def check_leakage(df: pd.DataFrame) -> bool:
    """
    Verify that no flight appears in more than one split.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'flight' and 'split' columns.

    Returns
    -------
    bool
        True  - leakage-free (PASS)
        False - leakage detected (FAIL)
    """
    flight_splits: Dict[str, set] = defaultdict(set)
    for _, row in df.iterrows():
        flight_splits[row["flight"]].add(row["split"])

    leakage = False
    for flight, splits in flight_splits.items():
        if len(splits) > 1:
            print(f"[LEAKAGE] Flight {flight} appears in splits: {splits}")
            leakage = True

    return not leakage


# ──────────────────────────────────────────────────────────────────────────────
# COPY LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_output_dirs(output_dir: Path) -> None:
    """Create all split / class sub-directories."""
    for split in ("train", "validation", "test"):
        for cls in CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)


def copy_images(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Copy processed PNGs from PROCESSED_DIR into the split sub-directories.

    Uses shutil.copy2() to preserve file timestamps. Existing destination
    files are skipped without error (idempotent re-runs).

    Parameters
    ----------
    df : pd.DataFrame
        Index DataFrame containing 'path', 'split', 'label', 'filename'
        columns.
    output_dir : Path
        Root of the split output (usually the same as PROCESSED_DIR).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added 'dst' column (destination path as str).
    """
    _prepare_output_dirs(output_dir)

    dsts: List[str] = []
    skipped = 0

    bar = tqdm(
        df.itertuples(index=False),
        total=len(df),
        unit="img",
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour="cyan",
    )

    for row in bar:
        src: Path = row.path
        dst: Path = output_dir / row.split / row.label / row.filename

        bar.set_postfix_str(f"{row.split}/{row.label}/{row.filename}", refresh=False)

        if not src.exists():
            print(f"\n[WARNING] Source file missing, skipping: {src}")
            dsts.append("MISSING")
            skipped += 1
            continue

        if dst.exists():
            # Already present - idempotent re-run, count but do not re-copy.
            dsts.append(str(dst))
            skipped += 1
            continue

        try:
            shutil.copy2(src, dst)
            dsts.append(str(dst))
        except OSError as exc:
            print(f"\n[ERROR] Could not copy {src.name}: {exc}")
            dsts.append("ERROR")
            skipped += 1

    if skipped:
        print(f"\n[INFO] {skipped} file(s) were skipped (already existed or errors).")

    result = df.copy()
    result["dst"] = dsts
    return result


# ──────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def verify_split(df_with_dst: pd.DataFrame, output_dir: Path) -> bool:
    """
    Perform post-copy verification.

    Checks
    ------
    1. Every source image appears exactly once across all splits.
    2. No missing destination files (copy failures).
    3. No flight exists in more than one split (leakage check).
    4. Mine and No_Mine sub-folders exist in every split.

    Parameters
    ----------
    df_with_dst : pd.DataFrame
        DataFrame returned by copy_images(), with a 'dst' column.
    output_dir : Path
        Root of the split output directory.

    Returns
    -------
    bool
        True if all checks pass.
    """
    all_pass = True

    # 1 - every image appears exactly once.
    filename_counts = Counter(df_with_dst["filename"])
    duplicates = {k: v for k, v in filename_counts.items() if v > 1}
    if duplicates:
        print(f"  [FAIL] Duplicate filenames detected: {list(duplicates.keys())[:5]}")
        all_pass = False
    else:
        print("  [PASS] Every processed image appears exactly once.")

    # 2 - no missing destinations.
    missing = df_with_dst[df_with_dst["dst"] == "MISSING"]
    if not missing.empty:
        print(f"  [FAIL] {len(missing)} file(s) could not be copied (source missing).")
        all_pass = False
    else:
        print("  [PASS] No missing source files.")

    # 3 - leakage check.
    leakage_free = check_leakage(df_with_dst)
    if leakage_free:
        print("  [PASS] No data leakage detected.")
    else:
        all_pass = False

    # 4 - Mine and No_Mine sub-folders exist in every split.
    folders_ok = True
    for split in ("train", "validation", "test"):
        for cls in CLASSES:
            folder = output_dir / split / cls
            if not folder.exists():
                print(f"  [FAIL] Missing folder: {folder}")
                all_pass = False
                folders_ok = False
    if folders_ok:
        print("  [PASS] All split sub-directories present.")

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────

def write_report(
    df: pd.DataFrame,
    leakage_free: bool,
    report_path: Path,
) -> None:
    """
    Write a plain-text split report to *report_path*.

    Parameters
    ----------
    df : pd.DataFrame
        Index DataFrame with 'split', 'label', 'flight' columns.
    leakage_free : bool
        Result of the leakage check.
    report_path : Path
        Destination file path.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = [
        "=" * 45,
        "FLIGHT-BASED SPLIT REPORT",
        "=" * 45,
        f"Generated : {timestamp}",
        "",
    ]

    # -- Flight assignments --------------------------------------------------
    lines += ["-- Flight Assignments ----------------------------------"]
    flight_split_map: Dict[str, str] = (
        df[["flight", "split"]].drop_duplicates().set_index("flight")["split"].to_dict()
    )
    for flight, split in sorted(flight_split_map.items()):
        lines.append(f"  {flight:20s}  ->  {split}")
    lines.append("")

    # -- Per-split image counts ---------------------------------------------
    lines += ["-- Images per Split ------------------------------------"]
    total_all = len(df)
    for split in ("train", "validation", "test"):
        sub     = df[df["split"] == split]
        mine    = len(sub[sub["label"] == "Mine"])
        no_mine = len(sub[sub["label"] == "No_Mine"])
        total   = len(sub)
        pct     = 100.0 * total / total_all if total_all else 0.0
        lines += [
            f"  {split.upper():10s}   total={total:4d}  ({pct:5.1f}%)",
            f"              Mine    = {mine}",
            f"              No_Mine = {no_mine}",
        ]
    lines.append("")

    # -- Overall totals -----------------------------------------------------
    lines += ["-- Overall Totals --------------------------------------"]
    total_mine   = len(df[df["label"] == "Mine"])
    total_nomine = len(df[df["label"] == "No_Mine"])
    lines += [
        f"  Total images : {total_all}",
        f"  Mine         : {total_mine}",
        f"  No_Mine      : {total_nomine}",
        "",
    ]

    # -- Leakage check -------------------------------------------------------
    lines += ["-- Leakage Check ---------------------------------------"]
    lines.append(f"  Result : {'PASS' if leakage_free else 'FAIL'}")
    lines.append("")
    lines.append("=" * 45)

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# CONSOLE OUTPUT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _print_header() -> None:
    """Print the section header banner."""
    print()
    print("=" * 41)
    print("FLIGHT-BASED DATASET SPLIT")
    print("=" * 41)
    print()


def _print_flight_assignments(assignment: Dict[str, str]) -> None:
    """Pretty-print the flight -> split assignment grouped by split."""
    splits: Dict[str, List[str]] = defaultdict(list)
    for flight, split in sorted(assignment.items()):
        splits[split].append(flight)

    labels = [
        ("Training Flights",  splits.get("train",      [])),
        ("Validation Flight", splits.get("validation", [])),
        ("Testing Flight",    splits.get("test",       [])),
    ]
    for title, flights in labels:
        print(title)
        for f in flights:
            print(f"  {f}")
        print()


def _print_statistics(df: pd.DataFrame) -> None:
    """Print per-split image counts to stdout."""
    for split in ("train", "validation", "test"):
        sub     = df[df["split"] == split]
        mine    = len(sub[sub["label"] == "Mine"])
        no_mine = len(sub[sub["label"] == "No_Mine"])
        total   = len(sub)
        print(f"{split.capitalize()} Images  ({total} total)")
        print(f"  Mine    : {mine}")
        print(f"  No Mine : {no_mine}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point for the flight-based dataset split.

    Steps
    -----
    1.  Validate that the processed dataset exists and contains images.
    2.  Reconstruct the flight -> image mapping from the raw dataset.
    3.  Assign each image a split label based on its flight.
    4.  Validate the index against files on disk.
    5.  Copy images into train / validation / test sub-directories.
    6.  Run post-copy verification.
    7.  Print statistics.
    8.  Write the split report.
    """
    _print_header()

    # -- Step 1 : Validate processed dataset ---------------------------------
    print("Validating Processed Dataset...")
    mine_count, no_mine_count = validate_processed_dir(PROCESSED_DIR)
    print(f"  Mine    : {mine_count} images")
    print(f"  No Mine : {no_mine_count} images")
    print()

    if not DATASET_ROOT.exists():
        print(f"[ERROR] Raw dataset not found at: {DATASET_ROOT.resolve()}")
        print("        This directory is required to rebuild the flight -> image index.")
        sys.exit(1)

    # -- Step 2 : Build the flight -> image index ----------------------------
    print("Rebuilding Flight -> Image Index...")
    df = build_flight_index(DATASET_ROOT, PROCESSED_DIR)

    # Cross-check index against disk.
    validate_index(df, PROCESSED_DIR)

    flights_found = sorted(df["flight"].unique())
    print(f"  Flights discovered : {flights_found}")
    print(f"  Index entries      : {len(df)}")
    print()

    # -- Step 3 : Assign splits ----------------------------------------------
    unassigned = set(flights_found) - set(FLIGHT_SPLIT_ASSIGNMENT.keys())
    if unassigned:
        print(
            f"[WARNING] {len(unassigned)} flight(s) not in FLIGHT_SPLIT_ASSIGNMENT "
            f"and will be skipped: {sorted(unassigned)}"
        )

    df["split"] = df["flight"].map(FLIGHT_SPLIT_ASSIGNMENT)
    df = df[df["split"].notna()].copy()  # Drop unassigned flights.

    if df.empty:
        print(
            "[ERROR] No images were assigned to any split. "
            "Check FLIGHT_SPLIT_ASSIGNMENT in the configuration section."
        )
        sys.exit(1)

    # -- Step 4 : Print flight assignments -----------------------------------
    _print_flight_assignments(FLIGHT_SPLIT_ASSIGNMENT)

    # -- Step 5 : Copy images ------------------------------------------------
    print("Copying Images...")
    print()
    df_copied = copy_images(df, OUTPUT_DIR)
    print()

    # -- Step 6 : Statistics -------------------------------------------------
    _print_statistics(df_copied)

    # -- Step 7 : Verification -----------------------------------------------
    print("Leakage Check")
    leakage_free = verify_split(df_copied, OUTPUT_DIR)
    status = "PASS" if leakage_free else "FAIL"
    print(f"\n  Result: {status}")
    print()

    # -- Step 8 : Report -----------------------------------------------------
    write_report(df_copied, leakage_free, REPORT_FILE)
    print(f"Report saved to")
    print(f"  {REPORT_FILE}")
    print()

    print("=" * 41)
    print("Finished")
    print("=" * 41)
    print()


if __name__ == "__main__":
    main()

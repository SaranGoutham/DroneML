"""
Cross-Format Duplicate Audit
DroneML

Determines whether JPG, R_JPG, and TIFF folders contain
duplicate, near-duplicate, or unique images by comparing
perceptual and average hashes across all formats.

Usage:
    python -m src.audit.check_cross_format_duplicates
"""

import logging
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so the import below works when run
# with  python -m src.audit.check_cross_format_duplicates
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import DATASET_DIR, RESULTS_DIR  # noqa: E402

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =====================================================
# CONSTANTS
# =====================================================

VALID_FORMATS: list[str] = ["JPG", "R_JPG", "TIFF"]

VALID_EXTENSIONS: set[str] = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
}

HAMMING_THRESHOLD: int = 5

# Configurable recommendation threshold: if the ratio of
# (exact + near duplicates) / total_images exceeds this value,
# the recommendation will be "Use TIFF Only".
DUPLICATE_RATIO_THRESHOLD: float = 0.30

# Output paths (derived from config)
CROSS_FORMAT_CSV: Path = RESULTS_DIR / "cross_format_duplicates.csv"
CROSS_FORMAT_SUMMARY: Path = RESULTS_DIR / "cross_format_summary.txt"
CROSS_FORMAT_STATS: Path = RESULTS_DIR / "cross_format_statistics.csv"

# =====================================================
# HELPERS
# =====================================================


def _is_image_file(path: Path) -> bool:
    """Return True if *path* has a recognised image extension."""
    return path.suffix.lower() in VALID_EXTENSIONS


def _collect_images(base_dir: Path, fmt: str) -> list[Path]:
    """Recursively collect image files under *base_dir* for a given format folder."""
    images: list[Path] = []
    for flight in sorted(base_dir.iterdir()):
        if not flight.is_dir():
            continue
        fmt_dir = flight / fmt
        if not fmt_dir.is_dir():
            continue
        for img_path in sorted(fmt_dir.rglob("*")):
            if img_path.is_file() and _is_image_file(img_path):
                images.append(img_path)
    return images


def _parse_zone(img_path: Path, fmt: str) -> str:
    """Extract the zone folder name from an image path.

    Expected structure: …/<flight>/<format>/<zone>/image.ext
    """
    # Walk up from the image file to find the zone folder (child of the
    # format folder).
    parts = img_path.parts
    try:
        fmt_idx = parts.index(fmt)
        return parts[fmt_idx + 1] if fmt_idx + 1 < len(parts) - 1 else "Unknown"
    except ValueError:
        return "Unknown"


def _parse_flight(img_path: Path, fmt: str) -> str:
    """Extract the flight date folder name from an image path."""
    parts = img_path.parts
    try:
        fmt_idx = parts.index(fmt)
        return parts[fmt_idx - 1] if fmt_idx >= 1 else "Unknown"
    except ValueError:
        return "Unknown"


def scan_images(base_dir: Path) -> pd.DataFrame:
    """Scan all images across JPG, R_JPG, and TIFF folders.

    Returns a DataFrame with metadata for every image found.
    """
    records: list[dict[str, Any]] = []
    corrupted_count: int = 0

    for fmt in VALID_FORMATS:
        print(f"Scanning {fmt}...")
        logger.info("Scanning %s ...", fmt)
        images = _collect_images(base_dir, fmt)
        logger.info("  Found %d images in %s", len(images), fmt)

        for img_path in tqdm(images, desc=f"  {fmt}", unit="img"):
            flight = _parse_flight(img_path, fmt)
            zone = _parse_zone(img_path, fmt)

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                    # Warn about 16-bit TIFF images (thermal data)
                    if fmt == "TIFF" and img.mode not in ("RGB", "L"):
                        logger.debug(
                            "TIFF image %s has mode '%s'; hashing will "
                            "use the rendered preview, not raw sensor values.",
                            img_path.name, img.mode,
                        )
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", img_path, exc)
                corrupted_count += 1
                continue

            records.append({
                "Flight": flight,
                "Format": fmt,
                "Zone": zone,
                "Filename": img_path.name,
                "Extension": img_path.suffix.lower(),
                "Width": width,
                "Height": height,
                "ImageMode": mode,
                "FileSize": img_path.stat().st_size,
                "AbsolutePath": str(img_path),
            })

    if corrupted_count:
        logger.warning("Skipped %d corrupted / unreadable images.", corrupted_count)

    return pd.DataFrame(records)


# =====================================================
# HASHING
# =====================================================


def compute_hashes(df: pd.DataFrame) -> pd.DataFrame:
    """Open every image and compute perceptual + average hashes.

    Adds columns ``phash`` and ``ahash`` (string representations).
    """
    print("Computing hashes...")
    logger.info("Computing perceptual and average hashes for %d images ...", len(df))

    phashes: list[str] = []
    ahashes: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Hashing", unit="img"):
        img_path = Path(row["AbsolutePath"])
        try:
            with Image.open(img_path) as img:
                # Convert to RGB to normalise modes before hashing
                rgb = img.convert("RGB")
                p = imagehash.phash(rgb)
                a = imagehash.average_hash(rgb)
                phashes.append(str(p))
                ahashes.append(str(a))
        except Exception as exc:
            logger.warning("Hash failed for %s: %s", img_path, exc)
            phashes.append("")
            ahashes.append("")

    df = df.copy()
    df["phash"] = phashes
    df["ahash"] = ahashes
    return df


# =====================================================
# COMPARISON
# =====================================================


def _hamming(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex-encoded imagehash strings."""
    hash1 = imagehash.hex_to_hash(h1)
    hash2 = imagehash.hex_to_hash(h2)
    return hash1 - hash2  # imagehash overloads __sub__ as Hamming distance


def compare_images(df: pd.DataFrame) -> tuple[pd.DataFrame, Counter[int], Counter[str]]:
    """Compare images across formats within the same (flight, zone) group.

    Only images sharing the same flight and zone are compared across
    formats, which dramatically reduces unnecessary comparisons.

    The ``ahash`` is used as secondary verification: a pair flagged by
    ``phash`` is only kept if its ``ahash`` Hamming distance is also
    within the threshold.

    Returns:
        df_pairs: DataFrame of duplicate / near-duplicate pairs.
        distance_histogram: Counter mapping Hamming distance → count
            (for *all* within-threshold pairs, useful for analysis).
        pair_format_counts: Counter mapping format-pair label
            (e.g. "JPG↔R_JPG") → number of flagged pairs.
    """
    print("Comparing images...")
    logger.info("Comparing images across formats (grouped by flight + zone) ...")

    # Only compare images that have valid hashes
    valid = df[(df["phash"] != "") & (df["ahash"] != "")].reset_index(drop=True)

    pairs: list[dict[str, Any]] = []
    distance_histogram: Counter[int] = Counter()
    pair_format_counts: Counter[str] = Counter()

    # Group images by (flight, zone) so we only compare within the same group
    grouped = valid.groupby(["Flight", "Zone"])

    total_comparisons = 0

    for (flight, zone), group_df in tqdm(grouped, desc="  Groups", unit="grp"):
        # Within this group, separate by format
        format_groups: dict[str, pd.DataFrame] = {
            fmt: grp.reset_index(drop=True)
            for fmt, grp in group_df.groupby("Format")
        }
        format_names = list(format_groups.keys())

        # Compare across every pair of formats within the group
        for fmt_a, fmt_b in combinations(format_names, 2):
            grp_a = format_groups[fmt_a]
            grp_b = format_groups[fmt_b]
            total_comparisons += len(grp_a) * len(grp_b)
            pair_label = f"{fmt_a}↔{fmt_b}"

            for _, row_a in grp_a.iterrows():
                for _, row_b in grp_b.iterrows():
                    phash_dist = _hamming(row_a["phash"], row_b["phash"])

                    if phash_dist <= HAMMING_THRESHOLD:
                        # Secondary verification with average hash
                        ahash_dist = _hamming(row_a["ahash"], row_b["ahash"])
                        if ahash_dist > HAMMING_THRESHOLD:
                            continue

                        if phash_dist == 0:
                            dup_type = "Exact Duplicate"
                        else:
                            dup_type = "Near Duplicate"

                        distance_histogram[phash_dist] += 1
                        pair_format_counts[pair_label] += 1

                        pairs.append({
                            "Image1": row_a["AbsolutePath"],
                            "Image2": row_b["AbsolutePath"],
                            "Flight1": row_a["Flight"],
                            "Flight2": row_b["Flight"],
                            "Format1": row_a["Format"],
                            "Format2": row_b["Format"],
                            "Zone1": row_a["Zone"],
                            "Zone2": row_b["Zone"],
                            "HashDistance": phash_dist,
                            "DuplicateType": dup_type,
                        })

    logger.info("  Total comparisons performed: %d", total_comparisons)
    logger.info("  Pairs flagged: %d", len(pairs))

    return pd.DataFrame(pairs), distance_histogram, pair_format_counts


# =====================================================
# REPORTING
# =====================================================


def _count_unique(df_images: pd.DataFrame, df_pairs: pd.DataFrame) -> int:
    """Count images that appear in zero duplicate / near-duplicate pairs.

    Uses full absolute paths to avoid false collisions when different
    folders contain files with the same basename.
    """
    if df_pairs.empty:
        return len(df_images)

    involved = set(df_pairs["Image1"]).union(set(df_pairs["Image2"]))
    return len(df_images[~df_images["AbsolutePath"].isin(involved)])


def write_reports(
    df_images: pd.DataFrame,
    df_pairs: pd.DataFrame,
    distance_histogram: Counter[int],
    pair_format_counts: Counter[str],
) -> None:
    """Write the three output files: CSV pairs, summary TXT, statistics CSV."""
    print("Writing reports...")
    logger.info("Writing reports to %s", RESULTS_DIR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Duplicate pairs CSV ----------------------------------------
    df_pairs.to_csv(CROSS_FORMAT_CSV, index=False)
    logger.info("  %s  (%d rows)", CROSS_FORMAT_CSV.name, len(df_pairs))

    # --- 2. Per-format statistics CSV -----------------------------------
    exact_mask = df_pairs["DuplicateType"] == "Exact Duplicate" if not df_pairs.empty else pd.Series(dtype=bool)
    near_mask = df_pairs["DuplicateType"] == "Near Duplicate" if not df_pairs.empty else pd.Series(dtype=bool)

    stats_rows: list[dict[str, Any]] = []

    for fmt in VALID_FORMATS:
        fmt_imgs = df_images[df_images["Format"] == fmt]
        n_images = len(fmt_imgs)

        if df_pairs.empty:
            stats_rows.append({
                "Format": fmt,
                "Images": n_images,
                "Unique": n_images,
                "Duplicates": 0,
                "NearDuplicates": 0,
            })
            continue

        # Count images from this format that appear in a pair
        fmt_exact = df_pairs[
            (exact_mask) & ((df_pairs["Format1"] == fmt) | (df_pairs["Format2"] == fmt))
        ]
        fmt_near = df_pairs[
            (near_mask) & ((df_pairs["Format1"] == fmt) | (df_pairs["Format2"] == fmt))
        ]

        # Unique *paths* involved in exact / near duplicate pairs
        exact_files: set[str] = set()
        near_files: set[str] = set()
        for _, r in fmt_exact.iterrows():
            if r["Format1"] == fmt:
                exact_files.add(r["Image1"])
            if r["Format2"] == fmt:
                exact_files.add(r["Image2"])
        for _, r in fmt_near.iterrows():
            if r["Format1"] == fmt:
                near_files.add(r["Image1"])
            if r["Format2"] == fmt:
                near_files.add(r["Image2"])

        dup_count = len(exact_files)
        near_count = len(near_files - exact_files)
        unique_count = n_images - len(exact_files | near_files)

        stats_rows.append({
            "Format": fmt,
            "Images": n_images,
            "Unique": unique_count,
            "Duplicates": dup_count,
            "NearDuplicates": near_count,
        })

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(CROSS_FORMAT_STATS, index=False)
    logger.info("  %s", CROSS_FORMAT_STATS.name)

    # --- 3. Summary TXT ------------------------------------------------
    total_images = len(df_images)
    jpg_count = len(df_images[df_images["Format"] == "JPG"])
    rjpg_count = len(df_images[df_images["Format"] == "R_JPG"])
    tiff_count = len(df_images[df_images["Format"] == "TIFF"])

    exact_count = int((df_pairs["DuplicateType"] == "Exact Duplicate").sum()) if not df_pairs.empty else 0
    near_count = int((df_pairs["DuplicateType"] == "Near Duplicate").sum()) if not df_pairs.empty else 0
    unique_count = _count_unique(df_images, df_pairs)

    # Recommendation logic (configurable threshold)
    dup_ratio = (exact_count + near_count) / max(total_images, 1)
    if dup_ratio > DUPLICATE_RATIO_THRESHOLD:
        recommendation = (
            "Use TIFF Only\n"
            "  High cross-format duplication detected.\n"
            "  Including all formats would introduce redundant data\n"
            "  and risk data leakage during training."
        )
    else:
        recommendation = (
            "Combine JPG + R_JPG + TIFF\n"
            "  Low cross-format duplication.\n"
            "  The formats capture sufficiently different information\n"
            "  and can be safely combined for training."
        )

    # Build Hamming distance histogram string
    hist_lines: list[str] = []
    if distance_histogram:
        for dist in sorted(distance_histogram.keys()):
            bar = "█" * distance_histogram[dist]
            hist_lines.append(f"    {dist:>2}  {bar}  ({distance_histogram[dist]})")
    else:
        hist_lines.append("    (no duplicates detected)")

    # Build per-format-pair counts string
    pair_lines: list[str] = []
    if pair_format_counts:
        for label, count in pair_format_counts.most_common():
            pair_lines.append(f"    {label:<16}: {count}")
    else:
        pair_lines.append("    (no duplicates detected)")

    summary_lines = [
        "=" * 50,
        "CROSS FORMAT AUDIT",
        "=" * 50,
        "",
        f"JPG Images          : {jpg_count}",
        f"R_JPG Images        : {rjpg_count}",
        f"TIFF Images         : {tiff_count}",
        f"Total Images        : {total_images}",
        "",
        f"Exact Duplicates    : {exact_count}",
        f"Near Duplicates     : {near_count}",
        f"Unique Images       : {unique_count}",
        "",
        "Hamming Distance Histogram:",
        *hist_lines,
        "",
        "Duplicates by Format Pair:",
        *pair_lines,
        "",
        f"Dup. Ratio Threshold: {DUPLICATE_RATIO_THRESHOLD:.0%}",
        f"Dup. Ratio Observed : {dup_ratio:.2%}",
        "",
        f"Recommendation      : {recommendation}",
        "",
        "=" * 50,
    ]

    summary_text = "\n".join(summary_lines)
    CROSS_FORMAT_SUMMARY.write_text(summary_text, encoding="utf-8")
    logger.info("  %s", CROSS_FORMAT_SUMMARY.name)

    # Print summary to console as well
    print()
    print(summary_text)


# =====================================================
# MAIN
# =====================================================


def main() -> None:
    """Entry point for the cross-format duplicate audit."""
    logger.info("Starting Cross-Format Duplicate Audit")
    logger.info("Dataset directory: %s", DATASET_DIR)

    # Step 1 – Scan
    df_images = scan_images(DATASET_DIR)
    logger.info("Scanned %d images total.", len(df_images))

    if df_images.empty:
        logger.error("No images found. Check DATASET_DIR: %s", DATASET_DIR)
        sys.exit(1)

    # Step 2 – Hash
    df_images = compute_hashes(df_images)

    # Step 3 – Compare (grouped by flight + zone)
    df_pairs, distance_histogram, pair_format_counts = compare_images(df_images)

    # Step 4 – Report
    write_reports(df_images, df_pairs, distance_histogram, pair_format_counts)

    print("Done.")
    logger.info("Audit complete.")


if __name__ == "__main__":
    main()

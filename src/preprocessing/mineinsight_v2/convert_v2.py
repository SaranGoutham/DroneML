"""
Master Dataset Reconstruction Module for MineInsight classification_v2
========================================================================

Orchestrates the conversion of MineInsight raw LWIR frames into the single
canonical classification_v2 crop tree:
    data/mineinsight/classification_v2/
        Mine/
        Known/
        Background/

And generates the master manifest and dual protocol split files:
    reports_v2/
        master_crop_manifest.csv
        protocol1_split.csv
        protocol2_split.csv

Requirements:
- Single canonical storage tree (zero crop duplication into split directories)
- Target 1:1 isotropic square crops with 20% margin resized to 224x224
- Background native center 224x224 tiles with thermal variance >= 2.0
- Preservation of provenance, timestamps, coordinates, and SHA-256 hashes
- Preservation of 14,236 quarantined Track 3 S2 orphan labels
- Zero modification to raw dataset, targets_list.yaml, or legacy crops
"""

import os
import io
import sys
import time
import shutil
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures

import numpy as np
from PIL import Image

from src.preprocessing.mineinsight_v2.config import (
    RAW_DIR,
    CLASSIFICATION_V2_DIR,
    MINE_DIR,
    KNOWN_DIR,
    BACKGROUND_DIR,
    REPORTS_V2_DIR,
    MASTER_MANIFEST_CSV,
    PROTOCOL1_SPLIT_CSV,
    PROTOCOL2_SPLIT_CSV,
    VALID_SEQUENCES,
    KNOWN_CLASS_IDS,
    MINE_CLASS_IDS,
    BACKGROUND_CLASS_ID,
    JPEG_QUALITY,
    CATEGORY_MINE,
    CATEGORY_KNOWN,
    CATEGORY_BACKGROUND,
)
from src.preprocessing.mineinsight_v2.targets_v2 import get_target_registry
from src.preprocessing.mineinsight_v2.crop_engine import (
    extract_isotropic_crop,
    extract_native_center_tile,
)
from src.preprocessing.mineinsight_v2.manifest import (
    CropManifestRecord,
    parse_timestamp_from_stem,
    export_manifest_csv,
    export_split_manifest_csv,
)
from src.preprocessing.mineinsight_v2.split_v2 import (
    assign_splits_to_records,
    verify_split_isolation,
)
from src.preprocessing.mineinsight_v2.validator_v2 import (
    run_full_preflight_validation,
    verify_reconstructed_dataset,
)
from src.preprocessing.mineinsight_v2.dry_run import execute_dry_run

logger = logging.getLogger(__name__)


def process_frame_chunk(
    chunk_items: List[Tuple[str, str, str, str, int, float]],
    mine_dir_str: str,
    known_dir_str: str,
    bg_dir_str: str,
    target_meta_map: Dict[int, Tuple[str, str]],
    jpeg_quality: int = 95,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Worker function to process a batch of frame tasks in parallel.
    
    Each item: (seq, stem, img_path_str, lbl_path_str, lbl_size, timestamp)
    Returns:
        (manifest_record_dicts, chunk_statistics)
    """
    records: List[Dict[str, Any]] = []
    stats = {
        "mine": 0,
        "known": 0,
        "bg": 0,
        "failed_crops": 0,
        "unreadable_images": 0,
        "invalid_boxes": 0,
    }

    for seq, stem, img_path_str, lbl_path_str, lbl_size, ts in chunk_items:
        parts = seq.split("_")
        track_name = f"track_{parts[1]}"
        seq_name = parts[2]

        # Case 1: Empty Frame -> Background Native Center Tile (k=1)
        if lbl_size == 0:
            try:
                with Image.open(img_path_str) as full_img:
                    full_img.load()
                    tile_img, (xmin, ymin, xmax, ymax) = extract_native_center_tile(full_img)
                    arr = np.array(tile_img)
                    if arr.ndim == 3:
                        arr = arr[:, :, 0]
                    std_dev = float(np.std(arr))
                    if std_dev < 2.0:
                        stats["failed_crops"] += 1
                        continue

                    crop_filename = f"{seq}_{stem}_bg01_soil_tile.jpg"
                    out_path = os.path.join(bg_dir_str, crop_filename)
                    buf = io.BytesIO()
                    tile_img.save(buf, format="JPEG", quality=jpeg_quality)
                    data = buf.getvalue()
                    file_sha256 = hashlib.sha256(data).hexdigest()
                    with open(out_path, "wb") as f_out:
                        f_out.write(data)

                    rel_path = f"classification_v2/Background/{crop_filename}"
                    records.append({
                        "crop_id": CropManifestRecord.generate_crop_id(rel_path),
                        "crop_rel_path": rel_path,
                        "category": CATEGORY_BACKGROUND,
                        "target_instance_id": BACKGROUND_CLASS_ID,
                        "target_name": "Natural_Terrain_Tile",
                        "semantic_subgroup": "Terrain_Tile",
                        "source_track": track_name,
                        "source_sequence": seq_name,
                        "source_image_name": os.path.basename(img_path_str),
                        "source_label_name": f"{stem}.txt",
                        "annotation_line_idx": 0,
                        "orig_x_center": 0.5,
                        "orig_y_center": 0.5,
                        "orig_width": 224.0 / 640.0,
                        "orig_height": 224.0 / 512.0,
                        "crop_xmin_px": xmin,
                        "crop_ymin_px": ymin,
                        "crop_xmax_px": xmax,
                        "crop_ymax_px": ymax,
                        "crop_width": 224,
                        "crop_height": 224,
                        "flight_timestamp_epoch": ts,
                        "protocol1_split": "unassigned",
                        "protocol2_split": "unassigned",
                        "file_sha256": file_sha256,
                    })
                    stats["bg"] += 1
            except Exception:
                stats["unreadable_images"] += 1
            continue

        # Case 2: Non-Empty Label -> Read Annotations
        try:
            with open(lbl_path_str, "r", encoding="utf-8") as f_lbl:
                lines = [line.strip() for line in f_lbl if line.strip()]
        except Exception:
            stats["unreadable_images"] += 1
            continue

        if not lines:
            # Empty file despite non-zero size (whitespace only)
            try:
                with Image.open(img_path_str) as full_img:
                    full_img.load()
                    tile_img, (xmin, ymin, xmax, ymax) = extract_native_center_tile(full_img)
                    arr = np.array(tile_img)
                    if arr.ndim == 3:
                        arr = arr[:, :, 0]
                    std_dev = float(np.std(arr))
                    if std_dev < 2.0:
                        stats["failed_crops"] += 1
                        continue

                    crop_filename = f"{seq}_{stem}_bg01_soil_tile.jpg"
                    out_path = os.path.join(bg_dir_str, crop_filename)
                    buf = io.BytesIO()
                    tile_img.save(buf, format="JPEG", quality=jpeg_quality)
                    data = buf.getvalue()
                    file_sha256 = hashlib.sha256(data).hexdigest()
                    with open(out_path, "wb") as f_out:
                        f_out.write(data)

                    rel_path = f"classification_v2/Background/{crop_filename}"
                    records.append({
                        "crop_id": CropManifestRecord.generate_crop_id(rel_path),
                        "crop_rel_path": rel_path,
                        "category": CATEGORY_BACKGROUND,
                        "target_instance_id": BACKGROUND_CLASS_ID,
                        "target_name": "Natural_Terrain_Tile",
                        "semantic_subgroup": "Terrain_Tile",
                        "source_track": track_name,
                        "source_sequence": seq_name,
                        "source_image_name": os.path.basename(img_path_str),
                        "source_label_name": f"{stem}.txt",
                        "annotation_line_idx": 0,
                        "orig_x_center": 0.5,
                        "orig_y_center": 0.5,
                        "orig_width": 224.0 / 640.0,
                        "orig_height": 224.0 / 512.0,
                        "crop_xmin_px": xmin,
                        "crop_ymin_px": ymin,
                        "crop_xmax_px": xmax,
                        "crop_ymax_px": ymax,
                        "crop_width": 224,
                        "crop_height": 224,
                        "flight_timestamp_epoch": ts,
                        "protocol1_split": "unassigned",
                        "protocol2_split": "unassigned",
                        "file_sha256": file_sha256,
                    })
                    stats["bg"] += 1
            except Exception:
                stats["unreadable_images"] += 1
            continue

        # Non-empty target objects in frame
        try:
            with Image.open(img_path_str) as full_img:
                full_img.load()
                for line_idx, line in enumerate(lines, start=1):
                    p = line.split()
                    if len(p) != 5:
                        stats["invalid_boxes"] += 1
                        continue
                    try:
                        cid = int(p[0])
                        xc = float(p[1])
                        yc = float(p[2])
                        w = float(p[3])
                        h = float(p[4])
                    except ValueError:
                        stats["invalid_boxes"] += 1
                        continue

                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        stats["invalid_boxes"] += 1
                        continue

                    if cid in MINE_CLASS_IDS:
                        cat = CATEGORY_MINE
                        out_dir = mine_dir_str
                        stats["mine"] += 1
                    elif cid in KNOWN_CLASS_IDS:
                        cat = CATEGORY_KNOWN
                        out_dir = known_dir_str
                        stats["known"] += 1
                    else:
                        stats["invalid_boxes"] += 1
                        continue

                    t_name, t_subgroup = target_meta_map.get(cid, ("Unknown", "Unknown"))
                    safe_name = t_name.lower().replace(" ", "_").replace("-", "_")
                    crop_filename = f"{seq}_{stem}_obj{line_idx:02d}_id{cid}_{safe_name}.jpg"
                    out_path = os.path.join(out_dir, crop_filename)

                    crop_img, (xmin, ymin, xmax, ymax) = extract_isotropic_crop(full_img, xc, yc, w, h)
                    buf = io.BytesIO()
                    crop_img.save(buf, format="JPEG", quality=jpeg_quality)
                    data = buf.getvalue()
                    file_sha256 = hashlib.sha256(data).hexdigest()
                    with open(out_path, "wb") as f_out:
                        f_out.write(data)

                    rel_path = f"classification_v2/{cat}/{crop_filename}"
                    records.append({
                        "crop_id": CropManifestRecord.generate_crop_id(rel_path),
                        "crop_rel_path": rel_path,
                        "category": cat,
                        "target_instance_id": cid,
                        "target_name": t_name,
                        "semantic_subgroup": t_subgroup,
                        "source_track": track_name,
                        "source_sequence": seq_name,
                        "source_image_name": os.path.basename(img_path_str),
                        "source_label_name": f"{stem}.txt",
                        "annotation_line_idx": line_idx,
                        "orig_x_center": xc,
                        "orig_y_center": yc,
                        "orig_width": w,
                        "orig_height": h,
                        "crop_xmin_px": xmin,
                        "crop_ymin_px": ymin,
                        "crop_xmax_px": xmax,
                        "crop_ymax_px": ymax,
                        "crop_width": 224,
                        "crop_height": 224,
                        "flight_timestamp_epoch": ts,
                        "protocol1_split": "unassigned",
                        "protocol2_split": "unassigned",
                        "file_sha256": file_sha256,
                    })
        except Exception:
            stats["unreadable_images"] += 1

    return records, stats


def run_reconstruction(
    num_workers: Optional[int] = None,
    chunk_size: int = 300,
) -> None:
    """
    Execute physical crop generation into single classification_v2 tree
    and export master manifest + dual split files.
    """
    total_start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING PHYSICAL CROP GENERATION: MineInsight classification_v2")
    logger.info("=" * 70)

    # 1. Pre-flight Validation Check
    preflight = run_full_preflight_validation()
    if preflight["preflight_status"] != "PASS":
        logger.error(f"Pre-flight validation FAILED! {preflight['yaml_errors'] + preflight['image_errors']}")
        sys.exit(1)
    logger.info("Pre-flight validation PASSED.")

    # 2. Setup Canonical Directories
    for d in [MINE_DIR, KNOWN_DIR, BACKGROUND_DIR, REPORTS_V2_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    reg = get_target_registry()
    target_meta_map = {
        cid: (meta.name, meta.semantic_subgroup)
        for cid, meta in reg.get_all_targets().items()
    }

    # 3. Fast Discovery Across Sequences
    logger.info("Scanning raw sequences for matching image-label pairs...")
    all_frame_tasks: List[Tuple[str, str, str, str, int, float]] = []

    for seq in VALID_SEQUENCES:
        img_dir = RAW_DIR / f"{seq}_lwir_images"
        lbl_dir = RAW_DIR / f"{seq}_lwir_labels_reproj"
        if not img_dir.exists():
            img_dir = RAW_DIR / f"{seq}_wir_images"
        if not lbl_dir.exists():
            lbl_dir = RAW_DIR / f"{seq}_wir_labels_reproj"

        img_stems: Dict[str, str] = {}
        with os.scandir(img_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    stem = entry.name.rsplit(".", 1)[0]
                    img_stems[stem] = entry.path

        lbl_files: Dict[str, Tuple[str, int]] = {}
        with os.scandir(lbl_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".txt"):
                    stem = entry.name.rsplit(".", 1)[0]
                    lbl_files[stem] = (entry.path, entry.stat(follow_symlinks=False).st_size)

        matching_stems = sorted(set(img_stems.keys()).intersection(lbl_files.keys()))
        for stem in matching_stems:
            img_p = img_stems[stem]
            lbl_p, sz = lbl_files[stem]
            ts = parse_timestamp_from_stem(stem)
            all_frame_tasks.append((seq, stem, img_p, lbl_p, sz, ts))

    total_frames = len(all_frame_tasks)
    logger.info(f"Discovered {total_frames:,} eligible frames across all sequences.")

    # 4. Chunk Tasks
    chunks = [
        all_frame_tasks[i : i + chunk_size]
        for i in range(0, total_frames, chunk_size)
    ]
    logger.info(f"Partitioned into {len(chunks)} parallel chunks (chunk_size={chunk_size}).")

    if num_workers is None:
        cpu_avail = os.cpu_count() or 4
        num_workers = min(8, cpu_avail)

    logger.info(f"Spawning worker pool with {num_workers} parallel worker processes...")

    all_records: List[CropManifestRecord] = []
    global_stats = {
        "mine": 0,
        "known": 0,
        "bg": 0,
        "failed_crops": 0,
        "unreadable_images": 0,
        "invalid_boxes": 0,
    }

    start_generation_time = time.time()
    last_log_time = start_generation_time
    completed_crops = 0
    target_crops_expected = 155103

    mine_str = str(MINE_DIR)
    known_str = str(KNOWN_DIR)
    bg_str = str(BACKGROUND_DIR)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {
            executor.submit(
                process_frame_chunk,
                chunk,
                mine_str,
                known_str,
                bg_str,
                target_meta_map,
                JPEG_QUALITY,
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                rec_dicts, chunk_stat = future.result()
                for k, v in chunk_stat.items():
                    global_stats[k] += v

                for r_d in rec_dicts:
                    all_records.append(CropManifestRecord(**r_d))

                completed_crops = global_stats["mine"] + global_stats["known"] + global_stats["bg"]
                now = time.time()

                # Progress reporting every 3 seconds or when milestones hit
                if now - last_log_time >= 3.0 or completed_crops >= target_crops_expected:
                    elapsed = now - start_generation_time
                    rate = completed_crops / max(0.1, elapsed)
                    remaining = max(0, target_crops_expected - completed_crops)
                    eta_sec = remaining / max(1.0, rate)
                    pct = (completed_crops / target_crops_expected) * 100.0
                    print(
                        f"[PROGRESS] {completed_crops:,} / {target_crops_expected:,} crops ({pct:.1f}%) "
                        f"| Mine: {global_stats['mine']:,} | Known: {global_stats['known']:,} | BG: {global_stats['bg']:,} "
                        f"| Rate: {rate:.1f} crops/s | ETA: {eta_sec/60:.1f} min"
                    )
                    last_log_time = now

            except Exception as e:
                logger.error(f"Error in chunk {chunk_idx}: {e}")

    gen_duration = time.time() - start_generation_time
    logger.info(f"\nCrop generation completed in {gen_duration:.2f} seconds ({len(all_records):,} crops generated).")

    # 5. Deterministic Record Sorting
    logger.info("Sorting manifest records deterministically...")
    all_records.sort(
        key=lambda r: (
            r.category,
            r.source_track,
            r.source_sequence,
            r.flight_timestamp_epoch,
            r.source_image_name,
            r.annotation_line_idx,
        )
    )

    # 6. Split Assignment
    logger.info("Assigning Protocol 1 and Protocol 2 splits...")
    assign_splits_to_records(all_records, guard_band_seconds=10.0)

    # 7. Export Manifests
    logger.info(f"Writing master manifest to {MASTER_MANIFEST_CSV}...")
    export_manifest_csv(all_records, MASTER_MANIFEST_CSV)

    logger.info(f"Writing Protocol 1 split to {PROTOCOL1_SPLIT_CSV}...")
    export_split_manifest_csv(all_records, "protocol1_split", PROTOCOL1_SPLIT_CSV)

    logger.info(f"Writing Protocol 2 split to {PROTOCOL2_SPLIT_CSV}...")
    export_split_manifest_csv(all_records, "protocol2_split", PROTOCOL2_SPLIT_CSV)

    # 8. Run Post-Reconstruction Verification
    logger.info("Executing post-generation verification suite...")
    v_results = verify_reconstructed_dataset(sha_sample_size=500)

    total_duration = time.time() - total_start_time

    # Calculate Disk Usage
    total_bytes = 0
    for p in [MINE_DIR, KNOWN_DIR, BACKGROUND_DIR]:
        with os.scandir(p) as it:
            for entry in it:
                if entry.is_file():
                    total_bytes += entry.stat(follow_symlinks=False).st_size

    disk_mb = total_bytes / (1024.0 * 1024.0)

    # 9. Format Final Report
    print("\n" + "=" * 70)
    print("MINEINSIGHT classification_v2 GENERATION & VERIFICATION REPORT")
    print("=" * 70)
    print(f"Status                       : {v_results['verification_status']}")
    print(f"Mine Crops Generated         : {global_stats['mine']:,} (Expected: 74,120)")
    print(f"Known Crops Generated        : {global_stats['known']:,} (Expected: 59,973)")
    print(f"Background Crops Generated   : {global_stats['bg']:,} (Expected: 21,010)")
    print(f"Total Canonical Crops        : {len(all_records):,} (Expected: 155,103)")
    print(f"Failed Crops                 : {global_stats['failed_crops']}")
    print(f"Unreadable Images            : {global_stats['unreadable_images']}")
    print(f"Invalid Bounding Boxes       : {global_stats['invalid_boxes']}")
    print(f"Disk Usage                   : {disk_mb:.2f} MB (~{disk_mb/1024:.2f} GB)")
    print(f"Elapsed Time                 : {total_duration:.2f} s ({total_duration/60:.2f} min)")
    print("-" * 70)
    print("VERIFICATION CHECKS:")
    print(f"- Disk Count Match           : {v_results['disk_counts']['total_match']} (Total: {v_results['disk_counts']['Total']:,})")
    print(f"- Master Manifest Row Count  : {v_results['manifest_metrics']['row_count_match']} (Rows: {v_results['manifest_metrics']['total_rows']:,})")
    print(f"- File vs Manifest Exact     : {v_results['file_vs_manifest']['is_exact_match']} (Missing: {v_results['file_vs_manifest']['missing_on_disk_count']}, Untracked: {v_results['file_vs_manifest']['untracked_on_disk_count']})")
    print(f"- Duplicate Crop IDs         : {v_results['manifest_metrics']['duplicate_crop_ids']}")
    print(f"- SHA-256 Sample Consistency : {v_results['sha256_audit']['is_consistent']} ({v_results['sha256_audit']['sampled_count']} crops sampled, {v_results['sha256_audit']['sha_mismatches']} mismatches)")
    print(f"- Protocol 1 Strict Disjoint : {v_results['protocol1_audit']['is_strictly_disjoint']} (Train/Val/Test overlap: {v_results['protocol1_audit']['train_test_overlap']})")
    print(f"- Raw Data & Targets Untouch : {v_results['raw_legacy_integrity']['targets_list_yaml_untouched']} (YAML: {v_results['raw_legacy_integrity']['targets_list_yaml_bytes']} bytes)")
    print(f"- Legacy Dataset Untouched   : {v_results['raw_legacy_integrity']['legacy_untouched']} (Legacy crops: {v_results['raw_legacy_integrity']['legacy_classification_files']:,})")
    print("=" * 70 + "\n")

    if v_results["verification_status"] != "PASS":
        logger.error(f"Verification FAILED with errors: {v_results['errors']}")
        sys.exit(1)
    else:
        logger.info("All verification checks PASSED. classification_v2 canonical dataset is fully verified.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MineInsight classification_v2 Reconstruction Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Execute dry-run scan only (no crops written)")
    parser.add_argument("--verify-only", action="store_true", help="Run verification checks on existing dataset")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=300, help="Frames per worker chunk (default: 300)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.dry_run:
        execute_dry_run()
    elif args.verify_only:
        v_res = verify_reconstructed_dataset()
        print(v_res)
    else:
        run_reconstruction(num_workers=args.workers, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()

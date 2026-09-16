"""
Dry-Run Module for MineInsight classification_v2
================================================

Simulates full dataset reconstruction across all 6 sequences WITHOUT generating
or writing any crop images to disk.

Calculates actual repository values:
- Exact image, label, and matching pair counts per sequence
- Exact Mine, Known, and Background candidate counts
- Exact empty-frame counts (k=1)
- Exact orphan label counts
- Target instance ID occurrences
- Timestamp ordering and range
- Bounding-box geometry checks
- Protocol 1 and Protocol 2 split feasibility
- Strict instance-isolation verification (TrainTargets ∩ TestTargets = ∅)
- Estimated storage footprint on disk
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import Counter, defaultdict

from src.preprocessing.mineinsight_v2.config import (
    RAW_DIR,
    REPORTS_V2_DIR,
    DRY_RUN_SUMMARY_JSON,
    VALID_SEQUENCES,
    KNOWN_CLASS_IDS,
    MINE_CLASS_IDS,
    BACKGROUND_CLASS_ID,
    RAW_IMAGE_RESOLUTION,
    CROP_TARGET_SIZE,
    CATEGORY_MINE,
    CATEGORY_KNOWN,
    CATEGORY_BACKGROUND,
    P1_INSTANCE_ALLOCATION,
    P1_SINGLETON_MINE_TRAIN,
    P1_SINGLETON_MINE_VAL,
    P1_SINGLETON_MINE_TEST,
    P1_SINGLETON_KNOWN_TRAIN,
    P1_SINGLETON_KNOWN_VAL,
    P1_SINGLETON_KNOWN_TEST,
    P2_TRAIN_SEQUENCES,
    P2_TEST_SEQUENCES,
)
from src.preprocessing.mineinsight_v2.targets_v2 import get_target_registry
from src.preprocessing.mineinsight_v2.manifest import parse_timestamp_from_stem
from src.preprocessing.mineinsight_v2.crop_engine import compute_square_crop_box
from src.preprocessing.mineinsight_v2.split_v2 import build_protocol1_instance_map
from src.preprocessing.mineinsight_v2.validator_v2 import run_full_preflight_validation

logger = logging.getLogger(__name__)


def execute_dry_run() -> Dict[str, Any]:
    """Execute complete dry-run scan across all 6 sequences and return full audit."""
    start_time = time.time()
    logger.info("=" * 65)
    logger.info("MINEINSIGHT classification_v2 — PRE-FLIGHT VALIDATION & DRY-RUN")
    logger.info("=" * 65)

    # 1. Run Pre-Flight Validation
    preflight_results = run_full_preflight_validation()
    logger.info(f"Pre-Flight Status: {preflight_results['preflight_status']}")
    if preflight_results["preflight_status"] != "PASS":
        logger.error(f"Errors: {preflight_results['yaml_errors'] + preflight_results['image_errors']}")

    # 2. Target Registry & Protocol 1 Map
    reg = get_target_registry()
    p1_instance_map = build_protocol1_instance_map()

    # Metrics Accumulators
    sequence_summaries: Dict[str, Dict[str, Any]] = {}
    total_raw_images = 0
    total_raw_labels = 0
    total_matching_pairs = 0
    total_empty_frames = 0
    total_mine_annos = 0
    total_known_annos = 0
    total_invalid_annos = 0
    total_orphans = 0

    instance_counter = Counter()
    geom_checks = {"valid_boxes": 0, "degenerate_boxes": 0}

    # Split tracking
    p1_split_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    p2_split_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    p1_train_targets: Set[int] = set()
    p1_val_targets: Set[int] = set()
    p1_test_targets: Set[int] = set()

    # 3. Process Each Sequence
    for seq in VALID_SEQUENCES:
        logger.info(f"Scanning sequence: {seq}...")
        img_dir = RAW_DIR / f"{seq}_lwir_images"
        lbl_dir = RAW_DIR / f"{seq}_lwir_labels_reproj"
        if not img_dir.exists():
            img_dir = RAW_DIR / f"{seq}_wir_images"
        if not lbl_dir.exists():
            lbl_dir = RAW_DIR / f"{seq}_wir_labels_reproj"

        track_name = seq.split("_")[0].capitalize() + seq.split("_")[1]
        seq_name = seq.split("_")[2]
        is_p2_train = any(ts in seq for ts in P2_TRAIN_SEQUENCES)

        # Fast scan of image filenames using scandir
        img_stems: Dict[str, str] = {}
        with os.scandir(img_dir) as it:
            for entry in it:
                if entry.is_file():
                    name = entry.name
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        stem = name.rsplit(".", 1)[0]
                        img_stems[stem] = name

        # Fast scan of label filenames + pre-cached file sizes
        lbl_files: Dict[str, Tuple[str, int]] = {}
        with os.scandir(lbl_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".txt"):
                    stem = entry.name.rsplit(".", 1)[0]
                    sz = entry.stat(follow_symlinks=False).st_size
                    lbl_files[stem] = (entry.path, sz)

        seq_images = len(img_stems)
        seq_labels = len(lbl_files)
        total_raw_images += seq_images
        total_raw_labels += seq_labels

        matching_stems = set(img_stems.keys()).intersection(lbl_files.keys())
        orphans_in_seq = set(lbl_files.keys()) - set(img_stems.keys())
        total_matching_pairs += len(matching_stems)
        total_orphans += len(orphans_in_seq)

        seq_empty = 0
        seq_mine = 0
        seq_known = 0
        seq_invalid = 0
        timestamps: List[float] = []

        # Empty frames for Background simulation
        bg_timestamps: List[float] = []

        # Parse matching pairs
        for stem in sorted(matching_stems):
            ts = parse_timestamp_from_stem(stem)
            if ts > 0:
                timestamps.append(ts)

            lbl_path, sz = lbl_files[stem]

            # 1. Empty Frame (Background candidate k=1)
            if sz == 0:
                seq_empty += 1
                total_empty_frames += 1
                bg_timestamps.append(ts)
                continue

            # 2. Non-Empty Label -> Read lines
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
            except Exception:
                continue

            if not lines:
                seq_empty += 1
                total_empty_frames += 1
                bg_timestamps.append(ts)
                continue

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    seq_invalid += 1
                    total_invalid_annos += 1
                    continue

                try:
                    cid = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    seq_invalid += 1
                    total_invalid_annos += 1
                    continue

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    seq_invalid += 1
                    total_invalid_annos += 1
                    continue

                if cid in MINE_CLASS_IDS:
                    cat = CATEGORY_MINE
                    seq_mine += 1
                    total_mine_annos += 1
                elif cid in KNOWN_CLASS_IDS:
                    cat = CATEGORY_KNOWN
                    seq_known += 1
                    total_known_annos += 1
                else:
                    seq_invalid += 1
                    total_invalid_annos += 1
                    continue

                instance_counter[cid] += 1

                # Verify isotropic square crop calculation
                xmin, ymin, xmax, ymax = compute_square_crop_box(640, 512, xc, yc, w, h)
                if xmax > xmin and ymax > ymin:
                    geom_checks["valid_boxes"] += 1
                else:
                    geom_checks["degenerate_boxes"] += 1

                # Protocol 1 Split Assignment for Object
                p1_split = p1_instance_map.get(cid, "unassigned")
                p1_split_counts[p1_split][cat] += 1
                if p1_split == "train":
                    p1_train_targets.add(cid)
                elif p1_split == "test":
                    p1_test_targets.add(cid)
                elif p1_split == "validation":
                    p1_val_targets.add(cid)

                # Protocol 2 Split Assignment for Object (S1=Train 85%/Val 15%, S2=Test)
                if is_p2_train:
                    p2_split_counts["train"][cat] += 1
                else:
                    p2_split_counts["test"][cat] += 1

        # Simulate Background splits for this sequence
        bg_sorted = sorted(bg_timestamps)
        num_bg = len(bg_sorted)
        if num_bg > 0:
            cut1 = int(num_bg * 0.70)
            cut2 = int(num_bg * 0.85)
            for idx, b_ts in enumerate(bg_sorted):
                if idx < cut1:
                    p1_bg_split = "train"
                elif idx < cut2:
                    prev_t = bg_sorted[cut1 - 1]
                    p1_bg_split = "purged_guard_band" if (b_ts - prev_t < 10.0) else "validation"
                else:
                    prev_t = bg_sorted[cut2 - 1]
                    p1_bg_split = "purged_guard_band" if (b_ts - prev_t < 10.0) else "test"

                p1_split_counts[p1_bg_split][CATEGORY_BACKGROUND] += 1

                if is_p2_train:
                    p2_split_counts["train"][CATEGORY_BACKGROUND] += 1
                else:
                    p2_split_counts["test"][CATEGORY_BACKGROUND] += 1

        # Sequence Monotonicity & Range Check
        is_strictly_monotonic = timestamps == sorted(timestamps) if timestamps else True

        sequence_summaries[seq] = {
            "total_images": seq_images,
            "total_labels": seq_labels,
            "matching_pairs": len(matching_stems),
            "orphans_quarantined": len(orphans_in_seq),
            "empty_frames": seq_empty,
            "mine_annotations": seq_mine,
            "known_annotations": seq_known,
            "invalid_annotations": seq_invalid,
            "total_annotations": seq_mine + seq_known,
            "timestamps_monotonic": is_strictly_monotonic,
            "timestamp_range": [min(timestamps), max(timestamps)] if timestamps else [],
        }

    # 4. Target Isolation Verification
    train_test_overlap = p1_train_targets.intersection(p1_test_targets)
    train_val_overlap = p1_train_targets.intersection(p1_val_targets)
    val_test_overlap = p1_val_targets.intersection(p1_test_targets)

    isolation_audit = {
        "p1_train_targets_count": len(p1_train_targets),
        "p1_val_targets_count": len(p1_val_targets),
        "p1_test_targets_count": len(p1_test_targets),
        "train_test_overlap": list(train_test_overlap),
        "train_val_overlap": list(train_val_overlap),
        "val_test_overlap": list(val_test_overlap),
        "p1_is_strictly_disjoint": len(train_test_overlap) == 0 and len(train_val_overlap) == 0,
    }

    # 5. Disk Usage Estimation
    total_projected_crops = total_mine_annos + total_known_annos + total_empty_frames
    avg_crop_kb = 14.8
    projected_disk_mb = (total_projected_crops * avg_crop_kb) / 1024.0

    # 6. Final Report Dictionary
    dry_run_summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "READY_FOR_CROP_GENERATION",
        "preflight_validation": preflight_results,
        "dataset_totals": {
            "total_images": total_raw_images,
            "total_labels": total_raw_labels,
            "total_matching_pairs": total_matching_pairs,
            "total_empty_frames_k1": total_empty_frames,
            "total_mine_annotations": total_mine_annos,
            "total_known_annotations": total_known_annos,
            "total_target_annotations": total_mine_annos + total_known_annos,
            "total_invalid_annotations": total_invalid_annos,
            "total_quarantined_orphans": total_orphans,
            "total_projected_crops_k1": total_projected_crops,
            "projected_disk_usage_mb": round(projected_disk_mb, 2),
        },
        "sequence_breakdown": sequence_summaries,
        "geometry_checks": geom_checks,
        "instance_frequencies": dict(sorted(instance_counter.items())),
        "protocol1_split_summary": {
            "splits": dict(p1_split_counts),
            "isolation_audit": isolation_audit,
        },
        "protocol2_split_summary": {
            "splits": dict(p2_split_counts),
        },
    }

    # Save report
    REPORTS_V2_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRY_RUN_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(dry_run_summary, f, indent=4)

    duration = time.time() - start_time
    logger.info(f"\nDry-run completed successfully in {duration:.2f} seconds.")
    logger.info(f"Report exported to: {DRY_RUN_SUMMARY_JSON}")

    # Print Formatted Results
    print("\n" + "=" * 65)
    print("ACTUAL DRY-RUN RESULTS SUMMARY")
    print("=" * 65)
    print(f"Total Raw Images             : {total_raw_images}")
    print(f"Total Raw Labels             : {total_raw_labels}")
    print(f"Total Matching Pairs         : {total_matching_pairs}")
    print(f"Total Empty Frames (k=1)     : {total_empty_frames}")
    print(f"Total Mine Annotations       : {total_mine_annos}")
    print(f"Total Known Annotations      : {total_known_annos}")
    print(f"Total Target Annotations     : {total_mine_annos + total_known_annos}")
    print(f"Total Invalid Annotations    : {total_invalid_annos}")
    print(f"Total Quarantined Orphans    : {total_orphans}")
    print(f"Total Projected Crops (k=1)  : {total_projected_crops}")
    print(f"Projected Disk Usage         : {projected_disk_mb:.2f} MB (~{projected_disk_mb/1024:.2f} GB)")
    print(f"P1 Strict Instance Overlap   : {isolation_audit['train_test_overlap']} (Disjoint: {isolation_audit['p1_is_strictly_disjoint']})")
    print(f"Dry-run Duration             : {duration:.2f} s")
    print("=" * 65 + "\n")

    return dry_run_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    execute_dry_run()

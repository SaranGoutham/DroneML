"""
Split Module for MineInsight v2
===============================

Implements two mutually exclusive, rigorous evaluation protocols:
1. Protocol 1: Physical Instance-Aware + Temporal-Safe Split (Zero object instance overlap).
2. Protocol 2: Flight-Holdout / Diurnal Generalization Split (Zero temporal flight correlation).
"""

import logging
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict

from src.preprocessing.mineinsight_v2.config import (
    CATEGORY_BACKGROUND,
    CATEGORY_MINE,
    CATEGORY_KNOWN,
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
from src.preprocessing.mineinsight_v2.manifest import CropManifestRecord

logger = logging.getLogger(__name__)


def build_protocol1_instance_map() -> Dict[int, str]:
    """
    Build a direct lookup map: target_instance_id -> 'train' | 'validation' | 'test'.
    Guarantees no physical instance is shared between splits.
    """
    instance_map: Dict[int, str] = {}

    # 1. Multi-instance targets
    for group_name, alloc in P1_INSTANCE_ALLOCATION.items():
        for split_name in ["train", "validation", "test"]:
            for tid in alloc.get(split_name, []):
                instance_map[tid] = split_name

    # 2. Singleton Mines
    for tid in P1_SINGLETON_MINE_TRAIN:
        instance_map[tid] = "train"
    for tid in P1_SINGLETON_MINE_VAL:
        instance_map[tid] = "validation"
    for tid in P1_SINGLETON_MINE_TEST:
        instance_map[tid] = "test"

    # 3. Singleton Known
    for tid in P1_SINGLETON_KNOWN_TRAIN:
        instance_map[tid] = "train"
    for tid in P1_SINGLETON_KNOWN_VAL:
        instance_map[tid] = "validation"
    for tid in P1_SINGLETON_KNOWN_TEST:
        instance_map[tid] = "test"

    return instance_map


def assign_splits_to_records(
    records: List[CropManifestRecord],
    guard_band_seconds: float = 10.0,
) -> None:
    """
    Assign both protocol1_split and protocol2_split in-place to records.
    """
    p1_instance_map = build_protocol1_instance_map()

    # Organize background records by sequence to assign continuous temporal chunks
    bg_by_seq: Dict[str, List[CropManifestRecord]] = defaultdict(list)
    for r in records:
        if r.category == CATEGORY_BACKGROUND:
            seq_key = f"{r.source_track.lower()}_{r.source_sequence.lower()}".replace("track", "track_").replace("track__", "track_")
            bg_by_seq[seq_key].append(r)

    # Assign Protocol 1 to Target Objects
    for r in records:
        if r.category != CATEGORY_BACKGROUND:
            tid = r.target_instance_id
            r.protocol1_split = p1_instance_map.get(tid, "unassigned")

    # Assign Protocol 1 to Background via continuous temporal blocks with guard bands
    for seq_name, seq_records in bg_by_seq.items():
        sorted_records = sorted(seq_records, key=lambda x: x.flight_timestamp_epoch)
        n = len(sorted_records)
        if n == 0:
            continue

        train_cut = int(n * 0.70)
        val_cut = int(n * 0.85)

        for i, r in enumerate(sorted_records):
            if i < train_cut:
                r.protocol1_split = "train"
            elif i < val_cut:
                # Check guard band with train end
                prev_t = sorted_records[train_cut - 1].flight_timestamp_epoch
                if r.flight_timestamp_epoch - prev_t < guard_band_seconds:
                    r.protocol1_split = "purged_guard_band"
                else:
                    r.protocol1_split = "validation"
            else:
                # Check guard band with val end
                prev_t = sorted_records[val_cut - 1].flight_timestamp_epoch
                if r.flight_timestamp_epoch - prev_t < guard_band_seconds:
                    r.protocol1_split = "purged_guard_band"
                else:
                    r.protocol1_split = "test"

    # Assign Protocol 2: Flight-Holdout (S1 = Train/Val, S2 = Test)
    for r in records:
        full_seq = f"{r.source_track.lower()}_{r.source_sequence.lower()}".replace("track", "track_").replace("track__", "track_")
        if full_seq in P2_TEST_SEQUENCES:
            r.protocol2_split = "test"
        elif full_seq in P2_TRAIN_SEQUENCES:
            # 85% train, 15% validation within S1
            r.protocol2_split = "train"
        else:
            r.protocol2_split = "unassigned"

    # S1 validation split: partition 15% of S1 as validation
    s1_records = [r for r in records if r.protocol2_split == "train"]
    by_target: Dict[int, List[CropManifestRecord]] = defaultdict(list)
    for r in s1_records:
        by_target[r.target_instance_id].append(r)

    for tid, t_records in by_target.items():
        sorted_t = sorted(t_records, key=lambda x: x.flight_timestamp_epoch)
        val_k = max(1, int(len(sorted_t) * 0.15))
        for r in sorted_t[-val_k:]:
            r.protocol2_split = "validation"


def verify_split_isolation(
    records: List[CropManifestRecord],
) -> Dict[str, Any]:
    """
    Verify that Protocol 1 has zero physical target instance overlap between train and test.
    """
    p1_train_targets: Set[int] = set()
    p1_test_targets: Set[int] = set()
    p1_val_targets: Set[int] = set()

    for r in records:
        if r.category == CATEGORY_BACKGROUND:
            continue
        tid = r.target_instance_id
        if r.protocol1_split == "train":
            p1_train_targets.add(tid)
        elif r.protocol1_split == "test":
            p1_test_targets.add(tid)
        elif r.protocol1_split == "validation":
            p1_val_targets.add(tid)

    train_test_overlap = p1_train_targets.intersection(p1_test_targets)
    train_val_overlap = p1_train_targets.intersection(p1_val_targets)
    val_test_overlap = p1_val_targets.intersection(p1_test_targets)

    return {
        "p1_train_targets_count": len(p1_train_targets),
        "p1_val_targets_count": len(p1_val_targets),
        "p1_test_targets_count": len(p1_test_targets),
        "train_test_overlap": list(train_test_overlap),
        "train_val_overlap": list(train_val_overlap),
        "val_test_overlap": list(val_test_overlap),
        "p1_is_strictly_disjoint": len(train_test_overlap) == 0 and len(train_val_overlap) == 0,
    }

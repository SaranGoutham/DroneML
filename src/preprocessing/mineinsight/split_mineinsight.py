"""Split MineInsight crops into train/validation/test folders.

The splitter groups crops by their source LWIR image, so multiple object crops
from one original image never cross splits.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from src.models.config import RANDOM_STATE
from src.preprocessing.mineinsight.mineinsight_config import (
    CLASSIFICATION_DIR,
    MINE_CLASS,
    NO_MINE_CLASS,
    PROCESSED_DIR,
    REPORTS_DIR,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class SplitStats:
    train: dict[str, int]
    validation: dict[str, int]
    test: dict[str, int]
    source_groups: int
    total_files: int


def source_group_id(path: Path) -> str:
    """Return the source-image group id encoded by convert_mineinsight.py."""
    marker = "_obj"
    stem = path.stem
    if marker not in stem:
        return stem
    return stem.rsplit(marker, 1)[0]


def iter_classification_images(classification_dir: Path) -> list[tuple[Path, str, str]]:
    """Collect ``(path, class_name, group_id)`` for all converted crops."""
    rows: list[tuple[Path, str, str]] = []
    for class_name in (MINE_CLASS, NO_MINE_CLASS):
        class_dir = classification_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Classification class directory not found: {class_dir}")
        for path in class_dir.iterdir():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                rows.append((path, class_name, source_group_id(path)))
    if not rows:
        raise ValueError(f"No MineInsight classification images found in {classification_dir}")
    return rows


def assign_groups(
    group_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, str]:
    """Assign source groups to train/validation/test deterministically."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    groups = sorted(set(group_ids))
    rng = random.Random(random_state)
    rng.shuffle(groups)

    n_groups = len(groups)
    train_cut = int(n_groups * train_ratio)
    val_cut = train_cut + int(n_groups * val_ratio)

    assignments: dict[str, str] = {}
    for group in groups[:train_cut]:
        assignments[group] = "train"
    for group in groups[train_cut:val_cut]:
        assignments[group] = "validation"
    for group in groups[val_cut:]:
        assignments[group] = "test"
    return assignments


def reset_output_dirs(output_dir: Path) -> None:
    """Create clean split/class folders under ``output_dir``."""
    for split in ("train", "validation", "test"):
        for class_name in (MINE_CLASS, NO_MINE_CLASS):
            class_dir = output_dir / split / class_name
            if class_dir.exists():
                shutil.rmtree(class_dir)
            class_dir.mkdir(parents=True, exist_ok=True)


def split_dataset(
    classification_dir: Path | str = CLASSIFICATION_DIR,
    output_dir: Path | str = PROCESSED_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> SplitStats:
    """Split converted MineInsight crops into train/validation/test folders."""
    classification_dir = Path(classification_dir)
    output_dir = Path(output_dir)

    rows = iter_classification_images(classification_dir)
    assignments = assign_groups(
        [group_id for _, _, group_id in rows],
        train_ratio,
        val_ratio,
        test_ratio,
        random_state,
    )
    reset_output_dirs(output_dir)

    counts: dict[str, dict[str, int]] = {
        "train": {MINE_CLASS: 0, NO_MINE_CLASS: 0},
        "validation": {MINE_CLASS: 0, NO_MINE_CLASS: 0},
        "test": {MINE_CLASS: 0, NO_MINE_CLASS: 0},
    }

    for path, class_name, group_id in rows:
        split = assignments[group_id]
        dest = output_dir / split / class_name / path.name
        shutil.copy2(path, dest)
        counts[split][class_name] += 1

    stats = SplitStats(
        train=counts["train"],
        validation=counts["validation"],
        test=counts["test"],
        source_groups=len(set(assignments)),
        total_files=len(rows),
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "split_report.json"
    import json

    report_path.write_text(json.dumps(asdict(stats), indent=4), encoding="utf-8")
    logger.info("MineInsight split report written to %s", report_path)
    return stats


def prevent_data_leakage(dataset_dir: Path | str = PROCESSED_DIR) -> list[str]:
    """Return source group ids that appear in more than one split."""
    dataset_dir = Path(dataset_dir)
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "validation", "test"):
        for class_name in (MINE_CLASS, NO_MINE_CLASS):
            for path in (dataset_dir / split / class_name).glob("*"):
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    split_by_group[source_group_id(path)].add(split)
    leaks = sorted(group for group, splits in split_by_group.items() if len(splits) > 1)
    if leaks:
        logger.error("Detected %d leaking source groups", len(leaks))
    else:
        logger.info("No source-image leakage detected.")
    return leaks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split MineInsight classification crops.")
    parser.add_argument("--classification-dir", type=Path, default=CLASSIFICATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    stats = split_dataset(
        classification_dir=args.classification_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    leaks = prevent_data_leakage(args.output_dir)
    print("MineInsight split complete")
    print(asdict(stats))
    print(f"Leakage groups: {len(leaks)}")


if __name__ == "__main__":
    main()

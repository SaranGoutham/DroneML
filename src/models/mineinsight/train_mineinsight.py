# -*- coding: utf-8 -*-
"""Train DroneML classifiers on the MineInsight LWIR dataset.

Expected dataset layout:

  data/mineinsight/processed/
    train/No_Mine, train/Mine
    validation/No_Mine, validation/Mine
    test/No_Mine, test/Mine

Usage:
  python -m src.models.mineinsight.train_mineinsight --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import BATCH_SIZE, EPOCHS, FINE_TUNE_EPOCHS, RANDOM_STATE
from src.models.training_utils import (
    RunPaths,
    compute_class_weights,
    create_image_datasets,
    select_model_specs,
    train_two_phase_model,
)
from src.preprocessing.mineinsight.mineinsight_config import (
    PROCESSED_DIR as MINEINSIGHT_PROCESSED_DIR,
    RESULTS_DIR as MINEINSIGHT_RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Force Mine to be the positive class for binary metrics.
CLASS_NAMES = ["No_Mine", "Mine"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DroneML models on the MineInsight classification splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=MINEINSIGHT_PROCESSED_DIR,
        help="MineInsight split directory containing train/validation/test.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=MINEINSIGHT_RESULTS_DIR,
        help="Directory for MineInsight checkpoints, models, histories, and reports.",
    )
    parser.add_argument(
        "--model",
        choices=["all", "mobilenetv3", "resnet50", "efficientnetb0"],
        default="all",
        help="Model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="Run only frozen-backbone training.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache normalized tf.data datasets in memory. Use only if RAM is sufficient.",
    )
    return parser.parse_args()


def summarize_results(results: dict[str, Any], comparison_path: Path) -> None:
    print("\n" + "=" * 72)
    print("  MINEINSIGHT TRAINING SUMMARY")
    print("=" * 72)
    print(f"  {'Model':<22} | {'Accuracy':>10} | {'AUC':>8} | {'Precision':>10} | {'Recall':>8}")
    print("-" * 72)
    for model_key, result in results.items():
        test = result["test_results"]
        print(
            f"  {result['display_name']:<22} | "
            f"{test.get('accuracy', 0):>10.4f} | "
            f"{test.get('auc', 0):>8.4f} | "
            f"{test.get('precision', 0):>10.4f} | "
            f"{test.get('recall', 0):>8.4f}"
        )
    print("=" * 72)
    print(f"  Comparison JSON: {comparison_path}")
    print("=" * 72 + "\n")


def train_mineinsight(args: argparse.Namespace) -> dict[str, Any]:
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    run_paths = RunPaths.from_root(args.results_dir)
    specs = select_model_specs(args.model)

    logger.info("Loading MineInsight datasets from %s", args.data_dir)
    train_ds, val_ds, test_ds, class_names = create_image_datasets(
        args.data_dir,
        class_names=CLASS_NAMES,
        batch_size=args.batch_size,
        random_seed=RANDOM_STATE,
        augment_train=True,
        cache=args.cache,
    )
    logger.info("Class order: %s", class_names)
    logger.info("Positive class for metrics: Mine")

    class_weights = compute_class_weights(args.data_dir / "train", class_names)

    results: dict[str, Any] = {}
    for spec in specs:
        logger.info("=" * 72)
        logger.info("Training %s on MineInsight", spec.display_name)
        logger.info("=" * 72)
        results[spec.key] = train_two_phase_model(
            spec=spec,
            run_paths=run_paths,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            class_weights=class_weights,
            epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune=not args.no_finetune,
            random_seed=RANDOM_STATE,
        )

    comparison = {
        key: {
            "display_name": result["display_name"],
            "model_key": result["model_key"],
            "final_model": result["final_model"],
            "history_json": result["history_json"],
            "test_results": result["test_results"],
            "best_val_accuracy": max(result["history"].get("val_accuracy", [0])),
            "best_val_auc": max(result["history"].get("val_auc", [0])),
            "class_order": class_names,
            "positive_class": "Mine",
        }
        for key, result in results.items()
    }
    comparison_path = run_paths.comparison / "mineinsight_comparison_results.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)

    summarize_results(results, comparison_path)
    return results


def main() -> None:
    train_mineinsight(parse_args())


if __name__ == "__main__":
    main()

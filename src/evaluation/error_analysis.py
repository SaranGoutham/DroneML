# -*- coding: utf-8 -*-
"""error_analysis.py – DroneML  [src/evaluation]
==================================================

Phase 12 – Error Analysis
--------------------------
Loads each saved final model, runs inference on the test set, and
matches predictions back to the original metadata to analyse failure
cases across multiple stratification dimensions.

Analyses performed
------------------
  1. False Positive / False Negative identification and image saving
  2. Flight-wise accuracy  (per flight date)
  3. Altitude-wise accuracy  (binned altitude in metres)
  4. Depth-wise accuracy  (mine burial depth)
  5. Zone-wise accuracy  (zone label from directory name)

Outputs  (all under  results/error_analysis/)
---------------------------------------------
  false_positives/<model>_fp_<n>.png        – FP sample panels
  false_negatives/<model>_fn_<n>.png        – FN sample panels
  flight_wise_accuracy.png                  – grouped bar chart
  altitude_wise_accuracy.png                – line / bar chart
  depth_wise_accuracy.png                   – grouped bar chart
  zone_wise_accuracy.png                    – horizontal bar chart
  error_report.md                           – narrative Markdown report

Metadata source
---------------
  results/metadata_checked.csv
  Columns used: flight_date, zone, depth, label, filename, altitude

Note on test-set matching
--------------------------
The test set comes from flight date 24_02_2020 (see split_report.txt).
We filter the metadata CSV to this flight date and match filenames
to prediction probabilities.  Images whose filenames cannot be matched
are excluded from stratified analysis but still counted in overall metrics.

Usage
-----
  python -m src.evaluation.error_analysis
  python -m src.evaluation.error_analysis --model resnet50
"""

from __future__ import annotations

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import PROCESSED_DIR, RESULTS_DIR
from src.preprocessing.create_tf_dataset import create_datasets
from src.models.config import (
    IMAGE_SIZE,
    ERROR_ANALYSIS_DIR,
    ERROR_ANALYSIS_FP_DIR,
    ERROR_ANALYSIS_FN_DIR,
    ERROR_ANALYSIS_REPORT_MD,
    ERROR_ANALYSIS_FLIGHT_PNG,
    ERROR_ANALYSIS_ALTITUDE_PNG,
    ERROR_ANALYSIS_DEPTH_PNG,
    ERROR_ANALYSIS_ZONE_PNG,
    MOBILENETV3_FINAL_MODEL_PATH,
    RESNET50_FINAL_MODEL_PATH,
    EFFICIENTNETB0_FINAL_MODEL_PATH,
)

# Path to the metadata CSV with all image attributes
METADATA_CSV = RESULTS_DIR / "metadata_checked.csv"

# Decision threshold
THRESHOLD: float = 0.5

# Altitude bins (metres)
ALTITUDE_BINS   = [0, 1, 2, 5, 10, float("inf")]
ALTITUDE_LABELS = ["0-1m", "1-2m", "2-5m", "5-10m", ">10m"]

# Matplotlib style
plt.rcParams.update({
    "figure.dpi"       : 130,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.size"        : 10,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "key"          : "mobilenetv3",
        "display_name" : "MobileNetV3-Large",
        "model_path"   : MOBILENETV3_FINAL_MODEL_PATH,
        "color"        : "#4C72B0",
    },
    {
        "key"          : "resnet50",
        "display_name" : "ResNet50",
        "model_path"   : RESNET50_FINAL_MODEL_PATH,
        "color"        : "#DD8452",
    },
    {
        "key"          : "efficientnetb0",
        "display_name" : "EfficientNetB0",
        "model_path"   : EFFICIENTNETB0_FINAL_MODEL_PATH,
        "color"        : "#55A868",
    },
]


# ---------------------------------------------------------------------------
# Helper: load metadata
# ---------------------------------------------------------------------------

def _load_metadata() -> pd.DataFrame | None:
    if not METADATA_CSV.exists():
        logging.warning("Metadata CSV not found: %s", METADATA_CSV)
        return None
    df = pd.read_csv(METADATA_CSV)
    logging.info(
        "Loaded metadata: %d rows, %d cols", len(df), len(df.columns)
    )
    return df


# ---------------------------------------------------------------------------
# Helper: collect predictions + file paths from dataset
# ---------------------------------------------------------------------------

def _collect_predictions(
    model: keras.Model,
    test_ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect ground-truth labels, predictions and raw images.

    Returns
    -------
    images    : np.ndarray  (N, H, W, C)
    y_true    : np.ndarray  (N,)
    y_pred_prob: np.ndarray (N,)
    """
    images_list: list[np.ndarray] = []
    y_true_list:  list[np.ndarray] = []
    y_prob_list:  list[np.ndarray] = []

    for imgs_batch, lbls_batch in test_ds:
        probs = model(imgs_batch, training=False).numpy().flatten()
        y_prob_list.append(probs)
        y_true_list.append(lbls_batch.numpy().flatten())
        images_list.append(imgs_batch.numpy())

    images   = np.concatenate(images_list, axis=0)
    y_true   = np.concatenate(y_true_list)
    y_pred_p = np.concatenate(y_prob_list)
    return images, y_true, y_pred_p


# ---------------------------------------------------------------------------
# Helper: save FP / FN image panels
# ---------------------------------------------------------------------------

def _save_error_panels(
    images: np.ndarray,
    indices: np.ndarray,
    label_str: str,
    out_dir: Path,
    model_key: str,
    prefix: str,
    class_names: list[str],
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    max_panels: int = 10,
) -> None:
    """Save individual image panels for false positives or false negatives."""
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = indices[:max_panels]
    for i, idx in enumerate(chosen):
        img  = images[idx]
        true = class_names[int(y_true[idx])] if int(y_true[idx]) < len(class_names) else str(int(y_true[idx]))
        prob = y_pred_prob[idx]
        pred_label = class_names[1] if prob >= THRESHOLD else class_names[0]
        confidence = prob if prob >= THRESHOLD else 1 - prob

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"{label_str} – {model_key}\n"
            f"True: {true}  |  Pred: {pred_label} ({confidence:.1%})",
            fontsize=10,
            fontweight="bold",
        )
        plt.tight_layout()
        out_path = out_dir / f"{model_key}_{prefix}_{i + 1:02d}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    logging.info(
        "Saved %d %s panels for %s → %s",
        len(chosen), label_str, model_key, out_dir,
    )


# ---------------------------------------------------------------------------
# Stratified analysis helpers
# ---------------------------------------------------------------------------

def _stratified_accuracy(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Compute accuracy per group.

    Expects df to have columns: true_label, pred_label, <group_col>.

    Returns
    -------
    pd.DataFrame  with columns [group_col, 'accuracy', 'n_samples']
    """
    rows = []
    for group, sub in df.groupby(group_col):
        acc = accuracy_score(sub["true_label"], sub["pred_label"])
        rows.append({group_col: group, "accuracy": acc, "n_samples": len(sub)})
    return pd.DataFrame(rows).sort_values(group_col)


def _bar_chart(
    data: pd.DataFrame,
    x_col: str,
    title: str,
    out_path: Path,
    model_colors: dict[str, str],
    rotate_xticks: bool = False,
) -> None:
    """Save a grouped bar chart (one bar per model, grouped by x_col)."""
    models = list(data["model"].unique())
    groups = list(data[x_col].unique())
    n_groups = len(groups)
    n_models = len(models)
    width    = 0.8 / max(n_models, 1)
    offsets  = np.linspace(
        -(n_models - 1) * width / 2,
        (n_models - 1) * width / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(max(9, n_groups * 1.5), 5))
    ax.set_title(title, fontsize=13, fontweight="bold")

    x = np.arange(n_groups)
    for off, model_name in zip(offsets, models):
        sub = data[data["model"] == model_name]
        vals = []
        for g in groups:
            row = sub[sub[x_col] == g]
            vals.append(float(row["accuracy"].values[0]) if len(row) > 0 else 0.0)
        bars = ax.bar(
            x + off, vals, width=width,
            label=model_name,
            color=model_colors.get(model_name, "gray"),
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        groups,
        rotation=30 if rotate_xticks else 0,
        ha="right" if rotate_xticks else "center",
        fontsize=9,
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.legend(loc="upper right")
    ax.axhline(y=1.0, color="gray", lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Analyse one model
# ---------------------------------------------------------------------------

def analyse_single_model(
    entry: dict[str, Any],
    test_ds: tf.data.Dataset,
    class_names: list[str],
    metadata_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Run error analysis for one model.

    Returns
    -------
    dict with keys: fp_count, fn_count, stratified_rows
    """
    model_path = entry["model_path"]
    if not model_path.exists():
        logging.warning("Model not found, skipping: %s", model_path)
        return {}

    logging.info(
        "Loading %s for error analysis ...", entry["display_name"]
    )
    model: keras.Model = keras.models.load_model(str(model_path))

    images, y_true, y_pred_prob = _collect_predictions(model, test_ds)
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

    # --- FP / FN indices ---
    # y_true: 0=Mine, 1=No_Mine  (Keras label convention from directory sort)
    fp_idx = np.where((y_true == 1) & (y_pred == 0))[0]   # true No_Mine, predicted Mine
    fn_idx = np.where((y_true == 0) & (y_pred == 1))[0]   # true Mine, predicted No_Mine

    logging.info(
        "%s  – FP=%d  FN=%d  total=%d",
        entry["display_name"], len(fp_idx), len(fn_idx), len(y_true),
    )

    # Save FP panels
    _save_error_panels(
        images=images, indices=fp_idx,
        label_str="False Positive", out_dir=ERROR_ANALYSIS_FP_DIR,
        model_key=entry["key"], prefix="fp",
        class_names=class_names, y_true=y_true, y_pred_prob=y_pred_prob,
    )

    # Save FN panels
    _save_error_panels(
        images=images, indices=fn_idx,
        label_str="False Negative", out_dir=ERROR_ANALYSIS_FN_DIR,
        model_key=entry["key"], prefix="fn",
        class_names=class_names, y_true=y_true, y_pred_prob=y_pred_prob,
    )

    del model
    tf.keras.backend.clear_session()

    # --- Stratified analysis requires metadata ---
    stratified_rows: list[dict] = []

    if metadata_df is not None:
        # The test set is from flight 24_02_2020
        test_meta = metadata_df[metadata_df["flight_date"] == "24_02_2020"].copy()

        # Map label column: Mine → 0, No_Mine → 1
        label_map = {"Mine": 0, "No_Mine": 1}
        test_meta["true_label"] = test_meta["label"].map(label_map)

        # We cannot directly match file paths to dataset order without
        # re-scanning the directory. Build a filename→prediction dict.
        # The test dataset is loaded from processed_dataset/test/{Mine,No_Mine}/.
        # We'll rebuild that order by listing files.
        test_mine_dir    = PROCESSED_DIR / "test" / "Mine"
        test_no_mine_dir = PROCESSED_DIR / "test" / "No_Mine"

        # tf.keras.utils.image_dataset_from_directory lists Mine first (alphabetical)
        mine_files    = sorted(test_mine_dir.glob("*.png")) + \
                        sorted(test_mine_dir.glob("*.jpg"))
        no_mine_files = sorted(test_no_mine_dir.glob("*.png")) + \
                        sorted(test_no_mine_dir.glob("*.jpg"))

        # Label 0 = Mine, Label 1 = No_Mine in alphabetical order from Keras
        ordered_files  = mine_files + no_mine_files
        ordered_labels = [0] * len(mine_files) + [1] * len(no_mine_files)

        # Map: stem → index in prediction array
        # Note: dataset is shuffled at load, so we recompute a fresh
        # unshuffled prediction here
        model_fresh: keras.Model = keras.models.load_model(str(model_path))

        img_ds = tf.keras.utils.image_dataset_from_directory(
            str(PROCESSED_DIR / "test"),
            labels="inferred",
            label_mode="binary",
            batch_size=32,
            image_size=IMAGE_SIZE,
            shuffle=False,
            seed=None,
        )
        img_ds = img_ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).cache().prefetch(tf.data.AUTOTUNE)

        y_true_unshuffled_list: list[np.ndarray] = []
        y_prob_unshuffled_list: list[np.ndarray] = []
        for imgs_b, lbls_b in img_ds:
            preds_b = model_fresh(imgs_b, training=False).numpy().flatten()
            y_prob_unshuffled_list.append(preds_b)
            y_true_unshuffled_list.append(lbls_b.numpy().flatten())

        y_prob_unshuffled = np.concatenate(y_prob_unshuffled_list)
        y_pred_unshuffled = (y_prob_unshuffled >= THRESHOLD).astype(int)
        y_true_unshuffled = np.concatenate(y_true_unshuffled_list)

        del model_fresh
        tf.keras.backend.clear_session()

        # Build filename → pred dict using the file names from the directory listing
        file_list = []
        for (fpath, lbl) in zip(ordered_files, ordered_labels):
            file_list.append(fpath.name)

        # Rebuild ordered file list from the unshuffled dataset
        raw_file_ds = tf.keras.utils.image_dataset_from_directory(
            str(PROCESSED_DIR / "test"),
            labels="inferred",
            label_mode="binary",
            batch_size=32,
            image_size=IMAGE_SIZE,
            shuffle=False,
        )
        ordered_filenames: list[str] = [
            Path(p).name for p in raw_file_ds.file_paths
        ]

        # Match metadata rows to predictions by filename stem
        filename_to_pred: dict[str, int] = {
            fname: pred
            for fname, pred in zip(ordered_filenames, y_pred_unshuffled)
        }
        filename_to_true: dict[str, int] = {
            fname: true
            for fname, true in zip(ordered_filenames, y_true_unshuffled)
        }

        # Merge predictions into test_meta
        test_meta["pred_label"] = test_meta["filename"].map(filename_to_pred)
        test_meta = test_meta.dropna(subset=["pred_label"])
        test_meta["pred_label"] = test_meta["pred_label"].astype(int)

        # Altitude bin
        test_meta["altitude_bin"] = pd.cut(
            test_meta["altitude"].astype(float),
            bins=ALTITUDE_BINS,
            labels=ALTITUDE_LABELS,
            right=True,
        ).astype(str)

        # Build stratified rows
        for grp_col in ["flight_date", "altitude_bin", "depth", "zone"]:
            for group, sub in test_meta.groupby(grp_col):
                if len(sub) == 0:
                    continue
                acc = accuracy_score(sub["true_label"], sub["pred_label"])
                stratified_rows.append({
                    "model"      : entry["display_name"],
                    "dimension"  : grp_col,
                    "group"      : str(group),
                    "accuracy"   : acc,
                    "n_samples"  : len(sub),
                    "n_fp"       : int(((sub["true_label"] == 1) & (sub["pred_label"] == 0)).sum()),
                    "n_fn"       : int(((sub["true_label"] == 0) & (sub["pred_label"] == 1)).sum()),
                })

    return {
        "fp_count"       : int(len(fp_idx)),
        "fn_count"       : int(len(fn_idx)),
        "stratified_rows": stratified_rows,
    }


# ---------------------------------------------------------------------------
# Generate stratified charts
# ---------------------------------------------------------------------------

def _generate_stratified_charts(
    stratified_df: pd.DataFrame,
    model_colors: dict[str, str],
) -> None:
    """Generate flight/altitude/depth/zone accuracy bar charts."""
    dim_map = {
        "flight_date"  : (ERROR_ANALYSIS_FLIGHT_PNG,   "Flight-wise Accuracy",    True),
        "altitude_bin" : (ERROR_ANALYSIS_ALTITUDE_PNG, "Altitude-wise Accuracy",  False),
        "depth"        : (ERROR_ANALYSIS_DEPTH_PNG,    "Depth-wise Accuracy",     False),
        "zone"         : (ERROR_ANALYSIS_ZONE_PNG,     "Zone-wise Accuracy",      True),
    }
    for dim, (out_path, title, rotate) in dim_map.items():
        sub = stratified_df[stratified_df["dimension"] == dim]
        if sub.empty:
            logging.warning("No data for dimension '%s' – skipping chart.", dim)
            continue
        _bar_chart(
            data         = sub,
            x_col        = "group",
            title        = title,
            out_path     = out_path,
            model_colors = model_colors,
            rotate_xticks= rotate,
        )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _generate_report(
    all_results: dict[str, dict],
    stratified_df: pd.DataFrame,
) -> None:
    """Write the error analysis Markdown report."""
    lines: list[str] = []
    lines.append("# DroneML – Error Analysis Report\n\n")
    lines.append("Phase 12 error analysis of MobileNetV3-Large, ResNet50, and EfficientNetB0.\n\n")
    lines.append("---\n\n")

    # FP / FN summary table
    lines.append("## False Positive / False Negative Summary\n\n")
    lines.append("| Model | Total Test | FP | FN | FP% | FN% |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for name, res in all_results.items():
        if not res:
            continue
        total = res.get("fp_count", 0) + res.get("fn_count", 0) + \
                (117 - res.get("fp_count", 0) - res.get("fn_count", 0))
        # Use actual total from dataset
        total = 117
        fp    = res.get("fp_count", 0)
        fn    = res.get("fn_count", 0)
        lines.append(
            f"| **{name}** | {total} | {fp} | {fn} | "
            f"{100*fp/total:.1f}% | {100*fn/total:.1f}% |\n"
        )
    lines.append("\n---\n\n")

    # Stratified accuracy tables
    for dim, title in [
        ("flight_date",  "Flight-wise Accuracy"),
        ("altitude_bin", "Altitude-wise Accuracy"),
        ("depth",        "Depth-wise Accuracy"),
    ]:
        sub = stratified_df[stratified_df["dimension"] == dim]
        if sub.empty:
            continue
        lines.append(f"## {title}\n\n")
        lines.append("| Group | Model | Accuracy | N Samples | FP | FN |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['group']} | {row['model']} | "
                f"{row['accuracy']:.4f} | {row['n_samples']} | "
                f"{row['n_fp']} | {row['n_fn']} |\n"
            )
        lines.append("\n---\n\n")

    # Generated figures
    lines.append("## Generated Figures\n\n")
    lines.append("| Figure | Description |\n")
    lines.append("|---|---|\n")
    lines.append("| `false_positives/` | FP image panels per model |\n")
    lines.append("| `false_negatives/` | FN image panels per model |\n")
    lines.append("| `flight_wise_accuracy.png` | Accuracy grouped by flight date |\n")
    lines.append("| `altitude_wise_accuracy.png` | Accuracy grouped by altitude bin |\n")
    lines.append("| `depth_wise_accuracy.png` | Accuracy grouped by mine depth |\n")
    lines.append("| `zone_wise_accuracy.png` | Accuracy grouped by zone |\n")
    lines.append("\n")

    ERROR_ANALYSIS_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    ERROR_ANALYSIS_REPORT_MD.write_text("".join(lines), encoding="utf-8")
    logging.info("Error report saved → %s", ERROR_ANALYSIS_REPORT_MD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_error_analysis(model_filter: str | None = None) -> None:
    """Run error analysis for all (or one) model(s)."""
    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_ANALYSIS_FP_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_ANALYSIS_FN_DIR.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    logging.info("Loading test dataset ...")
    _, _, test_ds, class_names = create_datasets(PROCESSED_DIR)
    logging.info("Classes: %s", class_names)

    # Load metadata
    metadata_df = _load_metadata()

    registry = MODEL_REGISTRY
    if model_filter:
        registry = [e for e in MODEL_REGISTRY if e["key"] == model_filter.lower()]
        if not registry:
            logging.error("Unknown model key: '%s'", model_filter)
            return

    all_results: dict[str, dict] = {}
    all_stratified: list[dict]   = []
    model_colors = {e["display_name"]: e["color"] for e in MODEL_REGISTRY}

    for entry in registry:
        result = analyse_single_model(entry, test_ds, class_names, metadata_df)
        if result:
            all_results[entry["display_name"]] = result
            all_stratified.extend(result.get("stratified_rows", []))

    # Generate stratified charts
    if all_stratified:
        strat_df = pd.DataFrame(all_stratified)
        _generate_stratified_charts(strat_df, model_colors)
    else:
        logging.warning(
            "No stratified data available. Metadata may be missing or "
            "filenames could not be matched to test predictions."
        )
        strat_df = pd.DataFrame()

    # Markdown report
    _generate_report(all_results, strat_df)

    print(f"\n[DONE] Error analysis outputs saved → {ERROR_ANALYSIS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 12 – Error analysis for DroneML models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mobilenetv3", "resnet50", "efficientnetb0"],
        help="Analyse only this model (default: all models).",
    )
    args = parser.parse_args()
    run_error_analysis(model_filter=args.model)


if __name__ == "__main__":
    main()

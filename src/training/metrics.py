"""
Multi-Class Evaluation Metrics Suite for classification_v2
==========================================================

Computes comprehensive evaluation metrics for 3-class thermal classification:
- Accuracy
- Macro & Weighted Precision, Recall, and F1-Score
- Per-class Precision, Recall, F1, and Support
- 3x3 Confusion Matrix (Raw counts & Row-normalized percentages)
- One-vs-Rest ROC-AUC (Macro, Weighted, and Per-Class)
- One-vs-Rest PR-AUC / Average Precision (Macro, Weighted, and Per-Class)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.training.config import CLASS_NAMES

logger = logging.getLogger(__name__)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Compute exhaustive multi-class classification evaluation metrics.
    
    Args:
        y_true: 1D array of ground truth integer class indices.
        y_pred_probs: 2D array of predicted class probabilities (N, num_classes).
        class_names: List of class name strings.
        
    Returns:
        Structured dictionary containing all computed metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)
    y_pred = np.argmax(y_pred_probs, axis=1)

    num_classes = len(class_names)
    total_samples = len(y_true)

    # 1. Overall & Aggregated Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), np.maximum(row_sums, 1), where=row_sums != 0)

    # 3. Per-Class Metrics & One-vs-Rest AUCs
    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    ovr_roc_aucs: List[float] = []
    ovr_pr_aucs: List[float] = []

    for c_idx, c_name in enumerate(class_names):
        c_true_bin = (y_true == c_idx).astype(int)
        c_pred_bin = (y_pred == c_idx).astype(int)
        c_probs = y_pred_probs[:, c_idx]

        c_prec = float(precision_score(c_true_bin, c_pred_bin, zero_division=0))
        c_rec = float(recall_score(c_true_bin, c_pred_bin, zero_division=0))
        c_f1 = float(f1_score(c_true_bin, c_pred_bin, zero_division=0))
        c_supp = int(np.sum(c_true_bin))

        # ROC-AUC (valid if both positive and negative samples exist)
        if len(np.unique(c_true_bin)) > 1:
            try:
                c_roc_auc = float(roc_auc_score(c_true_bin, c_probs))
                ovr_roc_aucs.append(c_roc_auc)
            except Exception:
                c_roc_auc = None
        else:
            c_roc_auc = None

        # PR-AUC / Average Precision
        if len(np.unique(c_true_bin)) > 1:
            try:
                c_pr_auc = float(average_precision_score(c_true_bin, c_probs))
                ovr_pr_aucs.append(c_pr_auc)
            except Exception:
                c_pr_auc = None
        else:
            c_pr_auc = None

        per_class_metrics[c_name] = {
            "class_index": c_idx,
            "support": c_supp,
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1_score": round(c_f1, 4),
            "roc_auc": round(c_roc_auc, 4) if c_roc_auc is not None else None,
            "pr_auc": round(c_pr_auc, 4) if c_pr_auc is not None else None,
        }

    macro_roc_auc = round(float(np.mean(ovr_roc_aucs)), 4) if ovr_roc_aucs else None
    macro_pr_auc = round(float(np.mean(ovr_pr_aucs)), 4) if ovr_pr_aucs else None

    # Multi-class macro/weighted ROC-AUC from sklearn if applicable
    try:
        sklearn_macro_auc = float(roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro"))
        sklearn_weighted_auc = float(roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="weighted"))
    except Exception:
        sklearn_macro_auc = macro_roc_auc
        sklearn_weighted_auc = None

    results = {
        "total_samples": total_samples,
        "overall_accuracy": round(acc, 4),
        "macro_metrics": {
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1_score": round(macro_f1, 4),
            "roc_auc": round(sklearn_macro_auc, 4) if sklearn_macro_auc is not None else None,
            "pr_auc": macro_pr_auc,
        },
        "weighted_metrics": {
            "precision": round(weighted_prec, 4),
            "recall": round(weighted_rec, 4),
            "f1_score": round(weighted_f1, 4),
            "roc_auc": round(sklearn_weighted_auc, 4) if sklearn_weighted_auc is not None else None,
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": class_names,
            "counts": cm.tolist(),
            "normalized": np.round(cm_norm, 4).tolist(),
        },
    }

    return results


def format_metrics_summary_table(results: Dict[str, Any]) -> str:
    """Produce a clean ASCII formatted summary of classification metrics."""
    lines: List[str] = []
    lines.append("\n" + "=" * 76)
    lines.append("                CLASSIFICATION_V2 EVALUATION SUMMARY")
    lines.append("=" * 76)
    lines.append(f"Total Test Samples: {results['total_samples']:,}")
    lines.append(f"Overall Accuracy  : {results['overall_accuracy'] * 100:.2f}%\n")

    lines.append(f"{'Class':<14} | {'Support':>8} | {'Precision':>10} | {'Recall':>8} | {'F1-Score':>8} | {'ROC-AUC':>8} | {'PR-AUC':>8}")
    lines.append("-" * 76)
    for c_name, m in results["per_class_metrics"].items():
        roc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        pr_str = f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A"
        lines.append(
            f"{c_name:<14} | {m['support']:>8,} | {m['precision']:>10.4f} | {m['recall']:>8.4f} | {m['f1_score']:>8.4f} | {roc_str:>8} | {pr_str:>8}"
        )
    lines.append("-" * 76)
    macro = results["macro_metrics"]
    macro_roc = f"{macro['roc_auc']:.4f}" if macro['roc_auc'] is not None else "N/A"
    macro_pr = f"{macro['pr_auc']:.4f}" if macro['pr_auc'] is not None else "N/A"
    lines.append(
        f"{'Macro Avg':<14} | {results['total_samples']:>8,} | {macro['precision']:>10.4f} | {macro['recall']:>8.4f} | {macro['f1_score']:>8.4f} | {macro_roc:>8} | {macro_pr:>8}"
    )

    weighted = results["weighted_metrics"]
    lines.append(
        f"{'Weighted Avg':<14} | {results['total_samples']:>8,} | {weighted['precision']:>10.4f} | {weighted['recall']:>8.4f} | {weighted['f1_score']:>8.4f} | {'—':>8} | {'—':>8}"
    )
    lines.append("=" * 76)

    # Confusion matrix
    cm_counts = results["confusion_matrix"]["counts"]
    cm_norm = results["confusion_matrix"]["normalized"]
    labels = results["confusion_matrix"]["labels"]

    lines.append("\nCONFUSION MATRIX (Predicted columns vs Actual rows):")
    col_hdr = "Actual / Pred"
    header = f"{col_hdr:<16} " + "".join(f"{lbl:>14}" for lbl in labels)
    lines.append(header)
    lines.append("-" * len(header))
    for i, actual_lbl in enumerate(labels):
        row_str = f"{actual_lbl:<16} "
        for j in range(len(labels)):
            cnt = cm_counts[i][j]
            pct = cm_norm[i][j] * 100.0
            row_str += f"{cnt:>7} ({pct:>4.1f}%)"
        lines.append(row_str)
    lines.append("=" * 76 + "\n")

    return "\n".join(lines)


def export_evaluation_artifacts(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save metrics JSON and human-readable text summary to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_metrics.json"
    summary_path = output_dir / "evaluation_summary.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    summary_text = format_metrics_summary_table(results)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info(f"Evaluation artifacts saved to {json_path.name} and {summary_path.name}")

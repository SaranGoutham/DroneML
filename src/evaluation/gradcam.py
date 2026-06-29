# -*- coding: utf-8 -*-
"""gradcam.py – DroneML  [src/evaluation]
==========================================

Phase 11 – Explainability (Grad-CAM + Heatmaps)
--------------------------------------------------
Generates Gradient-weighted Class Activation Maps (Grad-CAM) for each model
to visually explain which regions of a thermal image the model attends to
when predicting Mine vs No_Mine.

What it produces
----------------
  For each model × sample image:
    - Original thermal image
    - Grad-CAM heatmap  (jet colourmap)
    - Superimposed overlay (original + heatmap alpha-blend)
    - Side-by-side panel (3 images per figure)

  Additional grid: a 3-row × N-col grid showing one batch of images
  and their heatmaps for each model.

Outputs  (all under  results/gradcam/)
--------------------------------------
  gradcam/<model>/mine_<n>.png        – Mine samples
  gradcam/<model>/no_mine_<n>.png     – No_Mine samples
  gradcam/<model>/fp_<n>.png          – False Positive samples
  gradcam/<model>/fn_<n>.png          – False Negative samples
  gradcam/overlay_grid.png            – multi-model comparison grid

Architecture note
-----------------
Grad-CAM requires access to the gradient of the model output w.r.t. the
feature maps of the last convolutional layer.  For each backbone we use
the following target layer:

  MobileNetV3-Large  : "expanded_conv_14/project/BatchNorm"
  ResNet50           : "conv5_block3_out"
  EfficientNetB0     : "top_activation"

Usage
-----
  python -m src.evaluation.gradcam
  python -m src.evaluation.gradcam --model resnet50 --n_samples 8
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.config import PROCESSED_DIR
from src.preprocessing.create_tf_dataset import create_datasets
from src.models.config import (
    IMAGE_SIZE,
    GRADCAM_DIR,
    GRADCAM_MOBILENETV3_DIR,
    GRADCAM_RESNET50_DIR,
    GRADCAM_EFFICIENTNETB0_DIR,
    MOBILENETV3_FINAL_MODEL_PATH,
    RESNET50_FINAL_MODEL_PATH,
    EFFICIENTNETB0_FINAL_MODEL_PATH,
)

# Decision threshold
THRESHOLD: float = 0.5

# How many samples to visualise per class per model
DEFAULT_N_SAMPLES: int = 5

# ---------------------------------------------------------------------------
# Model registry – includes the target Grad-CAM layer name
# ---------------------------------------------------------------------------

MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "key"              : "mobilenetv3",
        "display_name"     : "MobileNetV3-Large",
        "model_path"       : MOBILENETV3_FINAL_MODEL_PATH,
        "gradcam_dir"      : GRADCAM_MOBILENETV3_DIR,
        # Last spatial conv layer inside the MobileNetV3Large backbone
        "target_layer"     : "Conv_1",          # last conv before GAP
        "color"            : "#4C72B0",
    },
    {
        "key"              : "resnet50",
        "display_name"     : "ResNet50",
        "model_path"       : RESNET50_FINAL_MODEL_PATH,
        "gradcam_dir"      : GRADCAM_RESNET50_DIR,
        # Last residual block output (before GAP)
        "target_layer"     : "conv5_block3_out",
        "color"            : "#DD8452",
    },
    {
        "key"              : "efficientnetb0",
        "display_name"     : "EfficientNetB0",
        "model_path"       : EFFICIENTNETB0_FINAL_MODEL_PATH,
        "gradcam_dir"      : GRADCAM_EFFICIENTNETB0_DIR,
        # Last activation before GAP
        "target_layer"     : "top_activation",
        "color"            : "#55A868",
    },
]


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

def _find_target_layer(model: keras.Model, layer_name: str) -> keras.Layer | None:
    """Search the model (and nested sub-models) for a layer matching layer_name."""
    # 1. Direct search
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
    # 2. Search inside nested sub-models (backbone is a sub-model)
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            for sublayer in layer.layers:
                if sublayer.name == layer_name:
                    return sublayer
    return None


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: keras.Model,
    target_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    """Compute the Grad-CAM heatmap for one image.

    Parameters
    ----------
    img_array : shape (1, H, W, C) float32 in [0, 1]
    model : the full Keras model
    target_layer_name : name of the convolutional layer to hook
    pred_index : class index to explain (default: argmax of model output)

    Returns
    -------
    heatmap : np.ndarray  shape (h, w)  values in [0, 1]
    """
    target_layer = _find_target_layer(model, target_layer_name)
    if target_layer is None:
        # Fallback: use the last Conv2D layer in the model
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                target_layer = layer
                logging.warning(
                    "Target layer '%s' not found. Falling back to '%s'.",
                    target_layer_name, layer.name,
                )
                break
        if target_layer is None:
            # Deep search inside sub-models
            for layer in model.layers:
                if isinstance(layer, keras.Model):
                    for sublayer in reversed(layer.layers):
                        if isinstance(sublayer, keras.layers.Conv2D):
                            target_layer = sublayer
                            logging.warning(
                                "Using fallback Grad-CAM layer '%s' inside '%s'.",
                                sublayer.name, layer.name,
                            )
                            break
                if target_layer is not None:
                    break
        if target_layer is None:
            raise RuntimeError("Could not find any Conv2D layer for Grad-CAM.")

    # Build a sub-model that outputs (target_layer_output, final_predictions)
    grad_model = keras.Model(
        inputs  = model.inputs,
        outputs = [target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        inputs      = tf.cast(img_array, tf.float32)
        conv_output, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of the class score w.r.t. the conv feature map
    grads = tape.gradient(class_channel, conv_output)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps
    conv_output = conv_output[0]
    heatmap     = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)

    # ReLU + normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# ---------------------------------------------------------------------------
# Overlay heatmap on image
# ---------------------------------------------------------------------------

def _overlay_heatmap(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2_COLORMAP_JET if False else None,
) -> np.ndarray:
    """Resize heatmap to image size and alpha-blend with the original.

    Parameters
    ----------
    img : np.ndarray  shape (H, W, 3)  float32 [0, 1]
    heatmap : np.ndarray  shape (h, w)  float32 [0, 1]
    alpha : heatmap opacity

    Returns
    -------
    overlay : np.ndarray  shape (H, W, 3)  float32 [0, 1]
    """
    h, w = img.shape[:2]
    # Use matplotlib to resize & colourise
    import cv2  # only imported here so the module stays usable without cv2
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    overlay = alpha * heatmap_rgb + (1 - alpha) * img
    return np.clip(overlay, 0, 1)


def _overlay_heatmap_pure(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Pure-matplotlib overlay (no OpenCV dependency).

    Resizes heatmap using numpy repeat and applies jet colourmap.
    """
    h, w = img.shape[:2]
    # Resize heatmap with nearest-neighbour
    scale_h = h / heatmap.shape[0]
    scale_w = w / heatmap.shape[1]
    heatmap_large = np.repeat(
        np.repeat(heatmap, int(np.ceil(scale_h)), axis=0),
        int(np.ceil(scale_w)), axis=1,
    )[:h, :w]

    # Apply jet colourmap
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap_large)[..., :3]    # (H, W, 3) float

    overlay = alpha * heatmap_rgb + (1 - alpha) * img
    return np.clip(overlay, 0, 1)


def overlay_heatmap(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay heatmap on image.  Uses OpenCV if available, else pure numpy."""
    try:
        import cv2  # noqa: F401
        return _overlay_heatmap(img, heatmap, alpha)
    except ImportError:
        return _overlay_heatmap_pure(img, heatmap, alpha)


# ---------------------------------------------------------------------------
# Save a Grad-CAM panel (original | heatmap | overlay)
# ---------------------------------------------------------------------------

def save_gradcam_panel(
    img: np.ndarray,
    heatmap: np.ndarray,
    out_path: Path,
    title: str,
    pred_label: str,
    true_label: str,
    confidence: float,
) -> None:
    """Save a 1×3 panel: original | heatmap | overlay."""
    cmap  = plt.get_cmap("jet")
    hm_h, hm_w = img.shape[:2]

    # Resize heatmap to image size for display
    scale_h = hm_h / max(heatmap.shape[0], 1)
    scale_w = hm_w / max(heatmap.shape[1], 1)
    heatmap_large = np.repeat(
        np.repeat(heatmap, int(np.ceil(scale_h)), axis=0),
        int(np.ceil(scale_w)), axis=1,
    )[:hm_h, :hm_w]

    overlay = overlay_heatmap(img, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"{title}  |  True: {true_label}  |  Pred: {pred_label} ({confidence:.1%})",
        fontsize=11,
        fontweight="bold",
    )

    axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_large, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Generate Grad-CAM for one model
# ---------------------------------------------------------------------------

def generate_gradcam_for_model(
    entry: dict[str, Any],
    test_ds: tf.data.Dataset,
    class_names: list[str],
    n_samples: int = DEFAULT_N_SAMPLES,
) -> None:
    """Generate Grad-CAM panels for Mine, No_Mine, FP, FN samples.

    Parameters
    ----------
    entry : model registry entry
    test_ds : unbatched or batched test dataset
    class_names : list of class label strings
    n_samples : number of samples per category
    """
    model_path = entry["model_path"]
    if not model_path.exists():
        logging.warning("Model not found, skipping: %s", model_path)
        return

    logging.info(
        "Loading %s for Grad-CAM from %s ...",
        entry["display_name"], model_path,
    )
    model: keras.Model = keras.models.load_model(str(model_path))

    out_dir: Path = entry["gradcam_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images + labels + predictions
    images_list: list[np.ndarray] = []
    labels_list: list[int]        = []
    probs_list:  list[float]      = []

    for imgs_batch, lbls_batch in test_ds:
        prbs = model(imgs_batch, training=False).numpy().flatten()
        for img, lbl, prob in zip(imgs_batch.numpy(), lbls_batch.numpy().flatten(), prbs):
            images_list.append(img)
            labels_list.append(int(lbl))
            probs_list.append(float(prob))

    images_arr = np.array(images_list)  # (N, H, W, C)
    labels_arr = np.array(labels_list)  # (N,)
    probs_arr  = np.array(probs_list)   # (N,)
    preds_arr  = (probs_arr >= THRESHOLD).astype(int)

    # Category indices
    # class_names[0] = Mine → label 0   class_names[1] = No_Mine → label 1
    mine_idx     = np.where(labels_arr == 0)[0]
    no_mine_idx  = np.where(labels_arr == 1)[0]
    fp_idx       = np.where((labels_arr == 1) & (preds_arr == 0))[0]  # predicted Mine, actually No_Mine
    fn_idx       = np.where((labels_arr == 0) & (preds_arr == 1))[0]  # predicted No_Mine, actually Mine

    categories = [
        ("mine",     mine_idx,    "Mine"),
        ("no_mine",  no_mine_idx, "No_Mine"),
        ("fp",       fp_idx,      "False Positive"),
        ("fn",       fn_idx,      "False Negative"),
    ]

    target_layer_name = entry["target_layer"]

    for cat_key, idx_arr, cat_label in categories:
        if len(idx_arr) == 0:
            logging.info(
                "No %s samples for %s – skipping.",
                cat_label, entry["display_name"],
            )
            continue

        # Sample up to n_samples
        chosen = idx_arr[:n_samples]
        logging.info(
            "Generating %d Grad-CAM panels for %s [%s] ...",
            len(chosen), entry["display_name"], cat_label,
        )

        for i, sample_idx in enumerate(chosen):
            img   = images_arr[sample_idx]           # (H, W, 3) float32 [0, 1]
            label = labels_arr[sample_idx]
            prob  = probs_arr[sample_idx]
            pred  = preds_arr[sample_idx]

            true_label = class_names[label] if label < len(class_names) else str(label)
            pred_label = class_names[pred]  if pred  < len(class_names) else str(pred)
            confidence = prob if pred == 1 else 1 - prob

            # Expand dims for model input: (1, H, W, C)
            img_input = img[np.newaxis, ...]

            try:
                heatmap = make_gradcam_heatmap(
                    img_input, model, target_layer_name, pred_index=0
                )
            except Exception as exc:
                logging.warning(
                    "Grad-CAM failed for sample %d of %s [%s]: %s",
                    sample_idx, entry["display_name"], cat_label, exc,
                )
                continue

            out_path = out_dir / f"{cat_key}_{i + 1:02d}.png"
            save_gradcam_panel(
                img        = img,
                heatmap    = heatmap,
                out_path   = out_path,
                title      = f"{entry['display_name']}",
                pred_label = pred_label,
                true_label = true_label,
                confidence = confidence,
            )
            logging.info("Saved Grad-CAM → %s", out_path)

    del model
    tf.keras.backend.clear_session()


# ---------------------------------------------------------------------------
# Multi-model overlay grid
# ---------------------------------------------------------------------------

def generate_overlay_grid(
    entries: list[dict[str, Any]],
    test_ds: tf.data.Dataset,
    class_names: list[str],
    n_samples: int = 4,
    out_path: Path = GRADCAM_DIR / "overlay_grid.png",
) -> None:
    """Generate a model × sample grid of Grad-CAM overlays.

    Rows = models, Cols = sample images.
    """
    # Collect the first n_samples images + labels from the test set
    sample_images: list[np.ndarray] = []
    sample_labels: list[int]        = []
    for imgs_batch, lbls_batch in test_ds:
        for img, lbl in zip(imgs_batch.numpy(), lbls_batch.numpy().flatten()):
            sample_images.append(img)
            sample_labels.append(int(lbl))
            if len(sample_images) >= n_samples:
                break
        if len(sample_images) >= n_samples:
            break

    available_entries = [e for e in entries if e["model_path"].exists()]
    n_models = len(available_entries)
    n_cols   = len(sample_images)

    if n_models == 0:
        logging.warning("No model files found for overlay grid.")
        return

    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_models),
    )
    if n_models == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Grad-CAM Overlay Grid – All Models × Test Samples",
        fontsize=14,
        fontweight="bold",
    )

    for row, entry in enumerate(available_entries):
        model: keras.Model = keras.models.load_model(str(entry["model_path"]))
        target_layer_name  = entry["target_layer"]

        for col, (img, lbl) in enumerate(zip(sample_images, sample_labels)):
            img_input = img[np.newaxis, ...]
            try:
                heatmap = make_gradcam_heatmap(img_input, model, target_layer_name, pred_index=0)
                overlay = overlay_heatmap(img, heatmap)
            except Exception as exc:
                overlay = img
                logging.warning("Grad-CAM skipped for grid cell [%d,%d]: %s", row, col, exc)

            ax = axes[row, col]
            ax.imshow(overlay)
            ax.axis("off")
            if row == 0:
                true_lbl = class_names[lbl] if lbl < len(class_names) else str(lbl)
                ax.set_title(f"True: {true_lbl}", fontsize=9)
            if col == 0:
                ax.set_ylabel(entry["short"] if "short" in entry else entry["display_name"],
                              fontsize=9, labelpad=5)

        del model
        tf.keras.backend.clear_session()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info("Overlay grid saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_gradcam(
    model_filter: str | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> None:
    """Generate Grad-CAM visualisations for all (or one) model(s)."""
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading test dataset ...")
    _, _, test_ds, class_names = create_datasets(PROCESSED_DIR)
    logging.info("Classes: %s", class_names)

    registry = MODEL_REGISTRY
    if model_filter:
        registry = [e for e in MODEL_REGISTRY if e["key"] == model_filter.lower()]
        if not registry:
            logging.error("Unknown model key: '%s'", model_filter)
            return

    for entry in registry:
        generate_gradcam_for_model(entry, test_ds, class_names, n_samples=n_samples)

    # Multi-model overlay grid (all models together)
    if not model_filter:
        logging.info("Generating multi-model overlay grid ...")
        generate_overlay_grid(MODEL_REGISTRY, test_ds, class_names, n_samples=4)

    print(f"\n[DONE] Grad-CAM outputs saved → {GRADCAM_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 11 – Grad-CAM explainability for DroneML models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mobilenetv3", "resnet50", "efficientnetb0"],
        help="Generate Grad-CAM only for this model (default: all).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of samples per category (default: {DEFAULT_N_SAMPLES}).",
    )
    args = parser.parse_args()
    run_gradcam(model_filter=args.model, n_samples=args.n_samples)


if __name__ == "__main__":
    main()

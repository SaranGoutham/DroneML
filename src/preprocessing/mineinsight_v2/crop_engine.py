"""
Crop Engine Module for MineInsight v2
=====================================

Implements isotropic 1:1 square crop extraction around YOLO bounding boxes
with 20% context margin, edge-safe inward shifting, Lanczos resampling to 224x224,
and exact coordinate tracking for master manifest provenance.
"""

import logging
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image

from src.preprocessing.mineinsight_v2.config import (
    CROP_TARGET_SIZE,
    CONTEXT_PADDING_RATIO,
    JPEG_QUALITY,
)

logger = logging.getLogger(__name__)


def compute_square_crop_box(
    img_w: int,
    img_h: int,
    x_center: float,
    y_center: float,
    w: float,
    h: float,
    padding_ratio: float = CONTEXT_PADDING_RATIO,
) -> Tuple[int, int, int, int]:
    """
    Compute integer pixel coordinates for a 1:1 square crop around a YOLO bounding box.
    
    Ensures:
    1. Aspect ratio is preserved (1:1 square).
    2. Context padding (default 20%) is applied symmetrically.
    3. Inward shifting if the square exceeds frame boundaries.
    4. Size does not exceed image dimensions.
    
    Args:
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        x_center: YOLO normalized center x in [0.0, 1.0].
        y_center: YOLO normalized center y in [0.0, 1.0].
        w: YOLO normalized width in [0.0, 1.0].
        h: YOLO normalized height in [0.0, 1.0].
        padding_ratio: Fractional context expansion (default 0.20 = 20%).
        
    Returns:
        (xmin, ymin, xmax, ymax) in integer pixel coordinates.
    """
    # Convert normalized dimensions to pixels
    box_w_px = w * img_w
    box_h_px = h * img_h
    cx_px = x_center * img_w
    cy_px = y_center * img_h

    # 1. Determine base square side length as max of width and height
    base_side = max(box_w_px, box_h_px)
    if base_side < 2.0:
        base_side = 2.0

    # 2. Add context margin (e.g. +20%)
    padded_side = base_side * (1.0 + padding_ratio)

    # 3. Constrain to maximum possible square that fits within the image
    max_allowable_side = min(img_w, img_h)
    side = min(padded_side, float(max_allowable_side))
    half_side = side / 2.0

    # 4. Compute candidate bounds centered on (cx_px, cy_px)
    xmin = cx_px - half_side
    ymin = cy_px - half_side
    xmax = cx_px + half_side
    ymax = cy_px + half_side

    # 5. Inward shifting if bounds exceed frame edges
    if xmin < 0:
        shift_x = -xmin
        xmin += shift_x
        xmax += shift_x
    elif xmax > img_w:
        shift_x = xmax - img_w
        xmin -= shift_x
        xmax -= shift_x

    if ymin < 0:
        shift_y = -ymin
        ymin += shift_y
        ymax += shift_y
    elif ymax > img_h:
        shift_y = ymax - img_h
        ymin -= shift_y
        ymax -= shift_y

    # 6. Final integer rounding and clamping
    int_xmin = int(round(max(0.0, xmin)))
    int_ymin = int(round(max(0.0, ymin)))
    int_xmax = int(round(min(float(img_w), xmax)))
    int_ymax = int(round(min(float(img_h), ymax)))

    # Guarantee exact square integer dimensions if rounding caused 1px disparity
    width = int_xmax - int_xmin
    height = int_ymax - int_ymin
    if width != height:
        target_dim = min(width, height)
        int_xmax = int_xmin + target_dim
        int_ymax = int_ymin + target_dim

    return int_xmin, int_ymin, int_xmax, int_ymax


def extract_isotropic_crop(
    image: Image.Image,
    x_center: float,
    y_center: float,
    w: float,
    h: float,
    padding_ratio: float = CONTEXT_PADDING_RATIO,
    target_size: Tuple[int, int] = CROP_TARGET_SIZE,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Extract an isotropic 1:1 square crop around the target and resize to target_size.
    
    Returns:
        (cropped_and_resized_image, (xmin, ymin, xmax, ymax))
    """
    img_w, img_h = image.size
    xmin, ymin, xmax, ymax = compute_square_crop_box(
        img_w, img_h, x_center, y_center, w, h, padding_ratio=padding_ratio
    )

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Degenerate crop box computed: ({xmin}, {ymin}, {xmax}, {ymax})")

    crop = image.crop((xmin, ymin, xmax, ymax))
    resized_crop = crop.resize(target_size, resample=Image.Resampling.LANCZOS)
    return resized_crop, (xmin, ymin, xmax, ymax)


def extract_native_center_tile(
    image: Image.Image,
    target_size: Tuple[int, int] = CROP_TARGET_SIZE,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Extract a native 1:1 central tile of exact target_size from an empty frame
    without any scaling or interpolation.
    
    Returns:
        (native_tile_image, (xmin, ymin, xmax, ymax))
    """
    img_w, img_h = image.size
    t_w, t_h = target_size

    if img_w < t_w or img_h < t_h:
        raise ValueError(f"Image ({img_w}x{img_h}) is smaller than requested tile ({t_w}x{t_h})")

    xmin = (img_w - t_w) // 2
    ymin = (img_h - t_h) // 2
    xmax = xmin + t_w
    ymax = ymin + t_h

    tile = image.crop((xmin, ymin, xmax, ymax))
    return tile, (xmin, ymin, xmax, ymax)

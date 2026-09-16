"""
Background Sampler Module for MineInsight v2
============================================

Extracts authentic, non-correlated background tiles from verified empty-label
frames. Implements central square tile extraction, thermal variance thresholding,
and temporal stride control.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

from src.preprocessing.mineinsight_v2.config import (
    CROP_TARGET_SIZE,
    BG_TEMPORAL_STRIDE_DEFAULT,
    BG_MIN_VARIANCE_STD_DEV,
)
from src.preprocessing.mineinsight_v2.crop_engine import extract_native_center_tile

logger = logging.getLogger(__name__)


class BackgroundSampler:
    """Manages background crop selection and quality control from empty-label frames."""

    def __init__(
        self,
        stride: int = BG_TEMPORAL_STRIDE_DEFAULT,
        min_variance: float = BG_MIN_VARIANCE_STD_DEV,
        target_size: Tuple[int, int] = CROP_TARGET_SIZE,
    ):
        self.stride = max(1, stride)
        self.min_variance = min_variance
        self.target_size = target_size

    def check_tile_quality(self, tile: Image.Image) -> Tuple[bool, float]:
        """
        Verify thermal variance across the tile.
        
        Returns:
            (is_valid, standard_deviation)
        """
        arr = np.array(tile)
        if arr.ndim == 3:
            # Grayscale channel is identical across R, G, B
            arr = arr[:, :, 0]
        std_dev = float(np.std(arr))
        is_valid = std_dev >= self.min_variance
        return is_valid, std_dev

    def filter_empty_frames(
        self,
        img_paths: List[Path],
        lbl_dir: Path,
    ) -> List[Path]:
        """
        Identify matching frames whose label file exists and is strictly empty (0 bytes or 0 lines).
        """
        empty_frames: List[Path] = []
        for img_path in sorted(img_paths):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            if lbl_path.stat().st_size == 0:
                empty_frames.append(img_path)
            else:
                try:
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if not lines:
                        empty_frames.append(img_path)
                except Exception:
                    continue
        return empty_frames

    def sample_sequence_candidates(
        self,
        empty_frames: List[Path],
    ) -> List[Path]:
        """
        Select empty frames by applying temporal stride k.
        """
        return empty_frames[:: self.stride]

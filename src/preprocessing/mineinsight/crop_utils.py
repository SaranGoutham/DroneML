"""
Crop Utilities Module
=====================

Reusable image cropping utility module for the MineInsight preprocessing pipeline.
This module ONLY provides image processing utilities.
"""

import logging
from typing import Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def yolo_to_pixel_bbox(
    img_w: int, img_h: int, x_center: float, y_center: float, w: float, h: float
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO normalized coordinates to absolute pixel coordinates.

    Parameters:
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.
        x_center (float): YOLO normalized x center.
        y_center (float): YOLO normalized y center.
        w (float): YOLO normalized width.
        h (float): YOLO normalized height.

    Returns:
        Tuple[int, int, int, int]: (xmin, ymin, xmax, ymax) in pixel coordinates.
    """
    xmin = int((x_center - w / 2.0) * img_w)
    ymin = int((y_center - h / 2.0) * img_h)
    xmax = int((x_center + w / 2.0) * img_w)
    ymax = int((y_center + h / 2.0) * img_h)
    
    return xmin, ymin, xmax, ymax


def clip_bbox(
    xmin: int, ymin: int, xmax: int, ymax: int, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """
    Clip bounding box coordinates to ensure they fall within the image boundaries.

    Parameters:
        xmin (int): Minimum x coordinate.
        ymin (int): Minimum y coordinate.
        xmax (int): Maximum x coordinate.
        ymax (int): Maximum y coordinate.
        img_w (int): Image width.
        img_h (int): Image height.

    Returns:
        Tuple[int, int, int, int]: Clipped (xmin, ymin, xmax, ymax).
    """
    return (
        max(0, xmin),
        max(0, ymin),
        min(img_w, xmax),
        min(img_h, ymax)
    )


def add_padding(
    xmin: int, ymin: int, xmax: int, ymax: int, img_w: int, img_h: int, padding_ratio: float
) -> Tuple[int, int, int, int]:
    """
    Add padding to a bounding box and clip it to image boundaries.

    Parameters:
        xmin (int): Minimum x coordinate.
        ymin (int): Minimum y coordinate.
        xmax (int): Maximum x coordinate.
        ymax (int): Maximum y coordinate.
        img_w (int): Image width.
        img_h (int): Image height.
        padding_ratio (float): The ratio of width/height to pad. Example: 0.10.

    Returns:
        Tuple[int, int, int, int]: Padded and clipped (xmin, ymin, xmax, ymax).
    """
    box_w = xmax - xmin
    box_h = ymax - ymin
    
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)
    
    new_xmin = xmin - pad_x
    new_ymin = ymin - pad_y
    new_xmax = xmax + pad_x
    new_ymax = ymax + pad_y
    
    return clip_bbox(new_xmin, new_ymin, new_xmax, new_ymax, img_w, img_h)


def validate_bbox(
    xmin: int, ymin: int, xmax: int, ymax: int, img_w: int, img_h: int
) -> None:
    """
    Validate a bounding box to ensure coordinates are valid and it represents a valid region.

    Parameters:
        xmin (int): Minimum x coordinate.
        ymin (int): Minimum y coordinate.
        xmax (int): Maximum x coordinate.
        ymax (int): Maximum y coordinate.
        img_w (int): Image width.
        img_h (int): Image height.

    Raises:
        ValueError: If coordinates are negative, exceed bounds, have zero size, or are incorrectly ordered.
    """
    if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
        raise ValueError(f"Bounding box ({xmin}, {ymin}, {xmax}, {ymax}) is outside image boundaries ({img_w}x{img_h}).")
    
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Invalid bounding box ordering: xmin({xmin}) >= xmax({xmax}) or ymin({ymin}) >= ymax({ymax}).")
        
    width = xmax - xmin
    height = ymax - ymin
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Bounding box has zero or negative dimensions: {width}x{height}.")
        
    if width < 2 or height < 2:
        raise ValueError(f"Extremely tiny bounding box: {width}x{height}.")


def crop_image(
    image: Union[Image.Image, np.ndarray], xmin: int, ymin: int, xmax: int, ymax: int
) -> Image.Image:
    """
    Crop the image using the given pixel bounding box.

    Parameters:
        image (Union[Image.Image, np.ndarray]): The input image, either a PIL Image or NumPy array.
        xmin (int): Minimum x coordinate.
        ymin (int): Minimum y coordinate.
        xmax (int): Maximum x coordinate.
        ymax (int): Maximum y coordinate.

    Returns:
        Image.Image: The cropped image as a PIL Image.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    return image.crop((xmin, ymin, xmax, ymax))


def resize_crop(
    image: Image.Image, target_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Resize a cropped image to a target size using high-quality resampling.

    Parameters:
        image (Image.Image): The cropped image.
        target_size (Tuple[int, int]): The target (width, height) to resize to.

    Returns:
        Image.Image: The resized image.
    """
    return image.resize(target_size, resample=Image.Resampling.LANCZOS)


def validate_crop(image: Image.Image) -> bool:
    """
    Verify that a crop is valid (has non-zero dimensions).

    Parameters:
        image (Image.Image): The cropped PIL Image.

    Returns:
        bool: True if crop is valid.

    Raises:
        ValueError: If crop width or height is zero.
    """
    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"Invalid crop dimensions: {image.width}x{image.height}.")
    return True


def process_crop(
    image: Union[Image.Image, np.ndarray],
    x_center: float,
    y_center: float,
    w: float,
    h: float,
    padding_ratio: float = 0.0,
    target_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Complete pipeline to convert a YOLO annotation into a cropped, resized PIL Image.

    Parameters:
        image (Union[Image.Image, np.ndarray]): Input image.
        x_center (float): YOLO normalized x center.
        y_center (float): YOLO normalized y center.
        w (float): YOLO normalized width.
        h (float): YOLO normalized height.
        padding_ratio (float): Ratio to expand the bounding box by.
        target_size (Tuple[int, int]): Target dimensions for the output crop.

    Returns:
        Image.Image: The final cropped and resized PIL Image.
    """
    if isinstance(image, np.ndarray):
        img_h, img_w = image.shape[:2]
    else:
        img_w, img_h = image.size
        
    xmin, ymin, xmax, ymax = yolo_to_pixel_bbox(img_w, img_h, x_center, y_center, w, h)
    
    if padding_ratio > 0:
        xmin, ymin, xmax, ymax = add_padding(xmin, ymin, xmax, ymax, img_w, img_h, padding_ratio)
    else:
        xmin, ymin, xmax, ymax = clip_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
        
    validate_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
    
    crop = crop_image(image, xmin, ymin, xmax, ymax)
    validate_crop(crop)
    
    resized_crop = resize_crop(crop, target_size=target_size)
    return resized_crop


def bbox_width(xmin: int, xmax: int) -> int:
    """Calculate bounding box width."""
    return xmax - xmin


def bbox_height(ymin: int, ymax: int) -> int:
    """Calculate bounding box height."""
    return ymax - ymin


def bbox_area(xmin: int, ymin: int, xmax: int, ymax: int) -> int:
    """Calculate bounding box area."""
    return bbox_width(xmin, xmax) * bbox_height(ymin, ymax)


if __name__ == "__main__":
    print("====================================")
    print("Crop Utilities Self Test")
    print("====================================\n")
    
    try:
        # Create a synthetic image (1000x1000 RGB)
        test_img = Image.new("RGB", (1000, 1000), color="blue")
        img_w, img_h = test_img.size
        
        # Synthetic YOLO annotation
        xc, yc, w, h = 0.5, 0.5, 0.2, 0.2
        
        # Test 1: YOLO -> Pixel
        xmin, ymin, xmax, ymax = yolo_to_pixel_bbox(img_w, img_h, xc, yc, w, h)
        assert (xmin, ymin, xmax, ymax) == (400, 400, 600, 600)
        print("YOLO -> Pixel : PASS")
        
        # Test 2: Padding
        p_xmin, p_ymin, p_xmax, p_ymax = add_padding(xmin, ymin, xmax, ymax, img_w, img_h, 0.10)
        assert (p_xmin, p_ymin, p_xmax, p_ymax) == (380, 380, 620, 620)
        print("Padding : PASS")
        
        # Test 3: Clipping
        c_xmin, c_ymin, c_xmax, c_ymax = clip_bbox(-10, -20, 1050, 1200, img_w, img_h)
        assert (c_xmin, c_ymin, c_xmax, c_ymax) == (0, 0, 1000, 1000)
        print("Clipping : PASS")
        
        # Test 4: Validate bbox (no exception means pass)
        validate_bbox(p_xmin, p_ymin, p_xmax, p_ymax, img_w, img_h)
        
        # Test 5: Crop
        cropped = crop_image(test_img, p_xmin, p_ymin, p_xmax, p_ymax)
        validate_crop(cropped)
        assert cropped.size == (240, 240)
        print("Crop : PASS")
        
        # Test 6: Resize
        resized = resize_crop(cropped, (224, 224))
        assert resized.size == (224, 224)
        print("Resize : PASS")
        
        # Test 7: Complete Pipeline
        final_img = process_crop(test_img, xc, yc, w, h, padding_ratio=0.10, target_size=(224, 224))
        assert final_img.size == (224, 224)
        print("Pipeline : PASS\n")
        
        print("====================================")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")

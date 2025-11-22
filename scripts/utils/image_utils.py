"""Image processing utilities"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image(image_path: Path, min_size: int = 224) -> bool:
    """
    Check if image is valid and meets minimum size requirements

    Args:
        image_path: Path to image file
        min_size: Minimum width/height in pixels

    Returns:
        True if valid, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()  # Check if corrupted

        # Reopen after verify (PIL requirement)
        img = Image.open(image_path)
        width, height = img.size

        if width < min_size or height < min_size:
            logger.warning(f"{image_path.name}: too small ({width}x{height})")
            return False

        return True

    except Exception as e:
        logger.error(f"Invalid image {image_path.name}: {e}")
        return False

def resize_image(image_path: Path, target_size: int = 512,
                keep_aspect: bool = True) -> Optional[np.ndarray]:
    """
    Resize image to target size

    Args:
        image_path: Path to image
        target_size: Target size (max dimension)
        keep_aspect: Maintain aspect ratio

    Returns:
        Resized image array or None if failed
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]

        if keep_aspect:
            # Resize maintaining aspect ratio
            if h > w:
                new_h = target_size
                new_w = int(w * (target_size / h))
            else:
                new_w = target_size
                new_h = int(h * (target_size / w))
        else:
            new_h = new_w = target_size

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return resized

    except Exception as e:
        logger.error(f"Failed to resize {image_path.name}: {e}")
        return None

def get_image_stats(image_dir: Path) -> dict:
    """
    Calculate statistics for images in directory

    Args:
        image_dir: Directory containing images

    Returns:
        Dictionary with image statistics
    """
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'avg_width': 0,
        'avg_height': 0,
        'min_size': float('inf'),
        'max_size': 0,
    }

    widths = []
    heights = []

    for img_file in image_dir.glob('*.jpg'):
        stats['total_images'] += 1

        try:
            img = Image.open(img_file)
            w, h = img.size

            widths.append(w)
            heights.append(h)
            stats['valid_images'] += 1

            min_dim = min(w, h)
            max_dim = max(w, h)

            stats['min_size'] = min(stats['min_size'], min_dim)
            stats['max_size'] = max(stats['max_size'], max_dim)

        except:
            stats['invalid_images'] += 1

    if widths:
        stats['avg_width'] = np.mean(widths)
        stats['avg_height'] = np.mean(heights)

    return stats

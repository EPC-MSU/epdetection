"""
Detection module for EyePoint P10/B10.
"""

from .detect import detect_BGA, detect_elements, detect_label
from .utils import save_detect_img


__all__ = ["detect_BGA", "detect_elements", "detect_label", "save_detect_img"]

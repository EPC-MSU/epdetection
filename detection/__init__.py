"""
Detection module for EyePoint P10/B10
"""

from .detect import detect_elements
from .detect import detect_label
from .detect import detect_BGA
from .detect import detect_BGA_params

__all__ = ["detect_elements", "detect_label", "detect_BGA", "detect_BGA_params"]

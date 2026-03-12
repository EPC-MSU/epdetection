"""
Detection module for EyePoint P10/B10.
"""

from .detect import detect_BGA, detect_BGA_params, detect_elements, detect_label, get_element_names_by_mode
from .utils import save_detect_img


__all__ = ["detect_BGA", "detect_BGA_params", "detect_elements", "detect_label", "get_element_names_by_mode",
           "save_detect_img"]

# To make pdoc generate documentation only for public functions
__pdoc__ = {
    "detect": False,
    "detect_nn": False,
    "train": False,
    "utils": False,
    "utilities": False
}

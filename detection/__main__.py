"""
This file use for console module call

For example:
python -m detection --image detection//tests//elm_test1/image.png --draw-elements
"""
import logging
import argparse
import os

import cv2

from .detect import detect_elements
from .utils import FakeGuiConnector, dump_elements, save_detect_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection and drawing bounding box.")
    parser.add_argument("--image", default=None,
                        help="Path to image to detect elements")
    parser.add_argument("--trh-prob", default=0.7, help="""Detector threshold""")
    parser.add_argument("--trh-corr-mult", default=1.5, help="""Detector multiplication threshold""")
    parser.add_argument("--find-one", action="store_false",
                        help="If true - find one element in file")
    parser.add_argument("--save-json-result", action="store_true",
                        help="Save found elements in json")
    parser.add_argument("--draw-elements", action="store_true",
                        help="Save image with drawed elements")
    cliargs = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", level=logging.INFO)
    if cliargs.image is None:
        raise Exception("For specify the path to the image, pleace use arg: --image PATH")

    logging.info("Detection runing...")
    gc = FakeGuiConnector()
    img = cv2.imread(cliargs.image)
    try:
        if img.shape[0]+img.shape[1] == 0:
            raise AttributeError
    except AttributeError:
        raise Exception("Image broken or invalid path.")

    if not os.path.exists(os.path.join("log")):
        os.makedirs(os.path.join("log"))
    if not os.path.exists(os.path.join("log", "main")):
        os.makedirs(os.path.join("log", "main"))
    cv2.imwrite(os.path.join("log", "main", "in_image.png"), img)

    result = detect_elements(gc, img, trh_corr_mult=cliargs.trh_corr_mult)

    logging.info("-" * 40)
    logging.info("Detected elements:")

    logging.info("Name ------------- Center ----- Rotation")
    for el in result:
        name = "{:<13}".format(el.name)
        center = "[{:<6}  {:<6}]".format(str(round(el.center[0], 1)), str(round(el.center[1], 1)))
        logging.info(f"""{name} {center}    {el.rotation}""")

    if cliargs.save_json_result:
        logging.info("-" * 40)
        dump_elements(os.path.join("log", "main", "board.json"), result)
        logging.info(f"""Detected elements saved to {os.path.join("log", "main", "board.json")}""")
    if cliargs.draw_elements:
        logging.info("-" * 40)
        save_detect_img(img, result, os.path.join("log", "main", "out_image.png"))
        logging.info(f"""Image with elements saved to {os.path.join("log", "main", "out_image.png")}""")

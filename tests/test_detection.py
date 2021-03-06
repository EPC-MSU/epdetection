import os
import sys
import json
import logging
import unittest

import cv2
import numpy as np

from detection.detect import detect_elements, detect_BGA, detect_label
from detection.detect import FakeGuiConnector
from epcore.elements.board import Board

from detection.utils import save_detect_img

"""
Run under virtual enviroment in top folder (epdetection):

python -m unittest discover tests
"""
DRAW_IMAGES = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("--%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def compare_elements(trueres, result, thd_in_pixels=12):
    tp = 0
    fp = 0
    total = len(trueres)
    for el1 in trueres:
        math = False
        for el2 in result:
            if el1.name == el2.name and el1.rotation == el2.rotation:
                if np.mean(np.abs(np.array(el1.center) - np.array(el2.center))) < thd_in_pixels:
                    math = True
        if math:
            tp += 1
        else:
            fp += 1
    return tp, fp, total


def load_test_data(test_folder_name):
    img = cv2.imread(os.path.join("tests", test_folder_name, "image.png"))

    with open(os.path.join("tests", test_folder_name, "board.json"), "r") as f:
        board_json = json.load(f, encoding="utf8")
    board = Board.create_from_json(board_json)
    if DRAW_IMAGES:
        if not os.path.exists(os.path.join("log")):
            os.makedirs(os.path.join("log"))
        if not os.path.exists(os.path.join("log", "tests")):
            os.makedirs(os.path.join("log", "tests"))
        if not os.path.exists(os.path.join("log", "tests", test_folder_name)):
            os.makedirs(os.path.join("log", "tests", test_folder_name))
        cv2.imwrite(os.path.join("log", "tests", test_folder_name, "image.png"), img)
        save_detect_img(img.copy(), board.elements, os.path.join("log", "tests", test_folder_name, "correct.png"))
    return img, board.elements


class DetectTests(unittest.TestCase):
    pass_trh = 0.2

    def test_elements_1(self, folder_name="elm_test1"):
        img, trueres = load_test_data(folder_name)
        result = detect_elements(FakeGuiConnector(), img)
        if DRAW_IMAGES:
            save_detect_img(img.copy(), result, os.path.join("log", "tests", folder_name, "detected.png"))
        tp, fp, total = compare_elements(result, trueres)
        score = tp / (total + fp)
        logger.info(f"TEST[1] Clf quality = {score:.2f} [Small board | {folder_name}]")
        self.assertTrue(score > self.pass_trh)

    def test_elements_2(self, folder_name="elm_test2"):
        img, trueres = load_test_data(folder_name)
        result = detect_elements(FakeGuiConnector(), img)
        if DRAW_IMAGES:
            save_detect_img(img.copy(), result, os.path.join("log", "tests", folder_name, "detected.png"))
        tp, fp, total = compare_elements(result, trueres)
        score = tp / (total + fp)
        logger.info(f"TEST[2] Clf quality = {score:.2f} [Sparse board | {folder_name}]")
        self.assertTrue(score > self.pass_trh)

    def test_elements_3(self, folder_name="elm_test3"):
        img, trueres = load_test_data(folder_name)
        result = detect_elements(FakeGuiConnector(), img)
        if DRAW_IMAGES:
            save_detect_img(img.copy(), result, os.path.join("log", "tests", folder_name, "detected.png"))
        tp, fp, total = compare_elements(result, trueres)
        score = tp / (total + fp)
        logger.info(f"TEST[3] Clf quality = {score:.2f} [Blue board | {folder_name}]")
        self.assertTrue(score > self.pass_trh)

    def test_bga_1(self, folder_name="bga_test1"):
        img, trueres = load_test_data(folder_name)
        result = detect_BGA(FakeGuiConnector(), img)
        if DRAW_IMAGES:
            save_detect_img(img.copy(), result, os.path.join("log", "tests", folder_name, "detected.png"))
        tp, fp, total = compare_elements(result, trueres, thd_in_pixels=10)
        score = tp / (total + fp)
        logger.info(f"TEST[4] Clf quality = {score:.2f} [BGA Detection | {folder_name}]")
        self.assertTrue(score > self.pass_trh)

    def test_calibration_1(self, folder_name="calibration_test1"):
        img, trueres = load_test_data(folder_name)
        result = detect_label(FakeGuiConnector(), img)
        if DRAW_IMAGES:
            save_detect_img(img.copy(), result, os.path.join("log", "tests", folder_name, "detected.png"))
        tp, fp, total = compare_elements(result, trueres)
        score = tp / (total + fp)
        logger.info(f"TEST[5] Clf quality = {score:.2f} [Calibration chessboard | {folder_name}]")
        self.assertTrue(score > self.pass_trh)

    def test_board_save(self, folder_name="elm_test1"):
        img, trueres = load_test_data(folder_name)
        board_json = Board(trueres).to_json()
        with open(os.path.join("log", "tests", folder_name, "test_board_save.json"), "w") as dump_file:
            json.dump(board_json, dump_file, separators=(",", ":"), indent=2)
        logger.info(f"TEST[6] Board json saved successfully [{folder_name} | test_board_save.json]")
        self.assertTrue(True)  # Test pass witout errors


if __name__ == "__main__":
    unittest.main()

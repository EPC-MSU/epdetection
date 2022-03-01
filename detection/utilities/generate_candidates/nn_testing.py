import json
import os

import numpy as np
import cv2
import jsonpickle

# import os
# import logging
# logging.getLogger("tensorflow").disabled = True
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# print("Loading tensorflow...")

from utilities.generate_candidates.detection_emulated import detect_by_nn


def convert_board(n=1):
    img = cv2.imread(f"utilities//tests//board_{n}//stitched.png")
    cv2.imwrite(f"utilities//tests//board_{n}//board_{n}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])


def load_board(n=1):
    return cv2.imread(f"utilities//tests//board_{n}//board_{n}.jpg")


def load_board_png(n=1):
    return cv2.imread(f"utilities//tests//board_{n}//stitched.png")


def load_elements_jsons(path):
    elements_json_list = []
    for top, dirs, files in os.walk(path):
        for file in files:
            if ".json" in file:
                name = os.path.join(top, file)
                with open(name) as f:
                    data = json.load(f)
                elements_json_list.append(data)
    return elements_json_list


def draw_true_cors(det, path="utilities//tests//board_1"):
    convert_board(n=1)
    board = load_board(n=1)
    elements_json_list = load_elements_jsons(path)
    color = (255, 0, 0)  # blue
    thickness = 1
    for js_el in elements_json_list:
        # c = det.names.index(js_el["name"])
        y1, x1 = js_el["bounding_zone"][0]
        y2, x2 = js_el["bounding_zone"][2]
        # shape = det.patterns[c].shape
        # y1, x1 = js_el["top_left_cor"]
        # x2 = x1 + shape[1]
        # y2 = y1 + shape[0]
        board = cv2.rectangle(board, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        # board = cv2.circle(board, (int(x1), int(y1)), 4, color, thickness)
    cv2.imwrite("utilities//tests//board_1//board_test.jpg", board)
    return board


def draw_predicted_cors(result, board, det, path="utilities//tests//board_1"):
    color = (0, 255, 0)
    thickness = 1
    for y1, x1, c, p in result:
        shape = det.patterns[c].shape
        # board = cv2.circle(board, (int(x1), int(y1)), 4, color, thickness)
        x2 = x1 + shape[1]
        y2 = y1 + shape[0]
        print(f"Adding element: ({float(x1)}, {float(y1)})")
        board = cv2.rectangle(board, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    cv2.imwrite("utilities//tests//board_1//board_predict.jpg", board)


if __name__ == "__main__":
    image = load_board_png()
    image = (image / 255.0).astype(np.float32)

    with open("utilities//tests//det_dump.json") as det_dump:
        frozen = json.load(det_dump)
    det = jsonpickle.decode(frozen)
    det.trh_corr_mult = 1.3

    only_pat_ids = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33,
                    34, 35, 36, 37,
                    38, 39, 40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                    66, 67, 68, 69,
                    70, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]

    result = detect_by_nn(image, det, find_rotations=True, only_pat_ids=only_pat_ids, debug_dir=None, find_one=False)
    print(f"Result len = {len(result)}")
    board = draw_true_cors(det, "utilities//tests//board_1")
    draw_predicted_cors(result, board, det)
    print("Cors was draw")

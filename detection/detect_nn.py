import os
import logging
import time

import cv2
import numpy as np
import torch
import tensorflow as tf
from detection.utilities.nn_train import New_Model

# from sklearn.mixture import GaussianMixture
# logging.getLogger("tensorflow").disabled = True
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# class PredictCallback(tf.keras.callbacks.Callback):
#     def __init__(self, logger, total_batches):
#         super().__init__()
#         self.logger = logger
#         self.total_batches = total_batches
#
#     def on_predict_batch_end(self, batch, logs=None):
#         self.logger.progressSignal_find4.emit(30 + int(30 * batch / self.total_batches))


class Stamp:
    def __init__(self):
        super().__init__()
        self.t_start = time.time()
        self.t_end = None

    def count(self, name):
        self.t_end = time.time()
        logging.debug(f"{name}  Time: {self.t_end - self.t_start}")
        self.t_start = self.t_end


def detect_by_one_model(rgb_image, det, model_info, non_overlap_hyp, find_one):
    """
    This function return elements list in format:
        (top_left_coordinate_y, top_left_coordinate_x, class_id, probability)

    Input:
    model_info -- from dumps model_info
    rgb_image -- image to scan
    non_overlap_hyp -- list of hypothesis of elements like
        (center_x, center_y, unknown_data, class_by_mathTemplate, matchTemplate_Probability)
    """
    t = Stamp()
    model_path = os.path.join(model_info["path"], model_info["modelname"]) + ".h5"
    start_mode = model_info["preparingcode"]["start_mode"]
    end_mode = model_info["preparingcode"]["end_mode"]
    # classes = model_info["classes"]  # flake
    data = model_info["preparingcode"]["additional"]
    classes_groups = model_info["classes_groups"]
    classes_groups_list = model_info["classes_groups_list"]

    # model_path, mode, classes, classes_extended = model_info
    # start_mode, end_mode, data = mode
    cv2.imwrite(os.path.join("log", "scanzone.jpg"), rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpg_image = cv2.imread(os.path.join("log", "scanzone.jpg"))
    t.count("save jpg")
    # Select hypothesis specified for this model
    specified_hyp = [hyp for hyp in non_overlap_hyp if hyp[4] in classes_groups_list]
    t.count("specified_hyp")

    specified_hyp = sorted(specified_hyp, key=lambda tup: tup[5], reverse=True)
    specified_hyp = specified_hyp[:int(len(specified_hyp) // 3)]

    if len(specified_hyp) == 0:
        return []

    if find_one and (len(specified_hyp) >= 150):
        specified_hyp = sorted(specified_hyp, key=lambda tup: tup[5], reverse=True)
        specified_hyp = specified_hyp[:150]

    logging.debug(f"--> Loading model {str(model_path)}...")
    model = New_Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.debug(f"Model {str(model_path)} loaded.")
    t.count("loading model")
    # ==== Run some code by start_mode
    logging.info("Preparing images for recognize...")
    if start_mode == "cut_normal":
        candidates, candidates_img = cut_rotate_and_normalize(det, jpg_image, specified_hyp)
        logging.info(f"Candidates found: {len(candidates)}")
    elif start_mode == "cut32":
        candidates, candidates_img = cut_rotate_and_normalize(det, jpg_image, specified_hyp, x32=True)
        logging.info(f"Candidates found: {len(candidates)}")
    elif start_mode == "cut32_bw":
        candidates, candidates_img = cut_rotate_and_normalize(det, jpg_image, specified_hyp, x32=True,
                                                              bw=True)
        logging.info(f"Candidates found: {len(candidates)}")
    else:
        logging.debug("Start mode unrecognized. Model skipped.")
        return []
    t.count("Preparing images")
    # ==== Do magic
    logging.info("Predicting...")
    total_batches = int(len(candidates_img) / 32)
    try:
        predict_arr = model(candidates_img)
    except ZeroDivisionError:
        predict_arr = model.predict(candidates_img)
    del model
    # logger.progressSignal_find4.emit(60)
    logging.info("Predict done.")
    t.count("Predict")

    # ==== Run some code by end_mode
    logging.info("Ending...")
    if end_mode == "end_normal":
        # threshold = det.trh_prob
        # if find_one:
        threshold = 0  # TODO: Fix threshold
        result = end_normal(det, threshold, predict_arr, candidates, classes_groups, t)
    elif end_mode == "end_thd":
        threshold = define_threshold(det.trh_prob, data)
        if find_one:
            threshold = 0
        result = end_normal(det, threshold, predict_arr, candidates, classes_groups, t)
    else:
        logging.debug("End mode unrecognized. Model skipped.")
        return []
    return result


def extended_classes(det, cl_id):
    same_classes = []  # Extend candidate id to all this class ids
    name = str(det.names[cl_id]).replace("/0", "")
    for i in range(len(det.names)):
        if str(det.names[i]).replace("/0", "") == name:
            same_classes.append(i)
    same_classes = list(set(same_classes))
    return same_classes


def cut_rotate_and_normalize(det, jpg_image, specified_hyp, x32=False, bw=False):
    """
    return:
    (candidates, candidates_img)
    """
    candidates = []
    candidates_img = []
    for i, (cy, cx, _, _, c, p) in enumerate(specified_hyp):
        x1 = int(cx - det.patterns[c].shape[1] / 2)  # Top left x
        y1 = int(cy - det.patterns[c].shape[0] / 2)  # Top left y
        x2 = int(cx + det.patterns[c].shape[1] / 2)  # Down right x
        y2 = int(cy + det.patterns[c].shape[0] / 2)  # Down right y
        patch = jpg_image[y1:y2, x1:x2]
        patch_rotated = np.rot90(patch, det.pat_rotations[c])
        if x32:
            patch_rotated = cv2.resize(patch_rotated, (32, 32), interpolation=cv2.INTER_AREA)
        if bw:
            patch_rotated = cv2.cvtColor(patch_rotated, cv2.COLOR_BGR2GRAY)
        candidates_img.append(patch_rotated)
        candidates.append((y1, x1, c, p))
        # logger.progressSignal_find4.emit(int(30 * i / len(specified_hyp)))

    candidates_img = np.array(candidates_img)
    candidates_img = candidates_img / 255.0
    logging.debug(f"Candidates: {len(candidates_img)}")
    if bw:
        candidates_img = candidates_img[..., tf.newaxis]
    # logger.progressSignal_find4.emit(30)
    return candidates, candidates_img


def define_threshold(det_thd_prob, data_thd):
    if data_thd is None:
        logging.debug(f"Threshold for this model not found. Use user value: {det_thd_prob}")
        return det_thd_prob
    else:
        try:
            thd = float(data_thd)
            if det_thd_prob >= 0.5:
                new_thd = thd + (1 - thd) / 0.5 * (det_thd_prob - 0.5)
            else:
                new_thd = thd / 0.5 * det_thd_prob
            logging.debug(f"Redefined threshold according to user threshold: {new_thd}")
            return new_thd
        except Exception as err:
            logging.error(f"{err} Use user value: {det_thd_prob}")
            return det_thd_prob


def end_normal(det, threshold, predict_arr, candidates, classes_groups, t):
    result = []

    # nn_classes_ext = dict()  # dict with nn output number and it'sextended classes
    # for i, cl in enumerate(classes):
    #     nn_classes_ext[i + 1] = extended_classes(det, cl)
    # print(predict_arr)
    for i, prob_arr in enumerate(predict_arr):
        candidate_class = candidates[i][2]
        actual_class, prob = find_class_id(prob_arr, classes_groups, candidate_class)
        if (prob > threshold) and (actual_class is not None):
            result.append((candidates[i][0], candidates[i][1], actual_class, prob))

    t.count("First result done")
    logging.debug(f"== Intersecting elements = {len(result)}")

    result = sorted(result, key=lambda tup: tup[3], reverse=True)
    t.count("Sorted")
    result = delete_intersecting(result)
    t.count("delete_intersecting")
    logging.debug(f"== Elements after averaging = {len(result)}")
    return result


def find_class_id(prob_arr, classes_groups, candidate_class):
    prob_arr[0] = 0

    for i, (key, nn_out_classes) in enumerate(classes_groups.items()):
        if candidate_class in nn_out_classes:
            pass
        else:
            prob_arr[int(key)] = 0

    return candidate_class, max(prob_arr)


def delete_intersecting(input_list, radius=6, method="average"):
    """
    This function groped same elements record to one element

    methods: "average", "best_probability", "center_weighted_probability"
    """
    try:
        kill = np.array(input_list)
        p_groups = [[kill[0]]]  # Add first point
        kill = np.delete(kill, 0, axis=0)

        while len(kill) > 0:
            index = 0
            # Extract last added point
            to_point = np.array(p_groups)
            if len(to_point.shape) == 1:
                p = to_point[to_point.shape[0] - 1][0]
            else:
                p = to_point[to_point.shape[0] - 1][to_point.shape[1] - 1]

            # Find closet point
            min_dist = 42000
            for i in range(len(kill)):
                dst = np.sqrt((kill[i][0] - p[0]) ** 2 + (kill[i][1] - p[1]) ** 2)
                if dst < min_dist:
                    min_dist = dst
                    index = i

            # Decision about point group
            if (abs(min_dist) <= radius) and (kill[index][2] == p[2]):
                p_groups[len(p_groups) - 1].append(kill[index])
            else:
                p_groups.append([kill[index]])
            kill = np.delete(kill, index, axis=0)

        # Averaging
        p_groups = np.array(p_groups)
        result = []
        if method == "average":
            for group in p_groups:
                g = np.array(group).T
                point = (int(np.average(g[0])), int(np.average(g[1])), int(g[2][0]), np.max(g[3]))
                result.append(point)
        elif method == "best_probability":
            for group in p_groups:
                g = np.array(group).T
                best_point_index = np.argmax(g[3])  # Best by probability
                point = group[best_point_index]
                point_tuple = tuple(map(tuple, point))
                result.append(point_tuple)
        else:
            result = input_list

    except Exception as err:
        result = input_list
        logging.error(err)
    return result

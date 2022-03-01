import os

import cv2
import numpy as np
import tensorflow as tf


# import logging
# logging.getLogger("tensorflow").disabled = True
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class PredictCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_of_models, total_batches, cur_model):
        super().__init__()
        self.num_of_models = num_of_models
        self.total_batches = total_batches
        self.cur_model = cur_model
        self.progres_per_model = None
        self.progres_total = None

    def on_predict_batch_end(self, batch, logs=None):
        self.progres_per_model = batch / self.total_batches
        self.progres_total = (self.cur_model + self.progres_per_model) / self.num_of_models


def detect_by_one_model(rgb_image, det, model_disc, cur_model, num_of_models, non_overlap_hyp, find_one=False):
    """
    This function return elements list in format:
        (top_left_cordinate_y, top_left_cordinate_x, class_id, probability)

    Input:
    model_disc -- from load_nn_models_disc
    rgb_image -- image to scan
    non_overlap_hyp -- list of hypothesis of elements like
        (center_x, center_y, unknown_data, class_by_mathTemplate, matchTemplate_Probability)
    """
    model_path, mode, classes, classes_extended = model_disc
    start_mode, end_mode, data = mode
    cv2.imwrite("log//scanzone.jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpg_image = cv2.imread("log//scanzone.jpg")

    # Select hypothesis specified for this model
    specified_hyp = []
    for i, hyp in enumerate(non_overlap_hyp):
        if hyp[4] in classes_extended:
            specified_hyp.append(hyp)
    if len(specified_hyp) == 0:
        return []
    if (find_one) and (len(specified_hyp) >= 300):
        specified_hyp = sorted(specified_hyp, key=lambda tup: tup[5], reverse=True)
        specified_hyp = specified_hyp[:300]

    print(f"--> Loading model {str(model_path)}...")
    model = tf.keras.models.load_model(model_path)
    print(f"Model {str(model_path)} loaded.")

    # ==== Run some code by start_mode
    print("Preparing images for recognize...")
    if start_mode == "cut_normal":
        candidates, candidates_img, cand_img_unchge = cut_rotate_and_normalaze(det, jpg_image, specified_hyp)
        print(f"Candidates found: {len(candidates)}")
    elif start_mode == "cut32":
        candidates, candidates_img, cand_img_unchge = cut_rotate_and_normalaze(det, jpg_image, specified_hyp, x32=True)
        print(f"Candidates found: {len(candidates)}")
    elif start_mode == "cut32_bw":
        candidates, candidates_img, cand_img_unchge = cut_rotate_and_normalaze(det, jpg_image, specified_hyp, x32=True,
                                                                               bw=True)
        print(f"Candidates found: {len(candidates)}")
    else:
        print("Start mode unrecognized. Model skipped.")
        return []

    # ==== Do magic
    print("Predicting...")
    total_batches = int(len(candidates_img) / 32)
    try:
        predict_arr = model.predict(candidates_img, batch_size=32,
                                    callbacks=[PredictCallback(num_of_models, total_batches, cur_model)])
    except ZeroDivisionError:
        predict_arr = model.predict(candidates_img)
    del model
    print("Predict done.")

    # ==== Run some code by end_mode
    print("Ending...")
    if end_mode == "end_normal":
        threshold = det.trh_prob
        if find_one:
            threshold = 0
        result = end_normal(det, threshold, predict_arr, candidates, classes, cand_img_unchge)
    elif end_mode == "end_thd":
        threshold = define_threshold(det.trh_prob, data)
        if find_one:
            threshold = 0
        result = end_normal(det, threshold, predict_arr, candidates, classes.cand_img_unchge)
    else:
        print("End mode unrecognized. Model skipped.")
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


def load_nn_models_disc(det, only_pat_ids, path="models"):
    """
    Returns models discription by model filename in format:
    >models_discr list with:
     >(model_path, (start_mode, end_mode, additional_data),
        [int_class1, int_class2], [int_class1, int_class2, int_class3, ...])

    Filename example:
    start_A.end_B.0.9666.23_24_25_26.h5

    start_A -- is start modificator used for run code before predict
    end_B -- is end modificator used for run code after predict
    0.9666 -- optional threshold or any data
    23_24_25_26 -- classes
    .h5 -- correct extention

    start_mode, end_mode and at least one class must by specified.
    additional_data is None if not specified.
    classes expands by det.names available classes with rotated variants
    """
    models_disc = []
    files = [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and f.endswith(".h5"))]
    for i, fname in enumerate(files):
        try:
            model_path = os.path.join(path, fname)  # Parsing name of model
            start_mode = fname.split(".")[0]
            end_mode = fname.split(".")[1]
            class_str = fname.split(".")[-2]

            if len(fname.split(".")) > 4:
                data = ".".join(fname.split(".")[2:-2])
            else:
                data = None

            mode = (start_mode, end_mode, data)
            classes = list(map(int, class_str.split("_")))  # list of classes with rot=0 by name of model

            pat_id_to_detect = []  # What classes need to detecting
            for cl in classes:
                if cl in only_pat_ids:
                    pat_id_to_detect.append(cl)
            if len(pat_id_to_detect) == 0:
                print(f"No elements selected for the model: {fname}")
                continue
            classes_extended = []  # Expand available classes with rotated variants
            for _, cl_id in enumerate(pat_id_to_detect):
                classes_extended += extended_classes(det, cl_id)

            classes_extended = list(set(classes_extended))
            models_disc.append((model_path, mode, classes, classes_extended))
        except Exception as err:
            print(f"{err}")

    return models_disc


def cut_rotate_and_normalaze(det, jpg_image, specified_hyp, x32=False, bw=False):
    """
    return:
    (candidates, candidates_img)
    """
    candidates = []
    candidates_img = []
    cand_img_unchge = []
    for i, v in enumerate(specified_hyp):
        cy = v[0]  # Center  of element Y
        cx = v[1]  # Center  of element X
        c = v[4]  # predicted Class of Element
        p = v[5]  # matchTemplate Probability

        x1 = int(cx - det.patterns[c].shape[1] / 2)  # Top left x
        y1 = int(cy - det.patterns[c].shape[0] / 2)  # Top left y
        x2 = int(cx + det.patterns[c].shape[1] / 2)  # Down right x
        y2 = int(cy + det.patterns[c].shape[0] / 2)  # Down right y

        try:
            patch = jpg_image[y1:y2, x1:x2]
            patch_rotated = np.rot90(patch, det.pat_rotations[c])
            cand_img_unchge.append(patch_rotated)
            if x32:
                patch_rotated = cv2.resize(patch_rotated, (32, 32), interpolation=cv2.INTER_AREA)
            if bw:
                patch_rotated = cv2.cvtColor(patch_rotated, cv2.COLOR_BGR2GRAY)
            candidates_img.append(patch_rotated)
            candidates.append((y1, x1, c, p))
        except Exception:
            pass

    candidates_img = np.array(candidates_img)
    candidates_img = candidates_img / 255.0
    if bw:
        candidates_img = candidates_img[..., tf.newaxis]
    return (candidates, candidates_img, cand_img_unchge)


def define_threshold(det_thd_prob, data_thd):
    if data_thd is None:
        print(f"Threshold for this model not found. Use user value: {det_thd_prob}")
        return det_thd_prob
    else:
        try:
            thd = float(data_thd)
            if det_thd_prob >= 0.5:
                new_thd = thd + (1 - thd) / 0.5 * (det_thd_prob - 0.5)
            else:
                new_thd = thd / 0.5 * det_thd_prob
            print(f"Redefined threshold according to user threshold: {new_thd}")
            return new_thd
        except Exception as err:
            print(f"{err} Use user value: {det_thd_prob}")
            return det_thd_prob


def end_normal(det, threshold, predict_arr, candidates, classes, cand_img_unchge):
    result = []

    nn_classes_ext = dict()  # dict with nn output number and it'sextended classes
    for i, cl in enumerate(classes):
        nn_classes_ext[i + 1] = extended_classes(det, cl)

    for i, prob_arr in enumerate(predict_arr):
        candidate_class = candidates[i][2]
        actual_class, prob = find_class_id(prob_arr, nn_classes_ext, candidate_class)

        if (prob > threshold) and (actual_class is not None):
            result.append((candidates[i][0], candidates[i][1], actual_class, prob))

    print(f"== Intersecting elements = {len(result)}")

    print(f"== Elements after averaging = {len(result)}")
    result = sorted(result, key=lambda tup: tup[3], reverse=True)
    result = delete_intersecting(result)

    return result


def find_class_id(prob_arr, nn_classes_ext, candidate_class):
    prob_arr[0] = 0

    for i, (key, nn_out_classes) in enumerate(nn_classes_ext.items()):
        if candidate_class in nn_out_classes:
            pass
        else:
            prob_arr[key] = 0

    return candidate_class, max(prob_arr)


def delete_intersecting(input_list, radius=6, method="average"):
    """
    This function groped same elements record to one element

    methods: "average", "best_probability", "center_weighted_probability"
    """
    try:
        kill = np.array(input_list)
        # Add first point
        p_groups = [[kill[0]]]
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

    except Exception:
        result = input_list
    return result

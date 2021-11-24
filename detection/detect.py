"""
detect.py - main file to detection functions.
See more in doc/readme.md
"""
__author__ = "medvdanil@gmail.com (Daniil Medvedev)"
__maintainer__ = "a.p.marakulin@gmail.com (Andrey Marakulin)"

import os
import logging
from ast import literal_eval

import numpy as np
import cv2
from cv2 import matchTemplate, TM_CCOEFF_NORMED

from epcore.elements.element import Element
from epcore.elements.pin import Pin
from detection.train import extract_hogs_opencv, Detector
from detection.utils import max_angle_bin, pins_to_array, remove_intersecting, idxrot, rgb2gray, \
    lqfp_bounding, find_nearest, distance, dist2, peak_k, select_one, fitSizes, \
    yield_patches, max_rect, pins_right_edge, remove_temp_dir, FakeGuiConnector, PIX_PER_MM

TRH_MAX_RECT = 0.25
TRH_CLOSP = 0.15
POS_MAX_DEVIATION = 0.3
FICT_PINS_N = 1.5
FICT_PINS_N_MAX = 5
PIN_IN_PATTERN = 0.35

clf_paths = {
    "PCB": {"dump": os.path.join(os.path.dirname(__file__), "dumps", "clf_types.dump"),
            "csv": os.path.join(os.path.dirname(__file__), "dumps", "types.csv")},
    "BGA": {"dump": os.path.join(os.path.dirname(__file__), "dumps", "clf_bga.dump"),
            "csv": os.path.join(os.path.dirname(__file__), "dumps", "bga.csv")},
    "label": {"dump": os.path.join(os.path.dirname(__file__), "dumps", "clf_label.dump"),
              "csv:": os.path.join(os.path.dirname(__file__), "dumps", "label.csv")}}


def get_element_names_by_mode(mode: str):
    """
    Return list of string names of elements by string mode.

    Parameters
    ----------
    mode : str
        'PCB', 'BGA' or 'label'
    Returns
    -------
    names : list
        Names of available elements for specific mode
    """
    print(os.getcwd())
    types_filename = clf_paths[mode]["csv"]
    element_names = []
    with open(types_filename) as types_file:
        for line in types_file.readlines():
            str_line = line.split(",")[0]
            if str_line != "not elem\n" and str_line != "name":
                element_names.append(str_line.strip())
    return element_names


def detect_elements(gc, img, trh_prob=0.7, trh_corr_mult=1.5,
                    find_one=False, elements_offset=(0, 0),
                    debug_dir=None, det_names=None, bga_szk=None):
    """
    Detect elements like smd on input image. Use det_names to specified what's to find.

    Parameters
    ----------
    gc : GuiConnector
        GuiConnector or FakeGuiConnector - class for progressbars and signals.
    img : np.array
        RGB uint8 array. Image for detection.
    trh_prob : float
        float from 0 to 1. Threshold for detection. Uses for probability calculations.
    trh_corr_mult : float
        Preliminary detector threshold.
    find_one : bool
        One element to search or many.
    elements_offset : tuple
        (x, y) offset detected image top-left corner from full-image top-left corener. Elements coordinates returned
        in full-image coordinate system.
    debug_dir : str
        Debug directory.
    det_names : list
        List of strings with name needed to detect. Exp: ["SMB", "2-SMD"] Other types ignored.
    bga_szk : float
        I don"t shure, probably this size of BGA pins.

    Returns
    -------
    elements : list
        List of detected Elements
    """
    clf_path = clf_paths["PCB"]
    elements = _detect_all(gc, img, clf_path, trh_prob, trh_corr_mult,
                           find_one, elements_offset, debug_dir, det_names, bga_szk)
    return elements


def detect_label(gc, img):
    """
    Detect chessboard on image.

    Parameters
    ----------
    gc : GuiConnector
        GuiConnector or FakeGuiConnector - class for progressbars and signals.
    img : np.array
        RGB uint8 array. Image for detection.
    Returns
    -------
    elements : list
        List with one label or empty list
    """
    clf_path = clf_paths["label"]
    labels = _detect_all(gc, img, clf_path)
    if len(labels) > 1:
        labels = [labels[0]]
    return labels


def detect_BGA(gc, img, trh_prob=0.7, trh_corr_mult=1.5,
               find_one=False, elements_offset=(0, 0),
               debug_dir=None, det_names=None, bga_szk=None):
    """
    Detect BGA elements for input image.

    Parameters
    ----------
    gc : GuiConnector
        GuiConnector or FakeGuiConnector - class for progressbars and signals.
    img : np.array
        RGB uint8 array. Image for detection.
    trh_prob : float
        float from 0 to 1. Threshold for detection. Uses for probability calculations.
    trh_corr_mult : float
        Preliminary detector threshold.
    find_one : bool
        One element to search or many.
    elements_offset : tuple
        (x, y) offset detected image top-left corner from full-image top-left corener. Elements coordinates returned
        in full-image coordinate system.
    debug_dir : str
        Debug directory.
    det_names : list
        List of strings with name needed to detect. Exp: ["SMB", "2-SMD"] Other types ignored.
    bga_szk : float
        I don"t shure, probably this size of BGA pins.

    Returns
    -------
    elements : list
        List of detected Elements
    """
    clf_path = clf_paths["BGA"]
    elements = _detect_all(gc, img, clf_path, trh_prob, trh_corr_mult,
                           find_one, elements_offset, debug_dir, det_names, bga_szk)
    return elements


def detect_BGA_params(gc, img):
    """
    Detect rotation params of bga image.

    Parameters
    ----------
    gc : GuiConnector
        GuiConnector or FakeGuiConnector - class for progressbars and signals.
    img : np.array
        RGB uint8 array. Image for detection.
    Returns
    -------
    tuple : tuple
        tuple with detected params: angle, pitch, points
    """
    pitch_step = 0.05
    max_pitch = 2.0
    elements = detect_BGA(gc, img)
    points = pins_to_array(elements)
    if len(points) < 2:
        return 0, 1.0, points
    vecs = np.zeros((len(points), 2))

    for i in range(len(points)):
        pt_i = points[i].copy()
        points[i] = np.inf
        j = np.argmin(((points - pt_i) ** 2).sum(axis=1))
        points[i] = pt_i
        vecs[i] = points[j] - pt_i

    ang, maxv = max_angle_bin(90, vecs)
    hist, ranges = np.histogram(
        np.sqrt(np.sum(vecs * vecs, axis=1)) / PIX_PER_MM,
        bins=int(max_pitch / pitch_step + 0.5), range=(pitch_step / 2, max_pitch + pitch_step / 2))

    pitch = (np.argmax(hist) + 1) * pitch_step
    logging.debug("Pitch = %.2f", pitch)
    return ang, pitch, points


def _detect_all(gc, img, clf_path, trh_prob=0.7, trh_corr_mult=1.5,
                find_one=False, elements_offset=(0, 0),
                debug_dir=None, det_names=None, bga_szk=None):
    """
    Main detection function. Used pretrained classifier, loaded from clf_path["dump"]. Prossed search for elements.
    """
    remove_temp_dir(debug_dir, find_one)
    logging.debug(f"""Loading classifier dump {clf_path["dump"]}""")
    det = Detector()
    det.load_from_file(clf_path["dump"])
    det = _tune_det(det, find_one, bga_szk, trh_corr_mult, trh_prob)
    ids = _filter_detection_ids(det, det_names, find_one, img)

    elements = _detect_common(gc, img, det, debug_dir=None, only_pat_ids=ids)
    elements = _filter_elements(elements, elements_offset, img, find_one, (det.clf is not None))

    logging.debug("Elements found: %d.", len(elements))
    return elements


def _detect(gc, image, det, find_rotations=False, only_pat_ids=None, debug_dir=None):
    """
    Main low-level detection function. Mathching and comporations processed here.
    """
    image_rgb = (image * 255).astype(np.uint8)
    im = rgb2gray(image)
    im8 = (im * 255).astype(np.uint8)
    logging.debug("img %s, resized pattern %s", im.shape, det.resized_shape)
    logging.debug("selected patterns: %s", only_pat_ids)

    # correlation
    non_overlap_hyp = []
    if only_pat_ids is None:
        en_patterns = enumerate(det.patterns)
    else:
        en_patterns = [(i, det.patterns[i]) for i in only_pat_ids]

    non_overlap_hyp = _detect_handle_en_patterns(gc, det, en_patterns, find_rotations, im, im8, non_overlap_hyp)

    # 1 class
    if det.clf is None:

        return _detect_without_clf(debug_dir, det, image_rgb, non_overlap_hyp)

    is_elem_class = np.array([0.0 if det.patterns[pat_i] is None else 1.0
                              for pat_i in det.clf.classes_])
    pat_to_class = [0] * len(det.patterns)
    for k, cluster_id in enumerate(det.clusters):
        pat_to_class[k] = np.where(det.clf.classes_ == cluster_id)[0][0]

    # max_rect
    logging.debug("%d hypothesis", len(non_overlap_hyp))
    if len(non_overlap_hyp) == 0:
        return []

    if debug_dir:
        np.array(non_overlap_hyp).dump("%snon_overlap_hyp.dump" % debug_dir)
        logging.debug("dump saved to %snon_overlap_hyp.dump", debug_dir)

    logging.debug("extract features & calc probabilities")
    probabilities = np.zeros((0, len(det.clf.classes_)), dtype=np.float64)

    for descriptors in extract_hogs_opencv(yield_patches(im, non_overlap_hyp, det), det.resized_shape):
        probabilities = np.vstack((probabilities, det.clf.predict_proba(descriptors)))
        gc.check_interruption()

    assert probabilities.shape[0] == len(non_overlap_hyp)
    matches = []
    for k, v in enumerate(non_overlap_hyp):
        pp = probabilities[k]
        p = pp[pat_to_class[v[4]]]
        if p > det.trh_prob:
            matches.append(tuple(v) + (pp, p))

    logging.debug("find maximums among %d patches" % len(matches))
    non_overlap = max_rect(matches, mode="mean")

    # clf
    logging.debug("found %s non-overlapping patches", len(non_overlap))
    logging.debug("classify")
    result = []
    for v in non_overlap:
        i = v[0] - det.patterns[v[4]].shape[0] // 2
        j = v[1] - det.patterns[v[4]].shape[1] // 2
        pp, val_corr = v[6], v[5]
        p_i = (pp * is_elem_class).argmax()
        c = det.clf.classes_[p_i]
        p_res = pp[p_i]
        patch = im[i:i + det.patterns[v[4]].shape[0], j:j + det.patterns[v[4]].shape[1]]
        assert patch.shape == det.patterns[v[4]].shape[:2]

        max_corr = 0
        for k, cluster_id in enumerate(det.clusters):
            cur_corr = -1.
            if cluster_id == det.clusters[c] and det.patterns[k] is not None and \
                    det.patterns[k].shape == det.patterns[v[4]].shape and \
                    det.pat_rotations[k] == det.pat_rotations[v[4]]:
                cur_corr = matchTemplate(patch, det.patterns[k], TM_CCOEFF_NORMED)[0][0]
            if cur_corr > max_corr:
                c = k
                max_corr = cur_corr
        if max_corr < det.parameters[c][1] * det.trh_corr_mult:
            continue

        # logging.debug("+ %d, %d corr=%.3f, c=%d" % (v[1], v[0], val_corr, c))
        # logging.debug("  p of classes: %s", ", ".join("%.2f" % p for p in pp))
        result.append((i, j, c, p_res))
        if debug_dir is not None:
            cv2.imsave(debug_dir + "c%d_%04d_%04d_%.2f_p%.2f_r%d.bmp" %
                       (det.pat_orig[c], v[1], v[0], val_corr, p_res, det.pat_rotations[c]),
                       np.rot90(image[i:i + det.patterns[c].shape[0],
                                j:j + det.patterns[c].shape[1]],
                                -det.pat_rotations[c]))

    logging.debug("result: %d patches", len(result))
    return result


def _detect_without_clf(debug_dir, det, image_rgb, non_overlap_hyp):
    non_overlap = max_rect(non_overlap_hyp, mode="mean")
    result = []
    for v in non_overlap:
        i = v[0] - det.patterns[v[4]].shape[0] // 2
        j = v[1] - det.patterns[v[4]].shape[1] // 2
        c = v[4]
        result.append((i, j, c, v[-1]))
        logging.debug("+ %d, %d corr=%.3f, c=%d" % (v[1], v[0], v[-1], c))
        if debug_dir is not None:
            filename = debug_dir + "c%d_%04d_%04d_%.2f_r%d.bmp" % (
                c, v[1], v[0], v[-1], det.pat_rotations[c])
            try:
                img_rot = np.rot90(image_rgb[i:i + det.patterns[c].shape[0],
                                   j:j + det.patterns[c].shape[1]],
                                   -det.pat_rotations[c])
                cv2.imsave(filename, img_rot)
            except Exception as e:
                logging.error("Cant save %s.", filename)
                logging.error(e)
    return result


def _detect_handle_en_patterns(gc, det, en_patterns, find_rotations, im, im8, non_overlap_hyp):
    gc.send_num_stages(len(en_patterns))
    for pat_i, pat in en_patterns:
        if pat is None:
            continue
        if det.pat_rotations[pat_i] != 0 and not find_rotations:
            continue
        if im.shape[0] < pat.shape[0] or im.shape[1] < pat.shape[1]:
            continue
        gc.check_interruption()
        trh_corr = det.parameters[pat_i][1] * det.trh_corr_mult
        pat = (rgb2gray(pat) * 255).astype(np.uint8)
        corr = matchTemplate(im8, pat, TM_CCOEFF_NORMED)
        shape07_half = int(pat.shape[0] * 0.35), int(pat.shape[1] * 0.35)

        matches = [(i + pat.shape[0] // 2, j + pat.shape[1] // 2,
                    shape07_half[0], shape07_half[1], pat_i, corr[i, j])
                   for i, j in peak_k(corr, trh_corr, k=3)]

        logging.debug("p%d: %d peaks" % (pat_i, len(matches)))
        non_overlap_hyp += max_rect(matches, trh=TRH_MAX_RECT)
        gc.send_next_stage()
    gc.reset_progress()
    return non_overlap_hyp


def _closest_peak(p0, img, det, only_pat_id=None, trh_mult=0.7, debug_dir=None):
    """
    Find patch which closest to p0
    Returns left top corner or (-1, -1) if not found
    """
    pos = (-1, -1)
    prob = 0.
    trh_old = det.trh_corr_mult, det.trh_prob
    det.trh_corr_mult *= trh_mult
    det.trh_prob *= trh_mult
    for v in _detect(FakeGuiConnector(), img, det, only_pat_ids=[only_pat_id], debug_dir=debug_dir):
        if pos[0] == -1 or v[3] > prob + TRH_CLOSP or \
                dist2((v[0], v[1]), p0) < dist2(pos, p0) and prob - v[3] < TRH_CLOSP:
            pos = v[0], v[1]
            prob = max(v[3], prob)
    det.trh_corr_mult, det.trh_prob = trh_old
    return pos


def _detect_2sides(image, det, i, j, pat_i, pinw, pin_y_offset,
                   patch_n_pins, edge_pins_n, debug_dir=None):
    im = rgb2gray(image)
    shp = det.patterns[pat_i].shape

    offset1 = int(pin_y_offset + 0.5)
    offset2 = int(pin_y_offset + pinw + 0.5)
    pin_img = det.patterns[pat_i][offset1:offset2, int(shp[1] * (1 - PIN_IN_PATTERN)):]
    pin_img = (pin_img + pin_img[::-1]) / 2.
    corners = [(i, j)]

    crop_l = 0
    crop_t = max(i - int(shp[0] * (POS_MAX_DEVIATION + 1)), 0)
    crop_b = min(i + int(shp[0] * POS_MAX_DEVIATION), image.shape[0])
    crop_r = min(j + int(shp[1] * POS_MAX_DEVIATION), image.shape[1])
    img_crp = image[crop_t:crop_b, crop_l:crop_r]
    pos = _closest_peak((0, 0), img_crp[:, ::-1], det, only_pat_id=pat_i, debug_dir=debug_dir)
    if pos[0] == -1:
        return
    corners.append((crop_t + pos[0] + shp[0],
                    crop_l + img_crp.shape[1] - pos[1] - 1 - shp[1]))
    crop_l = max(corners[-1][1] - int(shp[1] * POS_MAX_DEVIATION), 0)
    crop_r = min(corners[-1][1] + int(shp[1] * (1 + POS_MAX_DEVIATION)), image.shape[1])
    crop_t = 0
    img_crp = image[crop_t:crop_b, crop_l:crop_r]
    pos = _closest_peak((0, 0), img_crp[::-1, ::-1], det, only_pat_id=pat_i, debug_dir=debug_dir)
    if pos[0] == -1:
        return
    corners.append((crop_t + img_crp.shape[0] - pos[0] - 1 - shp[0],
                    crop_l + img_crp.shape[1] - pos[1] - 1 - shp[1]))

    crop_l = max(corners[0][1] - int(shp[1] * (1 + POS_MAX_DEVIATION)), 0)
    crop_r = min(corners[0][1] + int(shp[1] * (1 + POS_MAX_DEVIATION)), image.shape[1])
    crop_t = max(corners[-1][0] - int(shp[0] * POS_MAX_DEVIATION), 0)
    crop_b = min(corners[-1][0] + int(shp[0] * (1 + POS_MAX_DEVIATION)), image.shape[0])
    img_crp = image[crop_t:crop_b, crop_l:crop_r]
    pos = _closest_peak((0, 0), img_crp[::-1, :], det, only_pat_id=pat_i, debug_dir=debug_dir)

    if pos[0] == -1:
        return
    corners.append((crop_t + img_crp.shape[0] - pos[0] - 1 - shp[0],
                    crop_l + pos[1] + shp[1]))
    corners[1], corners[3] = corners[3], corners[1]  # ccw order
    corners = np.array(corners, dtype=np.float32)
    logging.debug("Found 4 corners")

    corners = np.round(corners).astype(np.int32)
    first_pin_shift = (pinw - pin_y_offset) % pinw
    p0p1_len = (distance(corners[0], corners[1]) + distance(corners[2], corners[3])) / 2
    n_pins = (p0p1_len - 2 * shp[0] + 2 * patch_n_pins * pinw - 2 * first_pin_shift) / pinw
    n_pins = find_nearest(edge_pins_n, n_pins)

    pins_right = pins_right_edge(im, pin_img, pin_y_offset,
                                 shp, corners[0], corners[1], pinw, n_pins, patch_n_pins=patch_n_pins)
    pins_right = [Pin(x=x, y=y) for x, y in pins_right]

    p0 = corners[3][0], image.shape[1] - corners[3][1] - 1
    p1 = corners[2][0], image.shape[1] - corners[2][1] - 1

    pins_left_mir = pins_right_edge(im[:, ::-1], pin_img, pin_y_offset,
                                    shp, p0, p1, pinw, n_pins, patch_n_pins=patch_n_pins)
    pins_left_mir = [Pin(x=x, y=y) for x, y in pins_left_mir]

    logging.debug("lpins: %d, rpins: %d" % (len(pins_right), len(pins_left_mir)))
    if len(pins_right) == 0 or len(pins_left_mir) == 0:
        return
    pins_left = []
    for p in pins_left_mir:
        pins_left.append(Pin(x=p.x, y=image.shape[1] - p.y - 1))

    # reversed for ccw order
    return corners, list(reversed(pins_right)), pins_left


# i, j - is coordinates of patch in left bottom corner
def _detect_multipin(image, det, i, j, pat_i, pinw,
                     pin_y_offset, patch_n_pins, edge_pins_n, debug_dir=None):
    name = det.names[pat_i]
    r = det.pat_rotations[pat_i]
    pat_i = det.pat_orig[pat_i]
    shp = det.patterns[pat_i].shape
    cr0_offset = idxrot((shp[0] - 1, shp[1] - 1), shp, -r)
    i, j = idxrot((i + cr0_offset[0], j + cr0_offset[1]), image.shape, r)
    image = np.rot90(image, -r)
    logging.debug("Detect 2 sides:")
    result = _detect_2sides(image, det, i, j, pat_i, pinw, pin_y_offset, patch_n_pins, edge_pins_n, debug_dir=debug_dir)
    logging.debug("result: %s", bool(result))
    if result is None:
        return
    corners, pins_bottom, pins_top = result
    # try to detect LQFP
    image_r = np.rot90(image, -1)
    i, j = idxrot(
        (corners[1][0] - pinw * FICT_PINS_N, corners[1][1] - pinw * FICT_PINS_N - shp[0]), image.shape, 1)
    crop_t = max(int(i - shp[0] / 2 - pinw * FICT_PINS_N_MAX + 0.5), 0)
    crop_b = min(int(i + shp[0] / 2 + pinw * FICT_PINS_N_MAX + 0.5), image_r.shape[0])
    crop_l = max(int(j - shp[1] / 2 - pinw * FICT_PINS_N_MAX + 0.5), 0)
    crop_r = min(int(j + shp[1] / 2 + pinw * FICT_PINS_N_MAX + 0.5), image_r.shape[1])
    logging.debug("SOIC detected, try to detect LQFP")

    pos = -1, -1
    result = None
    if "DIP" not in name:  # this condition must be in types.csv
        pos = _closest_peak((i - crop_t - shp[0] / 2, j - crop_l - shp[1] / 2),
                            image_r[crop_t: crop_b, crop_l: crop_r], det, only_pat_id=pat_i, debug_dir=debug_dir)
        i, j = np.array(pos) + (crop_t, crop_l) + shp
    if pos[0] != -1:
        result = _detect_2sides(image_r, det, i, j, pat_i, pinw,
                                pin_y_offset, patch_n_pins, edge_pins_n, debug_dir=debug_dir)
        logging.debug("result: %s", bool(result))

    pins_left = []
    pins_right = []
    if result is not None:
        corners_rr, pins_top_r, pins_bottom_r = result
        corners_r = []
        # pins_left, pins_right = pins_top_r, pins_bottom_r
        for p in pins_top_r:
            x, y = idxrot((p.x, p.y), image_r.shape, -1)
            pins_left.append(Pin(x=x, y=y))
        for p in pins_bottom_r:
            x, y = idxrot((p.x, p.y), image_r.shape, -1)
            pins_right.append(Pin(x=x, y=y))
        for p in corners_rr:
            corners_r.append(idxrot(p, image_r.shape, -1))
        # make corners for lqfp
        corners_r = np.array(corners_r[3:] + corners_r[:3])
        corners = np.array(corners)
        dcr = np.abs(corners - corners_r).mean()
        corners = (corners + corners_r) / 2
        corners = lqfp_bounding(corners, shp[1] - patch_n_pins * pinw - dcr / 2, -0.1 * shp[1])
        name = name.split("&")[0]
    else:
        name = name.split("&")[-1]

    h_pins = (len(pins_bottom) + len(pins_top))  # // 2 * 2
    w_pins = (len(pins_left) + len(pins_right))  # // 2 * 2

    rotated_pins = []
    for p in pins_bottom + pins_left + pins_top + pins_right:
        y, x = idxrot((p.x, p.y), image.shape, -r)
        rotated_pins.append(Pin(x=x, y=y))

    rotated_corners = np.array([idxrot(pt, image.shape, -r) for pt in corners])
    rotated_corners[:, [0, 1]] = rotated_corners[:, [1, 0]]

    elem = Element(pins=rotated_pins, set_automatically=True,
                   name=name, bounding_zone=rotated_corners, rotation=r)
    if r > 0 and w_pins > 0 and h_pins > 0:
        r = 0
        w_pins, h_pins = h_pins, w_pins
    elem = fitSizes(elem, h_pins, w_pins)
    return elem


def _detect_common(gc, image, det, debug_dir=None, only_pat_ids=None):
    pins_y_offset = []
    pins_w = []
    patch_n_pins = []
    elements = []
    pin_lists = []
    edge_pins_n = []
    multipin = np.zeros(len(det.patterns), dtype=bool)
    for pat_i, par in enumerate(det.parameters):
        pin_lists.append([])
        if det.patterns[pat_i] is None:
            continue
        if par[2] == "multipin":
            multipin[pat_i] = True
            continue
        for pin in literal_eval(par[2]):
            x, y = idxrot(pin, 1.0, -det.pat_rotations[pat_i])
            pin_lists[-1].append(Pin(x=x, y=y))

    for pat_i, pat in enumerate(det.patterns):
        if pat is None or det.parameters[pat_i][2] != "multipin":
            pins_y_offset.append(0)
            patch_n_pins.append(0)
            pins_w.append(0)
            edge_pins_n.append([])
        else:
            pins_y_offset.append(float(det.parameters[pat_i][3]))
            pins_w.append(float(det.parameters[pat_i][4]))
            patch_n_pins.append(int(det.parameters[pat_i][5]))
            edge_pins_n.append(sorted(literal_eval(det.parameters[pat_i][6])))

    if det.clf is None:
        _detect_common_return_elements(debug_dir, det, elements, gc, image, only_pat_ids, pat_i, pin_lists)
        return elements

    _detect_common_handle_detect(debug_dir, det, edge_pins_n, elements,
                                 gc, image, only_pat_ids, patch_n_pins, pin_lists,
                                 pins_w, pins_y_offset)
    return elements


def _detect_common_return_elements(debug_dir, det, elements, gc, image, only_pat_ids, pat_i, pin_lists):
    orig_pat = det.patterns[0]
    tmp = []
    szk_max = 1.0
    maxv = -1
    maxi = 0
    sizes_els = []
    if det.bga_szk is not None:  # TODO: I have no idea how this work with pat_i
        szk_list = [det.bga_szk]
    else:
        if len(det.parameters[pat_i]) > 3:
            szk_list = literal_eval(det.parameters[pat_i][3])
        else:
            szk_list = [1.0]

    for i, szk in enumerate(szk_list):
        det.patterns[0] = cv2.resize(orig_pat, (0, 0), fx=szk, fy=szk)
        els = _detect(gc, image, det, find_rotations=True,
                      debug_dir=debug_dir, only_pat_ids=only_pat_ids)

        val = np.sum(np.array(els)[:, -1] > 0.9) / len(els) if len(els) > 0 else 0.0
        sizes_els.append(els)
        tmp.append(val)
        if val > maxv:
            maxv = val
            szk_max = szk
            maxi = i

    det.bga_szk = szk_max
    if hasattr(gc, "BGASizeKSignal"):
        gc.BGASizeKSignal.emit(szk_max)
    logging.debug("elements size: %s, %s, %s", szk_max, maxv, tmp)
    det.patterns[0] = orig_pat
    for i, j, res_i, p in sizes_els[maxi]:
        shp = det.patterns[res_i].shape
        shp = np.round(np.array(shp, dtype=np.float32) * szk_max).astype(np.int32)
        pins = []
        for pin in pin_lists[res_i]:
            x = i + shp[0] * pin.x
            y = j + shp[1] * pin.y
            pins.append(Pin(x=y, y=x))

        bz = np.array([(i, j), (i + shp[0], j), (i + shp[0], j + shp[1]), (i, j + shp[1])])
        bz_xy = bz
        bz_xy[:, [0, 1]] = bz_xy[:, [1, 0]]
        elem = Element(pins=pins, set_automatically=True,
                       name=det.names[res_i], bounding_zone=bz_xy, rotation=det.pat_rotations[res_i])
        elem = fitSizes(elem)
        elem.width *= szk_max
        elem.height *= szk_max
        elements.append(elem)


def _detect_common_handle_detect(debug_dir, det, edge_pins_n, elements, gc, image,
                                 only_pat_ids, patch_n_pins, pin_lists, pins_w, pins_y_offset):
    for i, j, res_i, p in _detect(gc, image, det, find_rotations=True, debug_dir=debug_dir, only_pat_ids=only_pat_ids):
        if det.parameters[res_i][2] == "multipin":
            elem = _detect_multipin(image, det, i, j, res_i,
                                    pins_w[res_i], pins_y_offset[res_i],
                                    patch_n_pins[res_i], edge_pins_n[res_i], debug_dir=debug_dir)
            if elem is not None:
                elements.append(elem)
            continue
        shp = det.patterns[res_i].shape
        pins = []
        for pin in pin_lists[res_i]:
            x = i + shp[0] * pin.x
            y = j + shp[1] * pin.y
            pins.append(Pin(x=y, y=x))

        bz = np.array([(i, j), (i + shp[0], j), (i + shp[0], j + shp[1]), (i, j + shp[1])])
        bz_xy = bz
        bz_xy[:, [0, 1]] = bz_xy[:, [1, 0]]
        elem = Element(pins=pins, set_automatically=True,
                       name=det.names[res_i], bounding_zone=bz_xy, rotation=det.pat_rotations[res_i])
        elem = fitSizes(elem)
        elements.append(elem)


def _tune_det(det, find_one, bga_szk, trh_corr_mult, trh_prob):
    """
    Change detector params after load, replace with settings params
    """
    det.bga_szk = bga_szk
    det.trh_prob = trh_prob
    det.trh_corr_mult = trh_corr_mult
    det.trh_max_rect = TRH_MAX_RECT
    if find_one:
        det.trh_max_rect *= 1.8
    return det


def _filter_elements(elements, elements_offset, img, find_one, by_classifier):
    if len(elements) == 0:
        return []
    if find_one:
        elements = [elements[select_one(elements, img.shape[:2])]]
    for elem in elements:
        for i in range(len(elem.bounding_zone)):
            elem.bounding_zone[i] += np.array(elements_offset)
        for pin in elem.pins:
            pin.x += elements_offset[0]
            pin.y += elements_offset[1]
    if by_classifier:
        logging.debug("Removing intersecting elements (total %d)...", len(elements))
        elements = remove_intersecting(elements)
    return elements


def _filter_detection_ids(det, det_names, find_one, img):
    ids = list(range(len(det.patterns)))
    if det_names is not None:
        ids = []
        for i, n in enumerate(det.names):
            if n in det_names:
                ids.append(i)

    # don"t consider small patches
    filtered_ids = []
    for i in ids:
        if len(det.parameters[i]) >= 3:
            m_p = (det.parameters[i][2] == "multipin")
        else:
            m_p = False
        if det.patterns[i] is not None and (np.prod(det.patterns[i].shape[:2]) / np.prod(img.shape[:2]) > 0.1 or m_p):
            filtered_ids.append(i)
    if find_one:
        ids = filtered_ids
    return ids

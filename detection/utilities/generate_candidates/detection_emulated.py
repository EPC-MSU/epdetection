from typing import Iterable, List
import json
import numpy as np
import cv2
from cv2 import matchTemplate, TM_CCOEFF_NORMED

from skimage.transform import AffineTransform
from scipy.optimize import fmin_slsqp
from sklearn.externals import joblib
from skimage.io import imsave
from scipy.signal import correlate2d
from vision.train import train
from os import path, mkdir
from vision.utils import max_angle_bin, pins_to_array, EmptyLogger, remove_intersecting, idxrot, rgb2gray, \
    lqfp_bounding, find_nearest, distance, rect_metrics, dist2, fourier_sum, peak_k, polygon_center, polygon_square
from utilities.generate_candidates.detection_nn_emulated import load_nn_models_disc, detect_by_one_model, \
    extended_classes

from ast import literal_eval
from element import Element, Pin
from scipy.optimize import minimize_scalar
from scipy.fftpack import fft

classifier_image_shape = (60, 60)

trh_max_rect = 0.25
trh_closp = 0.15
pos_max_deviation = 0.3

min_pins = 2
pin_peak_trh = 0.05  # 0.25
fict_pins_n = 1.5
fict_pins_n_max = 5
pin_in_pattern = 0.35  # len of pin in pattern (fraction)
peak_fourier_trh = 0.85
pins_x_offset_mult = 0.56
skew_trh = 1.5

clf_dump_pattern = "clf_%s.dump"
elements_dump_file_name = "elements.json"


def loadSizesElements(elements):
    pix_per_mm = 23
    with open(path.join("svg", "sizes.csv")) as file:
        lines = file.readlines()
    for e in elements:
        for line in lines:
            if line.split(",")[0].strip() == e.reprName():
                e.width = float(line.split(",")[1].strip()) * pix_per_mm
                e.height = float(line.split(",")[2].strip()) * pix_per_mm
                break


def dump_elements(file_path: str, elements: Iterable[Element]) -> None:
    d = {
        "elements": [element.to_json() for element in elements]
    }
    with open(file_path, "w") as dump_file:
        json.dump(d, dump_file, separators=(",", ":"))  # Trim spaces


def load_elements(file_path) -> List[Element]:
    with open(file_path, "r") as dump_file:
        d = json.load(dump_file)
    elements = list(map(Element.from_json, d["elements"]))
    return elements


def select_one(elements, shape):
    max_d = distance(shape, (0, 0)) / 2
    max_score = 0.
    max_i = 0
    for i, elem in enumerate(elements):
        s = polygon_square(elem.bounding_zone) / (shape[0] * shape[1])
        d = 1. - distance(
            polygon_center(elem.bounding_zone),
            (shape[0] / 2., shape[1] / 2.)
        ) / max_d
        if (s + d) / 2 > max_score:
            max_score = (s + d) / 2
            max_i = i
    return max_i


def max_rect(matches, trh=0.0, mode="all"):
    if mode != "all" and mode != "mean":
        raise ValueError("mode must be 'all' or 'mean'")
    if len(matches) == 0:
        return matches
    min_b = list(matches[0][0:2])
    max_b = min_b.copy()
    min_val = matches[0][-1]
    for v in matches:
        min_b[0] = min(v[0] - v[2], min_b[0])
        min_b[1] = min(v[1] - v[3], min_b[1])
        max_b[0] = max(v[0] + v[2], max_b[0])
        max_b[1] = max(v[1] + v[3], max_b[1])
        min_val = min(v[-1], min_val)
    if min_b[0] < 0 or min_b[1] < 0:
        raise ValueError("rectangle is out of boundaries")

    offset = np.array(min_b)
    m = np.ones(max_b - offset, dtype=type(matches[0][-1])) * min_val
    for v in matches:
        m[v[0] - v[2] - offset[0]: v[0] + v[2] - offset[0], v[1] - v[3] - offset[1]: v[1] + v[3] - offset[1]] = \
            np.maximum(
                m[v[0] - v[2] - offset[0]: v[0] + v[2] - offset[0], v[1] - v[3] - offset[1]: v[1] + v[3] - offset[1]],
                v[-1])

    res = []
    if mode == "all":
        for v in matches:
            if m[v[0] - v[2] - offset[0]: v[0] + v[2] - offset[0],
               v[1] - v[3] - offset[1]: v[1] + v[3] - offset[1]].max() - v[-1] <= trh:
                res.append(v)
    if mode == "mean":
        max_matches = []
        for v in matches:
            if m[v[0] - v[2] - offset[0]: v[0] + v[2] - offset[0],
               v[1] - v[3] - offset[1]: v[1] + v[3] - offset[1]].max() - v[-1] <= trh:
                max_matches.append(v)
        # print("max_matches len", len(max_matches))
        i = 0
        while i < len(max_matches):
            v = max_matches[i]
            r, c = v[0:2]
            n = 1
            tail = max_matches[i][2:]
            j = i + 1
            while j < len(max_matches):
                if abs(v[0] - max_matches[j][0]) < tail[0] and \
                        abs(v[1] - max_matches[j][1]) < tail[3] and \
                        max_matches[j][4] == v[4]:
                    r += max_matches[j][0]
                    c += max_matches[j][1]
                    n += 1
                    if max_matches[j][-1] > tail[-1]:
                        tail = max_matches[j][2:]
                    del max_matches[j]
                else:
                    j += 1
            r = int(r / n + 0.5)
            c = int(c / n + 0.5)
            res.append((r, c) + tuple(tail))
            i += 1
    return res


def yield_patches(img, matches, det):
    for v in matches:
        i = v[0] - det.patterns[v[4]].shape[0] // 2
        j = v[1] - det.patterns[v[4]].shape[1] // 2
        yield cv2.resize(
            np.rot90(
                img[i:i + det.patterns[v[4]].shape[0], j:j + det.patterns[v[4]].shape[1]],
                -det.pat_rotations[v[4]]
            ),
            det.resized_shape[1::-1]
        )


def detect(image, det, find_rotations=False, only_pat_ids=None, debug_dir=None):
    im = rgb2gray(image)
    im8 = (im * 255).astype(np.uint8)
    print("img %s, resized pattern %s", im.shape, det.resized_shape)
    print("selected patterns: %s", only_pat_ids)

    # correlation
    non_overlap_hyp = []
    if only_pat_ids is None:
        en_patterns = enumerate(det.patterns)
    else:
        en_patterns = [(i, det.patterns[i]) for i in only_pat_ids]

    non_overlap_hyp = detect_handle_en_patterns(det, en_patterns, find_rotations, im, im8, non_overlap_hyp)
    # correlation

    # 1 class
    if det.clf is None:
        return detect_return_result(debug_dir, det, image, non_overlap_hyp)
    # 1 class

    is_elem_class = np.array([0.0 if det.patterns[pat_i] is None else 1.0
                              for pat_i in det.clf.classes_])
    pat_to_class = [0] * len(det.patterns)
    for k, cluster_id in enumerate(det.clusters):
        pat_to_class[k] = np.where(det.clf.classes_ == cluster_id)[0][0]
    # max_rect

    print("%d hypothesis", len(non_overlap_hyp))
    if len(non_overlap_hyp) == 0:
        return []

    if debug_dir:
        np.array(non_overlap_hyp).dump("%snon_overlap_hyp.dump" % debug_dir)
        print("dump saved to %snon_overlap_hyp.dump", debug_dir)

    print("extract features & calc probabilities")
    probabilities = np.zeros((0, len(det.clf.classes_)), dtype=np.float64)

    for descriptors in det.extract_batch_features(yield_patches(im, non_overlap_hyp, det), det.resized_shape):
        probabilities = np.vstack((probabilities, det.clf.predict_proba(descriptors)))

    assert probabilities.shape[0] == len(non_overlap_hyp)
    matches = []
    for k, v in enumerate(non_overlap_hyp):
        i = v[0] - det.patterns[v[4]].shape[0] // 2
        j = v[1] - det.patterns[v[4]].shape[1] // 2
        pp = probabilities[k]
        p = pp[pat_to_class[v[4]]]
        if p > det.trh_prob:
            matches.append(tuple(v) + (pp, p))

    print("find maximums among %d patches" % len(matches))
    non_overlap = max_rect(matches, mode="mean")
    # max_rect

    # clf
    print("found %s non-overlapping patches", len(non_overlap))
    print("classify")
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

        print("+ %d, %d corr=%.3f, c=%d" % (v[1], v[0], val_corr, c))
        print("  p of classes: %s", ", ".join("%.2f" % p for p in pp))
        result.append((i, j, c, p_res))
        if debug_dir is not None:
            imsave(debug_dir + "c%d_%04d_%04d_%.2f_p%.2f_r%d.bmp" %
                   (det.pat_orig[c], v[1], v[0], val_corr, p_res, det.pat_rotations[c]),
                   np.rot90(image[i:i + det.patterns[c].shape[0],
                            j:j + det.patterns[c].shape[1]],
                            -det.pat_rotations[c]))

    print("result: %d patches", len(result))
    return result


def detect_by_nn(image, det, find_rotations=False, only_pat_ids=None, debug_dir=None, find_one=False):
    # Preparing image
    # print(det.to_json())
    # frozen = jsonpickle.encode(det)
    # with open(f"virtual_boards//det_dump.json", "w") as json_file:
    #     json.dump(frozen, json_file)

    print(f"Selected patterns: {only_pat_ids}")
    print("Im here, what is that?")

    non_overlap_hyp = []
    if only_pat_ids is None:
        en_patterns = enumerate(det.patterns)
    else:
        en_patterns = [(i, det.patterns[i]) for i in only_pat_ids]

    im = rgb2gray(image)
    im8 = (im * 255).astype(np.uint8)
    print("img %s, resized pattern %s", im.shape, det.resized_shape)

    # Correlation function
    non_overlap_hyp = detect_handle_en_patterns(det, en_patterns, find_rotations, im, im8, non_overlap_hyp)

    if det.clf is None:
        return detect_return_result(debug_dir, det, image, non_overlap_hyp)

    print(f"Hypothesis: {len(non_overlap_hyp)}")
    if len(non_overlap_hyp) == 0:
        return []

    # Run neural network detection on founded candidates
    rgb_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    models_disc = load_nn_models_disc(det, only_pat_ids)  # Load models info, and use models one by one
    num_of_models = len(models_disc)
    if num_of_models > 0:
        print(f"{num_of_models} models you need found.")
    else:
        print("Models not found. Exit.")
        return []

    result = []
    for i, model_discr in enumerate(models_disc):
        result_by_one_model = detect_by_one_model(rgb_image, det, model_discr, i, num_of_models, non_overlap_hyp, False)
        [result.append(element) for _, element in enumerate(result_by_one_model)]
    if find_one:
        result = sorted(result, key=lambda tup: tup[3], reverse=True)
        result_best_prob = result[0]
        result_filter = []
        # If element with max prob is multipin, then add other multipins elements
        if det.parameters[result_best_prob[2]][2] == "multipin":
            ext_classes = extended_classes(det, result_best_prob[2])
            for res in result:
                # if det.parameters[res[2]][2] == "multipin":
                #     result_filter.append(res)
                if res[2] in ext_classes:
                    result_filter.append(res)
        else:
            result_filter.append(result_best_prob)
        result = result_filter

    return result


def detect_return_result(debug_dir, det, image, non_overlap_hyp):
    non_overlap = max_rect(non_overlap_hyp, mode="mean")
    result = []
    for v in non_overlap:
        i = v[0] - det.patterns[v[4]].shape[0] // 2
        j = v[1] - det.patterns[v[4]].shape[1] // 2
        c = v[4]
        result.append((i, j, c, v[-1]))
        print("+ %d, %d corr=%.3f, c=%d" % (v[1], v[0], v[-1], c))
        if debug_dir is not None:
            try:
                filename = debug_dir + "c%d_%04d_%04d_%.2f_r%d.bmp" % (
                    c, v[1], v[0], v[-1], det.pat_rotations[c])
                imsave(filename, np.rot90(image[i:i + det.patterns[c].shape[0],
                                          j:j + det.patterns[c].shape[1]],
                                          -det.pat_rotations[c]))
            except Exception:
                print("Can't save %s.", filename)
    return result


def detect_handle_en_patterns(det, en_patterns, find_rotations, im, im8, non_overlap_hyp):
    for pat_i, pat in en_patterns:
        if pat is None:
            continue
        if det.pat_rotations[pat_i] != 0 and not find_rotations:
            continue
        if im.shape[0] < pat.shape[0] or im.shape[1] < pat.shape[1]:
            continue
        trh_corr = det.parameters[pat_i][1] * det.trh_corr_mult
        pat = (rgb2gray(pat) * 255).astype(np.uint8)
        corr = matchTemplate(im8, pat, TM_CCOEFF_NORMED)
        shape07_half = int(pat.shape[0] * 0.35), int(pat.shape[1] * 0.35)

        matches = [(i + pat.shape[0] // 2, j + pat.shape[1] // 2,
                    shape07_half[0], shape07_half[1], pat_i, corr[i, j])
                   for i, j in peak_k(corr, trh_corr, k=3)]

        print("p%d: %d peaks" % (pat_i, len(matches)))
        non_overlap_hyp += max_rect(matches, trh=trh_max_rect)
    return non_overlap_hyp


def periodic_peaks(a, period, n):
    a.dump("a.dump")
    if len(a) < 3:
        return []

    def fourier_sum_abs(x):
        return -np.abs(fourier_sum(a, x))

    opt_res = minimize_scalar(fourier_sum_abs, method="Bounded",
                              bounds=[period - period * 0.04, period + period * 0.04],
                              options={"xatol": 0.01})
    period = opt_res.x
    offset = -np.angle(fourier_sum(a, period)) / np.pi / 2 * period % period
    print("pin spacing = %.3f, offset = %.3f" % (period, offset))
    # score = np.abs(fourier_sum(a, period)) * 2 / len(a)
    r = np.abs(fft(a))
    score = np.count_nonzero(r < r.max() * 0.2) / len(r)
    print("n = %d, score = %.3f" % (n, score))
    # import matplotlib.pyplot as plt
    # plt.plot(a); plt.show()
    if score < peak_fourier_trh:
        return []
    mid = (len(a) - 1) * 0.5
    if n % 2 == 0:
        k = np.ceil((mid - offset) / period)
    else:
        k = np.round((mid - offset) / period)
    pos = offset + k * period - (n // 2) * period
    res = np.arange(n) * period + pos
    print("%s", res)
    res = res[res > 0.5]
    res = res[res < len(a) - 0.5]
    return res


# p0 is right bottom corner of element
# p1 is right top corner of element
# returns list of pins coordinates
def pins_right_edge(img, pin, pins_y_offset, shp,
                    p0, p1, pitch, n_pins, patch_n_pins=2):
    res = []
    if p0[0] < p1[0]:
        p0, p1 = p1, p0
    # imshow(pin)
    skew = np.arctan2(p0[0] - p1[0], p0[1] - p1[1]) / np.pi * 180 % 90
    skew = 90 - skew if skew > 45 else skew
    print("skew = %.2f" % skew)
    if skew > skew_trh:
        print("skew = %.2f so large" % skew)
        cmean = (p0[1] + p1[1]) // 2
        p0 = p0[0], cmean
        p1 = p1[0], cmean
    t0 = p0[0] - shp[0] + (patch_n_pins - 1) * pin.shape[0], p0[1] - pin.shape[1]
    t1 = p1[0] + shp[0] - patch_n_pins * pin.shape[0] - pins_y_offset, p1[1] - pin.shape[1]

    crop_b = min(p0[0] - shp[0] + int((patch_n_pins + 0.5) * pin.shape[0] + 0.5), img.shape[0])
    crop_t = max(p1[0] + shp[0] - int((patch_n_pins + 0.5) * pin.shape[0] + 0.5), 0)

    def vertical_line(p0, p1, i0, i1, ln, r):
        res = []
        if p0[0] - p1[0] == 0:
            return np.array(res)
        k = (p0[1] - p1[1]) / (p0[0] - p1[0])
        for i in range(i0, i1):
            v = int((i - p0[0]) * k + p0[1] + 0.5)
            if v < ln:
                continue
            if v >= r:
                break
            res.append(v)
        return np.array(res)

    line_j = vertical_line(t0, t1, crop_t, crop_b - pin.shape[0] + 1, 0, img.shape[1] - pin.shape[1] + 1)
    # print(crop_b, crop_t, pin.shape[0], pitch)
    if len(line_j) == 0:
        return []
    crop_l = min(line_j)
    crop_r = max(line_j) + pin.shape[1]
    # img[crop_t: crop_b - pin.shape[0] + 1][(range(len(line_j)), line_j)] = 0; imshow(img);
    # imshow(img[crop_t: crop_b, crop_l: crop_r])
    # imsave("%.1f %.1f,%.1f %.1f.png" % (crop_t, crop_b, crop_l, crop_r), img[crop_t: crop_b, crop_l: crop_r])
    line_j -= crop_l
    # print(p0, p1, crop_l, crop_r, t0, t1)
    if crop_b - crop_t < pin.shape[0] or crop_r - crop_l < pin.shape[1]:
        return []
    mt = matchTemplate(img[crop_t:crop_b, crop_l:crop_r], pin, TM_CCOEFF_NORMED)
    corr = correlate2d(img[crop_t:crop_b, crop_l:crop_r], pin, mode="valid")
    corr *= mt
    corr /= corr.max()
    try:
        a = corr[(range(corr.shape[0]), line_j)]
    except IndexError:
        return []
    for peak in periodic_peaks(a, pitch, n_pins):
        res.append(Pin(
            rc=(
                crop_t + peak + pin.shape[0] // 2,
                crop_l + line_j[int(np.round(peak))] + pins_x_offset_mult * pin.shape[1]
            )))  # need move to names.csv

    print("%d pins along edge", len(res))
    return res


# find patch which closest to p0
# returns left top corner or (-1, -1) if not found
def closest_peak(p0, img, det, only_pat_id=None, trh_mult=0.7, debug_dir=None):
    pos = (-1, -1)
    prob = 0.
    trh_old = det.trh_corr_mult, det.trh_prob
    det.trh_corr_mult *= trh_mult
    det.trh_prob *= trh_mult
    for v in detect(EmptyLogger(), img, det, only_pat_ids=[only_pat_id], debug_dir=debug_dir):
        # print("dist2, 1-p:", dist2((v[0], v[1]), p0), (1 - v[3]))
        # dist2((v[0], v[1]), p0) * (1 - v[3]) ** 4 < \
        # dist2(pos, p0) * (1 - prob) ** 4:
        if pos[0] == -1 or v[3] > prob + trh_closp or \
                dist2((v[0], v[1]), p0) < dist2(pos, p0) and prob - v[3] < trh_closp:
            pos = v[0], v[1]
            prob = max(v[3], prob)
    det.trh_corr_mult, det.trh_prob = trh_old
    print("clp", pos, prob)
    return pos


def corners_alignment(gc, image, det, corners, pinw, pat_i, shp, debug_dir=None):
    pos_dev = pinw * 2.5
    top_k = 20
    cls_i = det.clf.classes_.searchsorted(det.clusters[pat_i])
    scales = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

    def chip_rect(values, vis=False):
        crn = np.reshape(values, (-1, 2)) + corners
        imgs = []
        for ci, (y, x) in enumerate(crn):
            scl = np.array(scales[ci]) / det.resized_shape[1::-1] * shp[1::-1]
            t = AffineTransform(
                scale=scl,
                translation=(x - shp[1] * scales[ci][0], y - shp[0] * scales[ci][1]))
            imgs.append(cv2.warpAffine(image, t._inv_matrix[:2], det.resized_shape[1::-1]))
        crn_patches = np.vstack(imgs)
        probabilities = np.zeros((0, len(det.clf.classes_)), dtype=np.float64)
        p2 = np.zeros((0, len(det.clf_lqfp.classes_)), dtype=np.float64)
        for descriptors in det.extract_batch_features(crn_patches, det.resized_shape):
            probabilities = np.vstack((probabilities, det.clf.predict_proba(descriptors)))
            p2 = np.vstack((p2, det.clf_lqfp.predict_proba(descriptors)))
        psum = np.mean(probabilities[:, cls_i] * p2[:, 1])
        rc = rect_metrics(crn)
        res = rc * 4 - psum
        if vis:
            print(values)
            print(" ".join(["%.3f" % p for p in probabilities[:, cls_i]]))
            print(" ".join(["%.3f" % p for p in p2[:, 1]]))
            print("value = %.4f = %.4f * k - %.4f" % (res, rc, psum))
            for ci, (y, x) in enumerate(crn):
                t = AffineTransform(
                    scale=scales[ci],
                    translation=(x - shp[1] * scales[ci][0], y - shp[0] * scales[ci][1]))
                img = cv2.warpAffine(image, t._inv_matrix[:2], shp[1::-1])
                imsave(debug_dir + "crn%d_%d_%d_p%.2f_q%.2f.bmp" % (
                    corners[0][0], corners[0][1], ci, probabilities[ci, cls_i], p2[ci, 1]), img)
            # imshow(crn_patches)
        return res

    gc.check_interruption()
    print("pos_dev", pos_dev)
    coeffs0, fx, its, imode, smode = \
        fmin_slsqp(chip_rect, [0.0] * (len(corners) * 2),
                   bounds=[[-pos_dev, pos_dev]] * (len(corners) * 2),
                   epsilon=0.1,
                   acc=0.01,
                   full_output=True)
    fx_min = fx
    tbr = None
    i_opt = None
    topl = []
    coeffs_opt = coeffs0.copy()
    for topr in range(-2, 3):
        for botr in range(-2, 3):
            for topr2 in range(-2, 3):
                for botr2 in range(-2, 3):
                    coeffs = coeffs0.copy()
                    coeffs[2] += pinw * topr / 2
                    coeffs[4] += pinw * topr2 / 2
                    coeffs[0] += pinw * botr / 2
                    coeffs[6] += pinw * botr2 / 2
                    v = chip_rect(coeffs)
                    topl.append((v, coeffs))
            gc.check_interruption()

    topl = sorted(topl, key=lambda x: x[0])
    for ti, (val, coeffs) in enumerate(topl[:top_k]):
        coeffs, fx, its, imode, smode = fmin_slsqp(
            chip_rect, coeffs,
            bounds=[[-pos_dev, pos_dev]] * (len(corners) * 2),
            epsilon=0.1,
            acc=0.01,
            full_output=True)
        gc.check_interruption()
        if fx < fx_min:
            fx_min = fx
            coeffs_opt = coeffs.copy()
            tbr = topr, botr
            i_opt = ti
        # chip_rect(coeffs, vis=True)
    print("opt values:", i_opt, tbr, fx_min, coeffs_opt)
    chip_rect(coeffs_opt, vis=True)
    # imshow(image[corners[2][0]: corners[0][0], corners[2][1] : corners[0][1]])
    return corners + np.reshape(coeffs_opt, (-1, 2))


def detect_2sides(gc, image, det, i, j, pat_i, pinw, pin_y_offset,
                  patch_n_pins, edge_pins_n, debug_dir=None):
    im = rgb2gray(image)
    shp = det.patterns[pat_i].shape

    int1 = int(pin_y_offset + 0.5)
    int2 = int(pin_y_offset + pinw + 0.5)
    int3 = int(shp[1] * (1 - pin_in_pattern))
    pin_img = det.patterns[pat_i][int1:int2, int3:]
    pin_img = (pin_img + pin_img[::-1]) / 2.
    corners = [(i, j)]
    # image[i-5:i+1, j-5:j+1] *= 0.3;    imshow(image); print(pat_i)
    crop_l = 0
    crop_t = max(i - int(shp[0] * (pos_max_deviation + 1)), 0)
    crop_b = min(i + int(shp[0] * pos_max_deviation), image.shape[0])
    crop_r = min(j + int(shp[1] * pos_max_deviation), image.shape[1])
    img_crp = image[crop_t:crop_b, crop_l:crop_r]
    pos = closest_peak((0, 0), img_crp[:, ::-1], det, only_pat_id=pat_i, debug_dir=debug_dir)
    if pos[0] == -1:
        return
    corners.append((crop_t + pos[0] + shp[0],
                    crop_l + img_crp.shape[1] - pos[1] - 1 - shp[1]))
    crop_l = max(corners[-1][1] - int(shp[1] * pos_max_deviation), 0)
    crop_r = min(corners[-1][1] + int(shp[1] * (1 + pos_max_deviation)), image.shape[1])
    crop_t = 0
    img_crp = image[crop_t:crop_b, crop_l:crop_r]
    pos = closest_peak((0, 0), img_crp[::-1, ::-1], det, only_pat_id=pat_i, debug_dir=debug_dir)
    if pos[0] == -1:
        return
    corners.append((crop_t + img_crp.shape[0] - pos[0] - 1 - shp[0],
                    crop_l + img_crp.shape[1] - pos[1] - 1 - shp[1]))

    crop_l = max(corners[0][1] - int(shp[1] * (1 + pos_max_deviation)), 0)
    crop_r = min(corners[0][1] + int(shp[1] * (1 + pos_max_deviation)), image.shape[1])
    crop_t = max(corners[-1][0] - int(shp[0] * pos_max_deviation), 0)
    crop_b = min(corners[-1][0] + int(shp[0] * (1 + pos_max_deviation)), image.shape[0])
    img_crp = image[crop_t:crop_b, crop_l:crop_r]
    pos = closest_peak((0, 0), img_crp[::-1, :], det, only_pat_id=pat_i, debug_dir=debug_dir)

    if pos[0] == -1:
        return
    corners.append((crop_t + img_crp.shape[0] - pos[0] - 1 - shp[0],
                    crop_l + pos[1] + shp[1]))
    corners[1], corners[3] = corners[3], corners[1]  # ccw order
    corners = np.array(corners, dtype=np.float32)
    print("Found 4 corners")

    # corners = corners_alignment(gc, image, det, corners, pinw, pat_i, shp, debug_dir=debug_dir)

    corners = np.round(corners).astype(np.int32)
    first_pin_shift = (pinw - pin_y_offset) % pinw
    p0p1_len = (distance(corners[0], corners[1]) + distance(corners[2], corners[3])) / 2
    n_pins = (p0p1_len - 2 * shp[0] + 2 * patch_n_pins * pinw - 2 * first_pin_shift) / pinw
    n_pins = find_nearest(edge_pins_n, n_pins)
    pins_right = pins_right_edge(
        gc, im, pin_img, pin_y_offset, shp, corners[0], corners[1], pinw, n_pins, patch_n_pins=patch_n_pins)
    p0 = corners[3][0], image.shape[1] - corners[3][1] - 1
    p1 = corners[2][0], image.shape[1] - corners[2][1] - 1
    pins_left_mir = pins_right_edge(
        gc, im[:, ::-1], pin_img, pin_y_offset, shp, p0, p1, pinw, n_pins, patch_n_pins=patch_n_pins)
    print("lpins: %d, rpins: %d" % (len(pins_right), len(pins_left_mir)))
    if len(pins_right) == 0 or len(pins_left_mir) == 0:
        return
    pins_left = []
    for p in pins_left_mir:
        pins_left.append(Pin(rc=(p[0], image.shape[1] - p[1] - 1)))

    # reversed for ccw order
    return corners, list(reversed(pins_right)), pins_left


# i, j - is coordinates of patch in left bottom corner
def detect_multipin(gc, image, det, i, j, pat_i, prob, pinw,
                    pin_y_offset, patch_n_pins, edge_pins_n, debug_dir=None):
    name = det.names[pat_i]
    r = det.pat_rotations[pat_i]
    pat_i = det.pat_orig[pat_i]
    shp = det.patterns[pat_i].shape
    cr0_offset = idxrot((shp[0] - 1, shp[1] - 1), shp, -r)
    i, j = idxrot((i + cr0_offset[0], j + cr0_offset[1]), image.shape, r)
    image = np.rot90(image, -r)
    print("Detect 2 sides:")
    result = detect_2sides(
        gc, image, det, i, j, pat_i, pinw, pin_y_offset, patch_n_pins, edge_pins_n, debug_dir=debug_dir)
    print("result: %s", bool(result))
    if result is None:
        return
    corners, pins_right, pins_left = result
    # try to detect LQFP
    image_r = np.rot90(image, -1)
    i, j = idxrot(
        (corners[1][0] - pinw * fict_pins_n, corners[1][1] - pinw * fict_pins_n - shp[0]), image.shape, 1)
    crop_t = max(int(i - shp[0] / 2 - pinw * fict_pins_n_max + 0.5), 0)
    crop_b = min(int(i + shp[0] / 2 + pinw * fict_pins_n_max + 0.5), image_r.shape[0])
    crop_l = max(int(j - shp[1] / 2 - pinw * fict_pins_n_max + 0.5), 0)
    crop_r = min(int(j + shp[1] / 2 + pinw * fict_pins_n_max + 0.5), image_r.shape[1])
    print("SOIC detected, try to detect LQFP")
    # imshow(image_r[crop_t: crop_b, crop_l: crop_r])
    pos = -1, -1
    result = None
    if "DIP" not in name:  # this condition must be in types.csv
        pos = closest_peak((i - crop_t - shp[0] / 2, j - crop_l - shp[1] / 2), image_r[crop_t: crop_b, crop_l: crop_r],
                           det, only_pat_id=pat_i, debug_dir=debug_dir)
        i, j = np.array(pos) + (crop_t, crop_l) + shp
    if pos[0] != -1:
        result = detect_2sides(
            gc, image_r, det, i, j, pat_i,
            pinw, pin_y_offset, patch_n_pins, edge_pins_n, debug_dir=debug_dir)
        print("result: %s", bool(result))
    # if result is not None and r != 0:
    #    return
    pins_top = []
    pins_bottom = []
    if result is not None:
        corners_rr, pins_top_r, pins_bottom_r = result
        corners_r = []
        # pins_top, pins_bottom = pins_top_r, pins_bottom_r
        for p in pins_top_r:
            pins_top.append(Pin(rc=idxrot(p.rc, image_r.shape, -1)))
        for p in pins_bottom_r:
            pins_bottom.append(Pin(rc=idxrot(p.rc, image_r.shape, -1)))
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
    h_pins = (len(pins_right) + len(pins_left))  # // 2 * 2
    w_pins = (len(pins_top) + len(pins_bottom))  # // 2 * 2
    rotated_pins = []
    rotated_corners = []
    for p in pins_right + pins_top + pins_left + pins_bottom:
        rotated_pins.append(Pin(rc=idxrot(p.rc, image.shape, -r)))

    i = 0
    side_idxes = []
    for pins_side in (pins_right, pins_top, pins_left, pins_bottom):
        if len(pins_side) > 0:
            side_idxes.append((i, i + len(pins_side)))
            i += len(pins_side)
    for pt in corners:
        rotated_corners.append(idxrot(pt, image.shape, -r))
    if r > 0 and w_pins > 0 and h_pins > 0:
        r = 0
        w_pins, h_pins = h_pins, w_pins
    return Element(pins=rotated_pins, name=name,
                   bounding_zone=rotated_corners, rotation=r, probability=prob,
                   w_pins=w_pins, h_pins=h_pins, side_idxes=side_idxes)


def detect_common(gc, image, det, debug_dir=None, only_pat_ids=None, find_one=False):
    # im = rgb2gray(image)
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
            pin_lists[-1].append(Pin(rc=idxrot(pin, 1.0, -det.pat_rotations[pat_i])))

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
        detect_common_return_elements(debug_dir, det, elements, gc, image, only_pat_ids, pat_i, pin_lists, find_one)
        return elements

    detect_common_handle_detect(debug_dir, det, edge_pins_n, elements, gc, image, only_pat_ids, patch_n_pins, pin_lists,
                                pins_w, pins_y_offset, find_one)
    gc.progressSignal_find4.emit(100)
    return elements


def detect_common_return_elements(debug_dir, det, elements, gc, image, only_pat_ids, pat_i, pin_lists, find_one=False):
    orig_pat = det.patterns[0]
    tmp = []
    szk_max = 1.0
    maxv = -1
    maxi = 0
    sizes_els = []
    if det.bga_szk is not None:
        szk_list = [det.bga_szk]
    else:
        if len(det.parameters[pat_i]) > 3:
            szk_list = literal_eval(det.parameters[pat_i][3])
        else:
            szk_list = [1.0]
    for i, szk in enumerate(szk_list):
        det.patterns[0] = cv2.resize(orig_pat, (0, 0), fx=szk, fy=szk)
        els = detect_by_nn(gc, image, det, find_rotations=True, debug_dir=debug_dir,
                           only_pat_ids=only_pat_ids, find_one=find_one)

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
    print("elements size: %s, %s, %s", szk_max, maxv, tmp)
    det.patterns[0] = orig_pat
    for i, j, res_i, p in sizes_els[maxi]:
        shp = det.patterns[res_i].shape
        shp = np.round(np.array(shp, dtype=np.float32) * szk_max).astype(np.int32)
        pins = []
        for pin in pin_lists[res_i]:
            pins.append(Pin(rc=(i + shp[0] * pin.r, j + shp[1] * pin.c)))
        elem = Element(pins=pins,
                       name=det.names[res_i],
                       bounding_zone=[(i, j), (i + shp[0], j), (i + shp[0], j + shp[1]), (i, j + shp[1])],
                       probability=p,
                       rotation=det.pat_rotations[res_i])
        elem.width *= szk_max
        elem.height *= szk_max
        elements.append(elem)
    gc.progressSignal_find4.emit(100)


def detect_common_handle_detect(debug_dir, det, edge_pins_n, elements, gc, image, only_pat_ids, patch_n_pins, pin_lists,
                                pins_w, pins_y_offset, find_one):
    res = detect_by_nn(gc, image, det, find_rotations=True, debug_dir=debug_dir, only_pat_ids=only_pat_ids,
                       find_one=find_one)

    counter = 0
    for i, j, res_i, p in res:
        gc.progressSignal_find4.emit(75 + int(25 * counter / len(res)))
        counter += 1
        if det.parameters[res_i][2] == "multipin":
            elem = detect_multipin(gc, image, det, i, j, res_i, p,
                                   pins_w[res_i], pins_y_offset[res_i],
                                   patch_n_pins[res_i], edge_pins_n[res_i], debug_dir=debug_dir)
            if elem is not None:
                elements.append(elem)
            continue
        shp = det.patterns[res_i].shape
        pins = []
        for pin in pin_lists[res_i]:
            pins.append(Pin(rc=(i + shp[0] * pin.r, j + shp[1] * pin.c)))
        elem = Element(pins=pins,
                       name=det.names[res_i],
                       bounding_zone=[(i, j), (i + shp[0], j), (i + shp[0], j + shp[1]), (i, j + shp[1])],
                       probability=p,
                       rotation=det.pat_rotations[res_i])
        elements.append(elem)


def rotate_and_detect(gc, img, det, debug_dir=None, only_pat_ids=None, find_one=False):
    elements = []
    elements += detect_common(gc, img, det, debug_dir=debug_dir, only_pat_ids=only_pat_ids, find_one=find_one)
    """ for r in rotations:
        if int(r) != r:

    center_old=(img.shape[0] / 2 - 0.5, img.shape[1] / 2 - 0.5)
    imgr = rotate(img, 45, resize=True)
    #imshow(imgr)
    center_new=(imgr.shape[0] / 2 - 0.5, imgr.shape[1] / 2 - 0.5)
    for elem in detect_lqfp(imgr, det):
        points = []
        for p in elem.pins:
            v = p[0] - center_new[0], p[1] - center_new[1]
            s45 = 1. /2 ** 0.5
            v = v[0] * s45 + v[1] * s45, -v[0] * s45 + v[1] * s45
            points.append((int(v[0] + center_old[0] + 0.5), int(v[1] + center_old[1] + 0.5)))
        elements.append(Element(points, name=elem.name))
    """

    return elements


def detect_all(gc, img, types_filename, always_train=False,
               trh_prob=0.5, trh_corr_mult=1.0,
               find_one=False, elements_offset=(0, 0),
               debug_dir=None, det_names=None, bga_szk=None):
    if debug_dir and not find_one:
        from shutil import rmtree
        try:
            rmtree("./%s/" % debug_dir)
        except Exception:
            print("Can't remove %s." % debug_dir)
    if debug_dir and not path.isdir(debug_dir):
        mkdir(debug_dir)

    train_folder, name = path.split(types_filename)
    name = name[:name.rfind(".")]
    dump_clf = path.join(train_folder, clf_dump_pattern % name)
    names = [None, "LQFP"]
    dets = []
    for n in names:
        dump_clf = path.join(train_folder, clf_dump_pattern % (name if n is None else n))
        if not always_train and path.isfile(dump_clf):
            print("Loading classifier dump")
            det = joblib.load(dump_clf)
        else:
            if not path.isfile(types_filename):
                print("File %s not found.", types_filename)
                return
            det = train(gc, types_filename, resized_shape=classifier_image_shape, only_cluster=n)
            joblib.dump(det, dump_clf, compress=9)
            print("Classifier saved to %s.", dump_clf)
        dets.append(det)

    det, ids = detect_all_find_one(bga_szk, det, det_names, dets, find_one, gc, img, trh_corr_mult, trh_prob)

    elements = rotate_and_detect(gc, img, det, debug_dir=debug_dir, only_pat_ids=ids, find_one=find_one)
    if len(elements) == 0:
        return []

    if find_one:
        elements = [elements[select_one(elements, img.shape[:2])]]
    for elem in elements:
        for i in range(len(elem.bounding_zone)):
            elem.bounding_zone[i] += np.array(elements_offset)
        elem.center += elements_offset
        for pin in elem.pins:
            pin.r += elements_offset[0]
            pin.c += elements_offset[1]
    if hasattr(det.clf, "closeSession"):
        det.clf.closeSession()
    if det.clf is not None:
        print("Removing intersecting elements (total %d)...", len(elements))
        elements = remove_intersecting(elements)
    gc.send_elements(elements)

    print("Elements found: %d.", len(elements))
    return elements


def detect_all_find_one(bga_szk, det, det_names, dets, find_one, gc, img, trh_corr_mult, trh_prob):
    det = dets[0]
    det.clf_lqfp = dets[1].clf
    det.bga_szk = bga_szk
    det.trh_prob = trh_prob
    det.trh_corr_mult = trh_corr_mult
    det.trh_max_rect = trh_max_rect
    gc.send_detector(det)
    if hasattr(det.clf, "openSession"):
        det.clf.openSession()
    ids = list(range(len(det.patterns)))
    if det_names is not None:
        ids = []
        for i, n in enumerate(det.names):
            if n in det_names:
                ids.append(i)
    # don't consider small patches
    i = 0
    filtered_ids = []
    for i in ids:
        m_p = (det.parameters[i][2] == "multipin")
        if det.patterns[i] is not None and (np.prod(det.patterns[i].shape[:2]) / np.prod(img.shape[:2]) > 0.1 or m_p):
            filtered_ids.append(i)
    if find_one:
        det.trh_max_rect *= 1.8
        ids = filtered_ids
    return det, ids


def detect_BGA_params(gc, img, types_filename, pix_per_mm):
    pitch_step = 0.05
    max_pitch = 2.0
    emp_logger = EmptyLogger()
    emp_logger.progressSignal_find3.emit = lambda x: gc.progressSignal_scan2.emit(x // 2)
    elements = detect_all(emp_logger, img, types_filename, trh_corr_mult=1.5, always_train=False)
    points = pins_to_array(elements)
    if len(points) < 2:
        return 0, 1.0, points
    vecs = np.zeros((len(points), 2))
    gc.set_iter_progress_bar(0.3)
    gc.check_point_progressbar()
    for i in range(len(points)):
        pt_i = points[i].copy()
        points[i] = np.inf
        j = np.argmin(((points - pt_i) ** 2).sum(axis=1))
        points[i] = pt_i
        vecs[i] = points[j] - pt_i
        gc.progressSignal_scan2.emit(50 + 50 * i / len(points))
    gc.set_iter_progress_bar(5)
    gc.check_point_progressbar()
    ang, maxv = max_angle_bin(90, vecs)
    hist, ranges = np.histogram(
        np.sqrt(np.sum(vecs * vecs, axis=1)) / pix_per_mm,
        bins=int(max_pitch / pitch_step + 0.5), range=(pitch_step / 2, max_pitch + pitch_step / 2))
    print(hist, ranges)
    pitch = (np.argmax(hist) + 1) * pitch_step
    print("pitch = %.2f", pitch)
    gc.progressSignal_scan2.emit(100)
    return ang, pitch, points

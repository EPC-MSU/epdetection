import os
import json
import logging
import warnings

import numpy as np
import cv2
from cv2 import matchTemplate, TM_CCOEFF_NORMED
from scipy.signal import correlate2d
from scipy.fftpack import fft
from scipy.optimize import minimize_scalar

SKEW_TRH = 1.5
PEAK_FOURIER_TRH = 0.85
PINS_X_OFFSET_MULT = 0.56
PIX_PER_MM = 23


class EmptyEmitter:
    def emit(self, *args):
        pass


class FakeGuiConnector:
    def __init__(self):
        super(FakeGuiConnector, self).__init__()

    def check_interruption(self):
        pass

    def check_skip(self):
        pass

    def send_next_stage(self):
        pass

    def send_num_stages(self):
        pass

    def send_skip_stages(self):
        pass


def remove_temp_dir(debug_dir, find_one):
    if debug_dir and not find_one:
        from shutil import rmtree
        try:
            rmtree("./%s/" % debug_dir)
        except OSError:
            logging.debug("Can't remove %s." % debug_dir)
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)


def dump_elements(file_path: str, elements) -> None:
    d = [element.to_json() for element in elements]
    with open(file_path, "w") as dump_file:
        json.dump(d, dump_file, separators=(",", ":"), indent=2)


def save_detect_img(img, elements, path):
    board = img
    for el in elements:
        corner_point = tuple(map(int, el.bounding_zone[0]))
        board = cv2.putText(board, el.name, corner_point, cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 0, 255), 1, cv2.LINE_AA)
        bz_points = np.array(el.bounding_zone, dtype=np.int32)
        bz_points = bz_points.reshape((-1, 1, 2))  # Add dim for cv2
        board = cv2.polylines(board, [bz_points], True, (255, 0, 255), 1)

        if el.rotation % 2 == 0:
            tl = (int(el.center[0] - el.height / 2), int(el.center[1] - el.width / 2))
            br = (int(el.center[0] + el.height / 2), int(el.center[1] + el.width / 2))
        else:
            tl = (int(el.center[0] - el.width / 2), int(el.center[1] - el.height / 2))
            br = (int(el.center[0] + el.width / 2), int(el.center[1] + el.height / 2))
        board = cv2.rectangle(board, tl, br, (255, 255, 255), 1)
        for pin in el.pins:
            board = cv2.circle(board, (int(pin.x), int(pin.y)), 3, (0, 0, 255), 1)
        board = cv2.drawMarker(board, (int(el.center[0]), int(el.center[1])), (255, 255, 0), cv2.MARKER_CROSS, 4, 1)
    cv2.imwrite(path, board)
    return board


def reprName(elem, h_pins, w_pins):
    name = elem.name.split("/")[0].split("\\")[0]
    repr_name = name
    if (w_pins >= 0) and elem.set_automatically:
        try:
            repr_name = name % (w_pins + h_pins)
        except TypeError:
            repr_name = (name + "-%d") % (w_pins + h_pins)
    return repr_name


def fitSizes(elem, h_pins=-1, w_pins=-1):
    """Load real sizes and fit element width and height"""
    if not elem.set_automatically:
        return

    # default border - rectangle around pins
    min_x = min([pin.x for pin in elem.pins]) if len(elem.pins) else 0
    max_x = max([pin.x for pin in elem.pins]) if len(elem.pins) else 0
    min_y = min([pin.y for pin in elem.pins]) if len(elem.pins) else 0
    max_y = max([pin.y for pin in elem.pins]) if len(elem.pins) else 0

    elem.width = max_x - min_x + PIX_PER_MM * 2
    elem.height = max_y - min_y + PIX_PER_MM * 2
    if elem.rotation % 2 == 0:
        elem.width, elem.height = elem.height, elem.width

    with open(os.path.join(os.path.join(os.path.dirname(__file__), "dumps", "sizes.csv"))) as file:
        for line in file.readlines():
            if line.split(",")[0].strip() == reprName(elem, h_pins, w_pins):
                elem.width = float(line.split(",")[1].strip()) * PIX_PER_MM
                elem.height = float(line.split(",")[2].strip()) * PIX_PER_MM
                break

    if len(elem.pins) > 0:  # Adjusts element rect to cover all pins.
        pin_coordinates = tuple(tuple([pin.x, pin.y]) for pin in elem.pins)
        pin_coordinates = np.array(pin_coordinates)
        pin_distances = np.abs(pin_coordinates - np.asarray(elem.center))
        max_distance = np.max(pin_distances, axis=0)

        size = (elem.width, elem.height) if elem.rotation % 2 == 1 else (elem.height, elem.width)
        scale_factor = np.max(2 * max_distance / np.asarray(size))

        if scale_factor > 1.0:  # Don't decrease element size
            elem.width *= scale_factor
            elem.height *= scale_factor
    return elem


# TODO: Duplicated with vision/utils.py
def rgb2gray(img):
    if len(img.shape) == 2:
        im = img
    else:
        im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    if im.dtype == np.float64:
        im = im.astype(np.float32)
    return im


# p is i, j in rotated image (rt times ccw)
# return i, j in original image, i.e. rotate clockwise
# TODO: Duplicated with vision/utils.py
def idxrot(p, img_shp, rt):
    p = np.array(p)
    if img_shp == 1.00:
        img_shp = (2, 2)
    for n in range(rt % 4):
        tmp = p[..., 1].copy()
        p[..., 1] = img_shp[0] - p[..., 0] - 1
        p[..., 0] = tmp
        img_shp = (img_shp[1], img_shp[0])
    return p


def dist2(a, b=0):
    return ((np.array(a) - np.array(b)) ** 2).sum()


def max_angle_bin(n_bins, vecs, weights=None):
    max_noise = 5.0
    if vecs.dtype == np.complex64 or vecs.dtype == np.complex128:
        ang = np.angle(vecs)
    else:
        ang = np.arctan2(vecs[..., 1], vecs[..., 0])
    ang *= n_bins * 2 / np.pi
    ang %= n_bins
    # print("np.mean ang =", np.mean((ang + n_bins // 2) % n_bins - n_bins // 2))
    bins, edges = np.histogram(ang, bins=n_bins, range=(0, n_bins), weights=weights)

    max_i = np.argmax(bins)
    max_v = bins[max_i]
    # ang.dump("tmp_ang.dump")
    max_noise *= n_bins / 90
    ang = (ang + (n_bins / 2 - max_i - 0.5)) % n_bins
    mask = (ang < n_bins / 2 + max_noise) * (ang > n_bins / 2 - max_noise)
    if weights is None:
        res = np.mean(ang[mask])
    else:
        res = np.dot(ang[mask], weights[mask]) / np.sum(weights[mask])
    res = (res - (n_bins / 2 - max_i - 0.5)) * 90 / n_bins
    if res > 45:
        res -= 90
    # calc stat, for not rect oriented PCB
    cnt = 0
    for i in range(n_bins):
        if bins[i] >= bins[max_i] * 0.8:
            cnt += 1
    if cnt / n_bins > 0.2:
        return 0, -1
    return res, max_v


def peak_k(img, trh, k=3):
    peak_sum = (img[1:-1, 1:-1] >= trh) * 1 * k
    peak_sum += (img[1:-1, 1:-1] >= img[1:-1, :-2]) * 1
    peak_sum += (img[1:-1, 1:-1] >= img[1:-1, 2:]) * 1
    peak_sum += (img[1:-1, 1:-1] >= img[:-2, 1:-1]) * 1
    peak_sum += (img[1:-1, 1:-1] >= img[2:, 1:-1]) * 1
    return np.transpose(np.nonzero(peak_sum >= 2 * k)) + 1


def _fourier_sum(a, period):
    s = 0.0 + 0.0j
    for i in range(len(a)):
        s += a[i] * np.exp(-2j * np.pi * i / period)
    return s


def _polygon_center(corners):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        a = np.mean(corners, axis=0)
    return a


def distance(pin1, pin2):
    return np.sqrt((pin1[0] - pin2[0]) ** 2 + (pin1[1] - pin2[1]) ** 2)


def select_one(elements, shape):
    max_d = distance(shape, (0, 0)) / 2
    max_score = 0.
    max_i = 0
    for i, elem in enumerate(elements):
        s = _polygon_square(elem.bounding_zone) / (shape[0] * shape[1])
        d = 1. - distance(
            _polygon_center(elem.bounding_zone),
            (shape[0] / 2., shape[1] / 2.)
        ) / max_d
        if (s + d) / 2 > max_score:
            max_score = (s + d) / 2
            max_i = i
    return max_i


def _polygon_square(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def find_nearest(a, n):
    i = np.array(a).searchsorted(n)
    if i == len(a) or i == 0:
        i -= int(i == len(a))
        return a[i]
    if n - a[i - 1] < a[i] - n:
        return a[i - 1]
    else:
        return a[i]


def _in_polygon(pt, corners):
    if np.any((pt < np.min(corners, axis=0)) | (pt > np.max(corners, axis=0))):
        return False
    ang = 0
    for i in range(len(corners)):
        p1 = corners[i] - pt
        p2 = corners[(i + 1) % len(corners)] - pt
        ang += np.arctan2(np.cross(p1, p2), np.dot(p1, p2))
    return ang >= 6.2  # >= 2pi - eps


def lqfp_bounding(corners, offset_package, offset_pin):
    new_corners = []
    mr = -1 + 0j
    mc = 0 - 1j
    for r in range(4):
        z = mr * offset_pin + mc * offset_package
        z += complex(*corners[r])
        new_corners.append((z.real, z.imag))
        z = mr * offset_package + mc * offset_package
        z += complex(*corners[r])
        new_corners.append((z.real, z.imag))
        z = mr * offset_package + mc * offset_pin
        z += complex(*corners[r])
        new_corners.append((z.real, z.imag))
        mr *= 1j
        mc *= 1j
    return new_corners


def remove_intersecting(elements):
    i = 0
    while i < len(elements):
        j = i + 1
        pci = _polygon_center(elements[i].bounding_zone)
        sqi = _polygon_square(elements[i].bounding_zone)
        while j < len(elements):
            pcj = elements[j].pins[0].x, elements[j].pins[0].y
            in_i = _in_polygon(pcj, elements[i].bounding_zone)
            in_j = _in_polygon(pci, elements[j].bounding_zone)
            if not in_i and not in_j:
                j += 1
                continue
            if in_i and in_j:
                sqj = _polygon_square(elements[j].bounding_zone)
                if sqi < sqj:
                    in_i = False
                else:
                    in_j = False
            if in_i and not in_j:
                del elements[j]
                continue
            if not in_i and in_j:
                del elements[i]
                i -= 1
                break
        i += 1
    return elements


# TODO: Duplicated with vision/utils.py
def pins_to_array(elements) -> np.ndarray:
    points = []
    for el in elements:
        for p in el.pins:
            points.append((p.x, p.y))
    points = np.array(points)
    return points


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
            np.maximum(m[v[0] - v[2] - offset[0]: v[0] + v[2] - offset[0],
                       v[1] - v[3] - offset[1]: v[1] + v[3] - offset[1]], v[-1])

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


def _periodic_peaks(a, period, n):
    if len(a) < 3:
        return []

    def fourier_sum_abs(x):
        return -np.abs(_fourier_sum(a, x))

    opt_res = minimize_scalar(fourier_sum_abs, method="Bounded",
                              bounds=[period - period * 0.04, period + period * 0.04],
                              options={"xatol": 0.01})
    period = opt_res.x
    offset = -np.angle(_fourier_sum(a, period)) / np.pi / 2 * period % period
    logging.debug("pin spacing = %.3f, offset = %.3f" % (period, offset))
    # score = np.abs(_fourier_sum(a, period)) * 2 / len(a)
    r = np.abs(fft(a))
    score = np.count_nonzero(r < r.max() * 0.2) / len(r)
    logging.debug("n = %d, score = %.3f" % (n, score))
    # import matplotlib.pyplot as plt
    # plt.plot(a); plt.show()
    if score < PEAK_FOURIER_TRH:
        return []
    mid = (len(a) - 1) * 0.5
    if n % 2 == 0:
        k = np.ceil((mid - offset) / period)
    else:
        k = np.round((mid - offset) / period)
    pos = offset + k * period - (n // 2) * period
    res = np.arange(n) * period + pos
    logging.debug("%s", res)
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
    skew = np.arctan2(p0[0] - p1[0], p0[1] - p1[1]) / np.pi * 180 % 90
    skew = 90 - skew if skew > 45 else skew
    logging.debug("skew = %.2f" % skew)
    if skew > SKEW_TRH:
        logging.debug("skew = %.2f so large" % skew)
        cmean = (p0[1] + p1[1]) // 2
        p0 = p0[0], cmean
        p1 = p1[0], cmean
    t0 = p0[0] - shp[0] + (patch_n_pins - 1) * pin.shape[0], p0[1] - pin.shape[1]
    t1 = p1[0] + shp[0] - patch_n_pins * pin.shape[0] - pins_y_offset, p1[1] - pin.shape[1]

    crop_b = min(p0[0] - shp[0] + int((patch_n_pins + 0.5) * pin.shape[0] + 0.5), img.shape[0])
    crop_t = max(p1[0] + shp[0] - int((patch_n_pins + 0.5) * pin.shape[0] + 0.5), 0)

    def vertical_line(p0, p1, i0, i1, lb, rb):
        res = []
        if p0[0] - p1[0] == 0:
            return np.array(res)
        k = (p0[1] - p1[1]) / (p0[0] - p1[0])
        for i in range(i0, i1):
            v = int((i - p0[0]) * k + p0[1] + 0.5)
            if v < lb:
                continue
            if v >= rb:
                break
            res.append(v)
        return np.array(res)

    line_j = vertical_line(t0, t1, crop_t, crop_b - pin.shape[0] + 1, 0, img.shape[1] - pin.shape[1] + 1)
    # print(crop_b, crop_t, pin.shape[0], pitch)
    if len(line_j) == 0:
        return []
    crop_l = min(line_j)
    crop_r = max(line_j) + pin.shape[1]

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

    for peak in _periodic_peaks(a, pitch, n_pins):
        res.append((crop_t + peak + pin.shape[0] // 2,
                    crop_l + line_j[int(np.round(peak))] + PINS_X_OFFSET_MULT * pin.shape[1]))
    logging.debug("%d pins along edge", len(res))
    return res

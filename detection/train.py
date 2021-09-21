import os
import logging
import time
from random import random
from itertools import islice
from datetime import datetime

import numpy as np
from cv2 import imread, matchTemplate, TM_CCOEFF_NORMED
from cv2 import HOGDescriptor, resize, INTER_AREA
from skimage.color import rgb2gray
from skimage.transform import AffineTransform, warp
from skimage import __version__ as skimage_version
from sklearn.externals import joblib
from sklearn import base
from sklearn.linear_model import SGDClassifier
from scipy.signal import convolve2d
from csv import reader as csv_reader

"""
def draw_rect(img, center, shape, color=[255, 0, 0]):
    clr = np.array(color)

    img[center[0], center[1]:center[1] + shape[1]] = clr
    img[center[0] + shape[0] - 1, center[1]:center[1] + shape[1]] = clr
    img[center[0]:center[0] + shape[0], center[1]] = clr
    img[center[0]:center[0] + shape[0], center[1] + shape[1] - 1] = clr
"""
trh_corr_default = 0.34

cellRows, cellCols = 4, 4
cpbRow, cpbCol = 3, 3
cells_per_block = (cpbRow, cpbCol)
binCount = 9
eps = 1e-6
blockRowCells = cells_per_block[0] * cellRows
blockColCells = cells_per_block[1] * cellCols
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 1
max_size = 1024 * 1024
intensity_features_rescale_f = 0.25
pinh2 = 7

edge_mode = "edge" if int(skimage_version.split(".")[1]) >= 12 else "nearest"

model_filename_pattern = "model_%s.ckpt"

# from vision.net import CNNClassifier
# clf = CNNClassifier(model_filename_pattern % name)

# classifier = svm.SVC(C=1, kernel="rbf", gamma=0.2, probability=True, verbose=10)

sobel = np.array([[-1 - 1j, -2, -1. + 1j],
                  [0 - 2j, 0, 0. + 2j],
                  [1 - 1j, 2, 1. + 1j]], dtype=np.complex64)


class Detector:
    def __init__(self):
        self.clf = None
        self.patterns = None
        self.names = None
        self.resized_shape = None
        self.parameters = None
        self.clusters = None
        self.trh_prob = 0.5
        self.trh_corr_mult = 1.0
        self.pat_rotations = None
        self.pat_orig = None
        self.bga_szk = None
        self.trh_max_rect = None
        self.savedate = None

    def save_to_file(self, filepath):
        self.savedate = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        d = {}
        for k in self.__dict__.keys():
            d[k] = getattr(self, k)
        joblib.dump(d, filepath, compress=9)

    def load_from_file(self, filepath):
        d = joblib.load(filepath)
        for k, v in d.items():
            setattr(self, k, v)


def multipin_corr(img, n):
    img = rgb2gray(img).astype(np.float32)
    h = img.shape[0] // n
    res = np.zeros((n, (h - pinh2 * 2 - 1) // pinh2), dtype=np.float32)
    for i in range(n):
        res[i] = matchTemplate(img[i * h: (i + 1) * h, img.shape[1] // 2:],
                               img[i * h: i * h + pinh2 * 2, img.shape[1] // 2:],
                               TM_CCOEFF_NORMED).ravel()[pinh2::pinh2]
    return res


def extra_features(img, n):
    if img.dtype == np.uint8:
        resized = resize(
            img.astype(np.float32) / 255., (0, 0),
            fx=intensity_features_rescale_f,
            fy=intensity_features_rescale_f,
            interpolation=INTER_AREA)
    else:
        resized = resize(
            img, (0, 0),
            fx=intensity_features_rescale_f,
            fy=intensity_features_rescale_f,
            interpolation=INTER_AREA)

    mpin1 = resize(resized[:, 10], (1, 2 * n), interpolation=INTER_AREA)
    mpin2 = resize(resized[:, 8], (1, 2 * n), interpolation=INTER_AREA)
    mpin_diff = mpin1 - mpin2
    mpin_diff = np.reshape(mpin_diff, (n, -1))
    mpin_diff = mpin_diff[:, 0:1]
    return np.hstack((np.reshape(resized, (n, -1)), multipin_corr(img, n), mpin_diff))


def extract_hogs_opencv(images, img_shape):
    if not hasattr(images, "__next__"):
        images = iter(images)
    blockSize = cellRows * cpbRow, cellCols * cpbCol
    blockStride = blockSize
    blockStride = blockStride[0] // 2, blockStride[1] // 2
    cellSize = cellRows, cellCols

    hog = HOGDescriptor(img_shape[1::-1], blockSize, blockStride, cellSize, binCount,
                        derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels)

    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = img_shape[1::-1]
    padding = 0, 0
    max_n = max_size // (img_shape[0] * img_shape[1])
    hog_len = hog.getDescriptorSize()
    while True:
        try:
            img_list = list(islice(images, 0, max_n))
            # img_list = [img_gen.next() for _ in range(0, max_n)]
            image = np.vstack(img_list)  # TODO: Remove warning
        except ValueError:
            break
        n = image.shape[0] // img_shape[0]
        extra = extra_features(image, n)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        locations = [(0, i) for i in range(0, image.shape[0], img_shape[0])]
        # print((n, hog_len),h.shape, image.shape, img_shape)
        hogs = np.reshape(hog.compute(image, winStride, padding, locations), (n, hog_len))
        yield np.hstack((hogs, extra))


def extract_hog(image, resized_shape):
    return next(extract_hogs_opencv(
        [resize(image, resized_shape[1::-1])], resized_shape)
    )[0]


def extract_gray(images, img_shape):
    max_n = max_size // (img_shape[0] * img_shape[1])
    while True:
        try:
            image = np.vstack(islice(images, 0, max_n))
        except ValueError:
            break
        n = image.shape[0] // img_shape[0]
        yield np.reshape(image, (n, img_shape[0] * img_shape[1]))

    """
    #return extract_hog_python(resize(image, resized_shape))
    return skhog(resize(image, resized_shape[1::-1]), visualise=False,
                orientations=binCount,
                pixels_per_cell=(cellRows, cellCols),
                cells_per_block=cells_per_block)
    """


def calc_cells(grad_polar, cell_r, cell_c, bin_cnt):
    v = np.zeros((int(grad_polar.shape[0] / cell_r), int(grad_polar.shape[1] / cell_c), bin_cnt),
                 dtype=np.float32, order="C")

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            # v[i][j] = np.histogram(np.imag(grad_polar[i*cell_r:(i+1)*cell_r, j*cell_c:(j+1)*cell_c]).ravel(),
            #                    bins=bin_cnt, range=(0, np.pi),
            #                weights=np.real(grad_polar[i*cell_r:(i+1)*cell_r, j*cell_c:(j+1)*cell_c]).ravel())[0]
            for line in grad_polar[i * cell_r:(i + 1) * cell_r, j * cell_c:(j + 1) * cell_c]:
                for abs_, ang in line:
                    ang -= eps
                    if ang < 0:
                        ang += np.pi
                    k = int(ang / np.pi * bin_cnt)
                    v[i][j][k] += abs_
    return v


def extract_hog_python(image):
    grad = convolve2d(image, sobel, boundary="symm", mode="same")
    grad_abs = np.absolute(grad)
    grad_ang = np.angle(grad)
    grad_polar = np.dstack((grad_abs, grad_ang))

    v = calc_cells(grad_polar, cellRows, cellCols, binCount)
    hog = np.zeros(
        (
            int(image.shape[0] / blockRowCells),
            int(image.shape[1] / blockRowCells),
            int(binCount * cpbRow * cpbCol)
        ),
        dtype=np.float32, order="C")
    for i in range(hog.shape[0]):
        for j in range(hog.shape[1]):
            hog[i][j] = v[i * cpbRow: (i + 1) * cpbRow, j * cpbCol:(j + 1) * cpbCol].ravel()
            hog[i][j] /= np.sqrt((hog[i][j] * hog[i][j]).sum() + eps)
    return hog.ravel()


def jitter(img, n):
    def rand_bias(a, b):
        r = random() * (b - a) * 2
        if r <= b - a:
            return a + r
        return -a - r + b - a

    center = np.array((img.shape[1] / 2 - 0.5, img.shape[0] / 2 - 0.5))
    for i in range(n):
        r = AffineTransform(translation=-center) + \
            AffineTransform(rotation=rand_bias(0.7, 2.0) * np.pi / 180) + \
            AffineTransform(translation=center) + \
            AffineTransform(translation=(rand_bias(0.7, 4.0), rand_bias(0.7, 4.0)))
        yield warp(img, r, mode=edge_mode)


# sym_mask has 3 bits responsible for horizontal,
# vertical and diagonal (.T) symmetry
def load_data(types_filename, extract_hogs_opencv,
              resized_shape=None, jitter_n=3,
              only_cluster=None, filenames=False):
    x = []
    y = []
    # x0_sym = []
    patterns = []
    fnames = []
    # n = 0
    folder_to_idx = dict()
    names = []
    parameters = []
    train_folder = os.path.split(types_filename)[0]
    clusters_file = os.path.join(train_folder, "clusters.txt")

    def extract_features(image, resized_shape):
        return next(extract_hogs_opencv([resize(image, resized_shape[1::-1])],
                                        resized_shape))[0]

    only_cluster_names = None
    if only_cluster is not None:
        f = open(clusters_file, "r")
        for line in f:
            spl = line.split(":")
            if len(spl) == 2 and spl[0] == only_cluster:
                only_cluster_names = [field.strip() for field in spl[1].split(",")]
                break
        if only_cluster_names is None:
            raise ValueError("cluster %s not found in %s" % (only_cluster, clusters_file))
    load_data_names_from_file(folder_to_idx, names, only_cluster, only_cluster_names, parameters, train_folder,
                              types_filename)

    n = len(names)
    for c in range(n):
        label_folder = os.path.join(train_folder, names[c])
        pat = None
        samples = 0
        pat, resized_shape, samples = load_data_handle_samples(c, extract_features, fnames, jitter_n, label_folder,
                                                               parameters, pat, resized_shape, samples, x, y)
        if pat is None:
            patterns.append(None)
        else:
            patterns.append((pat / samples).astype(np.float32))

    # join some classes into one
    clusters = np.arange(n)
    not_in_clust = set()
    for i, name in enumerate(names):
        if os.path.sep not in name:
            not_in_clust.add(name)
        else:
            names[i] = name.replace(os.path.sep, "/")
    if "not elem" in not_in_clust:
        not_in_clust.remove("not elem")
    if os.path.isfile(clusters_file):
        load_data_handle_cluster_file(clusters, clusters_file, folder_to_idx, not_in_clust)
    for name in not_in_clust:
        clusters[folder_to_idx[name]] = folder_to_idx[name][0]
    for i in range(len(y)):
        y[i] = clusters[y[i]]
    classes_count = []
    for i in range(n):
        # cnt = y.count(i)
        if clusters[i] != i:
            classes_count.append("%d with %d" % (i, clusters[i]))
        else:
            classes_count.append("%d: %d" % (i, y.count(i)))

    logging.debug("all: %d, %d classes: %s" % (len(y), n, ", ".join(classes_count)))
    logging.debug(f"n_features = {len(x[0])}, resized_shape = {resized_shape}")
    logging.debug("n_classes = %d" % len(set(clusters)))
    pat_rotations = [0] * n
    pat_orig = list(range(n))
    clusters = load_data_handle_pat_rotations(clusters, n, names, parameters, pat_orig, pat_rotations, patterns)

    det = Detector()
    det.patterns = patterns
    det.names = names
    det.parameters = parameters
    det.resized_shape = resized_shape
    det.clusters = list(range(len(patterns) + 1)) if clusters is None else clusters
    if pat_rotations is None:
        det.pat_rotations = [0] * len(patterns)
    else:
        det.pat_rotations = pat_rotations

    det.pat_orig = list(range(len(patterns) + 1)) if pat_orig is None else pat_orig
    if filenames:
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.int16), det, fnames
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int16), det


def load_data_handle_pat_rotations(clusters, n, names, parameters, pat_orig, pat_rotations, patterns):
    for pat_i in range(n):
        # all rotations are equal or it is not elem or
        # it is corner of multi pin element
        if parameters[pat_i][0] == 7 or patterns[pat_i] is None:
            continue
        rots = range(1, 4)
        # has center symmety
        if parameters[pat_i][0] & 4 and parameters[pat_i][2] == "multipin":
            continue
        if parameters[pat_i][0] == 3 or parameters[pat_i][2] == "multipin":
            rots = range(1, 2)
        for r in rots:
            patterns.append(np.rot90(patterns[pat_i], r))
            clusters = np.append(clusters, clusters[pat_i])
            parameters.append(parameters[pat_i])
            names.append(names[pat_i])
            pat_rotations.append(r)
            pat_orig.append(pat_i)
    return clusters


def load_data_handle_cluster_file(clusters, clusters_file, folder_to_idx, not_in_clust):
    with open(clusters_file, "r") as f:
        for line in f:
            spl = line.split(":")
            if len(spl) < 2:
                line = spl[0]
            else:
                cl_name, line = spl
            spl = line.split(",")
            num0 = None
            for field in spl:
                name = field.strip()
                if name not in folder_to_idx:
                    continue
                if name in not_in_clust:
                    not_in_clust.remove(name)
                elif "/" not in name:
                    ValueError("%s contains in %s twice" % (name, clusters_file))
                if num0 is None:
                    num0 = folder_to_idx[name][0]
                clusters[folder_to_idx[name]] = num0


def load_data_handle_samples(c, extract_features, fnames, jitter_n, label_folder, parameters, pat, resized_shape,
                             samples, x, y):
    for fimg in os.listdir(label_folder):
        try:
            fimg = os.path.join(label_folder, fimg)
            img = rgb2gray(imread(fimg)).astype(np.float32)
            if resized_shape is None:
                resized_shape = img.shape

            if parameters[c][1] < 1.0:
                if pat is None:
                    pat = img.copy()
                else:
                    pat += img
            samples += 1
            x.append(extract_features(img, resized_shape))
            y.append(c)
            fnames.append(fimg)
            for jimg in jitter(img, jitter_n):
                x.append(extract_features(jimg, resized_shape))
                y.append(c)
                fnames.append(fimg + ".j")

            if pat is None:
                continue
            if (parameters[c][0] & 4) and img.shape[0] != img.shape[1]:
                parameters[c][0] &= 3
                logging.debug("Can't transpose not square")
            for mask in range(1, 8):
                if (parameters[c][0] | mask) == parameters[c][0]:
                    img2 = img
                    if mask & 1:
                        img2 = img2[::-1]
                    if mask & 2:
                        img2 = img2[:, ::-1]
                    if mask & 4:
                        img2 = img2.T
                    x.append(extract_features(img2, resized_shape))
                    y.append(c)
                    fnames.append(fimg + ".%d" % mask)
                    for jimg in jitter(img, jitter_n):
                        x.append(extract_features(jimg, resized_shape))
                        y.append(c)
                        fnames.append(fimg + ".j")
                    pat += img2
                    samples += 1
        except OSError:
            pass
        except ValueError as msg:
            logging.debug(fimg + ": " + str(msg))
            raise
    return pat, resized_shape, samples


def load_data_names_from_file(folder_to_idx, names, only_cluster, only_cluster_names, parameters, train_folder,
                              types_filename):
    with open(types_filename, "r") as f:
        for row in csv_reader(f, quotechar=",", skipinitialspace=True):
            row = [field.strip() for field in row]
            if len(row) == 0 or row[0] == "name":
                continue
            if only_cluster is not None and not (row[0] in only_cluster_names or row[0] == "not elem"):
                continue
            all_dirs = [row[0]]
            # for i in os.walk(os.path.join(train_folder, row[0])):
            #     logging.debug(str(i))
            try:
                top, dirs, files = next(os.walk(os.path.join(train_folder, row[0])))
            except StopIteration:
                logging.error("Dataset is missing")
                exit(-1)
            for d in dirs:
                all_dirs.append(os.path.join(row[0], d))
            if row[0] in folder_to_idx:
                raise ValueError("all folders must be different")
            folder_to_idx[row[0]] = []
            for d in all_dirs:
                folder_to_idx[row[0]].append(len(names))
                names.append(d)
                if d.startswith("not"):
                    parameters.append([0, 1.0])
                    continue
                sym_mask = 0
                if len(row) > 3:
                    sym_mask = int(row[1]) | int(row[2]) << 1 | int(row[3]) << 2
                if sym_mask == 5:
                    logging.warning("Element has horizontal and diagonal symmetry,"
                                    " therefore it also has vertical symmetry")
                    sym_mask = 7
                if sym_mask == 6:
                    logging.warning("Element has vertical and diagonal symmetry,"
                                    " therefore it also has horizontal symmetry")
                    sym_mask = 7

                if len(row) > 4:
                    parameters.append([sym_mask, float(row[4])] + row[5:])
                else:
                    parameters.append([sym_mask, trh_corr_default])


def train(classifier, types_filename, resized_shape=None, only_cluster=None):
    x, y, det = load_data(types_filename, extract_hogs_opencv,
                          resized_shape=resized_shape, only_cluster=only_cluster)
    if np.count_nonzero(y) == 0:
        logging.debug("Found just 1 class. Classifier is not necessary")
        return det
    clf = base.clone(classifier)
    logging.debug(str(clf))
    logging.debug("training started")
    clf.fit(x, y)
    logging.debug("trained")
    # logging.debug("classes_:", str(clf.classes_))
    det.clf = clf
    return det


if __name__ == "__main__":
    start_time = time.time()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.debug("Parse dataset")
    clf_dump_pattern = "retrain_clf_%s.dump"
    classifier_image_shape = (60, 60)
    types_filename = os.path.join("detection", "dumps", "label.csv")
    train_folder, name = os.path.split(types_filename)
    name = name[:name.rfind(".")]
    dump_clf = os.path.join(train_folder, clf_dump_pattern % name)

    classifier = SGDClassifier(alpha=0.001, loss="log", max_iter=500, verbose=10, n_jobs=4)
    det = train(classifier, types_filename, resized_shape=classifier_image_shape, only_cluster=None)
    det.save_to_file(dump_clf)
    logging.debug(f"Run total time: {(time.time() - start_time):.1f} seconds")
    logging.debug("Classifier saved to %s.", dump_clf)

import os
import cv2
import numpy as np
import random

import matplotlib.pyplot as plt

__all__ = ["DataWorker"]


class DataLoaderError(Exception):
    pass


class LoadError(DataLoaderError):
    pass


class AutoModeError(DataLoaderError):
    pass


class DataWorker:
    def __init__(self, auto_mode: bool):
        super(DataWorker, self).__init__()
        self.images = None
        self.shape_2d = None
        self.classes = None
        self.class_weight = None
        self.classes_paths = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.auto_mode = auto_mode

    def rchars(self, n=8):
        alphabet = "abcdefghijklnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        return "".join([random.choice(alphabet) for _ in range(n)])

    def shuffle_in_unison(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        return a, b

    def load_dataset(self, classes_paths, shape_2d=(32, 32)):
        self.classes_paths = classes_paths
        self.shape_2d = shape_2d
        images, classes = list(), list()
        print("Loading dataset...")
        for actual_class in classes_paths:
            if actual_class == 1:
                print("")
            paths = classes_paths[actual_class]
            for path in paths:
                print("Loading " + path)
                for top, dirs, files in os.walk(path):
                    for i, name in enumerate(files):
                        if not os.path.isfile(top + "//" + name):
                            continue
                        try:
                            imag = cv2.imread(top + "//" + name)
                            imag = cv2.resize(imag, shape_2d, interpolation=cv2.INTER_AREA)
                            imag = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
                        except Exception:
                            continue
                        if imag.shape != shape_2d:
                            continue
                        images.append(imag)
                        classes.append(actual_class)

        if len(images) == 0:
            raise LoadError
        images, classes = self.shuffle_in_unison(images, classes)
        images, classes = np.array(images), np.array(classes)
        images, classes = images / 255.0, classes
        images = images[..., np.newaxis]
        if self.auto_mode is False:
            return images, classes
        else:
            self.images, self.classes = images, classes

    def define_class_weight(self, y_train=None, classes_paths=None):
        if self.auto_mode is True:
            class_weight = dict()
            for i in range(len(self.classes_paths)):
                class_weight[i] = (1 / np.count_nonzero(self.y_train == i)) * (len(self.y_train)) / len(
                    self.classes_paths)
                print(f"class_weight[{i}]: {class_weight[i]}")
            print("_" * 65)
            self.class_weight = class_weight
        else:
            class_weight = dict()
            for i in range(len(classes_paths)):
                class_weight[i] = (1 / np.count_nonzero(y_train == i)) * (len(y_train)) / len(classes_paths)
                print(f"class_weight[{i}]: {class_weight[i]}")
            print("_" * 65)
            self.class_weight = class_weight
            return self.class_weight

    def load_images(self, dataset_path):
        images = list()
        print("load_images: Raw images will not be save in this class variables")
        for top, dirs, files in os.walk(dataset_path):
            for i, name in enumerate(files):
                imag = cv2.imread(top + "//" + name)
                images.append(imag)
        return images

    def split_and_shuffle_auto(self, test_part=0.1):
        if self.auto_mode is False:
            raise AutoModeError("If auto-mode disabled, use split_and_shuffle with images and classes arguments.")
        train_len = int(len(self.classes) * (1 - test_part))
        x_train = self.images[:train_len]
        y_train = self.classes[:train_len]
        x_test = self.images[train_len:]
        y_test = self.classes[train_len:]
        print(f"Num of train images = {len(x_train)}")
        print(f"Num of test images = {len(x_test)}")
        x_train, y_train = self.shuffle_in_unison(x_train, y_train)
        x_test, y_test = self.shuffle_in_unison(x_test, y_test)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def split_and_shuffle(self, images, classes, test_part=0.1):
        if self.auto_mode is True:
            raise AutoModeError("If auto-mode enabled, use split_and_shuffle with only test_part argument.")
        train_len = int(len(classes) * (1 - test_part))
        x_train = images[:train_len]
        y_train = classes[:train_len]
        x_test = images[train_len:]
        y_test = classes[train_len:]
        print(f"Num of train images = {len(x_train)}")
        print(f"Num of test images = {len(x_test)}")
        x_train, y_train = self.shuffle_in_unison(x_train, y_train)
        x_test, y_test = self.shuffle_in_unison(x_test, y_test)
        return x_train, y_train, x_test, y_test

    def save_images(self, save_path, images, shape=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if shape is not None:
            for i, img in enumerate(images):
                if shape == img.shape:
                    cv2.imwrite(save_path + f"//{i}_" + self.rchars() + ".jpg", img)
        else:
            for i, img in enumerate(images):
                cv2.imwrite(save_path + f"//{i}_" + self.rchars() + ".jpg", img)

    def cut_center(self, from_path: str, into_path: str, from_shape: tuple, to_shape: tuple):
        for top, dirs, files in os.walk(from_path):
            print("Cutting center...")
            for i, name in enumerate(files):
                if (name.find(".bmp") > -1) or (name.find(".jpg") > -1):
                    img_orig = cv2.imread(top + "//" + name)
                else:
                    continue
                if img_orig.shape == to_shape:
                    new_path = top.replace(from_path, into_path)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    cv2.imwrite(new_path + "//" + name, img_orig)
                    continue
                if img_orig.shape == from_shape:
                    x1 = int((img_orig.shape[0] - to_shape[0]) / 2)
                    x2 = int((img_orig.shape[0] + to_shape[0]) / 2)
                    y1 = int((img_orig.shape[1] - to_shape[1]) / 2)
                    y2 = int((img_orig.shape[1] + to_shape[1]) / 2)
                    img = img_orig[x1:x2, y1:y2]

                    if img.shape != to_shape:
                        continue
                    if not os.path.exists(into_path):
                        os.makedirs(into_path)
                    cv2.imwrite(into_path + "//" + name, img)

    def save_plots(self, hist_fit, hist_eval, path, epochs, only_epoch_end=True):
        if only_epoch_end:
            y_acc_fit = hist_fit.acc_endepoch
            y_acc_eval = hist_eval.acc_endepoch
            y_loss_fit = hist_fit.loss_endepoch
            y_loss_eval = hist_eval.loss_endepoch
        else:
            y_acc_fit = hist_fit.acc
            y_acc_eval = hist_eval.acc
            y_loss_fit = hist_fit.loss
            y_loss_eval = hist_eval.loss

        # Accuracy plot
        fig, axs = plt.subplots()
        x_points_fit = [x * epochs / len(y_acc_fit) for x in range(len(y_acc_fit))]
        x_points_eval = [x * epochs / len(y_acc_eval) for x in range(len(y_acc_eval))]
        axs.plot(x_points_fit, y_acc_fit, label="Accuracy train")
        axs.plot(x_points_eval, y_acc_eval, label="Accuracy test")
        axs.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.yticks(np.arange(0.0, 1.05, 0.1))
        plt.grid(True, linestyle=":")
        plt.savefig(f"{path}/acc.png", dpi=400, bbox_inches="tight")

        # Loss plot
        fig, axs = plt.subplots()
        x_points_fit = [x * epochs / len(y_loss_fit) for x in range(len(y_loss_fit))]
        x_points_eval = [x * epochs / len(y_loss_eval) for x in range(len(y_loss_eval))]
        axs.plot(x_points_fit, y_loss_fit, label="Loss train")
        axs.plot(x_points_eval, y_loss_eval, label="Loss test")
        axs.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid(True, linestyle=":")
        plt.savefig(f"{path}/loss.png", dpi=400, bbox_inches="tight")
        print("Plots saved")

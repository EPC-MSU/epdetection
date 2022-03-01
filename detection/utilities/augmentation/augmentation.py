import os
import random
import shutil

import cv2
import numpy as np
import imutils

from utilities.dataworker import DataWorker


class Augmenter:
    def __init__(self):
        super(Augmenter, self).__init__()
        self.images = []
        self.dataworker = DataWorker(False)

    def random_procent_part(self, images, procent=50):
        images_local = images.copy()
        np.random.shuffle(images_local)
        count = int(procent / 100 * len(images_local))
        return images_local[:count]

    def hsv_random_shift(self, images, h_shift=10, s_shift=20, v_shift=30):
        for i, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
            img[:, :, 0] += np.random.randint(-h_shift, h_shift)
            img[:, :, 1] += np.random.randint(-s_shift, s_shift)
            img[:, :, 2] += np.random.randint(-v_shift, v_shift)
            img = np.clip(img, 0, 255)
            images[i] = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return images

    def brightness(self, images):
        for i, img in enumerate(images):
            cont = random.uniform(-0.8, 1.0)
            beta = random.randint(-50, 50)
            images[i] = cv2.convertScaleAbs(img.copy(), alpha=1.0 + int(cont) / 100, beta=int(beta))
        return images

    def color(self, images, dispersion=0.02):
        for i, img in enumerate(images):
            add = random.uniform(-dispersion, dispersion) * 255
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = hsv[..., 0] + add
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            images[i] = rgb
        return images

    def gaussian_noise(self, images, dispersion=0.1):
        for i, img in enumerate(images):
            gauss = np.random.normal(0, dispersion, img.size)
            gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype("uint8")
            img_gauss = cv2.add(img, gauss)
            images[i] = img_gauss
        return images

    def speckle_noise(self, images, dispersion=0.1):
        for i, img in enumerate(images):
            gauss = np.random.randint(int(-dispersion * 127), int(dispersion * 127), img.size)
            gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype("int16")

            img = cv2.add(img.astype("int16"), gauss)
            img = np.clip(img, 0, 255)

            images[i] = img.astype("uint8")
        return images

    def distortion(self, images):
        for i, img in enumerate(images):
            kernel = np.random.randint(3, 5)
            if kernel % 2 == 0:
                kernel += 1
            img = cv2.blur(img, (kernel, kernel))
            images[i] = img
        return images

    def rotate_and_shift(self, images, dispersion=0.025):
        for i, img in enumerate(images):
            rot_angle = np.random.uniform(-dispersion * 180, dispersion * 180)  # rotate
            img = imutils.rotate(img, rot_angle)
            images[i] = img

            x, y = np.random.randint(int(-dispersion * 150), int(dispersion * 150), (2,))  # shift
            M = np.float32([[1, 0, x], [0, 1, y]])
            images[i] = cv2.warpAffine(img, M, img.shape[:2])
        return images

    def flip_rotate(self, images, flip_axis="y", mode="flip_and_rotate_and_flip"):
        flip_rotate_images = []
        for i, img in enumerate(images):
            if mode == "rotate":
                img = imutils.rotate(img, 180)
                flip_rotate_images.append(img)
            elif mode == "flip":
                if flip_axis == "y":
                    flip_rotate_images.append(cv2.flip(img, 1))
                elif flip_axis == "x":
                    flip_rotate_images.append(cv2.flip(img, 0))
            elif mode == "flip_and_rotate_and_flip":
                if flip_axis == "y":
                    img_flip = cv2.flip(img.copy(), 1)
                elif flip_axis == "x":
                    img_flip = cv2.flip(img.copy(), 2)
                img_rot = imutils.rotate(img.copy(), 180)
                img_flip_rot = imutils.rotate(img_flip.copy(), 180)
                flip_rotate_images.append(img_flip)
                flip_rotate_images.append(img_rot)
                flip_rotate_images.append(img_flip_rot)
        return flip_rotate_images

    def augmentation_unit(self, images, load_path):
        images_gen = list()
        images_gen += images
        if "SMD" in load_path:
            images_gen += self.flip_rotate(images.copy())
        images_gen += self.hsv_random_shift(images.copy())
        images_gen += self.hsv_random_shift(images.copy())
        # images_gen += self.hsv_random_shift(images.copy(), h_shift=0, s_shift=0, v_shift=40)
        images_gen += self.speckle_noise(images.copy())
        images_gen += self.distortion(images.copy())
        return images_gen

    def execute_in_runtime(self, classes_paths):
        print("Augmentation started.")
        for i, (keys, value) in enumerate(classes_paths.items()):
            load_path = value[0]
        print(f"Loading images from {load_path}")
        images = self.dataworker.load_images(load_path)

        print("Starting...")
        images_gen = self.augmentation_unit(images, load_path)
        return images_gen

    def execute_and_save(self, classes_paths):
        for i, (keys, value) in enumerate(classes_paths.items()):
            load_path = value[0]
            if "incorrect" in value[0]:
                save_path = value[0].replace("incorrect", "false")
            else:
                save_path = value[0].replace("correct", "true")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            try:
                print(f"Deleting images from {save_path}")
                shutil.rmtree(save_path)  # Delete true dir
            except FileNotFoundError:
                pass

            print(f"Loading images from {load_path}")
            images = self.dataworker.load_images(load_path)

            print("Starting...")
            images_gen = self.augmentation_unit(images, load_path)

            print(f"Save {len(images_gen)} images to {save_path}")
            self.dataworker.save_images(save_path, images_gen, images[0].shape)


if __name__ == "__main__":
    CLASSES_PATHS = {
        0: ["train_x1//2-SMD//incorrect",
            "train_x1//SMA//incorrect",
            "train_x1//SMB//incorrect",
            "train_x1//SOD110//incorrect",
            "train_x1//SOD323F//incorrect",
            "train_x1//SOD523//incorrect",
            "train_x1//SOT23-5//incorrect",
            "train_x1//SOT23-6//incorrect",
            "train_x1//SOT143//incorrect",
            "train_x1//SOT323//incorrect",
            "train_x1//SOT323-5//incorrect",
            "train_x1//SOT343//incorrect",
            "train_x1//SOT363//incorrect",
            "train_x1//SOT523//incorrect",
            "train_x1//SOT723//incorrect",
            "train_x1//SMD0402_CL//incorrect",
            "train_x1//SMD0402_R//incorrect",
            "train_x1//SMD0603_CL//incorrect",
            "train_x1//SMD0603_D//incorrect",
            "train_x1//SMD0603_R//incorrect",
            "train_x1//SMD0805_CL//incorrect",
            "train_x1//SMD0805_R//incorrect",
            "train_x1//SMD1206_C//incorrect",
            "train_x1//SMD1206_R//incorrect",
            "train_x1//SMD1210_C//incorrect"],
        1: ["train_x1//SMD0402_CL//correct"],
        2: ["train_x1//SMD0402_R//correct"],
        3: ["train_x1//SMD0603_CL//correct"],
        4: ["train_x1//SMD0603_D//correct"],
        5: ["train_x1//SMD0603_R//correct"],
        6: ["train_x1//SMD0805_CL//correct"],
        7: ["train_x1//SMD0805_R//correct"],
        8: ["train_x1//SMD1206_C//correct"],
        9: ["train_x1//SMD1206_R//correct"],
        10: ["train_x1//SMD1210_C//correct"],
        11: ["train_x1//2-SMD//correct"],
        12: ["train_x1//SMA//correct"],
        13: ["train_x1//SMB//correct"],
        14: ["train_x1//SOD110//correct"],
        15: ["train_x1//SOD323F//correct"],
        16: ["train_x1//SOD523//correct"],
        17: ["train_x1//SOT23-5//correct"],
        18: ["train_x1//SOT23-6//correct"],
        19: ["train_x1//SOT143//correct"],
        20: ["train_x1//SOT323//correct"],
        21: ["train_x1//SOT323-5//correct"],
        22: ["train_x1//SOT343//correct"],
        23: ["train_x1//SOT363//correct"],
        24: ["train_x1//SOT523//correct"],
        25: ["train_x1//SOT723//correct"],
        26: ["train_x1//DIP-%D//correct"],
        27: ["train_x1//LQFP0.4-%d//correct"],
        28: ["train_x1//LQFP0.5-%d//correct"],
        29: ["train_x1//LQFP0.8-%d//correct"],
        30: ["train_x1//LQFP0.65-%d//correct"],
        31: ["train_x1//SOIC-%d//correct"]
    }

    aug = Augmenter()
    aug.execute_and_save(CLASSES_PATHS)

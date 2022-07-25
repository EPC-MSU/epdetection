import os
import random
import argparse

import cv2


def read_boards(boards_path):
    images, shapes = [], []
    for top, dirs, files in os.walk(boards_path):
        for i, name in enumerate(files):
            if os.path.isfile(os.path.join(top, name)) and (".jpg" in name):
                try:
                    img = cv2.imread(os.path.join(top, name))
                    img = img[int(img.shape[0] * 0.2): int(img.shape[0] * 0.8),
                              int(img.shape[1] * 0.2):int(img.shape[1] * 0.8)]
                except Exception as err:
                    print(f"ERROR: {err}")
                images.append(img)
                shapes.append(img.shape)
    return images, shapes


def available_to_gen(shapes_boards, shape_element):
    count = 0
    for i, shape in enumerate(shapes_boards):
        k = shape[0] // shape_element[0]
        n = shape[1] // shape_element[1]
        count += k * n
    return count


def read_elements(elements_path):
    paths, shapes = [], []
    top_dirs = []
    for top, dirs, files in os.walk(elements_path):
        top_dirs += dirs
        for i, name in enumerate(files):
            try:
                image = cv2.imread(os.path.join(top, name))
                shape = image.shape
            except Exception as err:
                print(f"ERROR: {err}")
            else:
                el_path_name = os.path.split(os.path.split(top)[0])[1]
                if el_path_name not in paths:
                    shapes.append(shape)
                    paths.append(el_path_name)
                break
    elems = dict(zip(paths, shapes))
    return elems


def save_images(gen_images, save_path):
    successful_saved = 0
    for i, img in enumerate(gen_images):
        alphabet = "abcdefghijklnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        unic_name = "".join([random.choice(alphabet) for _ in range(4)])
        cv2.imwrite(os.path.join(save_path, f"gen_{i + 1}_{unic_name}.jpg"), img)
        successful_saved += 1
    return successful_saved


def make_incorrect(images, shape, how_much=100):
    gen_images = []
    stop = False
    while True:
        img = random.choice(images)
        count = 0
        i_wight = range(0, img.shape[0], shape[0])
        j_hight = range(0, img.shape[1], shape[1])

        i = random.choice(i_wight)
        j = random.choice(j_hight)
        try:
            small_img = img[i:i + shape[0], j:j + shape[1]]
            if small_img.shape != shape:
                raise Exception
            if len(gen_images) < how_much:
                gen_images.append(small_img)
                count += 1
            else:
                stop = True
        except Exception:
            pass
        if stop:
            break
    return gen_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script help you to generate incorrect elements by cutting boards.")
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    required.add_argument("-el", dest="element_folder", type=str, required=True,
                          help="Name of floder with element (without full path). Example: SOT323")
    optional.add_argument("-n", dest="number", type=int, default=100,
                          help="How much incorrect elements need to generate.  \n Default: 100")
    optional.add_argument("--from_path", dest="boards_path", type=str, default="boards",
                          help="Path with boards to cut. Default: boards")
    optional.add_argument("--to_path", dest="elements_path", type=str, default="train_x1",
                          help="Path with elements folders. Default: train_x1")
    args = parser.parse_args()

    print("Loading elements shapes...")
    elems = read_elements(args.elements_path)
    if args.element_folder not in elems.keys():
        print(f"{args.element_folder} not found in {args.elements_path}")
        exit()
    else:
        print(f"{args.element_folder} found in {args.elements_path}\nshape: {elems[args.element_folder]}")

    print("Loading boards...")
    images, shapes_boards = read_boards(args.boards_path)

    a_to_gen = available_to_gen(shapes_boards, elems[args.element_folder])
    print(f"For this shape and boards available to generate images: {a_to_gen}")
    if a_to_gen < args.number:
        print("Not enough images to cut. Exit.")
        exit()

    print("Cutting...")
    gen_images = make_incorrect(images, shape=elems[args.element_folder], how_much=args.number)

    print("Saving...")
    save_path = os.path.join(os.path.join(args.elements_path, args.element_folder), "incorrect")
    if not (os.path.exists(save_path)):
        os.mkdir(save_path)
    successful_saved = save_images(gen_images, save_path)

    print(f"Successfully generated: {successful_saved}")

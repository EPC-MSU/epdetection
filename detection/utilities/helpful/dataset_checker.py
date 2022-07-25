import os
import cv2


def renamer(path, EXT=".jpg"):
    """
    Renamer files in folder as START_NUMBER+0, START_NUMBER+1, START_NUMBER+2, ...
    """
    print("START RENAMER")
    START_NUMBER = 0
    total_renames = 0
    for top, dirs, files in os.walk(path):
        for i, name in enumerate(files):
            if name[-4:] == EXT:
                new_name = "temp_" + str(START_NUMBER + i) + EXT
                os.rename(top + "/" + name, top + "//" + new_name)
                total_renames += 1
    for top, dirs, files in os.walk(path):
        for i, name in enumerate(files):
            if name[-4:] == EXT:
                new_name = str(START_NUMBER + i) + EXT
                os.rename(top + "/" + name, top + "//" + new_name)
    print("Total renames: " + str(total_renames))
    print("DONE RENAMER\n\n")


def resolution_remover(path, target_shape=(100, 100, 3), delete_with_diff_shape=True):
    print("RESOLUTION REMOVER")
    for top, dirs, files in os.walk(path):
        for i, name in enumerate(files):
            try:
                img = cv2.imread(top + "/" + name)
                if target_shape != img.shape:
                    if "%" not in top:
                        print("---Shape:" + str(img.shape) + top + "/" + name, end="")
                    if delete_with_diff_shape:
                        os.remove(top + "/" + name)
                        print("Remove done.")
            except Exception:
                subdir = top.replace(path, "")
                print("---Fail on :" + subdir + "/" + name)
    print("DONE RESOLUTION REMOVER\n\n")


def resolution_checker(path):
    """
    Resolution Checker
    False if different resolution in the same folder
    True if same resolution or type of element can contains different pins
    Fail if file corrupted
    """
    print("START CHECKER RESOLUTION")
    for top, dirs, files in os.walk(path):
        SAME_RES = True
        if len(files) > 0:
            if files[0][-4:] == ".bmp":
                try:
                    img = cv2.imread(top + "/" + files[0])
                    img_shape = img.shape
                except Exception:
                    subdir = top.replace(path, "")
                    print("---Fail on :" + subdir + "/" + files[0])
            for i, name in enumerate(files):
                try:
                    img = cv2.imread(top + "/" + name)
                    if img_shape != img.shape:
                        if top.find("%") < 0:
                            print("---False :" + top + "/" + name)
                            SAME_RES = False
                except Exception:
                    subdir = top.replace(path, "")
                    print("---Fail on :" + subdir + "/" + name)
        subdir = top.replace(path, "")
        print(str(SAME_RES) + " :" + subdir)
    print("DONE CHECKER RESOLUTION\n\n")


def folders_plot(path):
    """
    Make a folders plot with number of files in folders and shapes
    """
    print("FOLDER PLOT")
    for top, dirs, files in os.walk(path):
        try:
            top_default = top
            top = top.replace("\\", "/")
            top = top.split("/")[1:]
            depth = len(top)
            if top != 0:
                s = (depth - 1) * "      " + top[len(top) - 1]

            if "incorrect" in s or "incorrect" in top:
                continue

            s = s.replace("incorrect", "not cor")
            if len(files) > 0:
                s += ": " + str(len(files))
                img = cv2.imread(top_default + "\\" + files[0])

                s += (4 - len(s) // 8) * "\t"

                if ("/".join(top)).find("%") < 0:
                    s += "shape: " + str(img.shape)
                else:
                    s += "shape: different"
            s = s.replace(path, "")
            print(s)
            s = ""
        except Exception:
            pass
    print("END FOLDER PLOT\n\n")


if __name__ == "__main__":
    path = "train_x1"
    print("PATH: " + path)
    renamer(path)

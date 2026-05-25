import argparse
import os
import sys
from ast import literal_eval
import matplotlib.pyplot as plt
try:
    from detection import utils as ut
    from detection.detect import Detector
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from detection import utils as ut
    from detection.detect import Detector


def draw_pins(i: int, detector: Detector) -> None:
    """
    The function displays the pins on the component.

    Parameters
    ----------
    i : int
        Electronic component number.
    detector : Detector
        Detector object.
    """

    x_pins = []
    y_pins = []
    for pin in literal_eval(detector.parameters[i][2]):
        x, y = ut.idxrot(pin, 1.0, -detector.pat_rotations[i])
        x = detector.patterns[i].shape[0] * x
        y = detector.patterns[i].shape[1] * y
        x_pins.append(x)
        y_pins.append(y)

    plt.scatter(x_pins, y_pins, color="red", s=10, marker="o")


def visualize_pcb_components(dump_file: str, output_dir: str) -> None:
    """
    The function saves images of components that were used to train the classifier.

    Parameters
    ----------
    dump_file : srt
        Path to dump file where the classifier is saved.
    output_dir : str
        The folder in which to save component images.
    """

    detector = Detector()
    detector.load_from_file(dump_file)
    os.makedirs(output_dir, exist_ok=True)
    for i, pattern in enumerate(detector.patterns):
        if pattern is None:
            continue

        plt.imshow(pattern.T, origin="lower")
        plt.axis("off")
        name = f"{detector.names[i]} (rot={detector.pat_rotations[i]})"
        plt.title(name)

        if detector.parameters[i][2] != "multipin":
            draw_pins(i, detector)

        name = name.replace("/", "")
        file_path = os.path.join(output_dir, f"{name}.png")
        plt.savefig(file_path)
        print(f"File '{file_path}' saved")
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_file", help="Path to dump file where the classifier is saved")
    parser.add_argument("output_dir", help="The folder in which to save component images")
    parsed_args = parser.parse_args(sys.argv[1:])

    visualize_pcb_components(parsed_args.dump_file, parsed_args.output_dir)

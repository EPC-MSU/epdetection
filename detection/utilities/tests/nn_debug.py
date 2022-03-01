# import os
# import logging
# logging.getLogger("tensorflow").disabled = True
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# print("Loading tensorflow...")

import tensorflow as tf
import matplotlib.pyplot as plt

from utilities.dataworker import DataWorker

SHAPE_3D = (32, 32, 3)
SHAPE_2D = (32, 32)
SHAPE_2D1 = (32, 32, 1)

SAVE_FLODER = "models"
SAVE_SUBFOLDER = "ALL_LQFP_ELEMENTS_100"
MODEL_NAME = "cut32_bw.end_normal.0.5.21_22_23_26_24_27_29_32_31_30_17" \
             "_15_16_18_19_20_8_10_13_5_9_12_11_7_14_33_34_35_36_37_38"

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
    1: ["train_x1//SMD0402_CL//true"],
    2: ["train_x1//SMD0402_R//true"],
    3: ["train_x1//SMD0603_CL//true"],
    4: ["train_x1//SMD0603_D//true"],
    5: ["train_x1//SMD0603_R//true"],
    6: ["train_x1//SMD0805_CL//true"],
    7: ["train_x1//SMD0805_R//true"],
    8: ["train_x1//SMD1206_C//true"],
    9: ["train_x1//SMD1206_R//true"],
    10: ["train_x1//SMD1210_C//true"],
    11: ["train_x1//2-SMD//true"],
    12: ["train_x1//SMA//true"],
    13: ["train_x1//SMB//true"],
    14: ["train_x1//SOD110//true"],
    15: ["train_x1//SOD323F//true"],
    16: ["train_x1//SOD523//true"],
    17: ["train_x1//SOT23-5//true"],
    18: ["train_x1//SOT23-6//true"],
    19: ["train_x1//SOT143//true"],
    20: ["train_x1//SOT323//true"],
    21: ["train_x1//SOT323-5//true"],
    22: ["train_x1//SOT343//true"],
    23: ["train_x1//SOT363//true"],
    24: ["train_x1//SOT523//true"],
    25: ["train_x1//SOT723//true"],
    26: ["train_x1//DIP-%D//true"],
    27: ["train_x1//LQFP0.4-%d//true"],
    28: ["train_x1//LQFP0.5-%d//true"],
    29: ["train_x1//LQFP0.8-%d//true"],
    30: ["train_x1//LQFP0.65-%d//true"],
    31: ["train_x1//SOIC-%d//true"]}


def plot_probability(prob_classes, is_correct):
    x = range(0, len(prob_classes))  # my_colors = {0:"red", 1:"green"}

    x_green, y_green = [], []
    x_red, y_red = [], []

    for i, prob_class_i in enumerate(prob_classes):
        if is_correct[i] == 1:
            x_green.append(x[i])
            y_green.append(prob_class_i)
        elif is_correct[i] == 0:
            x_red.append(x[i])
            y_red.append(prob_class_i)

    fig, axs = plt.subplots()
    plt.title(SAVE_SUBFOLDER)
    plt.scatter(x_green, y_green, color="green", s=7, label="Correct")  # my_colors.get(0 or 1)
    plt.scatter(x_red, y_red, color="red", s=7, label="Incorrect")
    axs.set_axisbelow(True)
    plt.grid(True, linestyle=":")
    plt.ylabel("Probability")
    plt.xlabel("Candidate")
    axs.legend()
    plt.savefig(f"{SAVE_SUBFOLDER}//prob.png", dpi=400, bbox_inches="tight")
    print("Saved.")


# if __name__ == "__main__":
#     images, float_images, classes = DataWorker.load_dataset(CLASSES_PATHS)
#     print("Loading model...")
#     model = tf.keras.models.load_model(f"{SAVE_FLODER}/{SAVE_SUBFOLDER}/{MODEL_NAME}.h5")
#     print("Prediction...")
#     predict_arr = model.predict(float_images)
#
#     print("Working...")
#     prob_classes = []
#     is_correct = []
#     for i, predict in enumerate(predict_arr):
#         prob = predict[1:].max()
#         arg = (predict[1:].argmax() + 1)
#
#         prob_classes.append(prob)
#         if arg == classes[i]:
#             is_correct.append(1)
#             if arg != 0:
#                 p = str(prob)[:6]
#                 cv2.imwrite(f"log//cor_el//prob_{p}--class_{arg}.bmp", images[i])
#         else:
#             is_correct.append(0)
#             p = str(prob)[:6]
#             cv2.imwrite(f"log//incor//prob_{p}--class_{arg}.bmp", images[i])
#
#     prob_classes, is_correct = zip(*sorted(zip(prob_classes, is_correct)))
#
#     prob_classes = prob_classes[len(prob_classes) // 3 * 2:]
#     is_correct = is_correct[len(is_correct) // 3 * 2:]
#
#     print("Plotting...")
#     plot_probability(prob_classes, is_correct)


if __name__ == "__main__":
    model = tf.keras.models.load_model(f"{SAVE_FLODER}/{SAVE_SUBFOLDER}/{MODEL_NAME}.h5")

    data = DataWorker(auto_mode=False)
    images, classes = data.load_dataset(CLASSES_PATHS, SHAPE_2D)

    layer_outputs = [layer.output for layer in model.layers]
    layer_names = [layer.name for layer in model.layers]

    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

    test_im = images[2925]  # Random number
    activations = activation_model.predict(test_im.reshape(1, 32, 32, 1))

    for layer_i, layer_activation in enumerate(activations):
        if "conv" in str(model.layers[layer_i].name).lower():
            if layer_activation.shape[3] == 32:
                fig, axs = plt.subplots(4, 8, figsize=(15, 15))
                fig.suptitle(layer_names[layer_i], fontsize=18)
                [axi.set_axis_off() for axi in axs.ravel()]
                for i in range(layer_activation.shape[3]):
                    axs[i // 8, i % 8].imshow(layer_activation[0, :, :, i])

            if layer_activation.shape[3] == 64:
                fig, axs = plt.subplots(8, 8, figsize=(15, 15))
                fig.suptitle(layer_names[layer_i], fontsize=18)
                [axi.set_axis_off() for axi in axs.ravel()]
                for i in range(layer_activation.shape[3]):
                    axs[i // 8, i % 8].imshow(layer_activation[0, :, :, i])

            if layer_activation.shape[3] == 128:
                fig, axs = plt.subplots(16, 8, figsize=(15, 15))
                fig.suptitle(layer_names[layer_i], fontsize=18)
                [axi.set_axis_off() for axi in axs.ravel()]
                for i in range(layer_activation.shape[3]):
                    axs[i // 8, i % 8].imshow(layer_activation[0, :, :, i])

            plt.savefig(f"{SAVE_FLODER}/{SAVE_SUBFOLDER}/{layer_i}.png", dpi=400, bbox_inches="tight")
            plt.close(fig)
            plt.cla()
            plt.clf()

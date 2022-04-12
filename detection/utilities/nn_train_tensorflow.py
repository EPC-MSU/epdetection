import os
import json
import time

# import logging
# logging.getLogger("tensorflow").disabled = True
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# print("Loading tensorflow...")

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization

from utilities import dataworker


class FitHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.loss = []
        self.acc = []
        self.loss_endepoch = []
        self.acc_endepoch = []

    def __len__(self):
        return len(self.acc)

    def on_train_batch_end(self, batch, logs=None):
        self.loss.append(logs["loss"])
        self.acc.append(logs["accuracy"])
        if batch == self.params["steps"] - 1:
            self.loss_endepoch.append(logs["loss"])
            self.acc_endepoch.append(logs["accuracy"])


class EvalHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = []
        self.loss = []
        self.loss_endepoch = []
        self.acc_endepoch = []

    def __len__(self):
        return len(self.acc)

    def on_test_batch_end(self, batch, logs=None):
        self.loss.append(logs["loss"])
        self.acc.append(logs["accuracy"])
        if batch == self.params["steps"] - 1:
            self.loss_endepoch.append(logs["loss"])
            self.acc_endepoch.append(logs["accuracy"])


def new_model(shape, count_classes):
    model = tf.keras.models.Sequential([
        # BLOCK 1
        Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=shape),
        Dropout(0.25),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        Dropout(0.25),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),

        # BLOCK 2
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        Dropout(0.25),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        Dropout(0.25),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),

        # BLOCK 3
        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        Dropout(0.25),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        Dropout(0.25),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        Dropout(0.25),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(),
        Dense(40, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(count_classes, activation=tf.nn.softmax)])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
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
        31: ["train_x1//SOIC-%d//true"]
    }

    SHAPE_3D = (32, 32, 3)
    SHAPE_2D = (32, 32)
    SHAPE_2D1 = (32, 32, 1)

    SAVE_FLODER = "models"
    SAVE_SUBFOLDER = "NOISE_AUG_LONG_TRAIN"
    MODEL_NAME = "cut32_bw.end_normal.0.5.21_22_23_26_24_27_29_32_31_30_17" \
                 "_15_16_18_19_20_8_10_13_5_9_12_11_7_14_33_34_35_36_37_38"

    EPOCHS = 50
    BATCH_SIZE = 32

    data = dataworker.DataWorker(auto_mode=True)
    # == DEFINE MODEL
    model = new_model(SHAPE_2D1, count_classes=len(CLASSES_PATHS))  # -------------------------------
    model.summary()

    if not os.path.exists(SAVE_FLODER + "/" + SAVE_SUBFOLDER):
        os.makedirs(SAVE_FLODER + "/" + SAVE_SUBFOLDER)

    with open(f"{SAVE_FLODER}/{SAVE_SUBFOLDER}/summary.txt", "w+") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    with open(f"{SAVE_FLODER}//{SAVE_SUBFOLDER}//architecture.json", "w") as f:
        config = model.to_json()
        json.dump(config, f)

    #   Load model configuration
    # json_file = open(f"{SAVE_FLODER}/{SAVE_SUBFOLDER}/{MODEL_NAME}.json")
    # config = json.load(json_file)
    # model = tf.keras.models.model_from_json(config)
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #              loss="sparse_categorical_crossentropy",
    #              metrics=["accuracy"])

    # == LOADING DATASET
    data.load_dataset(CLASSES_PATHS, SHAPE_2D)

    # == PREPARETING DATASET
    data.split_and_shuffle_auto(0.2)

    data.define_class_weight()

    with open(f"{SAVE_FLODER}/{SAVE_SUBFOLDER}/parametrs.txt", "w+") as f:
        f.write("EPOCHS = ")
        f.write(str(EPOCHS))

        f.write("\nSHAPE_3D = ")
        f.write(str(SHAPE_3D))

        f.write("\nMODEL_NAME = ")
        f.write(str(MODEL_NAME))

        f.write("\nCLASSES_PATHS\n")
        for i, (key, value) in enumerate(CLASSES_PATHS.items()):
            f.write("\t" + str(key) + " : " + str(value) + "\n")

        f.write("class_weight\n")
        for i in range(len(data.class_weight)):
            f.write(f"\t{i} : {data.class_weight[i]}\n")

    # == TRAIN
    hist_fit, hist_eval = FitHistoryCallback(), EvalHistoryCallback()
    start_time = time.time()
    for epoch in range(EPOCHS):
        print(f"\nEPOCH: {epoch + 1}/{EPOCHS}", end="")
        if epoch > 0:
            spend = time.time() - start_time
            left = spend * (EPOCHS) / (epoch) - spend
            print("\nSpend time: {:.0f}s ({:.1f}m)\tTime left: {:.0f}s ({:.1f}m)".format(spend, spend / 60, left,
                                                                                         left / 60), end="")
        print("     ")  # Tensorflow eat this print ¯\_(ツ)_/¯
        model.fit(data.x_train, data.y_train, batch_size=BATCH_SIZE, callbacks=[hist_fit],
                  class_weight=data.class_weight)
        model.evaluate(data.x_test, data.y_test, batch_size=BATCH_SIZE, callbacks=[hist_eval])

    spend = time.time() - start_time
    print("\nSpend time: {:.0f}s ({:.0f}m)".format(spend, spend / 60))
    print("_" * 65)
    model.save(f"{SAVE_FLODER}/{SAVE_SUBFOLDER}/{MODEL_NAME}.h5")
    print("Model saved to")
    print(f"{SAVE_FLODER}/{SAVE_SUBFOLDER}/{MODEL_NAME}.h5")

    # == PLOTTING
    data.save_plots(hist_fit, hist_eval, f"{SAVE_FLODER}/{SAVE_SUBFOLDER}/", EPOCHS)
    print("End.")


if __name__ == "__main__":
    main()

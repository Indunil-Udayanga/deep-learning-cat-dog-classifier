import argparse
import subprocess
import sys
import time

def _install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

try:
    import opendatasets as od
except ImportError:
    _install("opendatasets")
    import opendatasets as od

try:
    import cv2
except ImportError:
    _install("opencv-python")
    import cv2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


BATCH_SIZE  = 32
IMAGE_SIZE  = (128, 128)
EPOCHS      = 20
SEED        = 42

DATASET_URL = "https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification"
TRAIN_DIR   = "cats-and-dogs-for-classification/cats_dogs/train"
TEST_DIR    = "cats-and-dogs-for-classification/cats_dogs/test"
MODEL_SAVE  = "cat_dog_cnn.keras"


def download_data():
    od.download(DATASET_URL)


def load_datasets():
    common = dict(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, seed=SEED)

    train_data = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, subset="training",   validation_split=0.1, **common
    )
    val_data = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, subset="validation", validation_split=0.1, **common
    )
    test_data = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, **{k: v for k, v in common.items() if k != "seed"}
    )

    class_names = train_data.class_names
    print("Classes:", class_names)
    return train_data, val_data, test_data, class_names


def visualise_batch(dataset, class_names):
    plt.figure(figsize=(10, 4))
    for images, labels in dataset.take(1):
        for i in range(10):
            ax = plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def preprocess(train_data, val_data, test_data):
    AUTOTUNE  = tf.data.AUTOTUNE
    normalise = lambda x, y: (x / 255.0, y)

    train_data = (train_data
                  .map(normalise, num_parallel_calls=AUTOTUNE)
                  .cache().shuffle(1000).prefetch(AUTOTUNE))
    val_data   = (val_data
                  .map(normalise, num_parallel_calls=AUTOTUNE)
                  .cache().prefetch(AUTOTUNE))
    test_data  = (test_data
                  .map(normalise, num_parallel_calls=AUTOTUNE)
                  .cache().prefetch(AUTOTUNE))
    return train_data, val_data, test_data


def build_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ], name="data_augmentation")

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(*IMAGE_SIZE, 3)),
        data_augmentation,

        tf.keras.layers.Conv2D(32,  kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64,  kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(32,  activation="relu"),
        tf.keras.layers.Dense(1,   activation="sigmoid"),
    ], name="cnn_from_scratch")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train(model, train_data, val_data):
    start   = time.time()
    history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data)
    print(f"Training time: {time.time() - start:.1f}s")
    return history


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     color="teal",   label="train")
    axes[0].plot(history.history["val_accuracy"], color="orange", label="val")
    axes[0].set_title("Accuracy"); axes[0].legend()

    axes[1].plot(history.history["loss"],     color="teal",   label="train")
    axes[1].plot(history.history["val_loss"], color="orange", label="val")
    axes[1].set_title("Loss"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history_cnn.png", dpi=120)
    plt.show()


def evaluate(model, test_data):
    precision = tf.keras.metrics.Precision()
    recall    = tf.keras.metrics.Recall()
    accuracy  = tf.keras.metrics.BinaryAccuracy()

    for X, y in test_data:
        yhat = model.predict(X, verbose=0)
        precision.update_state(y, yhat)
        recall.update_state(y, yhat)
        accuracy.update_state(y, yhat)

    print(f"Precision : {precision.result():.4f}")
    print(f"Recall    : {recall.result():.4f}")
    print(f"Accuracy  : {accuracy.result():.4f}")


def predict_single(model, class_names, img_path):
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb); plt.axis("off"); plt.show()

    resized = tf.image.resize(image_rgb, IMAGE_SIZE)
    scaled  = resized / 255.0
    input_  = np.expand_dims(scaled, axis=0)

    yhat  = model.predict(input_, verbose=0)
    label = class_names[1] if yhat[0][0] > 0.5 else class_names[0]
    print(f"Raw prediction : {yhat[0][0]:.4f}")
    print(f"Predicted class: {label}")
    return label


def main():
    parser = argparse.ArgumentParser(description="Cat vs Dog CNN pipeline")
    parser.add_argument("--img", default=None, help="Path to a single image for prediction only")
    args, _ = parser.parse_known_args()

    download_data()
    train_data, val_data, test_data, class_names = load_datasets()
    visualise_batch(train_data, class_names)
    train_data, val_data, test_data = preprocess(train_data, val_data, test_data)
    model   = build_model()
    history = train(model, train_data, val_data)
    plot_history(history)
    evaluate(model, test_data)
    model.save(MODEL_SAVE)

    img_path = args.img or f"{TEST_DIR}/cats/cat.4004.jpg"
    predict_single(model, class_names, img_path)


if __name__ == "__main__":
    main()

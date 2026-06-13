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


BATCH_SIZE      = 32
IMAGE_SIZE      = (128, 128)
EPOCHS_FROZEN   = 5
EPOCHS_FINETUNE = 3
FINETUNE_LAYERS = 20
SEED            = 42

DATASET_URL = "https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification"
TRAIN_DIR   = "cats-and-dogs-for-classification/cats_dogs/train"
TEST_DIR    = "cats-and-dogs-for-classification/cats_dogs/test"
MODEL_SAVE  = "cat_dog_xception.keras"

xception_preprocess = tf.keras.applications.xception.preprocess_input


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
    AUTOTUNE = tf.data.AUTOTUNE
    prep     = lambda x, y: (xception_preprocess(x), y)

    train_data = (train_data
                  .map(prep, num_parallel_calls=AUTOTUNE)
                  .cache().shuffle(1000).prefetch(AUTOTUNE))
    val_data   = (val_data
                  .map(prep, num_parallel_calls=AUTOTUNE)
                  .cache().prefetch(AUTOTUNE))
    test_data  = (test_data
                  .map(prep, num_parallel_calls=AUTOTUNE)
                  .cache().prefetch(AUTOTUNE))
    return train_data, val_data, test_data


def build_model():
    base_model = tf.keras.applications.Xception(
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(*IMAGE_SIZE, 3)),
        data_augmentation,
        base_model,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(32,  activation="relu"),
        tf.keras.layers.Dense(1,   activation="sigmoid"),
    ], name="xception_transfer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()
    return model, base_model


def train_frozen(model, train_data, val_data):
    start   = time.time()
    history = model.fit(train_data, epochs=EPOCHS_FROZEN, validation_data=val_data)
    print(f"Phase 1 time: {time.time() - start:.1f}s")
    return history


def fine_tune(model, base_model, train_data, val_data):
    base_model.trainable = True
    for layer in base_model.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    start      = time.time()
    history_ft = model.fit(train_data, epochs=EPOCHS_FINETUNE, validation_data=val_data)
    print(f"Phase 2 time: {time.time() - start:.1f}s")
    return history_ft


def plot_history(history, history_ft=None):
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    split    = len(acc)

    if history_ft:
        acc      += history_ft.history["accuracy"]
        val_acc  += history_ft.history["val_accuracy"]
        loss     += history_ft.history["loss"]
        val_loss += history_ft.history["val_loss"]

    epochs_range = range(len(acc))
    fig, axes    = plt.subplots(1, 2, figsize=(12, 4))

    for ax, train_vals, val_vals, title in zip(
        axes, [acc, loss], [val_acc, val_loss], ["Accuracy", "Loss"]
    ):
        ax.plot(epochs_range, train_vals, color="teal",   label="train")
        ax.plot(epochs_range, val_vals,   color="orange", label="val")
        if history_ft:
            ax.axvline(x=split - 0.5, color="gray", linestyle="--", label="fine-tune start")
        ax.set_title(title); ax.legend()

    plt.tight_layout()
    plt.savefig("training_history_xception.png", dpi=120)
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

    resized      = tf.image.resize(image_rgb, IMAGE_SIZE)
    preprocessed = xception_preprocess(resized)
    input_       = np.expand_dims(preprocessed, axis=0)

    yhat  = model.predict(input_, verbose=0)
    label = class_names[1] if yhat[0][0] > 0.5 else class_names[0]
    print(f"Raw prediction : {yhat[0][0]:.4f}")
    print(f"Predicted class: {label}")
    return label


def main():
    parser = argparse.ArgumentParser(description="Cat vs Dog Transfer Learning pipeline")
    parser.add_argument("--img",         default=None,        help="Single image path for prediction only")
    parser.add_argument("--no-finetune", action="store_true", help="Skip the fine-tuning phase")
    args, _ = parser.parse_known_args()

    download_data()
    train_data, val_data, test_data, class_names = load_datasets()
    visualise_batch(train_data, class_names)
    train_data, val_data, test_data = preprocess(train_data, val_data, test_data)

    model, base_model = build_model()
    history           = train_frozen(model, train_data, val_data)
    history_ft        = None if args.no_finetune else fine_tune(model, base_model, train_data, val_data)

    plot_history(history, history_ft)
    evaluate(model, test_data)
    model.save(MODEL_SAVE)

    img_path = args.img or f"{TEST_DIR}/cats/cat.4004.jpg"
    predict_single(model, class_names, img_path)


if __name__ == "__main__":
    main()

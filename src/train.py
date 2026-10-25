"""Tomato leaf disease classification with MobileNetV2 transfer learning.

Usage:
    python src/train.py --data data/raw --out results

`data/raw` holds one folder of images per class. The script splits the images 70/15/15 (stratified, seed 42),
trains a classification head on a frozen MobileNetV2 base and evaluates it on the held-out test set.
"""
import argparse
import os

import numpy as np
import splitfolders
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def make_generators(split_dir):
    """Augmented generator for training; rescaling only for validation and test."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=15, zoom_range=0.1, width_shift_range=0.1,
        height_shift_range=0.1, brightness_range=[0.8, 1.2], horizontal_flip=True, fill_mode="nearest",
    )
    plain_gen = ImageDataGenerator(rescale=1.0 / 255)
    flow = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
    train = train_gen.flow_from_directory(os.path.join(split_dir, "train"), shuffle=True, **flow)
    val = plain_gen.flow_from_directory(os.path.join(split_dir, "val"), shuffle=False, **flow)
    test = plain_gen.flow_from_directory(os.path.join(split_dir, "test"), shuffle=False, **flow)
    return train, val, test


def build_model(num_classes):
    """MobileNetV2 (ImageNet weights, frozen) with a small classification head."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,))
    for layer in base.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw", help="folder with one sub-folder of images per class")
    parser.add_argument("--split", default="data/split", help="where the train/val/test split is written")
    parser.add_argument("--out", default="results", help="output folder for the model and reports")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.isdir(os.path.join(args.split, "train")):
        splitfolders.ratio(args.data, output=args.split, seed=SEED, ratio=(0.7, 0.15, 0.15))
    train, val, test = make_generators(args.split)

    model = build_model(len(train.class_indices))
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(os.path.join(args.out, "best_model.keras"), monitor="val_accuracy", save_best_only=True),
    ]
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)

    y_prob = model.predict(test)
    y_pred = np.argmax(y_prob, axis=1)
    labels = list(test.class_indices.keys())
    report = classification_report(test.classes, y_pred, target_names=labels, digits=2)
    cm = confusion_matrix(test.classes, y_pred)
    with open(os.path.join(args.out, "classification_report.txt"), "w") as fh:
        fh.write(report + "\nConfusion matrix (rows = true, columns = predicted):\n" + str(cm) + "\n")
    print(report)


if __name__ == "__main__":
    main()

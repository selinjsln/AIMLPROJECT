# model.py
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(num_classes=47):
    model = models.Sequential(
        [
            # Input: gambar 28x28 grayscale
            layers.Input(shape=(28, 28, 1)),
            # Blok 1: Feature extraction level rendah (edge, garis)
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Blok 2: Feature extraction level tinggi (pola karakter)
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Flatten + Classifier
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),  # Output: 47 kelas
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model

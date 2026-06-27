# dataset_loader.py
import tensorflow as tf
import numpy as np


def load_emnist():
    # EMNIST 'balanced' subset: 47 kelas (0-9, A-Z, a-z subset)
    # Download otomatis lewat tensorflow_datasets
    import tensorflow_datasets as tfds

    (ds_train, ds_test) = tfds.load(
        "emnist/balanced", split=["train", "test"], as_supervised=True
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0  # Normalisasi 0-1
        image = tf.image.rot90(image, k=3)           # Rotate DULU (masih 28,28,1)
        image = tf.image.flip_left_right(image)      # Flip DULU (masih 28,28,1)
        image = tf.squeeze(image, axis=-1)           # Baru squeeze → (28,28)
        image = tf.expand_dims(image, axis=-1)       # Tambah channel kembali → (28,28,1)
        return image, label

    ds_train = ds_train.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test

# train.py
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from dataset_loader import load_emnist
from model import build_cnn


def train_and_evaluate():
    # --- Load Data ---
    print("Loading EMNIST dataset...")
    ds_train, ds_test = load_emnist()

    # --- Build Model ---
    model = build_cnn(num_classes=47)
    model.summary()

    # --- Training ---
    print("\nMulai training...")
    history = model.fit(
        ds_train,
        epochs=15,
        validation_data=ds_test,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True
            )
        ],
    )

    # --- Simpan Model ---
    model.save("ocr_model.keras")
    print("\nModel tersimpan: ocr_model.keras")

    # --- Plot Training Curve ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.show()
    print("Grafik tersimpan: training_curve.png")

    # --- Evaluasi ---
    print("\n=== Evaluasi pada Test Set ===")
    test_loss, test_acc = model.evaluate(ds_test)
    print(f"Test Accuracy : {test_acc*100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")


if __name__ == "__main__":
    train_and_evaluate()

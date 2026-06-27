import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf
import sys

EMNIST_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "ocr_model.keras")
    return tf.keras.models.load_model(model_path)


def fix_common_errors(text):
    """Post-processing: perbaiki kesalahan umum model."""
    result = ""
    for i, char in enumerate(text):
        prev_char = text[i - 1] if i > 0 else ""
        next_char = text[i + 1] if i < len(text) - 1 else ""
        is_between_letters = prev_char.isalpha() or next_char.isalpha()

        if char == "0" and is_between_letters:
            result += "O"
        elif char == "1" and is_between_letters:
            result += "I"
        elif char == "5" and is_between_letters:
            result += "S"
        elif char == "8" and is_between_letters:
            result += "B"
        else:
            result += char
    return result


def preprocess_image(image_path):
    """Tingkatkan kualitas gambar sebelum segmentasi."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Gambar tidak bisa dibaca: {image_path}")

    # 1. Resize kalau terlalu besar
    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, (1000, int(img.shape[0] * scale)))

    # 2. Tingkatkan kontras (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 3. Denoise
    img = cv2.fastNlMeansDenoising(img, h=10)

    # 4. Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    return img


def segment_characters(image_path):
    """Segmentasi gambar menjadi potongan per karakter + deteksi spasi."""
    # Pakai preprocess_image (CLAHE + denoise + sharpen)
    img = preprocess_image(image_path)

    # Blur ringan lalu threshold
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilasi: gabungkan titik yang putus (misal titik huruf i, j)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Deteksi connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    # Hitung median area untuk filter noise secara adaptif
    all_areas = [stats[i][4] for i in range(1, num_labels)]
    if not all_areas:
        return []
    median_area = np.median(all_areas)
    min_area = max(30, median_area * 0.05)

    characters = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # Filter noise
        if area < min_area or w < 4 or h < 4:
            continue

        # Abaikan blob terlalu lebar (kemungkinan 2 huruf nyambung)
        if w > h * 3:
            continue

        pad = 4
        char_crop = binary[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
        characters.append({"crop": char_crop, "x": x, "w": w, "h": h})

    # Urutkan kiri ke kanan
    characters.sort(key=lambda c: c["x"])

    if not characters:
        return []

    # Hitung gap antar karakter untuk deteksi spasi
    gaps = []
    for i in range(1, len(characters)):
        gap = characters[i]["x"] - (characters[i - 1]["x"] + characters[i - 1]["w"])
        if gap > 0:
            gaps.append(gap)

    # Threshold spasi = median gap * 2.0
    space_threshold = np.median(gaps) * 2.0 if gaps else 999

    # Sisipkan spasi kalau gap antar karakter cukup besar
    result = []
    for i, char in enumerate(characters):
        if i > 0:
            gap = char["x"] - (characters[i - 1]["x"] + characters[i - 1]["w"])
            if gap > space_threshold:
                result.append({"is_space": True})
        result.append({**char, "is_space": False})

    return result


def preprocess_char(char_crop):
    """Resize ke 28x28 dan normalisasi sebelum masuk model."""
    resized = cv2.resize(char_crop, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 28, 28, 1)


def predict_text(image_path):
    """Pipeline utama: gambar → teks."""
    model = load_model()
    characters = segment_characters(image_path)

    if not characters:
        return "Tidak ada karakter terdeteksi."

    result = ""
    for char_data in characters:
        # Kalau ini spasi, langsung tambahkan
        if char_data.get("is_space"):
            result += " "
            continue

        input_tensor = preprocess_char(char_data["crop"])
        prediction = model.predict(input_tensor, verbose=0)
        char_idx = prediction.argmax()
        confidence = prediction.max()

        if confidence > 0.4:
            result += EMNIST_LABELS[char_idx]

    return fix_common_errors(result.strip())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(predict_text(sys.argv[1]))
    else:
        print("Error: Masukkan path gambar sebagai argumen.")

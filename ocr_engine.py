# ocr_engine.py
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
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Gambar tidak bisa dibaca: {image_path}")

    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, (1000, int(img.shape[0] * scale)))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.fastNlMeansDenoising(img, h=10)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img


def group_into_lines(characters):
    if not characters:
        return []

    chars_sorted = sorted(characters, key=lambda c: c["y"] + c["h"] / 2)

    avg_height = np.median([c["h"] for c in chars_sorted])
    line_threshold = avg_height * 0.6  # toleransi variasi tinggi dalam 1 baris

    lines = []
    current_line = [chars_sorted[0]]
    current_center = chars_sorted[0]["y"] + chars_sorted[0]["h"] / 2

    for c in chars_sorted[1:]:
        center_y = c["y"] + c["h"] / 2
        if abs(center_y - current_center) <= line_threshold:
            current_line.append(c)
        else:
            lines.append(current_line)
            current_line = [c]
        current_center = np.mean([cc["y"] + cc["h"] / 2 for cc in current_line])

    lines.append(current_line)

    # Urutkan baris dari atas ke bawah, lalu tiap baris kiri ke kanan
    lines.sort(key=lambda line: np.mean([c["y"] + c["h"] / 2 for c in line]))
    for line in lines:
        line.sort(key=lambda c: c["x"])

    return lines


def segment_characters(image_path):
    img = preprocess_image(image_path)

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    all_areas = [stats[i][4] for i in range(1, num_labels)]
    if not all_areas:
        return []
    median_area = np.median(all_areas)
    min_area = max(30, median_area * 0.05)

    characters = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 4 or h < 4:
            continue
        if w > h * 3:
            continue
        pad = 4
        char_crop = binary[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
        characters.append({"crop": char_crop, "x": x, "y": y, "w": w, "h": h})

    if not characters:
        return []

    lines = group_into_lines(characters)

    result = []
    for line_idx, line in enumerate(lines):
        gaps = []
        for i in range(1, len(line)):
            gap = line[i]["x"] - (line[i - 1]["x"] + line[i - 1]["w"])
            if gap > 0:
                gaps.append(gap)
        space_threshold = np.median(gaps) * 2.0 if gaps else 999

        for i, char in enumerate(line):
            if i > 0:
                gap = char["x"] - (line[i - 1]["x"] + line[i - 1]["w"])
                if gap > space_threshold:
                    result.append({"is_space": True})
            result.append({**char, "is_space": False})

        if line_idx < len(lines) - 1:
            result.append({"is_space": True})  # pemisah antar baris

    return result


def preprocess_char(char_crop):
    h, w = char_crop.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = char_crop

    margin = max(2, int(size * 0.15))
    square = cv2.copyMakeBorder(
        square, margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=0
    )

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 28, 28, 1)


def predict_text(image_path):
    model = load_model()
    characters = segment_characters(image_path)
    if not characters:
        return "Tidak ada karakter terdeteksi."

    result = ""
    for char_data in characters:
        if char_data.get("is_space"):
            result += " "
            continue
        input_tensor = preprocess_char(char_data["crop"])
        prediction = model.predict(input_tensor, verbose=0)
        char_idx = prediction.argmax()
        confidence = prediction.max()
        if confidence > 0.4:
            result += EMNIST_LABELS[char_idx]
    return fix_common_errors(" ".join(result.split()))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(predict_text(sys.argv[1]))
    else:
        print("Error: Masukkan path gambar sebagai argumen.")
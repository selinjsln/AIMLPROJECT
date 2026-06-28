"""
ocr_engine.py — Hybrid OCR Pipeline
Menggabungkan CNN (trained on EMNIST) + TrOCR (Microsoft pre-trained)
untuk hasil yang lebih akurat pada berbagai jenis tulisan.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from PIL import Image

# ─── LABEL MAPPING EMNIST ────────────────────────────────────────────────────
EMNIST_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

# ─── MODEL CACHE (load sekali, reuse) ────────────────────────────────────────
_cnn_model = None
_trocr_proc = None
_trocr_model = None


def load_cnn():
    global _cnn_model
    if _cnn_model is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ocr_model.keras"
        )
        _cnn_model = tf.keras.models.load_model(model_path)
    return _cnn_model


def load_trocr():
    global _trocr_proc, _trocr_model
    if _trocr_proc is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        _trocr_proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
    return _trocr_proc, _trocr_model


# ═══════════════════════════════════════════════════════════════════════════════
# CNN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def preprocess_image_cnn(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if img.shape[1] > 1200:
        scale = 1200 / img.shape[1]
        img = cv2.resize(
            img, (1200, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA
        )
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.fastNlMeansDenoising(img, h=12)
    sharp_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharp_k)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dil_k = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, dil_k, iterations=1)
    return binary


def detect_lines(binary):
    h_proj = np.sum(binary, axis=1)
    h_smooth = np.convolve(h_proj, np.ones(5) / 5, mode="same")
    threshold = np.max(h_smooth) * 0.05
    in_line, start = False, 0
    raw_lines = []
    for i, val in enumerate(h_smooth):
        if val > threshold and not in_line:
            in_line, start = True, i
        elif val <= threshold and in_line:
            in_line = False
            if i - start > 8:
                raw_lines.append([start, i])
    if in_line:
        raw_lines.append([start, len(h_smooth) - 1])
    merged = []
    for line in raw_lines:
        if merged and line[0] - merged[-1][1] < 10:
            merged[-1][1] = line[1]
        else:
            merged.append(line)
    return merged


def segment_line_chars(line_binary):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(line_binary)
    if num_labels <= 1:
        return []
    areas = [stats[i][4] for i in range(1, num_labels)]
    median_area = np.median(areas)
    min_area = max(20, median_area * 0.04)
    chars = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 3 or h < 3:
            continue
        if w > h * 3.5:
            continue
        pad = 5
        crop = line_binary[
            max(0, y - pad) : min(line_binary.shape[0], y + h + pad),
            max(0, x - pad) : min(line_binary.shape[1], x + w + pad),
        ]
        chars.append({"crop": crop, "x": x, "w": w, "h": h})
    chars.sort(key=lambda c: c["x"])
    if not chars:
        return []
    gaps = [
        chars[i]["x"] - (chars[i - 1]["x"] + chars[i - 1]["w"])
        for i in range(1, len(chars))
        if chars[i]["x"] - (chars[i - 1]["x"] + chars[i - 1]["w"]) > 0
    ]
    if gaps:
        avg_w = np.mean([c["w"] for c in chars])
        space_threshold = max(np.median(gaps) * 1.8, avg_w * 0.5)
    else:
        space_threshold = 9999
    result = []
    for i, char in enumerate(chars):
        if i > 0:
            gap = char["x"] - (chars[i - 1]["x"] + chars[i - 1]["w"])
            if gap > space_threshold:
                result.append({"is_space": True})
        result.append({**char, "is_space": False})
    return result


def preprocess_char_cnn(crop):
    h, w = crop.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    square[
        (size - h) // 2 : (size - h) // 2 + h, (size - w) // 2 : (size - w) // 2 + w
    ] = crop
    margin = max(2, int(size * 0.15))
    square = cv2.copyMakeBorder(
        square, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0
    )
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)


def fix_common_errors(text):
    replacements = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
    chars = list(text)
    result = []
    for i, ch in enumerate(chars):
        if ch in (" ", "\n"):
            result.append(ch)
            continue
        prev = chars[i - 1] if i > 0 else ""
        nxt = chars[i + 1] if i < len(chars) - 1 else ""
        if ch in replacements and (prev.isalpha() or nxt.isalpha()):
            result.append(replacements[ch])
        else:
            result.append(ch)
    return "".join(result)


def predict_with_cnn(image_path):
    model = load_cnn()
    binary = preprocess_image_cnn(image_path)
    lines = detect_lines(binary)
    if not lines:
        return ""
    full_result = []
    for idx, (row_start, row_end) in enumerate(lines):
        line_binary = binary[row_start:row_end, :]
        tokens = segment_line_chars(line_binary)
        line_text = ""
        for token in tokens:
            if token.get("is_space"):
                line_text += " "
                continue
            crop = token.get("crop")
            if crop is None or crop.size == 0:
                continue
            tensor = preprocess_char_cnn(crop)
            prediction = model.predict(tensor, verbose=0)[0]
            char_idx = int(np.argmax(prediction))
            confidence = float(prediction[char_idx])
            if confidence > 0.35:
                line_text += EMNIST_LABELS[char_idx]
        if line_text.strip():
            full_result.append(line_text.strip())
    result = "\n".join(full_result)
    return fix_common_errors(result)


# ═══════════════════════════════════════════════════════════════════════════════
# TrOCR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def predict_with_trocr(image_path):
    processor, model = load_trocr()
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def predict_text(image_path):
    """
    Hybrid: CNN + TrOCR
    - CNN: bagus untuk karakter terpisah / printed text
    - TrOCR: bagus untuk tulisan tangan sambung / cursive
    - Pilih hasil terpanjang dan paling masuk akal
    """
    cnn_result = ""
    try:
        cnn_result = predict_with_cnn(image_path).strip()
    except Exception:
        pass

    trocr_result = ""
    try:
        trocr_result = predict_with_trocr(image_path).strip()
    except Exception:
        pass

    if not cnn_result and not trocr_result:
        return "Tidak ada karakter terdeteksi."
    if not cnn_result:
        return trocr_result
    if not trocr_result:
        return cnn_result

    cnn_words = len(cnn_result.split())
    trocr_words = len(trocr_result.split())

    # TrOCR jauh lebih panjang → pakai TrOCR
    if trocr_words > cnn_words * 1.5:
        return trocr_result

    # CNN sebanding atau lebih panjang → pakai CNN
    if cnn_words >= trocr_words * 0.8:
        return cnn_result

    return trocr_result


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(predict_text(sys.argv[1]))
    else:
        print("Error: Masukkan path gambar sebagai argumen.")
        print("Contoh: python ocr_engine.py gambar.png")

"""
trocr_engine.py — TrOCR fallback untuk karakter yang CNN-nya kurang yakin.

Cara kerja:
- Terima crop karakter (numpy array, hasil segmentasi dari ocr_engine.py)
- Kirim ke TrOCR (microsoft/trocr-base-handwritten)
- Kembalikan karakter hasil prediksi

Install dulu sebelum pakai:
    pip install transformers torch pillow
    (model ~1GB akan auto-download saat pertama kali dijalankan)
"""

from PIL import Image
import numpy as np

_processor = None
_model = None


def _load_trocr():
    """Lazy load TrOCR — hanya load sekali, bukan setiap prediksi."""
    global _processor, _model
    if _processor is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        print("[TrOCR] Loading model (pertama kali ~1GB, berikutnya dari cache)...")
        _processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        print("[TrOCR] Model siap.")
    return _processor, _model


def predict_char_trocr(char_crop_binary: np.ndarray) -> str:
    """
    Prediksi satu karakter menggunakan TrOCR.

    Args:
        char_crop_binary: numpy array (H, W), nilai 0-255, hasil binary dari segmentasi.
                          Latar belakang hitam (0), karakter putih (255).

    Returns:
        String karakter hasil prediksi (bisa lebih dari 1 karakter kalau TrOCR
        menghasilkan lebih, ambil karakter pertama saja).
    """
    processor, model = _load_trocr()

    # TrOCR butuh RGB image dengan latar PUTIH dan teks HITAM (kebalikan binary kita)
    inverted = 255 - char_crop_binary

    # Resize ke ukuran yang layak dibaca TrOCR (minimal ~32px)
    img = Image.fromarray(inverted).convert("RGB")
    w, h = img.size
    if w < 32 or h < 32:
        scale = max(32 / w, 32 / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Tambah padding putih supaya karakter tidak terlalu mepet tepi
    padded = Image.new("RGB", (img.width + 20, img.height + 20), (255, 255, 255))
    padded.paste(img, (10, 10))

    pixel_values = processor(images=padded, return_tensors="pt").pixel_values

    import torch

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=5)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Ambil karakter pertama saja (kita prediksi per-karakter)
    return text[0] if text else ""

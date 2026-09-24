"""Optional OCR fallback for ``docintel extract`` via Tesseract (``ocr`` extra + tesseract binary).

Not used in the PubTabNet evaluation, which uses the dataset's own cell text boxes.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from docintel.structure import Word


def words_from_tesseract_data(data: dict[str, list[Any]], min_conf: float = 30.0) -> list[Word]:
    """Convert ``pytesseract.image_to_data(..., output_type=DICT)`` output to Words."""
    words = []
    for i, text in enumerate(data.get("text", [])):
        text = str(text).strip()
        try:
            conf = float(data["conf"][i])
        except (KeyError, ValueError, TypeError):
            conf = -1.0
        if not text or conf < min_conf:
            continue
        x, y = float(data["left"][i]), float(data["top"][i])
        w, h = float(data["width"][i]), float(data["height"][i])
        words.append(Word(text, (x, y, x + w, y + h)))
    return words


def tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def scale_words(words: list[Word], factor: float) -> list[Word]:
    out = []
    for w in words:
        x1, y1, x2, y2 = w.bbox
        out.append(Word(w.text, (x1 / factor, y1 / factor, x2 / factor, y2 / factor)))
    return out


def ocr_words(image: Image.Image) -> list[Word]:  # pragma: no cover - needs tesseract binary
    import pytesseract

    # Table crops are often ~10 px text; Tesseract wants ~30 px, so upsample small images.
    factor = 3.0 if image.height < 400 else 1.0
    big = image.resize(
        (round(image.width * factor), round(image.height * factor)), Image.Resampling.LANCZOS
    )
    data = pytesseract.image_to_data(big, output_type=pytesseract.Output.DICT, config="--psm 11")
    return scale_words(words_from_tesseract_data(data), factor)

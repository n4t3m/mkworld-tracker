"""Detection package — screen detectors for Mario Kart game state."""

import os
import shutil

import pytesseract

# Auto-detect Tesseract path on Windows if not already on PATH.
if not shutil.which("tesseract"):
    _WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_WIN_TESSERACT):
        pytesseract.pytesseract.tesseract_cmd = _WIN_TESSERACT

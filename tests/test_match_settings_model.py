"""Model reading on a real custom-room settings card."""
from pathlib import Path

import cv2
import pytest

from mktracker.detection.match_settings import MatchSettings, MatchSettingsDetector
from mktracker.detection.name_model import NameModelReader


_FRAME = Path(__file__).parent / "fixtures" / "match_settings" / "12p-first-frame.jpg"
_EXPECTED = MatchSettings("150cc", "Two Teams", "Normal", "No COM", 12, "10 seconds")


def test_settings_model_path_parses_six_values_without_tesseract(monkeypatch):
    import pytesseract

    class Reader:
        disabled = False

        def predict_crops(self, crops):
            if len(crops) == 1:
                return [("The rules have been decided!", .99)]
            assert len(crops) == 6
            return [(value, .99) for value in
                    ("150cc", "Two Teams", "Normal.", "No COM", "12 Races", "10 seconds")]

    monkeypatch.setattr(pytesseract, "image_to_string",
                        lambda *args, **kwargs: pytest.fail("Settings OCR called Tesseract"))
    frame = cv2.imread(str(_FRAME))
    assert MatchSettingsDetector(Reader()).detect(frame) == _EXPECTED


@pytest.mark.skipif(not NameModelReader.available(), reason="Optional local model is unavailable")
def test_local_model_reads_real_settings_card():
    frame = cv2.imread(str(_FRAME))
    reader = NameModelReader()
    try:
        assert MatchSettingsDetector(reader).detect(frame) == _EXPECTED
    finally:
        reader.close()

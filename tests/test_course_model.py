"""Course banner crops for the trained recognizer."""
from pathlib import Path

import cv2
import pytest

from mktracker.detection.course_model import (
    course_banner_crops, race_start_card_crop, read_course_banner_model,
    read_race_start_card_model,
)
from mktracker.detection.name_model import NameModelReader


_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("filename", [
    "track_select/peach_stadium.png",
    "track_select/bowsers_castle_24p.png",
    "video_analysis/course-preview.jpg",
])
def test_course_banner_locator_finds_selected_course(filename):
    frame = cv2.imread(str(_FIXTURES / filename))
    assert len(course_banner_crops(frame)) == 1


@pytest.mark.parametrize("filename", [
    "track_select_negative/voting_screen.png",
    "track_select_negative/gameplay_overlay_bowsers.png",
])
def test_course_banner_locator_rejects_other_screens(filename):
    frame = cv2.imread(str(_FIXTURES / filename))
    assert course_banner_crops(frame) == []


def test_course_banner_model_normalizes_retro_badge():
    class Reader:
        def predict_crops(self, crops):
            assert len(crops) == 1
            return [("sEs Choco Island 1", .9)]

    frame = cv2.imread(str(_FIXTURES / "track_select" / "peach_stadium.png"))
    assert read_course_banner_model(frame, Reader())[0] == "SNES Choco Island 1"


def test_live_track_reader_uses_model_without_tesseract(monkeypatch):
    import pytesseract
    from mktracker.detection.track_select import TrackSelectDetector

    class Reader:
        disabled = False

        def predict_crops(self, crops):
            assert len(crops) == 1
            return [("Peach Stadium", .96)]

    monkeypatch.setattr(pytesseract, "image_to_string",
                        lambda *args, **kwargs: pytest.fail("Track OCR called Tesseract"))
    frame = cv2.imread(str(_FIXTURES / "track_select" / "peach_stadium.png"))
    assert TrackSelectDetector(Reader()).detect(frame) == {"track_name": "Peach Stadium"}


def test_race_start_course_card_locator_rejects_other_screens():
    for filename in ("race-start-course-card-24p.jpg",
                     "race-start-course-card-koopa-24p.jpg"):
        frame = cv2.imread(str(_FIXTURES / "video_analysis" / filename))
        assert race_start_card_crop(frame) is not None
    for filename in ("course-preview.jpg", "previous-course.jpg",
                     "result-animation.jpg"):
        negative = cv2.imread(str(_FIXTURES / "video_analysis" / filename))
        assert race_start_card_crop(negative) is None


def test_race_start_course_card_uses_trained_reader():
    frame = cv2.imread(str(_FIXTURES / "video_analysis" /
                           "race-start-course-card-24p.jpg"))

    class Reader:
        def predict_crops(self, crops):
            assert len(crops) == 1
            return [("SNEs Ghost Valley 2", .92)]

    assert read_race_start_card_model(frame, Reader())[0] == "SNES Ghost Valley 2"


@pytest.mark.parametrize("filename,expected", [
    ("race-start-course-card-24p.jpg", "SNES Ghost Valley 2"),
    ("race-start-course-card-koopa-24p.jpg", "SNES Koopa Beach 1"),
])
def test_real_model_reads_race_start_course_card(filename, expected):
    if not NameModelReader.available():
        pytest.skip("No local trained checkpoint configured")
    frame = cv2.imread(str(_FIXTURES / "video_analysis" / filename))
    reader = NameModelReader()
    try:
        result = read_race_start_card_model(frame, reader)
        assert result is not None and result[0] == expected
    finally:
        reader.close()

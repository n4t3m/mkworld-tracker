"""Model-first finish rank stays local when the HUD supports a reliable read."""
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

import mktracker.state_machine as state_machine_module
from mktracker.detection.hud_position import HudPositionReader
from mktracker.detection.hud_rank_model import read_finish_rank
from mktracker.detection.name_model import NameModelReader
from mktracker.state_machine import GameState, GameStateMachine, RaceInfo


FIXTURES = Path(__file__).parent / "fixtures" / "race_finish"


def test_trained_checkpoint_reads_tracked_finish_hud():
    if not NameModelReader.available():
        pytest.skip("No local dark checkpoint configured")
    frame = cv2.imread(str(FIXTURES / "finish_good_0418_04.png"))
    reader = NameModelReader()
    try:
        assert read_finish_rank(frame, reader, HudPositionReader()) == 12
    finally:
        reader.close()


def test_finish_rank_uses_shape_reader_when_model_misses():
    frame = cv2.imread(str(FIXTURES / "finish_good_0418_04.png"))

    class Reader:
        disabled = False

        @staticmethod
        def predict_crops(_crops):
            return [("Oh", .9)]

    assert read_finish_rank(frame, Reader(), HudPositionReader()) == 12


def test_finish_rank_rejects_non_hud_frame_without_model_call():
    frame = cv2.imread(str(FIXTURES / "bad_results.png"))

    class Reader:
        disabled = False

        @staticmethod
        def predict_crops(_crops):
            raise AssertionError("Model called on a results screen")

    assert read_finish_rank(frame, Reader(), HudPositionReader()) is None


def test_live_rank_uses_local_result_before_gemini(monkeypatch):
    machine = GameStateMachine()
    machine._races = [RaceInfo("Track", ())]
    monkeypatch.setattr(state_machine_module, "read_finish_rank",
                        lambda _frame, _reader, _verifier: 12)
    monkeypatch.setattr(state_machine_module, "request_race_rank",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Gemini called")))
    monkeypatch.setattr(machine, "_save_match_record", lambda: None)
    machine._read_rank_local_then_fallback(
        np.zeros((4, 4, 3), dtype=np.uint8), 1, 0,
        machine._match_seq, None, None)
    assert machine._races[0].race_rank == 12


def test_live_rank_keeps_gemini_fallback_for_hidden_hud(monkeypatch):
    machine = GameStateMachine()
    machine._races = [RaceInfo("Track", ())]
    monkeypatch.setattr(state_machine_module, "read_finish_rank",
                        lambda _frame, _reader, _verifier: None)
    calls = []

    def request(_frame, _race_num, callback, **_kwargs):
        calls.append(True)
        callback(7)

    monkeypatch.setattr(state_machine_module, "request_race_rank", request)
    monkeypatch.setattr(machine, "_save_match_record", lambda: None)
    machine._read_rank_local_then_fallback(
        np.zeros((4, 4, 3), dtype=np.uint8), 1, 0,
        machine._match_seq, None, None)
    assert calls == [True]
    assert machine._races[0].race_rank == 7


def test_live_rank_dispatches_local_read_off_capture_thread(monkeypatch, tmp_path):
    machine = GameStateMachine()
    machine._races = [RaceInfo("Track", ())]
    machine._state_entered_at = time.monotonic() - 3
    monkeypatch.setattr(state_machine_module, "_DEBUG_RANK_DIR", tmp_path)
    monkeypatch.setattr(machine, "_save_race_frame", lambda *_args: None)
    jobs = []

    class Thread:
        def __init__(self, **kwargs):
            jobs.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(state_machine_module.threading, "Thread", Thread)
    machine._handle_detecting_rank(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert len(jobs) == 1
    assert jobs[0]["target"] == machine._read_rank_local_then_fallback
    assert jobs[0]["daemon"] is True
    assert machine.state == GameState.READING_RACE_RESULTS

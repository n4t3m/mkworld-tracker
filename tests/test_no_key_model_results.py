"""Live no-key result boards use the checkpoint for their actual layout."""
from __future__ import annotations

import numpy as np
import pytest

import mktracker.detection.team_results as team_results
from mktracker.detection.match_settings import MatchSettings
from mktracker.state_machine import GameStateMachine


@pytest.mark.parametrize("colored", [False, True])
def test_no_key_team_result_selects_matching_model(monkeypatch, colored):
    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", "Three Teams", "Normal", "No COM", 12, "10 seconds")
    calls = []

    class Reader:
        disabled = False

        def __init__(self, label):
            self.label = label

        def read(self, _frame, grid):
            calls.append(self.label)
            return {**grid, "results": [(1, self.label + "Player")],
                    "confidences": {1: 95}}

    class Detector:
        @staticmethod
        def detect(_frame, **kwargs):
            assert kwargs["model_only"] is True
            grid = {"type": "race", "results": [], "row_centers": {1: 2}}
            if colored:
                grid["name_crop_x"] = (.68, .89)
            return grid

    machine._name_model = Reader("Dark")
    machine._color_name_model = Reader("Color")
    machine._result_detector = Detector()
    monkeypatch.setattr(team_results, "_bands",
                        lambda _frame: [(0, 1)] * (3 if colored else 0))
    monkeypatch.setattr(machine, "_save_race_frame", lambda *_args: None)
    machine._handle_reading_results_ocr(np.zeros((4, 4, 3), dtype=np.uint8))
    expected = "Color" if colored else "Dark"
    assert calls == [expected]
    assert machine._race_placements == {1: expected + "Player"}

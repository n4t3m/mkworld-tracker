"""Tests for the debug-mode pre/post placement frame capture in
``_handle_reading_results_gemini``.

These tests poke ``GameStateMachine`` internals directly: each frame is a
small uint8 array tagged with a unique value so we can later read it back
from disk and prove the right frame landed in the right slot. The
``has_race_results`` detector is monkeypatched per-test to script the
True/False sequence we want to feed the state machine.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

import mktracker.state_machine as sm_mod
from mktracker.detection.match_settings import MatchSettings
from mktracker.state_machine import (
    GameState,
    GameStateMachine,
    RaceInfo,
    _DEBUG_PLACEMENT_CONTEXT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(race_count: int = 4) -> MatchSettings:
    return MatchSettings(
        cc_class="150cc",
        teams="No Teams",
        items="Normal",
        com_difficulty="No COM",
        race_count=race_count,
        intermission="10 seconds",
    )


def _frame(tag: int) -> np.ndarray:
    """Tiny tagged frame — pixel (0,0,0) holds *tag* so we can identify it
    after a round-trip through cv2.imwrite/imread."""
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    f[0, 0, 0] = tag
    return f


def _read_tag(path: Path) -> int:
    img = cv2.imread(str(path))
    assert img is not None, f"failed to read {path}"
    return int(img[0, 0, 0])


def _setup_in_results_state(
    tmp_path: Path, *, debug_mode: bool, race_count: int = 4,
) -> GameStateMachine:
    sm = GameStateMachine()
    sm.debug_mode = debug_mode
    sm._match_settings = _settings(race_count=race_count)
    sm._match_dir = tmp_path / "match"
    sm._match_dir.mkdir(parents=True, exist_ok=True)
    sm._match_started_at = datetime.now()
    sm._races = [RaceInfo(track_name="X", players=("Alice", "Bob"))]
    sm._current_race = 1
    sm._state = GameState.READING_RACE_RESULTS
    return sm


def _script_has_results(
    monkeypatch: pytest.MonkeyPatch, sm: GameStateMachine, sequence: list[bool],
) -> list[bool]:
    """Make ``sm._result_detector.has_race_results`` return values from
    *sequence* in order (one per call). Returns the consumed list so the
    test can inspect it."""
    consumed: list[bool] = []
    seq_iter = iter(sequence)

    def _fake(_frame):
        v = next(seq_iter)
        consumed.append(v)
        return v

    monkeypatch.setattr(sm._result_detector, "has_race_results", _fake)
    return consumed


def _stub_finalise(monkeypatch: pytest.MonkeyPatch, calls: list[None]) -> None:
    """Replace ``request_race_results`` so finalise doesn't try to talk to
    Gemini. Records each call into *calls*."""
    def _noop(*_args, **_kwargs):
        calls.append(None)

    monkeypatch.setattr(sm_mod, "request_race_results", _noop)


def _placements_dir(sm: GameStateMachine) -> Path:
    return sm._match_dir / f"race_{sm._current_race:02d}" / "debug_placements"


# ---------------------------------------------------------------------------
# debug_mode=False: existing behaviour unchanged
# ---------------------------------------------------------------------------


class TestDebugDisabled:
    def test_no_pre_or_post_files_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        sm = _setup_in_results_state(tmp_path, debug_mode=False)
        finalise_calls: list[None] = []
        _stub_finalise(monkeypatch, finalise_calls)
        # Pre-frames F/F/F, two placement T/T, then F → transitions out.
        _script_has_results(
            monkeypatch, sm, [False, False, False, True, True, False],
        )

        for tag in range(6):
            sm._handle_reading_results_gemini(_frame(tag))

        assert not _placements_dir(sm).exists()
        assert finalise_calls == [None]

    def test_transitions_immediately_after_first_non_results_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        sm = _setup_in_results_state(tmp_path, debug_mode=False)
        _stub_finalise(monkeypatch, [])
        _script_has_results(monkeypatch, sm, [True, False])

        sm._handle_reading_results_gemini(_frame(0))
        assert sm._state is GameState.READING_RACE_RESULTS  # still here
        sm._handle_reading_results_gemini(_frame(1))
        # Single race in a 4-race match → next is WAITING_FOR_TRACK_PICK.
        assert sm._state is GameState.WAITING_FOR_TRACK_PICK


# ---------------------------------------------------------------------------
# debug_mode=True: pre-placement rolling buffer
# ---------------------------------------------------------------------------


class TestPrePlacementBuffer:
    def test_pre_buffer_dumped_on_first_placement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """3 pre-frames followed by a placement frame → 3 pre files."""
        sm = _setup_in_results_state(tmp_path, debug_mode=True)
        _stub_finalise(monkeypatch, [])
        _script_has_results(monkeypatch, sm, [False, False, False, True])

        for tag in [10, 11, 12, 99]:
            sm._handle_reading_results_gemini(_frame(tag))

        d = _placements_dir(sm)
        files = sorted(p.name for p in d.glob("pre_*.png"))
        assert files == ["pre_01.png", "pre_02.png", "pre_03.png"]
        # Order preserved: oldest pre frame becomes pre_01.
        assert _read_tag(d / "pre_01.png") == 10
        assert _read_tag(d / "pre_02.png") == 11
        assert _read_tag(d / "pre_03.png") == 12
        # Buffer cleared after dump.
        assert len(sm._pre_results_buffer) == 0
        assert sm._pre_results_dumped is True

    def test_pre_buffer_caps_at_window_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """8 pre-frames before the burst → only the most recent 5 are kept."""
        sm = _setup_in_results_state(tmp_path, debug_mode=True)
        _stub_finalise(monkeypatch, [])
        _script_has_results(
            monkeypatch, sm,
            [False] * 8 + [True],
        )

        # Tags 0..7 pre, tag 100 = first placement.
        for tag in list(range(8)) + [100]:
            sm._handle_reading_results_gemini(_frame(tag))

        d = _placements_dir(sm)
        pre_files = sorted(d.glob("pre_*.png"))
        assert len(pre_files) == _DEBUG_PLACEMENT_CONTEXT == 5
        # Most recent 5 are tags 3..7, in order.
        tags = [_read_tag(p) for p in pre_files]
        assert tags == [3, 4, 5, 6, 7]

    def test_no_pre_files_when_burst_starts_immediately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """First frame is a placement → debug_placements/ has no pre files."""
        sm = _setup_in_results_state(tmp_path, debug_mode=True)
        _stub_finalise(monkeypatch, [])
        _script_has_results(monkeypatch, sm, [True])

        sm._handle_reading_results_gemini(_frame(1))

        d = _placements_dir(sm)
        # Directory may not exist (helper only creates it on dump/post-write).
        if d.exists():
            assert list(d.glob("pre_*.png")) == []
        assert sm._pre_results_dumped is True

    def test_pre_buffer_dumped_only_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Even if has_results toggles within the window, the pre buffer is
        only written on the very first True transition."""
        sm = _setup_in_results_state(tmp_path, debug_mode=True)
        _stub_finalise(monkeypatch, [])
        # F, T (dump pre), then T again — must not re-write pre files.
        _script_has_results(monkeypatch, sm, [False, True, True])

        sm._handle_reading_results_gemini(_frame(50))
        sm._handle_reading_results_gemini(_frame(51))
        sm._handle_reading_results_gemini(_frame(52))

        d = _placements_dir(sm)
        pre_files = sorted(d.glob("pre_*.png"))
        assert len(pre_files) == 1
        assert _read_tag(pre_files[0]) == 50


# ---------------------------------------------------------------------------
# debug_mode=True: post-placement capture window
# ---------------------------------------------------------------------------


class TestPostPlacementWindow:
    def test_post_window_captures_5_frames_then_transitions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        sm = _setup_in_results_state(tmp_path, debug_mode=True)
        finalise_calls: list[None] = []
        _stub_finalise(monkeypatch, finalise_calls)
        # 1 placement, then 5 non-placement frames.
        _script_has_results(
            monkeypatch, sm, [True] + [False] * _DEBUG_PLACEMENT_CONTEXT,
        )

        # Placement frame.
        sm._handle_reading_results_gemini(_frame(0))
        assert sm._state is GameState.READING_RACE_RESULTS

        # Resolve the race dir up front — _transition() bumps _current_race
        # when it moves into WAITING_FOR_TRACK_PICK, so by the time we look
        # at the filesystem afterwards the helper would resolve to the
        # *next* race's directory.
        d = _placements_dir(sm)

        # First 4 non-placement frames stay in state.
        for tag in range(1, _DEBUG_PLACEMENT_CONTEXT):
            sm._handle_reading_results_gemini(_frame(tag))
            assert sm._state is GameState.READING_RACE_RESULTS, (
                f"transitioned too early after post frame {tag}"
            )
            assert finalise_calls == []

        # Fifth non-placement frame triggers finalise + transition.
        sm._handle_reading_results_gemini(_frame(_DEBUG_PLACEMENT_CONTEXT))
        assert finalise_calls == [None]
        assert sm._state is GameState.WAITING_FOR_TRACK_PICK

        post_files = sorted(d.glob("post_*.png"))
        assert [p.name for p in post_files] == [
            f"post_{i:02d}.png" for i in range(1, _DEBUG_PLACEMENT_CONTEXT + 1)
        ]
        # Tag preservation: post_NN.png contains the (NN)th post frame.
        for i, p in enumerate(post_files, start=1):
            assert _read_tag(p) == i

    def test_no_post_files_until_burst_ends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """While placements are still streaming, post files must not appear."""
        sm = _setup_in_results_state(tmp_path, debug_mode=True)
        _stub_finalise(monkeypatch, [])
        _script_has_results(monkeypatch, sm, [True, True, True])

        for tag in range(3):
            sm._handle_reading_results_gemini(_frame(tag))

        d = _placements_dir(sm)
        if d.exists():
            assert list(d.glob("post_*.png")) == []
        assert sm._post_results_count == 0


# ---------------------------------------------------------------------------
# State reset paths clear the debug buffers
# ---------------------------------------------------------------------------


class TestBufferLifecycle:
    def test_reset_clears_buffers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        sm = _setup_in_results_state(tmp_path, debug_mode=True)
        _stub_finalise(monkeypatch, [])
        # Fill the pre buffer and tick post counter.
        sm._pre_results_buffer.append(_frame(1))
        sm._pre_results_dumped = True
        sm._post_results_count = 3

        sm.reset()

        assert len(sm._pre_results_buffer) == 0
        assert sm._pre_results_dumped is False
        assert sm._post_results_count == 0

    def test_handle_race_ending_clears_buffers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Entering DETECTING_RACE_RANK from a FINISH frame should reset the
        debug buffers so the next race's window starts fresh."""
        sm = _setup_in_results_state(tmp_path, debug_mode=True)
        # Pretend a previous race left buffers populated.
        sm._pre_results_buffer.append(_frame(1))
        sm._pre_results_dumped = True
        sm._post_results_count = 4
        sm._state = GameState.WAITING_FOR_RACE_END

        # Stub the FINISH detector to immediately fire.
        monkeypatch.setattr(sm._finish_detector, "is_active", lambda _f: True)
        sm._handle_race_ending(_frame(0))

        assert len(sm._pre_results_buffer) == 0
        assert sm._pre_results_dumped is False
        assert sm._post_results_count == 0
        assert sm._state is GameState.DETECTING_RACE_RANK

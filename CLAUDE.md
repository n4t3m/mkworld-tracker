# MarioKartTableMaker — CLAUDE.md

## Project Overview
A Python app that watches a capture card's video feed of Mario Kart and automatically detects game state, match settings, and track names using computer vision and OCR.

## Tech Stack
- **PySide6** — Qt6 GUI (video display, source selector dropdown)
- **OpenCV** (`opencv-python`) — video capture and frame processing
- **pytesseract** — OCR via Tesseract (binary must be installed; auto-detected at `C:\Program Files\Tesseract-OCR\`)
- **Python >=3.10**, packaged with `pyproject.toml` + hatchling

## Project Structure
```
src/mktracker/
├── main.py                      # Entry point, logging setup, QApplication
├── state_machine.py             # GameStateMachine: WAITING_FOR_MATCH → MATCH_STARTED → RACING
├── capture/
│   └── video_source.py          # Camera enumeration (DirectShow) + VideoCapture wrapper
├── detection/
│   ├── __init__.py              # Shared Tesseract path auto-detection
│   ├── match_settings.py        # MatchSettingsDetector + MatchSettings dataclass
│   ├── track_select.py          # TrackSelectDetector (screen detection + OCR + fuzzy match)
│   └── tracks.py                # Canonical tuple of 30 track names
└── ui/
    └── main_window.py           # MainWindow: video display, calls state_machine.update() every ~500ms
```

## Architecture
- **Detectors** are stateless screen analyzers: `is_active(frame)` (fast) + `detect(frame)` (full pipeline with OCR).
- **GameStateMachine** orchestrates detectors based on current state. Called from UI timer.
- **main_window.py** owns the frame loop (30fps render, detection every 15th frame) and delegates all game logic to the state machine.

## State Machine Flow
1. **WAITING_FOR_MATCH** — polls `MatchSettingsDetector` for the "rules decided" screen
2. **MATCH_STARTED** — stores settings, waits 5 seconds, transitions to RACING
3. **RACING** — polls `TrackSelectDetector`, logs each track with race count (15s cooldown between OCR)

## Detection Patterns
- **Track selection screen**: left 42% of frame is very dark (player list panel), right side is colorful map. Track name is OCR'd from a tight banner ROI at y=33-37%, x=52-85%, upscaled 3x.
- **Match settings screen**: bright white card in center (mean brightness >150), bottom banner at y=94-98% contains "The rules have been decided!", settings parsed from whole card OCR at y=34-86%.
- Both detectors use fuzzy matching (`difflib.get_close_matches`) against known valid values to tolerate OCR noise.

## Test Data
- `testdata/trackselected/` — 5 screenshots of track selection (filename = track name)
- `testdata/match_settings/` — 2 screenshots of match settings screens
- `testdata/track_names.txt` — reference list of 30 tracks (canonical list lives in `tracks.py`)

## Conventions
- All game logic lives in `state_machine.py` and `detection/` — the UI just renders and forwards frames
- Debug/info logging via Python `logging` module (INFO level by default in main.py)
- State transitions are logged as `State: X -> Y`
- Track detections logged as `Race N/total: Track Name`
- OCR ROI coordinates are normalized (0-1) proportions of frame dimensions (1920x1080 reference)
- Tesseract `--psm 7` for single-line text, `--psm 6` for blocks

## Running
```bash
pip install -e .
mktracker
```

## GitHub
Private repo: github.com/n4t3m/MarioKartTableMaker (branch: master)

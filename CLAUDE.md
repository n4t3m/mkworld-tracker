# MarioKartTableMaker — CLAUDE.md

## Project Overview
A Python app that watches a capture card's video feed of Mario Kart and automatically detects game state, match settings, track names, and player names using computer vision and OCR.

## Tech Stack
- **PySide6** — Qt6 GUI (video display, source selector dropdown, state label, reset button)
- **OpenCV** (`opencv-python`) — video capture and frame processing
- **pytesseract** — OCR via Tesseract (binary must be installed; auto-detected at `C:\Program Files\Tesseract-OCR\`)
- **Python >=3.10**, packaged with `pyproject.toml` + hatchling

## Project Structure
```
src/mktracker/
├── main.py                      # Entry point, logging setup (INFO level), QApplication
├── state_machine.py             # GameStateMachine with 4 states, debug frame saving
├── capture/
│   └── video_source.py          # Camera enumeration (DirectShow) + VideoCapture wrapper
├── detection/
│   ├── __init__.py              # Shared Tesseract path auto-detection
│   ├── match_settings.py        # MatchSettingsDetector + MatchSettings dataclass
│   ├── player_reader.py         # PlayerReader: dynamic grid detection + per-cell OCR
│   ├── track_select.py          # TrackSelectDetector (screen detection + OCR + fuzzy match)
│   └── tracks.py                # Canonical tuple of 30 track names
└── ui/
    └── main_window.py           # MainWindow: video display, state label, reset button
```

## Architecture
- **Detectors** are screen analyzers: `is_active(frame)` (fast brightness check) + `detect(frame)` (full OCR pipeline).
- **PlayerReader** dynamically detects the player grid (row brightness bands + column profiling), OCRs each cell with OTSU thresholding, and cleans avatar-icon noise via regex.
- **GameStateMachine** orchestrates detectors based on current state. Called from UI timer every ~500ms.
- **main_window.py** owns the frame loop (30fps render, detection every 15th frame) and delegates all game logic to the state machine. Displays current state name in top-right toolbar and has a "Reset State" button.

## State Machine Flow
1. **WAITING_FOR_MATCH** — polls `MatchSettingsDetector` for the "rules decided" screen
2. **MATCH_STARTED** — stores settings, waits 5 seconds, transitions to RACING
3. **RACING** — polls `TrackSelectDetector` (15s cooldown managed here); on track detection, transitions to READING_PLAYERS
4. **READING_PLAYERS** — reads player names on the next frame (gives the player list time to load), logs the race, applies cooldown, returns to RACING

Track detection and player reading happen on **separate frames** because the player list animates in after the track name appears.

## Detection Patterns
- **Track selection screen**: left 42% of frame is very dark (player list panel), right side is colorful map. Track name is OCR'd from a tight banner ROI at y=33-37%, x=52-85%, upscaled 3x. Fuzzy-matched against 30 canonical track names.
- **Match settings screen**: bright white card in center (mean brightness >150), bottom banner at y=94-98% contains "The rules have been decided!", settings parsed from whole card OCR at y=34-86%. Each setting fuzzy-matched against known valid values.
- **Player names**: left panel has a 2-column grid of player boxes. Rows detected via horizontal brightness bands (merged if gap < 6px, filtered if height < 40px). Column split found from brightness profile dip. Right column width capped at 47% of frame. Each cell: skip 17% for avatar, OTSU threshold, `--psm 7` OCR. Names cleaned of leading avatar noise and trailing box-edge artefacts.

## Debug Frame Saving
Each match creates a timestamped folder under `debug_frames/` (gitignored):
- `match_settings.png` — frame that triggered match detection
- `race_01_Track_Name_track.png` — frame where track was detected
- `race_01_Track_Name_players.png` — frame where players were read

## Known Limitations
- **Special Unicode characters** (☆, π, ★, ♪, ⊃) in player names are not reliably OCR'd by Tesseract
- **Right-column player scores** are too dim in captures to read (< 10 levels of contrast); score detection was removed
- Player names with only special characters (e.g., "π") will be garbled

## Test Data
- `testdata/trackselected/` — 6 screenshots of track selection (filenames = track names)
- `testdata/match_settings/` — 2 screenshots of match settings screens
- `testdata/track_names.txt` — reference list of 30 tracks (canonical list lives in `tracks.py`)

## Conventions
- All game logic lives in `state_machine.py` and `detection/` — the UI just renders and forwards frames
- Debug/info logging via Python `logging` module (INFO level by default)
- State transitions logged as `State: X -> Y`
- Track detections logged as `Race N/total: Track Name` followed by player list
- OCR ROI coordinates are normalized (0-1) proportions of frame dimensions (1920x1080 reference)
- Tesseract `--psm 7` for single-line text, `--psm 6` for blocks, `--psm 8` for single word

## Running
```bash
pip install -e .
mktracker
```

## GitHub
Private repo: github.com/n4t3m/MarioKartTableMaker (branch: master)

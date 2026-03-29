# MarioKartTableMaker — CLAUDE.md

## Project Overview
A Python app that watches a capture card's video feed of Mario Kart and automatically detects game state, match settings, track names, and player names using computer vision and OCR.

## Tech Stack
- **PySide6** — Qt6 GUI (video display, source selector dropdown, state label, advance/reset buttons)
- **OpenCV** (`opencv-python`) — video capture and frame processing
- **pytesseract** — OCR via Tesseract (binary must be installed; auto-detected at `C:\Program Files\Tesseract-OCR\`)
- **Python >=3.10**, packaged with `pyproject.toml` + hatchling

## Project Structure
```
src/mktracker/
├── main.py                      # Entry point, logging setup (INFO level), QApplication
├── state_machine.py             # GameStateMachine with 5 states, debug frame saving
├── capture/
│   └── video_source.py          # Camera enumeration (DirectShow) + VideoCapture wrapper
├── detection/
│   ├── __init__.py              # Shared Tesseract path auto-detection
│   ├── match_settings.py        # MatchSettingsDetector + MatchSettings dataclass
│   ├── player_reader.py         # PlayerReader: dynamic grid detection + per-cell OCR
│   ├── race_finish.py           # RaceFinishDetector (FINISH! banner via HSV color detection)
│   ├── track_select.py          # TrackSelectDetector (screen detection + OCR + fuzzy match)
│   └── tracks.py                # Canonical tuple of 30 track names
└── ui/
    └── main_window.py           # MainWindow: video display, state label, advance/reset buttons
```

## Architecture
- **Detectors** are screen analyzers: `is_active(frame)` (fast check ~1ms) + `detect(frame)` (full OCR pipeline).
- **PlayerReader** dynamically detects the player grid (row brightness bands + column profiling), OCRs each cell with OTSU thresholding, and cleans avatar-icon noise via regex.
- **GameStateMachine** orchestrates detectors based on current state.
- **main_window.py** owns the frame loop (30fps render) and delegates all game logic to the state machine. Displays current state in top-right toolbar with "Advance State" and "Reset State" buttons.

## Threading Model
- All detection runs on the main thread every 15th frame (~500ms).

## State Machine Flow
1. **WAITING_FOR_MATCH** — polls `MatchSettingsDetector` for the "rules decided" screen
2. **MATCH_STARTED** — stores settings, waits 5 seconds, transitions to RACING
3. **WAITING_FOR_TRACK_PICK** — polls `TrackSelectDetector` (15s cooldown); on track detection, transitions to READING_PLAYERS_IN_RACE
4. **READING_PLAYERS_IN_RACE** — reads player names on the next frame (gives the player list time to load), transitions to WAITING_FOR_RACE_END
5. **WAITING_FOR_RACE_END** — polls `RaceFinishDetector` for the FINISH! banner, transitions back to WAITING_FOR_TRACK_PICK

## Detection Patterns
- **Track selection screen**: left 42% of frame is very dark (player list panel), right side is colorful map. Track name OCR'd from tight banner ROI at y=33-37%, x=52-85%, upscaled 3x. Fuzzy-matched against 30 canonical track names (difflib, cutoff 0.6).
- **Match settings screen**: bright white card in center (mean brightness >150), bottom banner at y=94-98% contains "The rules have been decided!", settings parsed from whole card OCR at y=34-86%. Each setting fuzzy-matched against known valid values.
- **Player names**: left panel 2-column grid. Rows detected via horizontal brightness bands (merged if gap < 6px, filtered if height < 40px). Column split from brightness profile dip. Each cell: skip 17% for avatar, OTSU threshold, `--psm 7` OCR. Names cleaned of leading avatar noise and trailing artefacts.
- **Race finish screen**: FINISH! banner detected via HSV masking for orange-yellow text (H 15-35, S>150, V>180) in center ROI (x=15-82%, y=33-58%). Validated by pixel ratio bounds (0.20-0.35) and requiring orange in all 5 vertical strips (rejects GO! text and partial animations).

## Debug Frame Saving
- Each match creates a timestamped folder under `debug_frames/` (gitignored)
- `match_settings.png`, `race_NN_Track_track.png`, `race_NN_Track_players.png`

## Known Limitations
- **Special Unicode characters** (☆, π, ★, ♪, ⊃) in player names are not reliably OCR'd by Tesseract

## Test Data
- `testdata/trackselected/` — 6 screenshots of track selection (filenames = track names)
- `testdata/match_settings/` — 2 screenshots of match settings screens
- `testdata/track_names.txt` — reference list of 30 tracks (canonical list lives in `tracks.py`)
- `testdata/samplerace.mp4` — full race at 2560x1440
- `testdata/realsamplerace.mp4` — full race at 1920x1080 (real capture card output)
- `testdata/race_finish/` — 1 FINISH! screenshot + 4 negative frames (GO!, partial animation, track select, results)

## Conventions
- All game logic lives in `state_machine.py` and `detection/` — the UI just renders and forwards frames
- Debug/info logging via Python `logging` module (INFO level by default)
- State transitions logged as `State: X -> Y`
- Track detections logged as `Race N/total: Track Name` followed by player list
- OCR ROI coordinates are normalized (0-1) proportions of frame dimensions (1920x1080 reference)
- Tesseract `--psm 7` for single-line text (player cells), `--psm 6` for blocks (match settings), `--psm 8` for single word

## Running
```bash
pip install -e .
mktracker
```

## GitHub
Private repo: github.com/n4t3m/MarioKartTableMaker (branch: master)

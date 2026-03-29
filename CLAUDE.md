# MarioKartTableMaker — CLAUDE.md

## Project Overview
A Python app that watches a capture card's video feed of Mario Kart and automatically detects game state, match settings, track names, player names, and race results using computer vision and OCR.

## Tech Stack
- **PySide6** — Qt6 GUI (video display, source selector dropdown, state label, advance/reset buttons)
- **OpenCV** (`opencv-python`) — video capture and frame processing
- **pytesseract** — OCR via Tesseract (binary must be installed; auto-detected at `C:\Program Files\Tesseract-OCR\`)
- **Python >=3.10**, packaged with `pyproject.toml` + hatchling

## Project Structure
```
src/mktracker/
├── main.py                      # Entry point, logging setup (INFO level), QApplication
├── state_machine.py             # GameStateMachine with 6 states, debug frame saving
├── capture/
│   └── video_source.py          # Camera enumeration (DirectShow) + VideoCapture wrapper
├── detection/
│   ├── __init__.py              # Shared Tesseract path auto-detection
│   ├── match_settings.py        # MatchSettingsDetector + MatchSettings dataclass
│   ├── player_reader.py         # PlayerReader: dynamic grid detection + per-cell OCR
│   ├── race_result.py           # RaceResultDetector + RaceResultAccumulator
│   ├── track_select.py          # TrackSelectDetector (screen detection + OCR + fuzzy match)
│   └── tracks.py                # Canonical tuple of 30 track names
└── ui/
    └── main_window.py           # MainWindow: video display, state label, advance/reset buttons
```

## Architecture
- **Detectors** are screen analyzers: `is_active(frame)` (fast check ~1ms) + `read_results(frame)` / `detect(frame)` (full OCR pipeline).
- **PlayerReader** dynamically detects the player grid (row brightness bands + column profiling), OCRs each cell with OTSU thresholding, and cleans avatar-icon noise via regex.
- **RaceResultDetector** uses per-row OCR with dynamic row offset detection. Row offset found by minimizing average brightness at separator positions across the full panel width. Multi-pass preprocessing per row (raw + threshold 180 fallback).
- **RaceResultAccumulator** collects placement readings across multiple frames and resolves via majority-vote consensus with deduplication.
- **GameStateMachine** orchestrates detectors based on current state.
- **main_window.py** owns the frame loop (30fps render) and delegates all game logic to the state machine. Displays current state in top-right toolbar with "Advance State" and "Reset State" buttons.

## Threading Model
- **Normal states** (WAITING_FOR_MATCH, MATCH_STARTED, RACING, READING_PLAYERS): detection runs on the main thread every 15th frame (~500ms).
- **RACE_ACTIVE**: `is_active()` + OCR runs in a background thread every 3rd frame (~100ms) to detect the results screen start. Requires 3+ placements in a single frame to confirm.
- **READING_RESULTS**: frames are buffered cheaply (just `copy()`, no OCR) every 3rd frame via background thread. Only `is_active()` runs per frame (~1ms). When the results screen ends (3 consecutive non-active frames), all buffered frames (~80 over 8 seconds) are batch-processed through per-row OCR in the background thread.
- A `threading.Lock` ensures only one background OCR runs at a time; frames arriving during OCR are skipped.

## State Machine Flow
1. **WAITING_FOR_MATCH** — polls `MatchSettingsDetector` for the "rules decided" screen
2. **MATCH_STARTED** — stores settings, waits 5 seconds, transitions to RACING
3. **RACING** — polls `TrackSelectDetector` (15s cooldown); on track detection, transitions to READING_PLAYERS
4. **READING_PLAYERS** — reads player names on the next frame (gives the player list time to load), transitions to RACE_ACTIVE
5. **RACE_ACTIVE** — polls `RaceResultDetector.is_active()` every ~100ms; when 3+ placements found, transitions to READING_RESULTS
6. **READING_RESULTS** — buffers frames cheaply; on screen end, batch OCR all frames, finalizes via `RaceResultAccumulator`, logs placements, returns to RACING

## Detection Patterns
- **Track selection screen**: left 42% of frame is very dark (player list panel), right side is colorful map. Track name OCR'd from tight banner ROI at y=33-37%, x=52-85%, upscaled 3x. Fuzzy-matched against 30 canonical track names (difflib, cutoff 0.6).
- **Match settings screen**: bright white card in center (mean brightness >150), bottom banner at y=94-98% contains "The rules have been decided!", settings parsed from whole card OCR at y=34-86%. Each setting fuzzy-matched against known valid values.
- **Player names**: left panel 2-column grid. Rows detected via horizontal brightness bands (merged if gap < 6px, filtered if height < 40px). Column split from brightness profile dip. Each cell: skip 17% for avatar, OTSU threshold, `--psm 7` OCR. Names cleaned of leading avatar noise and trailing artefacts.
- **Race results**: panel at x=48-88% of frame (tuned for 1920x1080 capture). `is_active()` uses percentage-based horizontal edge density (>10% of rows with Sobel-Y > 40). Per-row OCR with dynamic offset: row height 0.0713 (77px at 1080p), offset found by minimizing mean brightness at separator positions. Each row tried with raw image then threshold 180. Regex: `(\d{1,2})(?!\d).{0,20}?([A-Z][A-Za-z_.  ]+)` with lowercase fallback. Name cleanup strips avatar prefixes, trailing junk, and short fragments.

## Debug Frame Saving
- Each match creates a timestamped folder under `debug_frames/` (gitignored)
- `match_settings.png`, `race_NN_Track_track.png`, `race_NN_Track_players.png`
- Result frames saved to `debug_frames/results/` with sequential numbering (works with manual advance)
- Batch OCR frames saved as `batch_NNN_M.png` during finalization

## Known Limitations
- **Gold/silver/bronze rows** (places 1-3) in race results have styled numbers that OCR frequently misreads
- **Special Unicode characters** (☆, π, ★, ♪, ⊃) in player names are not reliably OCR'd by Tesseract
- **Right-column player scores** are too dim in captures to read; score detection was removed
- **Panel x-position** is hardcoded for 1920x1080 capture; other resolutions may need adjustment
- Race result batch OCR takes several seconds after the results screen ends (runs in background thread)

## Test Data
- `testdata/trackselected/` — 6 screenshots of track selection (filenames = track names)
- `testdata/match_settings/` — 2 screenshots of match settings screens
- `testdata/track_names.txt` — reference list of 30 tracks (canonical list lives in `tracks.py`)
- `testdata/raceresult/` — sample videos of race finishes with results screens
- `testdata/samplerace.mp4` — full race at 2560x1440
- `testdata/realsamplerace.mp4` — full race at 1920x1080 (real capture card output)

## Conventions
- All game logic lives in `state_machine.py` and `detection/` — the UI just renders and forwards frames
- Debug/info logging via Python `logging` module (INFO level by default)
- State transitions logged as `State: X -> Y`
- Track detections logged as `Race N/total: Track Name` followed by player list
- Race results logged as `Final race results for Track (N placements):` with per-placement lines
- OCR ROI coordinates are normalized (0-1) proportions of frame dimensions (1920x1080 reference)
- Tesseract `--psm 7` for single-line text (per-row results, player cells), `--psm 6` for blocks (match settings), `--psm 8` for single word

## Running
```bash
pip install -e .
mktracker
```

## GitHub
Private repo: github.com/n4t3m/MarioKartTableMaker (branch: master)

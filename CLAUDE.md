# MarioKartTableMaker — CLAUDE.md

## Project Overview
A Python app that watches a capture card's video feed of Mario Kart and automatically detects game state, match settings, track names, player names, and race result placements using computer vision and OCR.

## Tech Stack
- **PySide6** — Qt6 GUI (video display, source selector dropdown, state label, advance/reset buttons, match settings editor, settings tab)
- **OpenCV** (`opencv-python`) — video capture and frame processing
- **pytesseract** — OCR via Tesseract (binary must be installed; auto-detected at `C:\Program Files\Tesseract-OCR\`)
- **python-dotenv** — loads/saves `.env` for Gemini API key and model persistence
- **Python >=3.10**, packaged with `pyproject.toml` + hatchling

## Project Structure
```
src/mktracker/
├── main.py                      # Entry point, logging setup (INFO level), QApplication
├── state_machine.py             # GameStateMachine with 7 states, debug frame saving
├── gemini_client.py             # Gemini API key/model persistence (.env) and health check
├── capture/
│   └── video_source.py          # Camera enumeration (DirectShow) + VideoCapture wrapper
├── detection/
│   ├── __init__.py              # Shared Tesseract path auto-detection
│   ├── match_results.py         # MatchResultDetector: final CONGRATULATIONS! screen OCR (two-teams)
│   ├── match_settings.py        # MatchSettingsDetector + MatchSettings dataclass
│   ├── player_reader.py         # PlayerReader: dynamic grid detection + per-cell OCR
│   ├── race_finish.py           # RaceFinishDetector (FINISH! banner via HSV color detection)
│   ├── race_results.py          # RaceResultDetector: post-race placement reading (teams & no-teams)
│   ├── track_select.py          # TrackSelectDetector (screen detection + OCR + fuzzy match)
│   └── tracks.py                # Canonical tuple of 30 track names
└── ui/
    └── main_window.py           # MainWindow: video display, state label, tabbed settings panel, controls
```

## Architecture
- **Detectors** are screen analyzers: `is_active(frame)` (fast check ~1ms) + `detect(frame)` (full OCR pipeline).
- **PlayerReader** dynamically detects the player grid (row brightness bands + column profiling), OCRs each cell with OTSU thresholding, and cleans avatar-icon noise via regex.
- **RaceResultDetector** reads placement/name pairs from the post-race results screen. Two preprocessing paths: hybrid threshold `(gray>170 & blue>100)` for no-teams mode, blur+Otsu on HSV V-channel for team-colour mode. Gold first-place bar handled with hybrid threshold in both modes. Uses `image_to_data` for word-level OCR with x-position zones to classify placement numbers, names, and scores. Gap-aware sequential placement fix infers missing placements from y-spacing. Adaptive-V cluster counting on a narrow strip (x=0.84-0.90) detects `+` signs to distinguish race results from overall standings.
- **GameStateMachine** orchestrates detectors based on current state. Accumulates race result placements across multiple frames with quality-based overwriting (frames with more detected rows can overwrite earlier lower-quality detections).
- **main_window.py** owns the frame loop (30fps render) and delegates all game logic to the state machine. Displays current state in top-right toolbar with "Advance State" and "Reset State" buttons. Right panel is a `QTabWidget` with a **Match** tab (match settings editor, syncs bidirectionally with state machine) and a **Settings** tab (Gemini API key + model configuration).
- **gemini_client.py** handles Gemini API key and model persistence in `.env` via `python-dotenv`. Health check uses a GET request to `/v1beta/models/{model}` (no tokens consumed). `_VerifyThread` runs the check off the main thread.

## Threading Model
- All detection runs on the main thread every 15th frame (~500ms).
- During READING_RACE_RESULTS, detection runs every 3rd frame (~100ms) to capture scrolling results.

## State Machine Flow
1. **WAITING_FOR_MATCH** — polls `MatchSettingsDetector` for the "rules decided" screen
2. **MATCH_STARTED** — stores settings, waits 5 seconds, transitions to WAITING_FOR_TRACK_PICK
3. **WAITING_FOR_TRACK_PICK** — polls `TrackSelectDetector` (15s cooldown); on track detection, transitions to READING_PLAYERS_IN_RACE
4. **READING_PLAYERS_IN_RACE** — reads player names on the next frame (gives the player list time to load), transitions to WAITING_FOR_RACE_END
5. **WAITING_FOR_RACE_END** — polls `RaceFinishDetector` for the FINISH! banner, transitions to READING_RACE_RESULTS
6. **READING_RACE_RESULTS** — reads placement/name pairs from scrolling results via `RaceResultDetector`, accumulates across frames, transitions to WAITING_FOR_TRACK_PICK when overall standings (no `+` signs) are detected or after 30s timeout
7. **FINALIZING_MATCH** — polls `MatchResultDetector` for the CONGRATULATIONS! screen; captures final standings (name + points) and logs them

## Detection Patterns
- **Track selection screen**: left 42% of frame is very dark (player list panel), right side is colorful map. Track name OCR'd from tight banner ROI at y=33-37%, x=52-85%, upscaled 3x. Fuzzy-matched against 30 canonical track names (difflib, cutoff 0.6).
- **Match settings screen**: bright white card in center (mean brightness >150), bottom banner at y=94-98% contains "The rules have been decided!", settings parsed from whole card OCR at y=34-86%. Each setting fuzzy-matched against known valid values.
- **Player names**: left panel 2-column grid. Rows detected via horizontal brightness bands (merged if gap < 6px, filtered if height < 40px). Column split from brightness profile dip. Each cell: skip 17% for avatar, OTSU threshold, `--psm 7` OCR. Names cleaned of leading avatar noise and trailing artefacts.
- **Race finish screen**: FINISH! banner detected via HSV masking for orange-yellow text (H 15-35, S>150, V>180) in center ROI (x=15-82%, y=33-58%). Validated by pixel ratio bounds (0.20-0.35) and requiring orange in all 5 vertical strips (rejects GO! text and partial animations).
- **Race results (no-teams)**: right-side result bars (x=0.56-0.98). Hybrid threshold `(gray>170) & (blue>100)` isolates white text on grey/gold bars. Words classified by x-position: placement numbers (<0.62), icons (0.62-0.66, skipped), names (0.66-0.82), scores/plus (>0.82). `+` detected via OCR word matching `^[+#]\d+$`.
- **Race results (teams)**: red/blue bars where text takes on bar colour (grayscale ~100). Gaussian blur + Otsu on HSV V-channel for name reading. Gold first-place bar top 10% replaced with hybrid threshold. `+` detected via adaptive-V cluster counting on a narrow strip (3+ clusters = race results). Minimum 3 detected name-rows required to filter gameplay frame noise.

## Match Settings UI
- Right panel is a `QTabWidget` with two tabs: **Match** and **Settings**
- **Match tab**: dropdowns for Class, Teams, Items, COM, Intermission and a spinbox for Race Count. On startup, UI defaults are pushed to the state machine (settings always available even when advancing past match detection). When match settings are detected from video, UI updates to show detected values (label: "Detected"). Reset State wipes detected settings and re-applies current UI values as manual. The `teams` setting controls which race result preprocessing path is used.
- **Settings tab**: Gemini API key field (password-masked, with eye toggle to reveal), model name field (defaults to `gemma-3-27b-it`), Save button, and Verify Key button. Status label shows verification result in green/red/grey. A `●` dot indicator in the toolbar reflects the API key status at a glance without opening the tab. Auto-verifies on startup if a key is stored.

## Gemini API
- Key and model stored in `.env` (gitignored) via `python-dotenv`
- `gemini_client.py` exposes `load_api_key`, `save_api_key`, `load_model`, `save_model`, `verify_api_key`
- Health check: `GET /v1beta/models/{model}?key={key}` — no tokens consumed
- Default model: `gemma-3-27b-it`
- Verification runs on a background `QThread` to avoid blocking the UI; concurrent verify requests are ignored while one is in flight

## Debug Frame Saving
- Each match creates a timestamped folder under `debug_frames/` (gitignored)
- `match_settings.png`, `race_NN_Track_track.png`, `race_NN_Track_players.png`, `match_results.png`
- Race result frames saved to `debug_placements/` (gitignored) when new placements are captured (temporary debugging)

## Known Limitations
- **Special Unicode characters** (☆, π, ★, ♪, ⊃) in player names are not reliably OCR'd by Tesseract
- **Team-mode OCR quality**: text on coloured bars has only ~7-10 levels of V-channel contrast with the background, causing some names to be partially garbled

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
- Race result placements logged as numbered list when overall standings are detected
- OCR ROI coordinates are normalized (0-1) proportions of frame dimensions (1920x1080 reference)
- Tesseract `--psm 7` for single-line text (player cells), `--psm 6` for blocks (match settings, race results), `--psm 8` for single word

## Running
```bash
pip install -e .
mktracker
```

## GitHub
Private repo: github.com/n4t3m/MarioKartTableMaker (branch: master)

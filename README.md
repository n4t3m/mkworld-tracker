# MarioKartTableMaker

## Prerequisites

### Capture card

The app reads frames from a USB capture card via the platform's native video backend (DirectShow on Windows, AVFoundation on macOS, V4L2 on Linux) and lets you pick the device from the source dropdown in the toolbar. Any card that can deliver a **1920x1080** feed from your Switch will work — the detection ROIs and OCR thresholds are tuned against that resolution.

### Tesseract OCR

The program utilizes the Tesseract binary for OCR. It must be installed separately prior to running this application.

- **Windows**: install via the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki). The default install location (`C:\Program Files\Tesseract-OCR\`) is auto-detected. If you install it anywhere else, add that directory to your `PATH`.
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr` (or your distro's equivalent)

If Tesseract is not on `PATH` and not at the default Windows location, the GUI will launch but OCR detection will throw `TesseractNotFoundError` on the first frame and the program becomes effectively unusable.

### Gemini API key (optional but strongly recommended)

The app uses a mix of OCR (via Tesseract) and Gemini models to detect game state. Tesseract handles the always-on detectors (match settings, track name, player names); Gemini, when configured, replaces OCR for race rank, race results, and final standings — all of which are noticeably more reliable with the LLM.

The Gemini models we use are **free** on Google AI Studio's free tier. Grab a key from [Google AI Studio](https://aistudio.google.com/app/apikey) and paste it into the **Settings** tab in the app; it'll be persisted to a local `.env` file. Without a key, the app still runs but falls back to OCR for everything.

Suggested models (configurable from the Settings tab):

- `gemma-4-31b-it` _(default)_ — newer, larger Gemma; better at recovering noisy or partially obscured frames at the cost of a bit more latency.
- `gemma-3-27b-it` — open-weights Gemma 3; a solid lighter-weight baseline that's accurate enough across all three tasks.
- `gemini-3.1-flash-lite-preview` — the best of the three when you can get it, but free-tier capacity is often exhausted during peak hours and requests will fail with a quota error until demand dies down. Worth trying first; fall back to one of the Gemma models if you start seeing rate-limit errors.

## Running

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run mktracker
```

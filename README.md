# Audio segmentation with YAMNet

This repository provides a Python CLI tool to segment audio into:

- `music`
- `human_voice`
- `silence_very_low`
- `silence_low`
- `other`

## Features

- Uses **YAMNet** for semantic audio labeling.
- Detects **silence from volume (RMS dBFS)** rather than model predictions.
- Uses two silence thresholds:
  - `--very-low-db` (default: `-50 dBFS`)
  - `--low-db` (default: `-40 dBFS`)
- Detects class changes on a coarse grid, then refines boundary times using a finer interval.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python yamnet_segmenter.py input_audio.wav --output segments.json
```

Useful options:

```bash
python yamnet_segmenter.py input_audio.wav \
  --very-low-db -52 \
  --low-db -42 \
  --coarse-step 0.48 \
  --refine-step 0.02 \
  --analysis-window 0.96 \
  --output segments.json
```

Output format:

```json
{
  "input": "input_audio.wav",
  "sample_rate": 16000,
  "very_low_db": -50.0,
  "low_db": -40.0,
  "segments": [
    {"start": 0.0, "end": 1.28, "label": "silence_very_low"},
    {"start": 1.28, "end": 5.74, "label": "human_voice"},
    {"start": 5.74, "end": 12.11, "label": "music"}
  ]
}
```

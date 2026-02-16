# Audio Segmentation Tool

A Python CLI that segments an audio/video file into semantic regions using a **two-stage pipeline**:

1. **Coarse segmentation** over fixed-size chunks.
2. **Boundary refinement** around label transitions.

The tool emits segment boundaries with one of four labels:

- `Silence`
- `Silence2`
- `Speech`
- `Music`

## Features

- Uses `ffmpeg` to decode many input media formats.
- Resamples to mono 16 kHz for model compatibility.
- Uses YAMNet (TensorFlow Hub) to classify non-silent audio.
- Splits low-energy audio into two silence classes (`Silence`, `Silence2`).
- Refines coarse boundaries with a sliding-window local search.
- Outputs both human-readable lines and machine-readable JSON.

## Requirements

- Python 3.9+
- `ffmpeg` available on `PATH`
- Python dependencies from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python segment_audio.py <input_audio_or_video_file>
```

Optional flags:

- `--indent <int>`: pretty-print JSON output indent (default: `2`).

Example:

```bash
python segment_audio.py sample.wav --indent 2
```

## Output format

The tool prints:

1. A per-segment summary line:
   - `Segment 001 | 00:00:00 - 00:00:08 | type=Speech`
2. A JSON array with fields:
   - `start` / `end` (seconds)
   - `start_hms` / `end_hms` (`HH:MM:SS`)
   - `label` and `type` (same value)

## Notes

- The first run may take longer because TensorFlow Hub downloads YAMNet artifacts.
- Classification quality depends on audio quality and domain.
- Very short transitions may be smoothed by coarse chunking and refinement heuristics.

For algorithm details, see `docs/algorithm.md`.

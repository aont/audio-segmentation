# Algorithm Overview

This document explains the segmentation algorithm used by `segment_audio.py`.

## High-level pipeline

The system performs segmentation in **two stages**:

1. **Coarse pass (chunk-level labeling)**
2. **Fine pass (boundary refinement)**

This structure balances speed and temporal precision.

---

## 1) Audio loading and normalization

Input media is decoded with `ffmpeg` and converted to:

- **Mono** (`-ac 1`)
- **16 kHz sample rate** (`-ar 16000`)
- **32-bit float PCM** (`-f f32le`)

If the decoded waveform amplitude exceeds `1.0`, audio is normalized by dividing by the maximum absolute value.

Why this matters:

- YAMNet expects a 16 kHz mono waveform.
- A stable amplitude range makes energy-based silence rules predictable.

---

## 2) Label taxonomy

The output labels are:

- `Silence`: very low-energy audio (`RMS dBFS < -55`)
- `Silence2`: low but non-negligible energy (`-55 <= RMS dBFS < -45`)
- `Speech`
- `Music`

The first two are rule-based; the last two are model-based.

---

## 3) Coarse segmentation (fixed chunks)

### 3.1 Chunking

The waveform is split into contiguous chunks of:

- `CHUNK_SEC = 8.0` seconds

For each chunk, a label is produced using `classify_chunk`.

### 3.2 Chunk classification logic

For each chunk:

1. Compute RMS energy in dBFS.
2. If RMS is below silence thresholds, output `Silence` or `Silence2`.
3. Otherwise run YAMNet and compare aggregated speech vs music probabilities.

### 3.3 YAMNet decision rule

- YAMNet outputs per-frame class scores across AudioSet classes.
- Scores are averaged across frames.
- Two class groups are pre-built from class names:
  - **Speech group**: classes whose names include keywords like `speech`, `conversation`, `laugh`, `choir`, etc.
  - **Music group**: classes containing keywords like `music`, `instrument`, `piano`, `jazz`, `rock`, etc.
- Final decision:
  - `Speech` if `sum(speech_group_scores) >= sum(music_group_scores)`
  - Else `Music`

### 3.4 Initial merge

After labeling all chunks, adjacent chunks with the same label are merged to form coarse segments.

---

## 4) Boundary refinement (local search)

The coarse stage is efficient but boundaries are quantized to chunk edges (8s). Refinement improves transition timing.

For each boundary between segment `i` and `i+1`:

- Let coarse boundary be `t0`
- Let left/right labels be `L` and `R`

### 4.1 Search region and windows

A local search region is created around `t0`:

- Radius: `FINE_SEARCH_RADIUS_SEC = 4.0`
- Window size: `FINE_WIN_SEC = 0.975`
- Hop: `FINE_HOP_SEC = 0.2`

Sliding windows are classified across this region.

### 4.2 Best split objective

For each candidate split index `k` in the window sequence:

- `left_score`: count of windows `[0..k]` classified as `L`
- `right_score`: count of windows `[k+1..end]` classified as `R`
- objective: maximize `left_score + right_score`

The first `k` with maximal score is chosen.

Refined boundary time is:

- `refined = window_start[k] + FINE_WIN_SEC`

### 4.3 Monotonic constraints

Refined boundaries are clamped so they remain ordered and valid:

- boundary `i` cannot move before boundary `i-1`
- boundary `i` cannot move past boundary `i+1` (coarse upper bound for intermediate boundaries)
- all boundaries must stay in `[0, total_duration]`

This prevents overlap and negative-length segments.

---

## 5) Final segment reconstruction

Using the refined boundaries:

- Rebuild `[start, end, label]` segments
- Skip zero/negative-length segments
- Merge neighboring segments with identical labels (if created by refinement side effects)
- Ensure the final segment reaches `total_duration`

---

## 6) Complexity characteristics

Let:

- `N` = audio duration in seconds
- coarse chunks ≈ `N / 8`
- `B` = number of coarse boundaries
- windows per boundary ≈ `search_span / hop` (here around 30–40 depending on edge clipping)

Then runtime is dominated by model inference for:

- one pass per coarse chunk, plus
- one pass per fine window around each boundary

The two-stage design keeps the expensive fine analysis local to transitions.

---

## 7) Practical behavior

- Long homogeneous sections are processed quickly and usually robust.
- Boundary precision is typically better than coarse 8s edges due to local search.
- Rapid alternations (e.g., speech over music, abrupt edits) may still be simplified by the speech-vs-music and energy-threshold framework.

---

## 8) Tunable constants

Key constants in `segment_audio.py`:

- `SILENCE_DBFS = -55.0`
- `LOW_NOISE_DBFS = -45.0`
- `CHUNK_SEC = 8.0`
- `FINE_SEARCH_RADIUS_SEC = 4.0`
- `FINE_WIN_SEC = 0.975`
- `FINE_HOP_SEC = 0.2`

Adjusting these changes sensitivity, smoothness, and speed.

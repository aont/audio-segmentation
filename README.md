# YAMNet Audio Segmentation Tool

This repository provides a Python tool that segments an input WAV audio file into intervals labeled as:

- `Silent1` (very quiet)
- `Silent2` (moderately quiet)
- `Speech`
- `Music`

## File

- `yamnet_segment_tool.py`

## Usage

```bash
python yamnet_segment_tool.py input.wav \
  --silent1-db -55 \
  --silent2-db -40 \
  --tyam 0.96 \
  --tfine 0.1 \
  --log-level DEBUG
```

The script outputs JSON with:

- `step1_intervals`: merged coarse intervals from Step 1
- `t1_list`: coarse boundary list
- `t2_list`: refined boundary list from Step 2

## Notes

- Input should be a WAV file readable by TensorFlow's `tf.audio.decode_wav`.
- YAMNet is loaded from TF-Hub (`google/yamnet/1`).
- Debug logs are emitted with `--log-level DEBUG`.

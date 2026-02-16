#!/usr/bin/env python3
"""Audio segmentation tool using YAMNet for Music/Speech/Silence boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn


LOGGER = logging.getLogger("yamnet_segmenter")
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)


@dataclass
class Interval:
    start: float
    end: float
    label: str


class TensorflowLazyRuntime:
    """Lazily imports tensorflow and tensorflow_hub only when needed."""

    def __init__(self) -> None:
        self._tf = None
        self._hub = None
        self._yamnet = None
        self._labels: Optional[List[str]] = None

    def tf(self):
        if self._tf is None:
            import tensorflow as tf  # noqa: PLC0415

            self._tf = tf
        return self._tf

    def hub(self):
        if self._hub is None:
            import tensorflow_hub as hub  # noqa: PLC0415

            self._hub = hub
        return self._hub

    def yamnet(self):
        if self._yamnet is None:
            LOGGER.debug("Loading YAMNet from TF Hub: %s", YAMNET_HANDLE)
            self._yamnet = self.hub().load(YAMNET_HANDLE)
        return self._yamnet

    def labels(self) -> List[str]:
        if self._labels is None:
            self._labels = load_yamnet_labels()
        return self._labels


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_yamnet_labels() -> List[str]:
    import urllib.request

    try:
        with urllib.request.urlopen(YAMNET_LABELS_URL, timeout=10) as response:
            content = response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to download YAMNet labels from {YAMNET_LABELS_URL}: {exc}"
        ) from exc

    rows = list(csv.DictReader(content.splitlines()))
    labels = [row["display_name"] for row in rows]
    if not labels:
        raise RuntimeError("Downloaded YAMNet labels were empty.")
    return labels


def np_resample(audio: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    """NumPy linear interpolation resampling replacement for tf.signal.resample."""
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.array([], dtype=np.float32)

    duration = audio.size / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    resampled = np.interp(dst_x, src_x, audio)
    return resampled.astype(np.float32)


def _ffmpeg_decode_to_mono16k(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    LOGGER.debug("Falling back to ffmpeg decoding: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=False, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required for non-WAV decoding but was not found in PATH."
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr or 'unknown error'}")

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("ffmpeg decoded empty audio stream.")
    return audio


def load_audio_mono16k(path: Path, runtime: TensorflowLazyRuntime) -> np.ndarray:
    """Decode audio to mono 16k. Try TensorFlow WAV decode first, then ffmpeg fallback."""
    raw = path.read_bytes()
    tf = runtime.tf()

    try:
        wav, sr = tf.audio.decode_wav(raw, desired_channels=1)
        wav_np = wav.numpy().squeeze(axis=-1).astype(np.float32)
        sr_val = int(sr.numpy())
        LOGGER.debug("TensorFlow WAV decode succeeded, sample_rate=%d", sr_val)
        return np_resample(wav_np, src_sr=sr_val, dst_sr=16000)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("TensorFlow WAV decode failed, trying ffmpeg fallback: %s", exc)

    return _ffmpeg_decode_to_mono16k(path)


def to_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if rms <= 1e-12:
        return -np.inf
    return 20.0 * math.log10(rms)


def classify_top_label(top_label: str) -> str:
    low = top_label.lower()
    if "music" in low:
        return "Music"
    if "speech" in low:
        return "Speech"
    if "silence" in low or "quiet" in low:
        return "Silent2"
    return "Other"


def run_yamnet(audio_mono16k: np.ndarray, runtime: TensorflowLazyRuntime) -> Tuple[str, float, str]:
    tf = runtime.tf()
    scores, _, _ = runtime.yamnet()(tf.convert_to_tensor(audio_mono16k, dtype=tf.float32))
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_idx = int(np.argmax(mean_scores))
    top_score = float(mean_scores[top_idx])
    labels = runtime.labels()
    top_label = labels[top_idx] if top_idx < len(labels) else f"class_{top_idx}"
    coarse = classify_top_label(top_label)
    return coarse, top_score, top_label


def split_ranges(duration_s: float, chunk_s: float) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    t = 0.0
    while t < duration_s:
        end = min(duration_s, t + chunk_s)
        points.append((t, end))
        t = end
    return points


def map_to_primary(label: str) -> str:
    if label in {"Silent1", "Silent2"}:
        return "Silent"
    if label in {"Music", "Speech"}:
        return label
    return "Speech"


def merge_same(intervals: Iterable[Interval]) -> List[Interval]:
    merged: List[Interval] = []
    for item in intervals:
        if not merged or merged[-1].label != item.label:
            merged.append(Interval(item.start, item.end, item.label))
        else:
            merged[-1].end = item.end
    return merged


def postprocess_silent_bridges(intervals: List[Interval]) -> List[Interval]:
    out = [Interval(x.start, x.end, x.label) for x in intervals]
    i = 1
    while i < len(out) - 1:
        prev_i, curr_i, next_i = out[i - 1], out[i], out[i + 1]
        if curr_i.label != "Silent":
            i += 1
            continue

        triple = (prev_i.label, curr_i.label, next_i.label)
        midpoint = (curr_i.start + curr_i.end) / 2.0
        if triple == ("Speech", "Silent", "Speech"):
            prev_i.end = next_i.end
            del out[i : i + 2]
            continue
        if triple == ("Music", "Silent", "Music"):
            prev_i.end = midpoint
            next_i.start = midpoint
            del out[i]
            i += 1
            continue
        if triple == ("Music", "Silent", "Speech"):
            prev_i.end = midpoint
            next_i.start = midpoint
            del out[i]
            i += 1
            continue
        if triple == ("Speech", "Silent", "Music"):
            prev_i.end = midpoint
            next_i.start = midpoint
            del out[i]
            i += 1
            continue
        i += 1

    return merge_same(out)


def classify_window(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    db_silent: float,
    runtime: TensorflowLazyRuntime,
) -> Tuple[str, float, str]:
    s = max(0, int(round(start * sample_rate)))
    e = min(audio.size, int(round(end * sample_rate)))
    chunk = audio[s:e]
    db = to_dbfs(chunk)
    if db <= db_silent:
        return "Silent1", 1.0, f"dB={db:.2f} <= {db_silent:.2f}"

    coarse, score, top_label = run_yamnet(chunk, runtime)
    if coarse == "Silent2":
        return "Silent2", score, top_label
    return coarse, score, top_label


def step1(
    audio: np.ndarray,
    tyam: float,
    tsilence: float,
    db_silent: float,
    runtime: TensorflowLazyRuntime,
) -> List[Interval]:
    duration_s = audio.size / 16000.0
    silence_ranges = split_ranges(duration_s, tsilence)
    raw: List[Interval] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Step1 coarse classification", total=len(silence_ranges))
        for seg_start, seg_end in silence_ranges:
            label, score, why = classify_window(
                audio, 16000, seg_start, seg_end, db_silent, runtime
            )
            LOGGER.debug(
                "[Step1-silence] %.2f-%.2f => %s (score=%.3f reason=%s)",
                seg_start,
                seg_end,
                label,
                score,
                why,
            )
            if label in {"Silent1", "Silent2"}:
                raw.append(Interval(seg_start, seg_end, "Silent"))
                progress.advance(task)
                continue

            for yam_start, yam_end in split_ranges(seg_end - seg_start, tyam):
                abs_start = seg_start + yam_start
                abs_end = seg_start + yam_end
                y_label, y_score, y_why = classify_window(
                    audio, 16000, abs_start, abs_end, db_silent, runtime
                )
                primary = map_to_primary(y_label)
                LOGGER.debug(
                    "[Step1-yam] %.2f-%.2f => %s(primary=%s score=%.3f reason=%s)",
                    abs_start,
                    abs_end,
                    y_label,
                    primary,
                    y_score,
                    y_why,
                )
                raw.append(Interval(abs_start, abs_end, primary))
            progress.advance(task)

    merged = merge_same(raw)
    LOGGER.debug("Merged intervals before postprocess: %s", merged)
    final = postprocess_silent_bridges(merged)
    LOGGER.debug("Intervals after postprocess: %s", final)
    return final


def candidate_distance(
    left_label: str,
    right_label: str,
    expected_left: str,
    expected_right: str,
) -> float:
    distance = 0.0
    distance += 0.0 if left_label == expected_left else 1.0
    distance += 0.0 if right_label == expected_right else 1.0
    if left_label == right_label:
        distance += 0.2
    return distance


def step2_refine(
    audio: np.ndarray,
    intervals: List[Interval],
    tyam: float,
    tfine: float,
    db_silent: float,
    runtime: TensorflowLazyRuntime,
) -> List[float]:
    boundaries = [x.end for x in intervals[:-1]]
    refined: List[float] = []
    total_dur = audio.size / 16000.0

    for i, t1 in enumerate(boundaries):
        left_expected = intervals[i].label
        right_expected = intervals[i + 1].label
        t_start = max(0.0, t1 - tyam / 2.0)
        t_end = min(total_dur, t1 + tyam / 2.0)
        candidates = np.arange(t_start, t_end + 1e-9, tfine)

        best_t = t1
        best_d = float("inf")
        for t2 in candidates:
            left_s, left_e = max(0.0, t2 - tyam), t2
            right_s, right_e = t2, min(total_dur, t2 + tyam)

            left_label, _, left_reason = classify_window(
                audio, 16000, left_s, left_e, db_silent, runtime
            )
            right_label, _, right_reason = classify_window(
                audio, 16000, right_s, right_e, db_silent, runtime
            )
            left_primary = map_to_primary(left_label)
            right_primary = map_to_primary(right_label)
            dist = candidate_distance(
                left_primary,
                right_primary,
                left_expected,
                right_expected,
            )
            LOGGER.debug(
                "[Step2] t1=%.3f t2=%.3f left=%s(%s) right=%s(%s) expected=(%s,%s) dist=%.3f",
                t1,
                t2,
                left_label,
                left_reason,
                right_label,
                right_reason,
                left_expected,
                right_expected,
                dist,
            )
            if dist < best_d:
                best_d = dist
                best_t = float(t2)

        LOGGER.info(
            "[Step2] boundary %d t1=%.3f refined t2=%.3f (distance=%.3f)",
            i,
            t1,
            best_t,
            best_d,
        )
        refined.append(best_t)

    return refined


def log_segment_results(intervals: Sequence[Interval], refined_boundaries: Sequence[float]) -> None:
    """Log final segment identification results."""
    LOGGER.info("Segment identification result (%d segments):", len(intervals))
    for idx, interval in enumerate(intervals):
        LOGGER.info(
            "  Segment %d: %.3f-%.3f sec => %s",
            idx,
            interval.start,
            interval.end,
            interval.label,
        )

    if not refined_boundaries:
        LOGGER.info("No internal boundaries were refined.")
        return

    LOGGER.info("Refined boundaries (t2): %s", ", ".join(f"{x:.3f}" for x in refined_boundaries))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment audio into Music/Speech/Silence using YAMNet.",
    )
    parser.add_argument("audio_path", type=Path, help="Path to input audio (wav/m4a/etc).")
    parser.add_argument("--DB_silent", type=float, default=-45.0, help="Silence dBFS threshold.")
    parser.add_argument("--tyam", type=float, default=0.96, help="YAMNet frame length in seconds.")
    parser.add_argument(
        "--tsilence",
        type=float,
        default=3.0,
        help="Coarse silence detection frame length in seconds.",
    )
    parser.add_argument("--tfine", type=float, default=0.1, help="Step2 fine step in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write result JSON.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    runtime = TensorflowLazyRuntime()
    audio = load_audio_mono16k(args.audio_path, runtime)
    LOGGER.info("Loaded audio: %.2f sec", audio.size / 16000.0)

    coarse = step1(audio, args.tyam, args.tsilence, args.DB_silent, runtime)
    t1_list = [x.end for x in coarse[:-1]]
    t2_list = step2_refine(audio, coarse, args.tyam, args.tfine, args.DB_silent, runtime)
    log_segment_results(coarse, t2_list)

    payload = {
        "intervals": [
            {"start": x.start, "end": x.end, "label": x.label}
            for x in coarse
        ],
        "t1_list": t1_list,
        "t2_list": t2_list,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Wrote output JSON to %s", args.output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Segment audio into Silent1/Silent2/Speech/Music intervals using YAMNet.

Step 1 (coarse): classify fixed 0.96s YAMNet-sized windows.
Step 2 (refinement): refine each coarse boundary at 0.1s granularity.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

TYAM = 0.96
TFINE = 0.1

LABEL_SILENT1 = "Silent1"
LABEL_SILENT2 = "Silent2"
LABEL_SPEECH = "Speech"
LABEL_MUSIC = "Music"


@dataclass
class Segment:
    start: float
    end: float
    label: str


@dataclass
class BoundaryRefineResult:
    t1: float
    t2: float
    distance: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Audio segmentation with YAMNet + silence thresholds. "
            "TensorFlow imports are delayed until actual processing."
        )
    )
    p.add_argument("input_audio", type=Path, help="Path to input audio file (wav/m4a/etc)")
    p.add_argument("--silent1-db", type=float, default=-50.0, help="dBFS threshold for Silent1")
    p.add_argument("--silent2-db", type=float, default=-35.0, help="dBFS threshold for Silent2")
    p.add_argument("--yamnet-handle", default="https://tfhub.dev/google/yamnet/1", help="TF Hub YAMNet handle")
    p.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate (Hz)")
    p.add_argument("--tyam", type=float, default=TYAM, help="YAMNet frame length (seconds)")
    p.add_argument("--tfine", type=float, default=TFINE, help="Refinement step length (seconds)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--output-json", type=Path, default=None, help="Optional path to JSON output")
    return p


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _resample_np(signal: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return signal.astype(np.float32, copy=False)
    if signal.size == 0:
        return signal.astype(np.float32, copy=False)

    duration = signal.shape[0] / float(src_sr)
    dst_n = max(1, int(round(duration * dst_sr)))
    src_t = np.linspace(0.0, duration, num=signal.shape[0], endpoint=False)
    dst_t = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    out = np.interp(dst_t, src_t, signal).astype(np.float32)
    logging.debug("Resampled audio from %d Hz to %d Hz (%d -> %d samples)", src_sr, dst_sr, signal.shape[0], out.shape[0])
    return out


def _decode_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    logging.debug("Running ffmpeg fallback: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required for non-WAV decoding but was not found on PATH. "
            "Install ffmpeg and retry."
        ) from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed for '{path}': {stderr or 'unknown ffmpeg error'}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg decoded zero samples for '{path}'.")
    logging.debug("ffmpeg decoded %d mono samples at %d Hz", audio.size, sample_rate)
    return audio


def load_audio_mono16k(path: Path, sample_rate: int) -> np.ndarray:
    """Load audio to mono float32 at sample_rate.

    First attempts TensorFlow WAV decode.
    If WAV header parsing fails, falls back to ffmpeg decoding.
    """
    logging.info("Loading audio: %s", path)

    # Delayed TensorFlow import so --help does not trigger heavy imports.
    import tensorflow as tf  # noqa: WPS433

    raw = path.read_bytes()
    try:
        wav, sr = tf.audio.decode_wav(raw, desired_channels=1)
        audio = tf.squeeze(wav, axis=-1).numpy().astype(np.float32)
        src_sr = int(sr.numpy())
        logging.debug("TensorFlow WAV decode succeeded: sr=%d, samples=%d", src_sr, audio.shape[0])
        return _resample_np(audio, src_sr, sample_rate)
    except Exception as exc:
        logging.debug("TensorFlow WAV decode failed (%s). Falling back to ffmpeg.", exc)
        return _decode_with_ffmpeg(path, sample_rate)


def rms_dbfs(signal: np.ndarray, eps: float = 1e-12) -> float:
    if signal.size == 0:
        return -120.0
    rms = math.sqrt(float(np.mean(np.square(signal))))
    return 20.0 * math.log10(max(rms, eps))


class YAMNetClassifier:
    def __init__(self, handle: str) -> None:
        # Delayed imports, only when actually needed.
        import tensorflow as tf  # noqa: WPS433
        import tensorflow_hub as hub  # noqa: WPS433

        self.tf = tf
        logging.info("Loading YAMNet model from %s", handle)
        self.model = hub.load(handle)
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        self.class_names = self._read_class_names(class_map_path)
        self.speech_indices = self._find_indices(["Speech", "Conversation", "Narration", "Inside, small room"])  # broad speech-ish
        self.music_indices = self._find_indices(["Music", "Singing", "Song", "Musical instrument", "Choir"])
        logging.debug("Loaded %d class names", len(self.class_names))

    def _read_class_names(self, class_map_path: str) -> List[str]:
        names: List[str] = []
        with open(class_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["display_name"])
        return names

    def _find_indices(self, keywords: Sequence[str]) -> List[int]:
        out: List[int] = []
        lowered = [k.lower() for k in keywords]
        for i, name in enumerate(self.class_names):
            lname = name.lower()
            if any(k in lname for k in lowered):
                out.append(i)
        return out

    def infer_scores(self, audio_window: np.ndarray) -> np.ndarray:
        wav = self.tf.convert_to_tensor(audio_window, dtype=self.tf.float32)
        scores, _, _ = self.model(wav)
        return scores.numpy()

    def speech_music_label(self, audio_window: np.ndarray) -> Tuple[str, float, float]:
        scores = self.infer_scores(audio_window)
        mean_scores = np.mean(scores, axis=0)
        speech_score = float(np.sum(mean_scores[self.speech_indices])) if self.speech_indices else 0.0
        music_score = float(np.sum(mean_scores[self.music_indices])) if self.music_indices else 0.0
        label = LABEL_SPEECH if speech_score >= music_score else LABEL_MUSIC
        return label, speech_score, music_score


def classify_window(
    window: np.ndarray,
    silent1_db: float,
    silent2_db: float,
    yamnet: YAMNetClassifier,
) -> Tuple[str, dict]:
    db = rms_dbfs(window)
    meta = {"dbfs": db}
    if db <= silent1_db:
        meta["reason"] = "silent1-threshold"
        return LABEL_SILENT1, meta
    if db <= silent2_db:
        meta["reason"] = "silent2-threshold"
        return LABEL_SILENT2, meta

    label, speech_score, music_score = yamnet.speech_music_label(window)
    meta.update(
        {
            "reason": "yamnet",
            "speech_score": speech_score,
            "music_score": music_score,
        }
    )
    return label, meta


def chunk_audio(audio: np.ndarray, sample_rate: int, window_s: float) -> List[np.ndarray]:
    w = int(round(window_s * sample_rate))
    out: List[np.ndarray] = []
    for start in range(0, audio.shape[0], w):
        piece = audio[start : start + w]
        if piece.shape[0] < w:
            piece = np.pad(piece, (0, w - piece.shape[0]))
        out.append(piece)
    return out


def merge_segments(labels: Sequence[str], window_s: float, n_samples: int, sample_rate: int) -> List[Segment]:
    if not labels:
        return []
    segs: List[Segment] = []
    start_i = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            segs.append(Segment(start_i * window_s, i * window_s, cur))
            start_i = i
            cur = labels[i]
    total_dur = n_samples / float(sample_rate)
    segs.append(Segment(start_i * window_s, min(len(labels) * window_s, total_dur), cur))
    return segs


def step1_coarse(
    audio: np.ndarray,
    sample_rate: int,
    tyam: float,
    silent1_db: float,
    silent2_db: float,
    yamnet: YAMNetClassifier,
) -> Tuple[List[Segment], List[float], List[str]]:
    windows = chunk_audio(audio, sample_rate, tyam)
    labels: List[str] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Step 1 Coarse Classification", total=len(windows))
        for idx, window in enumerate(windows):
            label, meta = classify_window(window, silent1_db, silent2_db, yamnet)
            labels.append(label)
            logging.debug("step1 window=%d t=[%.2f,%.2f] label=%s meta=%s", idx, idx * tyam, (idx + 1) * tyam, label, meta)
            progress.advance(task)

    merged = merge_segments(labels, tyam, audio.shape[0], sample_rate)
    boundaries = [seg.end for seg in merged[:-1]]
    logging.info("Step 1 produced %d merged segments and %d boundaries", len(merged), len(boundaries))
    return merged, boundaries, labels


def _extract_window(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    win = int(round((end_s - start_s) * sample_rate))
    s = int(round(start_s * sample_rate))
    e = s + win
    out = np.zeros((win,), dtype=np.float32)
    src_s = max(s, 0)
    src_e = min(e, audio.shape[0])
    if src_e > src_s:
        dst_s = src_s - s
        dst_e = dst_s + (src_e - src_s)
        out[dst_s:dst_e] = audio[src_s:src_e]
    return out


def label_distance(pred: str, target: str) -> float:
    if pred == target:
        return 0.0
    if pred.startswith("Silent") and target.startswith("Silent"):
        return 0.4
    if pred.startswith("Silent") != target.startswith("Silent"):
        return 1.5
    return 1.0


def step2_refine(
    audio: np.ndarray,
    sample_rate: int,
    tyam: float,
    tfine: float,
    boundaries: Sequence[float],
    merged_segments: Sequence[Segment],
    silent1_db: float,
    silent2_db: float,
    yamnet: YAMNetClassifier,
) -> List[BoundaryRefineResult]:
    out: List[BoundaryRefineResult] = []

    for i, t1 in enumerate(boundaries):
        l1_before = merged_segments[i].label
        l1_after = merged_segments[i + 1].label
        logging.debug("Refining boundary t1=%.3f with labels before=%s after=%s", t1, l1_before, l1_after)

        t_start = t1 - tyam / 2
        t_end = t1 + tyam / 2
        n = int(math.floor((t_end - t_start) / tfine)) + 1
        candidates = [t_start + k * tfine for k in range(max(n, 1))]

        best_t2 = t1
        best_dist = float("inf")
        for t2 in candidates:
            left = _extract_window(audio, sample_rate, t2 - tyam, t2)
            right = _extract_window(audio, sample_rate, t2, t2 + tyam)

            left_label, left_meta = classify_window(left, silent1_db, silent2_db, yamnet)
            right_label, right_meta = classify_window(right, silent1_db, silent2_db, yamnet)

            d = label_distance(left_label, l1_before) + label_distance(right_label, l1_after)
            logging.debug(
                "step2 t1=%.3f t2=%.3f left=%s right=%s dist=%.3f left_meta=%s right_meta=%s",
                t1,
                t2,
                left_label,
                right_label,
                d,
                left_meta,
                right_meta,
            )
            if d < best_dist:
                best_dist = d
                best_t2 = t2

        result = BoundaryRefineResult(t1=t1, t2=best_t2, distance=best_dist)
        out.append(result)
        logging.info("Refined boundary t1=%.3f -> t2=%.3f (distance=%.3f)", t1, best_t2, best_dist)

    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if args.silent1_db > args.silent2_db:
        parser.error("--silent1-db should be <= --silent2-db (more quiet threshold first).")

    audio = load_audio_mono16k(args.input_audio, args.sample_rate)
    yamnet = YAMNetClassifier(args.yamnet_handle)

    merged, t1_list, _ = step1_coarse(
        audio,
        sample_rate=args.sample_rate,
        tyam=args.tyam,
        silent1_db=args.silent1_db,
        silent2_db=args.silent2_db,
        yamnet=yamnet,
    )

    refined = step2_refine(
        audio,
        sample_rate=args.sample_rate,
        tyam=args.tyam,
        tfine=args.tfine,
        boundaries=t1_list,
        merged_segments=merged,
        silent1_db=args.silent1_db,
        silent2_db=args.silent2_db,
        yamnet=yamnet,
    )

    payload = {
        "config": {
            "tyam": args.tyam,
            "tfine": args.tfine,
            "silent1_db": args.silent1_db,
            "silent2_db": args.silent2_db,
            "sample_rate": args.sample_rate,
        },
        "segments_step1": [segment.__dict__ for segment in merged],
        "t1_list": t1_list,
        "t2_list": [r.t2 for r in refined],
        "refine_details": [r.__dict__ for r in refined],
    }

    text = json.dumps(payload, indent=2)
    if args.output_json:
        args.output_json.write_text(text, encoding="utf-8")
        logging.info("Wrote output JSON: %s", args.output_json)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

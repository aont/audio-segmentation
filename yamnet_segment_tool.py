#!/usr/bin/env python3
"""Audio segmentation tool using YAMNet.

This script performs a two-step segmentation pipeline:
1. Coarse classification on fixed YAMNet windows (`tyam`).
2. Boundary refinement around label transitions using a finer time step (`tfine`).

Labels: Silent1, Silent2, Speech, Music.
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rich.progress import track


TARGET_SAMPLE_RATE = 16000
TYAM_DEFAULT = 0.96
TFINE_DEFAULT = 0.1
STEP1_LABEL_HOLD_FRAMES = 4


@dataclass
class ClassificationResult:
    label: str
    rms_db: float
    speech_prob: float
    music_prob: float
    details: Dict[str, float]


@dataclass
class Interval:
    start: float
    end: float
    label: str


class YAMNetSegmenter:
    @staticmethod
    def _merge_adjacent_same_label(intervals: Sequence[Interval]) -> List[Interval]:
        merged: List[Interval] = []
        for seg in intervals:
            if not merged:
                merged.append(Interval(start=seg.start, end=seg.end, label=seg.label))
                continue

            if merged[-1].label == seg.label:
                merged[-1].end = seg.end
            else:
                merged.append(Interval(start=seg.start, end=seg.end, label=seg.label))
        return merged

    @staticmethod
    def _apply_step1_transition_rules(intervals: Sequence[Interval]) -> List[Interval]:
        """Apply Step 1 transition rules on coarse intervals.

        Rules:
        - Speech/Music - Silent(1 or 2)+ - opposite label
          => remove silence and set boundary at silence midpoint.
        - Speech/Music - Silent2+ - same label
          => merge to a single Speech/Music interval.
        """

        voice_labels = {"Speech", "Music"}
        silence_labels = {"Silent1", "Silent2"}

        working = [Interval(start=s.start, end=s.end, label=s.label) for s in intervals]
        transformed: List[Interval] = []

        i = 0
        while i < len(working):
            current = working[i]
            if current.label not in voice_labels:
                transformed.append(current)
                i += 1
                continue

            j = i + 1
            while j < len(working) and working[j].label in silence_labels:
                j += 1

            has_silence_run = j > i + 1
            has_right_voice = j < len(working) and working[j].label in voice_labels
            if not has_silence_run or not has_right_voice:
                transformed.append(current)
                i += 1
                continue

            right = working[j]
            silence_run = working[i + 1 : j]

            if current.label != right.label:
                silent_start = silence_run[0].start
                silent_end = silence_run[-1].end
                mid = (silent_start + silent_end) / 2.0

                transformed.append(Interval(start=current.start, end=mid, label=current.label))
                working[j] = Interval(start=mid, end=right.end, label=right.label)
                i = j
                continue

            if all(seg.label == "Silent2" for seg in silence_run):
                transformed.append(Interval(start=current.start, end=right.end, label=current.label))
                i = j + 1
                continue

            transformed.append(current)
            i += 1

        return YAMNetSegmenter._merge_adjacent_same_label(transformed)

    def __init__(
        self,
        silent1_db: float,
        silent2_db: float,
        tyam: float = TYAM_DEFAULT,
        tfine: float = TFINE_DEFAULT,
    ) -> None:
        if silent1_db >= silent2_db:
            raise ValueError("silent1_db must be lower than silent2_db (more quiet).")
        self.silent1_db = silent1_db
        self.silent2_db = silent2_db
        self.tyam = tyam
        self.tfine = tfine

        import tensorflow_hub as hub

        logging.info("Loading YAMNet model from TF-Hub...")
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.class_names = self._load_class_names()
        self.speech_indices, self.music_indices = self._collect_target_class_indices(self.class_names)
        logging.info(
            "Model loaded. speech_classes=%d music_classes=%d",
            len(self.speech_indices),
            len(self.music_indices),
        )

    def _load_class_names(self) -> List[str]:
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        names: List[str] = []
        with open(class_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["display_name"])
        return names

    @staticmethod
    def _collect_target_class_indices(class_names: Sequence[str]) -> Tuple[List[int], List[int]]:
        speech_keywords = ("speech", "conversation", "narration")
        music_keywords = (
            "music",
            "singing",
            "choir",
            "humming",
            "rapping",
            "musical instrument",
        )

        speech_idx: List[int] = []
        music_idx: List[int] = []
        for i, name in enumerate(class_names):
            lname = name.lower()
            if any(k in lname for k in speech_keywords):
                speech_idx.append(i)
            if any(k in lname for k in music_keywords):
                music_idx.append(i)
        return speech_idx, music_idx

    @staticmethod
    def _rms_db(segment: np.ndarray, eps: float = 1e-10) -> float:
        if len(segment) == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(segment))))
        return 20.0 * math.log10(max(rms, eps))

    def _slice_with_padding(self, audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
        start = int(round(start_s * TARGET_SAMPLE_RATE))
        end = int(round(end_s * TARGET_SAMPLE_RATE))
        if end <= start:
            return np.zeros(1, dtype=np.float32)

        src_start = max(0, start)
        src_end = min(len(audio), end)
        chunk = audio[src_start:src_end]

        left_pad = max(0, -start)
        right_pad = max(0, end - len(audio))
        if left_pad > 0 or right_pad > 0:
            chunk = np.pad(chunk, (left_pad, right_pad), mode="constant")
        return chunk.astype(np.float32, copy=False)

    def classify_window(self, segment: np.ndarray) -> ClassificationResult:
        rms_db = self._rms_db(segment)

        if rms_db <= self.silent1_db:
            return ClassificationResult(
                label="Silent1",
                rms_db=rms_db,
                speech_prob=0.0,
                music_prob=0.0,
                details={
                    "speech_prob": 0.0,
                    "music_prob": 0.0,
                    "silence_db": rms_db,
                },
            )
        if rms_db <= self.silent2_db:
            return ClassificationResult(
                label="Silent2",
                rms_db=rms_db,
                speech_prob=0.0,
                music_prob=0.0,
                details={
                    "speech_prob": 0.0,
                    "music_prob": 0.0,
                    "silence_db": rms_db,
                },
            )

        scores, _, _ = self.model(segment)
        score_np = scores.numpy()
        mean_scores = score_np.mean(axis=0) if score_np.ndim == 2 else score_np
        speech_prob = float(mean_scores[self.speech_indices].sum()) if self.speech_indices else 0.0
        music_prob = float(mean_scores[self.music_indices].sum()) if self.music_indices else 0.0

        label = "Speech" if speech_prob >= music_prob else "Music"

        details = {
            "speech_prob": speech_prob,
            "music_prob": music_prob,
            "silence_db": rms_db,
        }
        return ClassificationResult(
            label=label,
            rms_db=rms_db,
            speech_prob=speech_prob,
            music_prob=music_prob,
            details=details,
        )

    def classify_at(self, audio: np.ndarray, start_s: float, end_s: float) -> ClassificationResult:
        segment = self._slice_with_padding(audio, start_s, end_s)
        return self.classify_window(segment)

    def step1_coarse(self, audio: np.ndarray, duration_s: float) -> Tuple[List[Interval], List[float]]:
        logging.info("Step 1: coarse classification started (tyam=%.2fs)", self.tyam)
        n_segments = int(math.ceil(duration_s / self.tyam))
        coarse: List[Interval] = []
        forced_labels: Dict[int, str] = {}
        for i in track(range(n_segments), description="Step 1 (coarse classification)", total=n_segments):
            start = i * self.tyam
            end = min((i + 1) * self.tyam, duration_s)

            if i in forced_labels:
                coarse.append(Interval(start=start, end=end, label=forced_labels[i]))
                logging.debug(
                    "[STEP1] seg=%04d %.2f-%.2f label=%s (skipped; copied from previous YAMNet result)",
                    i,
                    start,
                    end,
                    forced_labels[i],
                )
                continue

            result = self.classify_at(audio, start, end)
            coarse.append(Interval(start=start, end=end, label=result.label))
            logging.debug(
                "[STEP1] seg=%04d %.2f-%.2f label=%s db=%.2f speech=%.4f music=%.4f",
                i,
                start,
                end,
                result.label,
                result.rms_db,
                result.speech_prob,
                result.music_prob,
            )

            if result.label in {"Speech", "Music"}:
                for skip_idx in range(i + 1, min(i + 1 + STEP1_LABEL_HOLD_FRAMES, n_segments)):
                    forced_labels[skip_idx] = result.label

        merged = self._merge_adjacent_same_label(coarse)
        merged = self._apply_step1_transition_rules(merged)

        t1_list = [interval.end for interval in merged[:-1]]
        logging.info("Step 1 complete. segments=%d merged_intervals=%d boundaries=%d", n_segments, len(merged), len(t1_list))
        return merged, t1_list

    @staticmethod
    def _distance(expected_label: str, observed: ClassificationResult) -> float:
        mismatch = 0.0 if observed.label == expected_label else 1.0
        confidence = observed.details.get("speech_prob", 0.0) if expected_label == "Speech" else 0.0
        confidence = observed.details.get("music_prob", 0.0) if expected_label == "Music" else confidence

        if expected_label == "Silent1":
            confidence = 1.0 if observed.rms_db <= -60 else 0.0
        elif expected_label == "Silent2":
            confidence = 1.0 if observed.rms_db <= -40 else 0.0

        return mismatch + 0.25 * (1.0 - max(0.0, min(1.0, confidence)))

    def step2_refine(
        self,
        audio: np.ndarray,
        duration_s: float,
        intervals: Sequence[Interval],
        t1_list: Sequence[float],
    ) -> List[float]:
        logging.info("Step 2: boundary refinement started (tfine=%.2fs)", self.tfine)
        refined: List[float] = []

        for idx, t1 in enumerate(t1_list):
            l1_before = intervals[idx].label
            l1_after = intervals[idx + 1].label

            cmin = max(t1 - STEP1_LABEL_HOLD_FRAMES * self.tyam / 2.0, self.tyam)
            cmax = min(t1 + STEP1_LABEL_HOLD_FRAMES * self.tyam / 2.0, duration_s - self.tyam)
            if cmin > cmax:
                logging.debug(
                    "[STEP2] boundary=%d t1=%.2f skipped (candidate range empty: %.2f..%.2f)",
                    idx,
                    t1,
                    cmin,
                    cmax,
                )
                refined.append(t1)
                continue

            candidates = np.arange(cmin, cmax + 1e-9, self.tfine)
            best_t = t1
            best_dist = float("inf")

            for t2 in candidates:
                left = self.classify_at(audio, t2 - self.tyam, t2)
                right = self.classify_at(audio, t2, t2 + self.tyam)
                dist = self._distance(l1_before, left) + self._distance(l1_after, right)

                logging.debug(
                    "[STEP2] b=%d t1=%.2f t2=%.2f left=%s right=%s dist=%.4f",
                    idx,
                    t1,
                    t2,
                    left.label,
                    right.label,
                    dist,
                )

                if dist < best_dist:
                    best_dist = dist
                    best_t = float(t2)

            logging.info(
                "[STEP2] boundary=%d t1=%.2f -> t2=%.2f (before=%s after=%s min_dist=%.4f)",
                idx,
                t1,
                best_t,
                l1_before,
                l1_after,
                best_dist,
            )
            refined.append(best_t)

        logging.info("Step 2 complete. refined_boundaries=%d", len(refined))
        return refined


def load_audio_mono16k(audio_path: Path) -> Tuple[np.ndarray, float]:
    import tensorflow as tf

    try:
        binary = tf.io.read_file(str(audio_path))
        wav, sample_rate = tf.audio.decode_wav(binary, desired_channels=1)
        audio = tf.squeeze(wav, axis=-1).numpy().astype(np.float32)
        sample_rate_value = int(sample_rate.numpy())
    except tf.errors.InvalidArgumentError:
        audio = _decode_with_ffmpeg(audio_path)
        sample_rate_value = TARGET_SAMPLE_RATE

    if sample_rate_value != TARGET_SAMPLE_RATE:
        audio = _resample_audio(audio, sample_rate_value, TARGET_SAMPLE_RATE)
        sample_rate_value = TARGET_SAMPLE_RATE

    duration_s = len(audio) / float(sample_rate_value)
    return audio, duration_s


def _resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate <= 0:
        raise ValueError(f"Invalid source sample rate: {src_rate}")
    if len(audio) == 0 or src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)

    out_len = max(1, int(round(len(audio) * dst_rate / src_rate)))
    src_x = np.arange(len(audio), dtype=np.float64)
    dst_x = np.linspace(0.0, len(audio) - 1, out_len, dtype=np.float64)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _decode_with_ffmpeg(audio_path: Path) -> np.ndarray:
    ffmpeg_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-",
    ]

    try:
        proc = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Audio is not a WAV file TensorFlow can decode and `ffmpeg` is not installed. "
            "Install ffmpeg or provide a PCM WAV input."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr or 'unknown error'}") from exc

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("Decoded audio is empty.")
    return audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment audio using YAMNet + silence thresholds.")
    parser.add_argument("audio_path", type=Path, help="Input audio file path (WAV recommended; other formats via ffmpeg).")
    parser.add_argument("--silent1-db", type=float, default=-55.0, help="Very quiet threshold in dBFS (Silent1).")
    parser.add_argument("--silent2-db", type=float, default=-40.0, help="Moderately quiet threshold in dBFS (Silent2).")
    parser.add_argument("--tyam", type=float, default=TYAM_DEFAULT, help="YAMNet frame length in seconds (default 0.96).")
    parser.add_argument("--tfine", type=float, default=TFINE_DEFAULT, help="Refinement step in seconds (default 0.1).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.audio_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.audio_path}")

    logging.info("Loading audio: %s", args.audio_path)
    audio, duration_s = load_audio_mono16k(args.audio_path)
    logging.info("Audio loaded. duration=%.2fs, samples=%d", duration_s, len(audio))

    segmenter = YAMNetSegmenter(
        silent1_db=args.silent1_db,
        silent2_db=args.silent2_db,
        tyam=args.tyam,
        tfine=args.tfine,
    )

    merged, t1_list = segmenter.step1_coarse(audio, duration_s)
    t2_list = segmenter.step2_refine(audio, duration_s, merged, t1_list)

    output = {
        "config": {
            "silent1_db": args.silent1_db,
            "silent2_db": args.silent2_db,
            "tyam": args.tyam,
            "tfine": args.tfine,
            "duration_s": duration_s,
        },
        "step1_intervals": [interval.__dict__ for interval in merged],
        "t1_list": t1_list,
        "t2_list": t2_list,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

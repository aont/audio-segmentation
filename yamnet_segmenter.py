#!/usr/bin/env python3
"""Segment audio into music, human_voice, and silence intervals using YAMNet + volume thresholds.

Usage:
    python yamnet_segmenter.py input.wav --output segments.json
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)


@dataclass
class Segment:
    start: float
    end: float
    label: str

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "label": self.label,
        }


class YAMNetAudioSegmenter:
    def __init__(
        self,
        very_low_db: float = -50.0,
        low_db: float = -40.0,
        coarse_step: float = 0.48,
        refine_step: float = 0.02,
        analysis_window: float = 0.96,
    ) -> None:
        self.very_low_db = very_low_db
        self.low_db = low_db
        self.coarse_step = coarse_step
        self.refine_step = refine_step
        self.analysis_window = analysis_window

        self.model = hub.load(YAMNET_HANDLE)
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> List[str]:
        cache_path = Path(tempfile.gettempdir()) / "yamnet_class_map.csv"
        if not cache_path.exists():
            urllib.request.urlretrieve(YAMNET_CLASS_MAP_URL, cache_path)

        names: List[str] = []
        with cache_path.open("r", encoding="utf-8") as f:
            # first line is header
            next(f)
            for line in f:
                fields = line.strip().split(",")
                # CSV layout: index,mid,display_name
                names.append(fields[2])
        return names

    def _rms_db(self, samples: np.ndarray) -> float:
        if samples.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        if rms <= 1e-10:
            return -120.0
        return 20.0 * math.log10(rms)

    def _silence_label(self, db: float) -> str | None:
        if db <= self.very_low_db:
            return "silence_very_low"
        if db <= self.low_db:
            return "silence_low"
        return None

    def _bucket_yamnet_class(self, class_name: str) -> str:
        c = class_name.lower()

        voice_keywords = [
            "speech",
            "conversation",
            "narration",
            "whisper",
            "chant",
            "singing",
            "vocal",
            "choir",
            "yell",
            "shout",
            "screaming",
            "laugh",
        ]
        music_keywords = [
            "music",
            "musical",
            "orchestra",
            "instrument",
            "drum",
            "guitar",
            "piano",
            "violin",
            "synthesizer",
            "song",
            "rapping",
            "hip hop",
        ]

        if any(k in c for k in voice_keywords):
            return "human_voice"
        if any(k in c for k in music_keywords):
            return "music"
        return "other"

    def _classify_non_silent_window(self, samples: np.ndarray) -> str:
        if samples.size == 0:
            return "other"

        waveform = tf.convert_to_tensor(samples, dtype=tf.float32)
        scores, _, _ = self.model(waveform)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        class_idx = int(np.argmax(mean_scores))
        return self._bucket_yamnet_class(self.class_names[class_idx])

    def label_window(self, samples: np.ndarray) -> str:
        db = self._rms_db(samples)
        silence = self._silence_label(db)
        if silence is not None:
            return silence
        return self._classify_non_silent_window(samples)

    def _window_at_time(self, audio: np.ndarray, sr: int, t: float) -> np.ndarray:
        half = self.analysis_window / 2.0
        start_t = max(0.0, t - half)
        end_t = min(len(audio) / sr, t + half)
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        return audio[start_idx:end_idx]

    def _label_at_time(self, audio: np.ndarray, sr: int, t: float) -> str:
        return self.label_window(self._window_at_time(audio, sr, t))

    def _merge_adjacent(self, segments: Sequence[Segment]) -> List[Segment]:
        if not segments:
            return []
        merged: List[Segment] = [Segment(segments[0].start, segments[0].end, segments[0].label)]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg.label == prev.label and seg.start <= prev.end + 1e-6:
                prev.end = max(prev.end, seg.end)
            else:
                merged.append(Segment(seg.start, seg.end, seg.label))
        return merged

    def _refine_boundary(
        self,
        audio: np.ndarray,
        sr: int,
        left_time: float,
        right_time: float,
        left_label: str,
        right_label: str,
    ) -> float:
        if right_time <= left_time:
            return left_time

        t = left_time
        last_left = left_time
        while t <= right_time:
            current = self._label_at_time(audio, sr, t)
            if current == right_label:
                return max(left_time, min(t, right_time))
            if current == left_label:
                last_left = t
            t += self.refine_step

        return (last_left + right_time) / 2.0

    def segment(self, audio: np.ndarray, sr: int) -> List[Segment]:
        duration = len(audio) / sr
        if duration == 0:
            return []

        centers = list(np.arange(self.analysis_window / 2.0, duration, self.coarse_step))
        if not centers:
            centers = [duration / 2.0]

        coarse_labels = [self._label_at_time(audio, sr, t) for t in centers]

        boundaries = [0.0]
        for i in range(len(centers) - 1):
            t_left = centers[i]
            t_right = centers[i + 1]
            left_label = coarse_labels[i]
            right_label = coarse_labels[i + 1]

            if left_label == right_label:
                continue
            refined = self._refine_boundary(audio, sr, t_left, t_right, left_label, right_label)
            boundaries.append(refined)

        boundaries.append(duration)
        boundaries = sorted(set(max(0.0, min(duration, b)) for b in boundaries))

        segments: List[Segment] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end - start <= 1e-4:
                continue
            mid = (start + end) / 2.0
            label = self._label_at_time(audio, sr, mid)
            segments.append(Segment(start, end, label))

        return self._merge_adjacent(segments)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Input audio file path")
    p.add_argument("--output", type=Path, default=Path("segments.json"), help="Output JSON path")
    p.add_argument("--very-low-db", type=float, default=-50.0, help="Very low volume threshold in dBFS")
    p.add_argument("--low-db", type=float, default=-40.0, help="Normal low volume threshold in dBFS")
    p.add_argument("--coarse-step", type=float, default=0.48, help="Initial class-change scan interval in seconds")
    p.add_argument("--refine-step", type=float, default=0.02, help="Boundary refinement scan interval in seconds")
    p.add_argument("--analysis-window", type=float, default=0.96, help="Analysis window duration in seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    audio, sr = librosa.load(args.input.as_posix(), sr=16000, mono=True)

    segmenter = YAMNetAudioSegmenter(
        very_low_db=args.very_low_db,
        low_db=args.low_db,
        coarse_step=args.coarse_step,
        refine_step=args.refine_step,
        analysis_window=args.analysis_window,
    )
    segments = segmenter.segment(audio, sr)

    payload = {
        "input": args.input.as_posix(),
        "sample_rate": sr,
        "very_low_db": args.very_low_db,
        "low_db": args.low_db,
        "segments": [s.to_dict() for s in segments],
    }

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(segments)} segments to {args.output}")


if __name__ == "__main__":
    main()

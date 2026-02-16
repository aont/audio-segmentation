#!/usr/bin/env python3
"""Two-stage (coarse -> fine) audio segmentation using YAMNet.

Labels:
- Silence
- Silence2
- Speech
- Music
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

TARGET_SR = 16000
CHUNK_SEC = 8.0
SILENCE_DBFS = -55.0
LOW_NOISE_DBFS = -45.0
FINE_SEARCH_RADIUS_SEC = 4.0
FINE_WIN_SEC = 0.975
FINE_HOP_SEC = 0.2

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)

SPEECH_KEYWORDS = (
    "speech",
    "conversation",
    "narration",
    "monologue",
    "babbling",
    "laugh",
    "giggle",
    "chuckle",
    "snicker",
    "crying",
    "whimper",
    "wail",
    "screaming",
    "yell",
    "shout",
    "chant",
    "choir",
    "vocal",
)

MUSIC_KEYWORDS = (
    "music",
    "singing",
    "song",
    "musical",
    "orchestra",
    "band",
    "instrument",
    "guitar",
    "piano",
    "drum",
    "violin",
    "saxophone",
    "trumpet",
    "flute",
    "synth",
    "bass",
    "harp",
    "cello",
    "bell",
    "hip hop",
    "jazz",
    "rock",
    "classical",
    "electronic",
)


@dataclass
class Segment:
    start: float
    end: float
    label: str


class YAMNetClassifier:
    def __init__(self) -> None:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies. Install with: pip install tensorflow tensorflow-hub"
            ) from exc

        self.tf = tf
        self.hub = hub
        self.model = hub.load(YAMNET_URL)
        self.speech_indices, self.music_indices = self._load_class_indices()

    def _load_class_indices(self) -> tuple[np.ndarray, np.ndarray]:
        class_map_path = self.tf.keras.utils.get_file(
            fname="yamnet_class_map.csv",
            origin=YAMNET_CLASS_MAP_URL,
        )

        speech_indices: List[int] = []
        music_indices: List[int] = []

        with open(class_map_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                index = int(row["index"])
                name = row["display_name"].lower()

                if any(k in name for k in SPEECH_KEYWORDS):
                    speech_indices.append(index)
                if any(k in name for k in MUSIC_KEYWORDS):
                    music_indices.append(index)

        if not speech_indices or not music_indices:
            raise RuntimeError(
                "Could not build speech/music class index groups from YAMNet class map."
            )

        return np.array(sorted(set(speech_indices))), np.array(sorted(set(music_indices)))

    def classify_non_silent(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return "Silence"

        scores, _, _ = self.model(audio)
        scores_np = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)

        speech_score = float(np.sum(mean_scores[self.speech_indices]))
        music_score = float(np.sum(mean_scores[self.music_indices]))
        return "Speech" if speech_score >= music_score else "Music"


def dbfs_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if rms <= 1e-12:
        return -120.0
    return 20.0 * math.log10(rms)


def dbfs_peak(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-12:
        return -120.0
    return 20.0 * math.log10(peak)


def load_audio_ffmpeg(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("Decoded audio is empty. Unsupported file or decode failure.")

    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 1.0:
        audio = audio / max_abs
    return audio


def classify_chunk(audio: np.ndarray, classifier: YAMNetClassifier) -> str:
    rms_db = dbfs_rms(audio)
    _peak_db = dbfs_peak(audio)

    if rms_db < SILENCE_DBFS:
        return "Silence"
    if rms_db < LOW_NOISE_DBFS:
        return "Silence2"
    return classifier.classify_non_silent(audio)


def build_coarse_segments(
    audio: np.ndarray,
    classifier: YAMNetClassifier,
    on_chunk_done: Callable[[], None] | None = None,
) -> List[Segment]:
    total_dur = len(audio) / TARGET_SR
    if total_dur <= 0:
        return []

    raw_segments: List[Segment] = []
    pos = 0.0
    while pos < total_dur:
        end = min(total_dur, pos + CHUNK_SEC)
        s = int(round(pos * TARGET_SR))
        e = int(round(end * TARGET_SR))
        label = classify_chunk(audio[s:e], classifier)
        raw_segments.append(Segment(pos, end, label))
        if on_chunk_done is not None:
            on_chunk_done()
        pos = end

    if not raw_segments:
        return []

    merged = [raw_segments[0]]
    for seg in raw_segments[1:]:
        last = merged[-1]
        if seg.label == last.label:
            last.end = seg.end
        else:
            merged.append(seg)
    return merged


def classify_window(audio: np.ndarray, start: float, win: float, classifier: YAMNetClassifier) -> str:
    s = max(0, int(round(start * TARGET_SR)))
    e = min(len(audio), int(round((start + win) * TARGET_SR)))
    return classify_chunk(audio[s:e], classifier)


def refine_boundary(
    audio: np.ndarray,
    t0: float,
    left_label: str,
    right_label: str,
    classifier: YAMNetClassifier,
    total_dur: float,
    on_window_done: Callable[[], None] | None = None,
) -> float:
    lo = max(0.0, t0 - FINE_SEARCH_RADIUS_SEC)
    hi = min(total_dur - FINE_WIN_SEC, t0 + FINE_SEARCH_RADIUS_SEC - FINE_WIN_SEC)
    if hi <= lo:
        return t0

    starts = np.arange(lo, hi + 1e-9, FINE_HOP_SEC, dtype=np.float64)
    if starts.size == 0:
        return t0

    labels: List[str] = []
    for st in starts:
        labels.append(classify_window(audio, float(st), FINE_WIN_SEC, classifier))
        if on_window_done is not None:
            on_window_done()

    n = len(labels)
    left_match = np.array([1 if lb == left_label else 0 for lb in labels], dtype=np.int32)
    right_match = np.array([1 if lb == right_label else 0 for lb in labels], dtype=np.int32)

    left_prefix = np.cumsum(left_match)
    right_suffix = np.cumsum(right_match[::-1])[::-1]

    best_k = 0
    best_score = -1
    for k in range(n):
        left_score = int(left_prefix[k])
        right_score = int(right_suffix[k + 1]) if k + 1 < n else 0
        score = left_score + right_score
        if score > best_score:
            best_score = score
            best_k = k

    refined = float(starts[best_k] + FINE_WIN_SEC)
    return min(max(0.0, refined), total_dur)


def refine_segments(
    audio: np.ndarray,
    segments: Sequence[Segment],
    classifier: YAMNetClassifier,
    on_boundary_done: Callable[[], None] | None = None,
) -> List[Segment]:
    if len(segments) <= 1:
        return list(segments)

    total_dur = len(audio) / TARGET_SR
    boundaries = [segments[i].end for i in range(len(segments) - 1)]

    refined_boundaries: List[float] = []
    prev = 0.0
    for i, t0 in enumerate(boundaries):
        left_label = segments[i].label
        right_label = segments[i + 1].label
        r = refine_boundary(audio, t0, left_label, right_label, classifier, total_dur)

        min_b = prev
        max_b = total_dur if i == len(boundaries) - 1 else boundaries[i + 1]
        r = min(max(r, min_b), max_b)

        refined_boundaries.append(r)
        prev = r
        if on_boundary_done is not None:
            on_boundary_done()

    out: List[Segment] = []
    st = 0.0
    for i, seg in enumerate(segments):
        ed = refined_boundaries[i] if i < len(refined_boundaries) else total_dur
        if ed <= st:
            continue
        if out and out[-1].label == seg.label:
            out[-1].end = ed
        else:
            out.append(Segment(st, ed, seg.label))
        st = ed

    if out and out[-1].end < total_dur:
        out[-1].end = total_dur

    return out


def segments_to_dicts(segments: Sequence[Segment]) -> List[dict]:
    return [
        {"start": round(s.start, 3), "end": round(s.end, 3), "label": s.label}
        for s in segments
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage audio segmentation using YAMNet")
    p.add_argument("audio", help="Input audio/video file path")
    p.add_argument("--indent", type=int, default=2, help="JSON output indent")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    audio = load_audio_ffmpeg(args.audio, TARGET_SR)
    classifier = YAMNetClassifier()

    total_dur = len(audio) / TARGET_SR
    chunk_count = int(math.ceil(total_dur / CHUNK_SEC)) if total_dur > 0 else 0

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        coarse_task = progress.add_task("ステップ1: 粗い区間分割", total=chunk_count)
        coarse = build_coarse_segments(
            audio,
            classifier,
            on_chunk_done=lambda: progress.advance(coarse_task),
        )

        boundary_count = max(0, len(coarse) - 1)
        refine_task = progress.add_task("ステップ2: 境界の微調整", total=boundary_count)
        final_segments = refine_segments(
            audio,
            coarse,
            classifier,
            on_boundary_done=lambda: progress.advance(refine_task),
        )

    print(json.dumps(segments_to_dicts(final_segments), indent=args.indent, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

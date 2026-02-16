import unittest

import numpy as np

from segment_audio import FINE_WIN_SEC, Segment, postprocess_step2_segments, refine_boundary


class PostprocessStep2Tests(unittest.TestCase):
    def test_rule3_repeats_until_no_active_silence_active(self):
        segments = [
            Segment(0.0, 10.0, "Music"),
            Segment(10.0, 12.0, "Silence"),
            Segment(12.0, 20.0, "Speech"),
            Segment(20.0, 22.0, "Silence"),
            Segment(22.0, 30.0, "Music"),
        ]

        out = postprocess_step2_segments(segments)

        # All silence between active labels should be removed by midpoint splitting.
        self.assertTrue(all(s.label in {"Music", "Speech"} for s in out))
        self.assertEqual(out[0].start, 0.0)
        self.assertEqual(out[-1].end, 30.0)

    def test_rule2_repeats_until_no_speech_silence_speech(self):
        segments = [
            Segment(0.0, 5.0, "Speech"),
            Segment(5.0, 6.0, "Silence"),
            Segment(6.0, 7.0, "Speech"),
            Segment(7.0, 8.0, "Silence"),
            Segment(8.0, 10.0, "Speech"),
        ]

        out = postprocess_step2_segments(segments)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].label, "Speech")
        self.assertEqual(out[0].start, 0.0)
        self.assertEqual(out[0].end, 10.0)


class _FakeClassifier:
    def __init__(self, labels):
        self._labels = list(labels)
        self._i = 0

    def classify_non_silent(self, audio):
        label = self._labels[self._i]
        self._i += 1
        return label


class RefineBoundaryTests(unittest.TestCase):
    def test_refine_boundary_uses_midpoint_of_transition_windows(self):
        labels = ["Speech", "Speech", "Music", "Music", "Music"]
        classifier = _FakeClassifier(labels)

        total_dur = 1.775
        audio = np.full(int(round(16000 * total_dur)), 0.1, dtype=np.float32)
        refined = refine_boundary(
            audio=audio,
            t0=0.1,
            left_label="Speech",
            right_label="Music",
            classifier=classifier,
            total_dur=total_dur,
        )

        expected = ((0.2 + FINE_WIN_SEC / 2.0) + (0.4 + FINE_WIN_SEC / 2.0)) / 2.0
        self.assertAlmostEqual(refined, expected, places=6)


if __name__ == "__main__":
    unittest.main()

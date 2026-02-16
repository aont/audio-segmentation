import unittest

from segment_audio import Segment, postprocess_step2_segments


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


if __name__ == "__main__":
    unittest.main()

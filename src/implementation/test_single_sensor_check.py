"""Small no-hardware regression tests for single_sensor_check."""

import unittest

import numpy as np

from single_sensor_check import frames_in_first_5s, phase_stats


class PhaseStatsTest(unittest.TestCase):
    def test_mismatched_snr_does_not_crash(self):
        frame = {
            "pts": np.zeros((2, 4), dtype=np.float32),
            "snr": np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
            "_phase_elapsed": 15.0,
        }
        stats = phase_stats([frame])
        self.assertEqual(stats["snr_mismatch"], 1)
        self.assertEqual(stats["snr_fraction"], 0.0)


class FirstFiveTest(unittest.TestCase):
    def test_window_starts_at_phase_collection_not_countdown(self):
        # Phase A collection starts at t=105 (after a 5 s countdown from 100).
        frames = [{"t": 105.1, "_phase_start": 105.0},
                  {"t": 109.9, "_phase_start": 105.0},
                  {"t": 111.0, "_phase_start": 105.0}]
        self.assertEqual(len(frames_in_first_5s(frames)), 2)

    def test_no_frames_is_empty(self):
        self.assertEqual(frames_in_first_5s([]), [])


if __name__ == "__main__":
    unittest.main()

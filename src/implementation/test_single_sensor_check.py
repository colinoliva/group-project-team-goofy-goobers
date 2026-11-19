"""Small no-hardware regression tests for single_sensor_check."""

import unittest

import numpy as np

from single_sensor_check import phase_stats


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


if __name__ == "__main__":
    unittest.main()

"""Tests for the live 2D mapper helpers."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "prototype", "software"))

from live_2d_mapper import complete_frame, estimate_position, parse_distance_line  # noqa: E402


class TestLive2DMapperHelpers(unittest.TestCase):
    def test_parse_distance_line_accepts_uart_format(self):
        parsed = parse_distance_line("S1,dist=13.2")
        self.assertEqual(parsed, ("S1", 13.2))

    def test_parse_distance_line_rejects_bad_input(self):
        self.assertIsNone(parse_distance_line("garbage"))

    def test_complete_frame_requires_all_three_sensors(self):
        self.assertFalse(complete_frame({"S1": 1.0, "S2": 2.0}))
        self.assertTrue(complete_frame({"S1": 1.0, "S2": 2.0, "S3": 3.0}))

    def test_estimate_position_recovers_known_point(self):
        from mapping_mock_data import S1, S2, S3

        true_xy = np.array([10.0, 5.0])
        distances = {
            "S1": float(np.linalg.norm(true_xy - S1)),
            "S2": float(np.linalg.norm(true_xy - S2)),
            "S3": float(np.linalg.norm(true_xy - S3)),
        }
        estimate = estimate_position(distances)
        self.assertIsNotNone(estimate)
        np.testing.assert_allclose(estimate, true_xy, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
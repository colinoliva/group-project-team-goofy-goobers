"""Tests for ultrasonic trilateration mock mapping."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "prototype", "software"))

from mapping_mock_data import (  # noqa: E402
    S1,
    S2,
    S3,
    compute_positions,
    data,
    trilaterate,
)


class TestTrilateration(unittest.TestCase):
    def test_trilaterate_recovers_known_point(self):
        true_xy = np.array([10.0, 5.0])
        r1 = float(np.linalg.norm(true_xy - S1))
        r2 = float(np.linalg.norm(true_xy - S2))
        r3 = float(np.linalg.norm(true_xy - S3))
        est = trilaterate(S1, S2, S3, r1, r2, r3)
        self.assertIsNotNone(est)
        np.testing.assert_allclose(est, true_xy, rtol=1e-5, atol=1e-5)

    def test_compute_positions_skips_none_and_counts_rows(self):
        positions = compute_positions((S1, S2, S3), data)
        # Four rows in `data` have a missing distance; 12 rows have all three ranges
        self.assertEqual(positions.shape, (12, 2))

    def test_compute_positions_empty_when_all_none(self):
        rows = [(None, None, None)]
        out = compute_positions((S1, S2, S3), rows)
        self.assertEqual(out.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()

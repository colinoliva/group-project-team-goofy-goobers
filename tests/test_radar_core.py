"""Hardware-free checks for radar point filtering and voxel accumulation."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "implementation"))

from radar_core import MultiVoxelMap, filter_points  # noqa: E402


def test_filter_points_applies_range_and_snr_thresholds():
    points = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
    ])
    snr = np.array([12.0, 5.0, 30.0])

    xyz, keep = filter_points(points, snr, min_range=0.5, max_range=3.0, min_snr=8.0)

    np.testing.assert_array_equal(keep, [True, False, False])
    np.testing.assert_array_equal(xyz, points[:, :3])


def test_multi_voxel_map_counts_frames_and_sensor_support():
    voxel_map = MultiVoxelMap(voxel=0.5, n_sensors=2)
    point = np.array([[0.1, 0.1, 0.1]])

    voxel_map.add(point, sensor_idx=0)
    voxel_map.add(point, sensor_idx=0)
    voxel_map.add(point, sensor_idx=1)

    positions, owners = voxel_map.occupied(min_hits=3, min_sensors=1)
    np.testing.assert_allclose(positions, [[0.25, 0.25, 0.25]])
    np.testing.assert_array_equal(owners, [0])
    assert voxel_map.occupied(min_hits=3, min_sensors=2)[0].shape == (0, 3)

    voxel_map.add(point, sensor_idx=1)
    positions, owners = voxel_map.occupied(min_hits=4, min_sensors=2)
    np.testing.assert_allclose(positions, [[0.25, 0.25, 0.25]])
    np.testing.assert_array_equal(owners, [0])

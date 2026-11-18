"""Feature extraction for DBSCAN radar clusters."""

import numpy as np


FEATURE_NAMES = [
    "n_points",
    "height_extent",
    "z_top",
    "z_centroid",
    "width_x",
    "depth_y",
    "xy_extent",
    "footprint_area",
    "point_density",
    "doppler_mean",
    "doppler_std",
    "doppler_max_abs",
    "fraction_moving",
    "snr_mean",
    "snr_max",
    "range_to_nearest_sensor",
    "n_points_per_range_squared",
]


def extract_features(xyz, vel, snr=None, sensor_pos=None):
    """Extract a fixed feature dictionary from one cluster."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    vel = np.asarray(vel, dtype=np.float64).reshape(-1)
    n = int(len(xyz))
    if n == 0:
        features = {name: 0.0 for name in FEATURE_NAMES}
        features["snr_mean"] = np.nan
        features["snr_max"] = np.nan
        features["range_to_nearest_sensor"] = np.nan
        features["n_points_per_range_squared"] = 0.0
        return features

    z_min = float(np.min(xyz[:, 2]))
    z_max = float(np.max(xyz[:, 2]))
    width_x = float(np.ptp(xyz[:, 0]))
    depth_y = float(np.ptp(xyz[:, 1]))
    footprint = width_x * depth_y
    xy_extent = max(width_x, depth_y)
    density = n / footprint if footprint > 1e-6 else float(n)
    vel = vel[:n]
    if len(vel) != n:
        vel = np.pad(vel, (0, max(0, n - len(vel))), constant_values=np.nan)[:n]
    finite_vel = vel[np.isfinite(vel)]
    if len(finite_vel):
        doppler_mean = float(np.mean(finite_vel))
        doppler_std = float(np.std(finite_vel))
        doppler_max = float(np.max(np.abs(finite_vel)))
        fraction_moving = float(np.mean(np.abs(finite_vel) > 0.1))
    else:
        doppler_mean = doppler_std = doppler_max = fraction_moving = 0.0

    if snr is None:
        snr_mean = snr_max = np.nan
    else:
        snr = np.asarray(snr, dtype=np.float64).reshape(-1)[:n]
        if len(snr) != n:
            snr = np.pad(snr, (0, max(0, n - len(snr))), constant_values=np.nan)[:n]
        finite_snr = snr[np.isfinite(snr)]
        snr_mean = float(np.mean(finite_snr)) if len(finite_snr) else np.nan
        snr_max = float(np.max(finite_snr)) if len(finite_snr) else np.nan

    if sensor_pos is None:
        nearest_range = float(np.min(np.linalg.norm(xyz, axis=1)))
    else:
        sensors = np.asarray(sensor_pos, dtype=np.float64).reshape(-1, 3)
        if len(sensors):
            distances = np.linalg.norm(xyz[:, None, :] - sensors[None, :, :], axis=2)
            nearest_range = float(np.min(distances))
        else:
            nearest_range = float(np.min(np.linalg.norm(xyz, axis=1)))
    normalized = n / (nearest_range * nearest_range) if nearest_range > 1e-6 else float(n)

    return {
        "n_points": float(n),
        "height_extent": float(z_max - z_min),
        "z_top": z_max,
        "z_centroid": float(np.mean(xyz[:, 2])),
        "width_x": width_x,
        "depth_y": depth_y,
        "xy_extent": xy_extent,
        "footprint_area": float(footprint),
        "point_density": float(density),
        "doppler_mean": doppler_mean,
        "doppler_std": doppler_std,
        "doppler_max_abs": doppler_max,
        "fraction_moving": fraction_moving,
        "snr_mean": snr_mean,
        "snr_max": snr_max,
        "range_to_nearest_sensor": nearest_range,
        "n_points_per_range_squared": float(normalized),
    }


def to_vector(features):
    """Convert a feature dictionary to the fixed feature order."""
    return np.asarray([float(features[name]) for name in FEATURE_NAMES], dtype=np.float64)

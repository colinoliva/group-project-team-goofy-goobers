#!/usr/bin/env python3
"""Check sensor poses by comparing where each sensor sees the same moving person.

usage: python3 check_poses.py [data/<session>] [--target X Y Z] [--radius R]
       (default session: newest in data/; default target 0 0 1.27, radius 0.7 m)

Pair check: for every 0.5 s window in which at least 2 sensors each have >= 3
moving points (|Doppler| >= 0.1 m/s), each sensor's moving points are moved to
the world frame with that sensor's T from meta.json and averaged into one
centroid. With correct poses and one person, the centroids nearly coincide.
Prints, per sensor pair, the median and 90th percentile centroid distance and
the median offset (dx, dy, dz) from the first sensor's centroid to the second's.

Target check: for each sensor, the moving points whose world position is within
--radius of --target (the tape mark). Prints their count, median world position,
offset from the target, and a suggested yaw correction:
    measured bearing - expected bearing,  bearing = atan2(x, y) in the sensor frame
of the median point and of the target. Adding it to the sensor's yaw should turn
the measured bearing onto the target. Nothing is written; sensors.mac.json is
never edited.
"""
import argparse
import glob
import math
import os
from itertools import combinations

import numpy as np

from load_session import load_session
from radar_core import apply_T, pose_to_T, world_to_sensor

WINDOW_S = 0.5
V_MOVING = 0.1
MIN_POINTS = 3


def sensor_transforms(meta):
    """Return {sensor index: (name, 4x4 T)} from meta.json's sensor_layout."""
    out = {}
    for i, entry in enumerate(meta.get("sensor_layout", [])):
        T = entry.get("T")
        if isinstance(T, dict):          # recorder fell back to storing the pose
            T = pose_to_T(**T)
        out[i] = (entry.get("name", f"sensor{i}"), np.asarray(T, dtype=float))
    return out


def bearing_deg(p):
    """Bearing of a sensor-frame point in degrees: 0 = straight ahead, + = right."""
    return math.degrees(math.atan2(p[0], p[1]))


def wrap_deg(a):
    """Wrap an angle to (-180, 180]."""
    return -((-a + 180.0) % 360.0 - 180.0)


def moving_world_points(frames, Ts):
    """Yield (frame, sensor index, world xyz of that frame's moving points)."""
    for fr in frames:
        pts = fr["pts"]
        if len(pts) == 0 or fr["sensor"] not in Ts:
            continue
        moving = pts[np.abs(pts[:, 3]) >= V_MOVING]
        if len(moving):
            yield fr, fr["sensor"], apply_T(Ts[fr["sensor"]][1], moving[:, :3].astype(float))


def pair_check(frames, Ts):
    t0 = frames[0]["t"]
    windows = {}                         # window index -> sensor index -> list of world points
    for fr, s, world in moving_world_points(frames, Ts):
        w = int((fr["t"] - t0) // WINDOW_S)
        windows.setdefault(w, {}).setdefault(s, []).append(world)

    offsets = {pair: [] for pair in combinations(sorted(Ts), 2)}
    used = 0
    for per_sensor in windows.values():
        centroids = {}
        for s, chunks in per_sensor.items():
            pts = np.concatenate(chunks)
            if len(pts) >= MIN_POINTS:
                centroids[s] = pts.mean(axis=0)
        if len(centroids) < 2:
            continue
        used += 1
        for a, b in combinations(sorted(centroids), 2):
            offsets[(a, b)].append(centroids[b] - centroids[a])

    n_windows = int((frames[-1]["t"] - t0) // WINDOW_S) + 1
    print(f"\nPAIR CHECK: {used} of {n_windows} windows of {WINDOW_S} s have >= 2 sensors "
          f"with >= {MIN_POINTS} moving points (|Doppler| >= {V_MOVING} m/s)")
    print(f"{'pair':<10}{'windows':>9}{'median (m)':>13}{'p90 (m)':>10}"
          f"{'dx':>8}{'dy':>8}{'dz':>8}   (median offset, second minus first)")
    for (a, b), off in offsets.items():
        label = f"{Ts[a][0]}-{Ts[b][0]}"
        if not off:
            print(f"{label:<10}{0:>9}{'n/a':>13}{'n/a':>10}")
            continue
        off = np.asarray(off)
        d = np.linalg.norm(off, axis=1)
        med = np.median(off, axis=0)
        print(f"{label:<10}{len(d):>9}{np.median(d):>13.2f}{np.percentile(d, 90):>10.2f}"
              f"{med[0]:>+8.2f}{med[1]:>+8.2f}{med[2]:>+8.2f}")


def target_check(frames, Ts, target, radius):
    near = {s: [] for s in Ts}
    for _, s, world in moving_world_points(frames, Ts):
        keep = np.linalg.norm(world - target, axis=1) <= radius
        if keep.any():
            near[s].append(world[keep])

    print(f"\nTARGET CHECK: moving points within {radius:g} m of target "
          f"({target[0]:g}, {target[1]:g}, {target[2]:g})")
    for s, (name, T) in Ts.items():
        if not near[s]:
            print(f"{name}: 0 points")
            continue
        pts = np.concatenate(near[s])
        med = np.median(pts, axis=0)
        off = med - target
        measured = bearing_deg(world_to_sensor(T, med))
        expected = bearing_deg(world_to_sensor(T, target))
        correction = wrap_deg(measured - expected)
        yaw_now = math.degrees(math.atan2(T[1, 0], T[0, 0]))   # yaw from T (pitch/roll 0)
        print(f"{name}: {len(pts)} points | median world ({med[0]:+.2f}, {med[1]:+.2f}, {med[2]:+.2f}) | "
              f"offset dx {off[0]:+.2f} dy {off[1]:+.2f} dz {off[2]:+.2f} m")
        sensor_xy = T[:2, 3]
        to_target = target[:2] - sensor_xy
        expected_range = float(np.linalg.norm(to_target))
        measured_range = float(np.linalg.norm(med[:2] - sensor_xy))
        radial_dir = to_target / expected_range if expected_range > 0 else np.array([0.0, 1.0])
        right_dir = np.array([radial_dir[1], -radial_dir[0]])   # sensor's right, seen from the sensor
        radial = float(off[:2] @ radial_dir)
        lateral = float(off[:2] @ right_dir)
        print(f"{'':<4}horizontal range measured {measured_range:.2f} m, expected {expected_range:.2f} m | "
              f"offset radial {radial:+.2f} m (+ = farther), lateral {lateral:+.2f} m (+ = sensor's right)")
        print(f"{'':<4}bearing measured {measured:+.1f} deg, expected {expected:+.1f} deg -> "
              f"suggested yaw correction {correction:+.1f} deg "
              f"(yaw {yaw_now:.1f} -> {wrap_deg(yaw_now + correction):.1f}, not applied)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", nargs="?", help="session folder (default: newest in data/)")
    ap.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 1.27],
                    metavar=("X", "Y", "Z"), help="world position of the tape mark")
    ap.add_argument("--radius", type=float, default=0.7, help="target radius in m")
    args = ap.parse_args()

    path = args.path
    if path is None:
        sessions = sorted(d for d in glob.glob("data/*") if os.path.isdir(d))
        if not sessions:
            raise SystemExit("no sessions found in data/")
        path = sessions[-1]
    meta, frames, _ = load_session(path)
    Ts = sensor_transforms(meta)
    print(f"session: {path}")
    for i, (name, T) in Ts.items():
        print(f"  sensor {i} = {name}, position {T[:3, 3].round(3).tolist()}")
    if not frames:
        raise SystemExit("session has no frames")

    pair_check(frames, Ts)
    target_check(frames, Ts, np.asarray(args.target, dtype=float), args.radius)


if __name__ == "__main__":
    main()

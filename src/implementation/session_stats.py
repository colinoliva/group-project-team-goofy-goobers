#!/usr/bin/env python3
"""Per-sensor summary of a recorded session (built on load_session.py).

usage: python3 session_stats.py data/<session folder>

Why: the overall summary pools all sensors. For an EMPTY-room session the useful
numbers are per sensor: how many points per frame, and how many of them are
"moving" (|Doppler| >= 0.1 m/s). Ghost points from sensors interfering with each
other, or from a passer-by, show up as moving points in a room that should have none.
The 10 s blocks show WHEN they happened.
"""
import sys

import numpy as np

V_MOVE = 0.1
BLOCK_S = 10.0


def sensor_stats(frames, sensor, duration):
    """Summary numbers for one sensor's frames (list of dicts with t, frame, pts)."""
    fr = [f for f in frames if f["sensor"] == sensor]
    counts = np.array([len(f["pts"]) for f in fr])
    pts = [f["pts"] for f in fr if len(f["pts"])]
    v = np.abs(np.vstack(pts)[:, 3]) if pts else np.empty(0)
    numbers = np.sort(np.array([f["frame"] for f in fr], dtype=np.int64))
    gaps = np.diff(numbers)
    dropped = int(np.clip(gaps - 1, 0, None).sum())
    moving = int((v >= V_MOVE).sum())
    return {
        "frames": len(fr), "fps": len(fr) / max(duration, 1e-9), "dropped": dropped,
        "pts_per_frame": counts.sum() / max(len(fr), 1),
        "moving_per_frame": moving / max(len(fr), 1),
        "moving_share": moving / max(len(v), 1),
        "p99_v": float(np.percentile(v, 99)) if len(v) else 0.0,
        "max_v": float(v.max()) if len(v) else 0.0,
    }


def block_table(frames, sensors, t0, duration):
    """Moving points per frame, per sensor, in BLOCK_S windows."""
    n_blocks = int(np.ceil(duration / BLOCK_S))
    rows = []
    for b in range(n_blocks):
        lo, hi = t0 + b * BLOCK_S, t0 + (b + 1) * BLOCK_S
        row = []
        for s in sensors:
            fr = [f for f in frames if f["sensor"] == s and lo <= f["t"] < hi]
            mv = sum(int((np.abs(f["pts"][:, 3]) >= V_MOVE).sum()) for f in fr if len(f["pts"]))
            row.append(mv / max(len(fr), 1))
        rows.append((b * BLOCK_S, row))
    return rows


def report(frames):
    """Print the per-sensor table and the block table."""
    sensors = sorted({f["sensor"] for f in frames})
    t0 = min(f["t"] for f in frames)
    duration = max(f["t"] for f in frames) - t0
    print(f"duration {duration:.1f} s, sensors {sensors}")
    print(f"{'sensor':>6} {'frames':>7} {'fps':>5} {'drop':>5} {'pts/fr':>7} "
          f"{'mov/fr':>7} {'mov %':>6} {'p99|v|':>7} {'max|v|':>7}")
    for s in sensors:
        r = sensor_stats(frames, s, duration)
        print(f"{'S' + str(s + 1):>6} {r['frames']:>7d} {r['fps']:>5.1f} {r['dropped']:>5d} "
              f"{r['pts_per_frame']:>7.2f} {r['moving_per_frame']:>7.2f} "
              f"{r['moving_share'] * 100:>5.1f}% {r['p99_v']:>7.2f} {r['max_v']:>7.2f}")
    print("\nmoving points per frame in 10 s blocks")
    print(f"{'from s':>7}" + "".join(f"{'S' + str(s + 1):>8}" for s in sensors))
    for start, row in block_table(frames, sensors, t0, duration):
        print(f"{start:>7.0f}" + "".join(f"{v:>8.2f}" for v in row))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    from load_session import load_session
    meta, frames, events = load_session(sys.argv[1])
    print("scenario:", meta.get("scenario"), "| people:", meta.get("people"), "| events:", len(events))
    report(frames)


if __name__ == "__main__":
    main()

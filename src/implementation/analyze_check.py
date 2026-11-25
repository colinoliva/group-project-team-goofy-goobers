#!/usr/bin/env python3
"""Show WHERE the points are in a single_sensor_check recording.

usage: python3 analyze_check.py [check_S1_....npz ...]
       (no argument: the newest check_*.npz in the current folder)

For each phase (A empty, B standing, C walking) it prints, in points per frame:
  * static points (|Doppler| < 0.1 m/s) per range bin
  * moving points (|Doppler| >= 0.1 m/s) per range bin
and, under each table, the difference to phase A. A person who is really seen
shows up as a positive number in the range bin where they stood (or walked).

Expected npz keys: counts (points per frame), phase (one letter per frame),
pts (N x 4: x, y, z, doppler). Sensor frame: x right, y forward, z up.
"""
import glob
import os
import sys

import numpy as np

EDGES = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 10]
V_STATIC = 0.1


def load(path):
    """Return counts, per-frame phase letters, points (N x 4)."""
    d = np.load(path, allow_pickle=True)
    print(f"\n=== {os.path.basename(path)}")
    print("keys:", {k: d[k].shape for k in d.files})
    for k in ("sensor", "cfg_path", "cfg_sha256", "note"):
        if k in d.files:
            print(f"{k}: {d[k].item()}")
    for k in ("counts", "phase", "pts"):
        if k not in d.files:
            raise SystemExit(f"missing key '{k}'; paste the keys line above so the script can be adapted")
    counts = np.asarray(d["counts"]).astype(int)
    pts = np.asarray(d["pts"], float)
    phase = np.array([p.decode() if isinstance(p, bytes) else str(p) for p in np.asarray(d["phase"])])
    if len(phase) != len(counts):
        raise SystemExit(f"phase has {len(phase)} entries but counts has {len(counts)}; need one per frame")
    if counts.sum() != len(pts):
        raise SystemExit(f"counts sum to {counts.sum()} but pts has {len(pts)} rows")
    return counts, phase, pts


def table(title, sel, phases, n_frames, ph_pt, rng):
    """Print points per frame per range bin for the points selected by `sel`."""
    print(f"\n{title}  (points per frame)")
    print("range bin (m)  " + "".join(f"{a:>5.1f}-{b:<4.1f}" for a, b in zip(EDGES[:-1], EDGES[1:])))
    rows = {}
    for p in phases:
        h = np.histogram(rng[(ph_pt == p) & sel], EDGES)[0] / max(n_frames[p], 1)
        rows[p] = h
        print(f"{p + ' (' + str(n_frames[p]) + ' fr)':<14s}" + "".join(f"{v:>10.2f}" for v in h))
    base = phases[0]
    for p in phases[1:]:
        print(f"{p + ' minus ' + base:<14s}" + "".join(f"{v:>+10.2f}" for v in rows[p] - rows[base]))


def analyze(path):
    counts, phase, pts = load(path)
    phases = sorted(set(phase.tolist()))
    ph_pt = np.repeat(phase, counts)
    n_frames = {p: int((phase == p).sum()) for p in phases}
    xyz, vel = pts[:, :3], pts[:, 3]
    rng = np.linalg.norm(xyz, axis=1)
    az = np.degrees(np.arctan2(xyz[:, 0], xyz[:, 1]))
    static = np.abs(vel) < V_STATIC

    print("\nphase summary")
    for p in phases:
        m = ph_pt == p
        mv = m & ~static
        print(f"{p}: {m.sum() / max(n_frames[p], 1):5.1f} pts/frame | static {static[m].mean() * 100:3.0f}% | "
              f"moving pts/frame {mv.sum() / max(n_frames[p], 1):4.1f} "
              f"(+v {int((vel[mv] > 0).sum())}, -v {int((vel[mv] < 0).sum())}) | "
              f"median az {np.median(az[m]) if m.any() else 0:+5.0f} deg, "
              f"median z {np.median(xyz[m, 2]) if m.any() else 0:+.2f} m")

    table("STATIC points", static, phases, n_frames, ph_pt, rng)
    table("MOVING points", ~static, phases, n_frames, ph_pt, rng)


def main():
    paths = sys.argv[1:] or sorted(glob.glob("check_*.npz"))[-1:]
    if not paths:
        raise SystemExit("no check_*.npz found in this folder")
    for path in paths:
        analyze(path)


if __name__ == "__main__":
    main()

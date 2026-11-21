#!/usr/bin/env python3
"""Turn recorded sessions into labeled cluster rows for train_classifier.py.

usage: python3 build_dataset.py data/<session> [more sessions] --out datasets/real.csv
           [--poses sensors.mac.json] [--window-s 0.5] [--learn-s 30] [--segment-s 0]

Per session:
  * sensor T from meta.json's sensor_layout, or from the pose blocks in --poses
    (matched by sensor name, built with radar_core.pose_to_T)
  * each frame's points -> world frame (same 0.3-9 m sensor-range filter as
    radar_dash_viewer.py), keeping Doppler and SNR
  * background learned (tracking.Background) from |Doppler| < 0.05 points in the
    first --learn-s seconds after the session start, then frozen
  * active period: between the first "go" event and the first "done" after it if
    both exist, otherwise everything after the learning period; the learning
    period is always skipped
  * frames from ALL sensors pooled into non-overlapping --window-s windows;
    foreground = |Doppler| >= 0.1 or not background (as in the viewer);
    tracking.cluster + tracking.detections; one row per detection
  * label from meta scenario: walk, two_people, stand, sit -> human;
    empty, fan, chair -> nonhuman; other scenarios are skipped

With --segment-s > 0 each row's session_id is <folder>_segN, N = number of whole
--segment-s chunks from the active start to the start of the row's window, so
train_classifier.py treats chunks as separate groups. Adjacent chunks of one
session share the room, background and person, so held-out scores between them
are optimistic compared with a split by whole session.

Rows are appended (the file is never overwritten). A session whose session_id
is already in the output file is skipped, so rerunning cannot duplicate rows.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

from features import FEATURE_NAMES, extract_features, to_vector
from load_session import load_session
from radar_core import apply_T, filter_points, pose_to_T
from tracking import Background, cluster, detections

LABELS = {"walk": "human", "two_people": "human", "stand": "human", "sit": "human",
          "empty": "nonhuman", "fan": "nonhuman", "chair": "nonhuman"}
V_LEARN = 0.05          # |Doppler| below this is used to learn the background
V_MOVE = 0.1            # radar_dash_viewer.py --vmove default
EPS = 0.4               # radar_dash_viewer.py --eps default
MIN_RANGE, MAX_RANGE = 0.3, 9.0    # radar_dash_viewer.py filter_points call
COLUMNS = FEATURE_NAMES + ["label", "session_id", "person"]


POSE_WARN_M = 0.05      # warn when --poses moves a sensor this far from the session's meta.json


def sensor_transforms(meta, poses_path=None, label="session"):
    """Return {sensor index: (name, 4x4 T)}; --poses overrides by sensor name.

    With --poses, warns (does not stop) when a sensor's position differs by more than
    POSE_WARN_M from the pose recorded in the session's meta.json: --poses must match the
    rig the session was recorded on.
    """
    override = {}
    if poses_path:
        data = json.loads(Path(poses_path).read_text())
        entries = data.get("sensors", []) if isinstance(data, dict) else data
        for s in entries:
            if isinstance(s, dict) and isinstance(s.get("pose"), dict):
                override[s["name"]] = pose_to_T(**s["pose"])
    out, moved = {}, []
    for i, entry in enumerate(meta.get("sensor_layout", [])):
        name = entry.get("name", f"sensor{i}")
        T = entry.get("T")
        if isinstance(T, dict):
            T = pose_to_T(**T)
        if name in override:
            if T is not None:
                d = float(np.linalg.norm(override[name][:3, 3] - np.asarray(T, float)[:3, 3]))
                if d > POSE_WARN_M:
                    moved.append((name, np.asarray(T, float)[:3, 3], override[name][:3, 3], d))
            T = override[name]
        out[i] = (name, np.asarray(T, dtype=float))
    if poses_path:
        missing = [n for n, _ in out.values() if n not in override]
        if missing:
            print(f"  warning: no pose in {poses_path} for {missing}; using meta.json T")
    if moved:
        print(f"  WARNING: --poses {poses_path} does not match the poses recorded in {label}'s "
              f"meta.json (> {POSE_WARN_M * 100:.0f} cm):")
        for name, old, new, d in moved:
            print(f"    {name}: meta.json ({old[0]:+.3f}, {old[1]:+.3f}, {old[2]:.3f}) -> --poses "
                  f"({new[0]:+.3f}, {new[1]:+.3f}, {new[2]:.3f}), moved {d:.3f} m")
        print("    --poses must match the rig the session was recorded on (rig v1 vs v2); "
              "continuing with --poses as given")
    return out


def world_frames(frames, Ts):
    """Return [(t, world xyz, doppler, snr)] for every frame of a known sensor."""
    out = []
    for fr in frames:
        if fr["sensor"] not in Ts or len(fr["pts"]) == 0:
            continue
        pts = fr["pts"]
        xyz, keep = filter_points(pts, fr["snr"], MIN_RANGE, MAX_RANGE, 0.0)
        if not keep.any():
            continue
        out.append((fr["t"], apply_T(Ts[fr["sensor"]][1], xyz[keep]),
                    pts[keep, 3].astype(float), np.asarray(fr["snr"], dtype=float)[keep]))
    return out


def active_period(events, learn_end):
    """(start, end) of the labeled period; end None means until the last frame."""
    go = next((t for t, name in events if name.strip().lower() == "go"), None)
    done = None
    if go is not None:
        done = next((t for t, name in events if name.strip().lower() == "done" and t > go), None)
    if go is not None and done is not None:
        return max(go, learn_end), done, "go/done events"
    return learn_end, None, "after learning period (no go+done events)"


def existing_sessions(path):
    """Session ids already in the output CSV; raises if its header differs."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if header != COLUMNS:
            raise SystemExit(f"{path} has a different header; refusing to append")
        idx = COLUMNS.index("session_id")
        return {row[idx] for row in reader if len(row) > idx}


def build_session(path, args, writer):
    session_id = os.path.basename(os.path.normpath(path))
    meta, frames, events = load_session(path)
    scenario = meta.get("scenario", "")
    label = LABELS.get(scenario)
    print(f"\n=== {session_id}  scenario={scenario!r}  people={meta.get('people', '')!r}")
    if scenario in args.skip_scenarios:
        print(f"  skipped: scenario {scenario!r} is in --skip-scenarios")
        return 0
    if label is None:
        print(f"  skipped: scenario {scenario!r} has no label")
        return 0
    if not frames:
        print("  skipped: no frames")
        return 0

    Ts = sensor_transforms(meta, args.poses, label=session_id)
    sensor_pos = np.asarray([T[:3, 3] for _, T in Ts.values()])
    start = float(meta.get("start_epoch", frames[0]["t"]))
    learn_end = start + args.learn_s
    wf = world_frames(frames, Ts)

    bg = Background()
    for t, xyz, vel, _ in wf:
        if t < learn_end:
            bg.learn(xyz[np.abs(vel) < V_LEARN])
    bg.freeze()

    a_start, a_end, how = active_period(events, learn_end)
    last_t = frames[-1]["t"]
    a_end = last_t if a_end is None else min(a_end, last_t)
    print(f"  background: {len(bg.occupied)} voxels from first {args.learn_s:g} s | "
          f"active {a_start - start:.1f}-{a_end - start:.1f} s ({how})")
    if a_end <= a_start:
        print("  skipped: no active period after learning")
        return 0

    windows = {}
    for t, xyz, vel, snr in wf:
        if a_start <= t < a_end:
            windows.setdefault(int((t - a_start) // args.window_s), []).append((xyz, vel, snr))

    n_windows = int(np.ceil((a_end - a_start) / args.window_s))
    rows = with_cluster = 0
    sizes = []              # (n_points, xy_extent, height_extent) of each kept cluster
    seg_rows = {}           # segment number -> rows written
    for w, parts in sorted(windows.items()):
        row_id = session_id
        if args.segment_s > 0:
            seg = int((w * args.window_s + 1e-9) // args.segment_s)
            row_id = f"{session_id}_seg{seg}"
        xyz = np.vstack([p[0] for p in parts])
        vel = np.concatenate([p[1] for p in parts])
        snr = np.concatenate([p[2] for p in parts])
        idx = np.flatnonzero((np.abs(vel) >= V_MOVE) | ~bg.is_background(xyz))
        if len(idx) == 0:
            continue
        dets = detections(xyz[idx], cluster(xyz[idx], EPS))
        if dets:
            with_cluster += 1
        for d in dets:
            sel = idx[d["idx"]]
            feats = extract_features(xyz[sel], vel[sel], snr[sel], sensor_pos)
            vec = to_vector(feats)
            sizes.append((feats["n_points"], feats["xy_extent"], feats["height_extent"]))
            writer.writerow([repr(float(v)) for v in vec] + [label, row_id, meta.get("people", "")])
            rows += 1
            if args.segment_s > 0:
                seg_rows[seg] = seg_rows.get(seg, 0) + 1

    share = with_cluster / n_windows if n_windows else 0.0
    print(f"  windows: {n_windows} | clusters kept: {rows} | per label: {{{label!r}: {rows}}} | "
          f"windows with >= 1 cluster: {with_cluster}/{n_windows} ({share:.0%})")
    if args.segment_s > 0:
        n_seg = int(np.ceil((a_end - a_start) / args.segment_s))
        counts = [seg_rows.get(k, 0) for k in range(n_seg)]
        print(f"  segments of {args.segment_s:g} s: {n_seg} | with >= 1 cluster: {len(seg_rows)} | "
              f"rows per segment min/median/max {min(counts)}/{np.median(counts):g}/{max(counts)} | "
              f"session_id {session_id}_seg0..{n_seg - 1}")
    if sizes:
        med = np.median(np.asarray(sizes), axis=0)
        print(f"  median of kept clusters: n_points {med[0]:.1f} | xy_extent {med[1]:.2f} m | "
              f"height_extent {med[2]:.2f} m")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build labeled cluster rows from recorded sessions.")
    ap.add_argument("sessions", nargs="+", help="data/<session> folders")
    ap.add_argument("--out", required=True, help="CSV to append to, e.g. datasets/real.csv")
    ap.add_argument("--poses", default=None, help="sensors JSON with pose blocks (overrides meta.json T)")
    ap.add_argument("--window-s", dest="window_s", type=float, default=0.5)
    ap.add_argument("--learn-s", dest="learn_s", type=float, default=30.0)
    ap.add_argument("--segment-s", dest="segment_s", type=float, default=0.0,
                    help="split each session into chunks of this many seconds (session_id <folder>_segN); 0 = off")
    ap.add_argument("--skip-scenarios", dest="skip_scenarios", default="",
                    help="comma-separated scenarios to skip, e.g. pose_check,fan")
    args = ap.parse_args()
    args.skip_scenarios = {x.strip() for x in args.skip_scenarios.split(",") if x.strip()}
    if args.segment_s < 0:
        raise SystemExit("--segment-s must be >= 0")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {re.sub(r"_seg\d+$", "", sid) for sid in existing_sessions(out)}   # folder names
    new_file = not out.exists() or out.stat().st_size == 0
    total = 0
    with open(out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(COLUMNS)
        for path in args.sessions:
            sid = os.path.basename(os.path.normpath(path))
            if sid in done:
                print(f"\n=== {sid}\n  skipped: already in {out}")
                continue
            total += build_session(path, args, writer)
    print(f"\n{'created' if new_file else 'appended to'} {out}: {total} new rows")


if __name__ == "__main__":
    main()

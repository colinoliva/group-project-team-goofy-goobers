#!/usr/bin/env python3
"""Replay recorded sessions through the viewer's detection pipeline and score it.

usage: python3 eval_detection.py SESSION[:LEARN_S] ... [--poses sensors.mac.json]
           [--window-s 0.0] [--eps 0.4] [--min-samples 4] [--vmove 0.1] [--min-points 5]
           [--confirm-hits 3] [--gate 0.9] [--max-misses 25] [--min-new-points 6]
           [--min-sensors 1] [--init-moving-min 0] [--moving-tracks-only]
           [--sweep PARAM=v1,v2,...]
       e.g. python3 eval_detection.py data/..._walk:45 data/..._empty:45 --sweep window-s=0,0.3,0.5,1.0

Pipeline per 0.1 s tick, as radar_dash_viewer.py runs it with --no-classify:
  world points (T from meta.json or --poses; 0.3-9 m sensor-range filter)
  -> background learned from |Doppler| < 0.05 points for the first LEARN_S seconds, then frozen
  -> foreground = |Doppler| >= vmove or not background
  -> pool of the last window-s seconds -> tracking.cluster(eps, min-samples)
  -> tracking.detections(min-points) -> [min-sensors gate]
  -> Tracker.update(min-new-points, ...)  [init-moving-min gate on NEW tracks only]
min-new-points is Tracker's min_new_points (6 in the viewer): a detection needs that many
points to start a track, so with min-points 5 a 5-point detection can only extend one.
init-moving-min exists only in this script; the min-sensors gate (passes_sensor_gate) is
also used by radar_dash_viewer.py --sparse.

Session roles from meta["scenario"]: walk, two_people, pose_check = person present;
empty, fan, chair = quiet. Metrics are taken over the active window: go -> done from
events.csv when both exist; otherwise, for person sessions, inferred from moving points
(the span of the 5 s blocks after learning whose mean over sensors of moving points per
sensor frame, |Doppler| >= 0.1, is >= 0.3; a fixed rule, not tuned); otherwise (quiet
sessions, or no such block) from the end of learning to the last frame. The tracker
itself runs from the end of learning. Reads data/ only; writes nothing.
"""
import argparse
import os
from collections import deque

import numpy as np

from build_dataset import active_period, sensor_transforms
from load_session import load_session
from radar_core import apply_T, filter_points
from tracking import Background, Tracker, cluster, detections

TICK_S = 0.1
V_LEARN = 0.05
MIN_RANGE, MAX_RANGE = 0.3, 9.0
PERSON_ROLES = {"walk", "two_people", "pose_check"}
QUIET_ROLES = {"empty", "fan", "chair"}
DEFAULTS = {"window-s": 0.0, "eps": 0.4, "min-samples": 4, "vmove": 0.1, "min-points": 5,
            "confirm-hits": 3, "gate": 0.9, "max-misses": 25, "min-new-points": 6,
            "min-sensors": 1, "init-moving-min": 0}
INT_PARAMS = {"min-samples", "min-points", "confirm-hits", "max-misses", "min-new-points",
              "min-sensors", "init-moving-min"}
ACTIVE_BLOCK_S = 5.0        # active-window inference for person sessions without go/done
ACTIVE_MIN_MOVING = 0.3     # moving points per sensor frame, mean over sensors
ACTIVE_VMOVE = 0.1


class GatedTracker(Tracker):
    """Tracker whose NEW tracks need >= init_moving_min moving points in their detection.

    Detections that fail the gate are passed on with n = -1, so Tracker.update never
    starts a track from them, but they can still be matched to (extend) existing tracks.
    Also records in self.moved the ids of tracks that were ever given a detection with
    >= 1 moving point (Tracker.update runs with confirm_hits 1 so that its assignment
    covers unconfirmed tracks too; the confirm_hits filter is then applied here, as
    Tracker.update would apply it).
    """

    def __init__(self, init_moving_min=0, **kwargs):
        super().__init__(**kwargs)
        self.init_moving_min = init_moving_min
        self.moved = set()

    def update(self, dets, t):
        if self.init_moving_min > 0:
            dets = [d if d.get("n_moving", 0) >= self.init_moving_min else {**d, "n": -1}
                    for d in dets]
        confirm_hits, self.confirm_hits = self.confirm_hits, 1
        try:
            tracks, assign = super().update(dets, t)
        finally:
            self.confirm_hits = confirm_hits
        self.moved.update(tid for di, tid in assign.items() if dets[di].get("n_moving", 0) > 0)
        confirmed = [tr for tr in tracks if tr.hits >= confirm_hits]
        ids = {tr.id for tr in confirmed}
        return confirmed, {di: tid for di, tid in assign.items() if tid in ids}


def parse_session(arg, default_learn):
    """'path' or 'path:LEARN_S' -> (path, learn_s)."""
    path, _, tail = arg.rpartition(":")
    if path:
        try:
            return path, float(tail)
        except ValueError:
            pass
    return arg, default_learn


def infer_active(frames, merged, start, n_learn):
    """(start, end) spanning the 5 s blocks after learning with enough moving points, or None."""
    per_block = int(round(ACTIVE_BLOCK_S / TICK_S))
    t_fr = np.array([fr["t"] for fr in frames])
    s_fr = np.array([fr["sensor"] for fr in frames])
    sensors = sorted(set(s_fr.tolist()))
    active = []
    for k0 in range(n_learn, len(merged), per_block):
        k1 = min(k0 + per_block, len(merged))
        t0, t1 = start + k0 * TICK_S, start + k1 * TICK_S
        rates = []
        for s in sensors:
            n_fr = int(np.sum((s_fr == s) & (t_fr >= t0) & (t_fr < t1)))
            n_mv = sum(int(np.sum((np.abs(merged[k][1]) >= ACTIVE_VMOVE) & (merged[k][2] == s)))
                       for k in range(k0, k1))
            rates.append(n_mv / n_fr if n_fr else 0.0)
        if rates and np.mean(rates) >= ACTIVE_MIN_MOVING:
            active.append((t0, t1))
    return (active[0][0], active[-1][1]) if active else None


def prepare(path, learn_s, poses):
    """Load a session, bin world points into ticks, learn and freeze the background."""
    meta, frames, events = load_session(path)
    if not frames:
        raise SystemExit(f"{path}: no frames")
    Ts = sensor_transforms(meta, poses, label=os.path.basename(os.path.normpath(path)))
    start = float(meta.get("start_epoch", frames[0]["t"]))
    n_ticks = int((frames[-1]["t"] - start) // TICK_S) + 1
    ticks = [[] for _ in range(n_ticks)]
    n_frames = np.zeros((n_ticks, max(Ts) + 1 if Ts else 0), int)   # frames per tick per sensor
    for fr in frames:
        if fr["sensor"] not in Ts:
            continue
        k = int((fr["t"] - start) // TICK_S)
        if k < 0:
            continue
        n_frames[k, fr["sensor"]] += 1
        if len(fr["pts"]) == 0:
            continue
        xyz, keep = filter_points(fr["pts"], fr["snr"], MIN_RANGE, MAX_RANGE, 0.0)
        if keep.any():
            snr = (np.full(int(keep.sum()), np.nan) if fr["snr"] is None
                   else np.asarray(fr["snr"], dtype=float)[keep])
            ticks[k].append((apply_T(Ts[fr["sensor"]][1], xyz[keep]),
                             fr["pts"][keep, 3].astype(float),
                             np.full(int(keep.sum()), fr["sensor"]), snr))
    merged, snrs = [], []
    for parts in ticks:
        if parts:
            merged.append(tuple(np.concatenate([p[j] for p in parts]) for j in range(3)))
            snrs.append(np.concatenate([p[3] for p in parts]))
        else:
            merged.append((np.empty((0, 3)), np.empty(0), np.empty(0, int)))
            snrs.append(np.empty(0))

    learn_end = start + learn_s
    n_learn = int(np.ceil(learn_s / TICK_S))
    bg = Background()
    for xyz, vel, _ in merged[:n_learn]:          # learn per tick, as the viewer does
        bg.learn(xyz[np.abs(vel) < V_LEARN])
    bg.freeze()

    a_start, a_end, how = active_period(events, learn_end)
    a_end = frames[-1]["t"] if a_end is None else min(a_end, frames[-1]["t"])
    scenario = meta.get("scenario", "")
    role = "person" if scenario in PERSON_ROLES else "quiet" if scenario in QUIET_ROLES else None
    if role == "person" and how != "go/done events":
        inferred = infer_active(frames, merged, start, n_learn)
        if inferred:
            a_start, a_end = inferred
            how = (f"inferred from moving points: 5 s blocks with >= {ACTIVE_MIN_MOVING:g} "
                   f"moving pts per sensor frame")
    return {"name": os.path.basename(os.path.normpath(path)), "scenario": scenario, "role": role,
            "ticks": merged, "snr": snrs, "n_frames": n_frames,
            "sensor_names": {i: name for i, (name, _) in Ts.items()},
            "n_learn": n_learn, "bg": bg, "start": start,
            "active": (a_start, a_end), "how": how, "learn_s": learn_s}


def passes_sensor_gate(d, sid, min_sensors):
    """min-sensors gate: True when at least min_sensors distinct sensors each contribute
    >= 2 of the detection's points. sid holds the sensor id of every clustered point and
    d["idx"] indexes into it (tracking.detections). Also used by radar_dash_viewer.py --sparse.
    """
    _, counts = np.unique(sid[d["idx"]], return_counts=True)
    return int(np.sum(counts >= 2)) >= int(min_sensors)


def replay(sess, p, moving_only=False):
    """Run the pipeline with parameters p; return metrics over the active window.

    moving_only: count a confirmed track only once it has been given a detection with
    >= 1 moving point (so tracks living purely on static residual points are left out).
    """
    tracker = GatedTracker(init_moving_min=int(p["init-moving-min"]), gate=p["gate"],
                           max_misses=int(p["max-misses"]), confirm_hits=int(p["confirm-hits"]),
                           min_new_points=int(p["min-new-points"]))
    bg, pool = sess["bg"], deque()
    a_start, a_end = sess["active"]
    counts, ids, first_seen, series = [], set(), {}, []
    for k in range(sess["n_learn"], len(sess["ticks"])):
        t = sess["start"] + k * TICK_S
        xyz, vel, sid = sess["ticks"][k]
        idx = np.flatnonzero((np.abs(vel) >= p["vmove"]) | ~bg.is_background(xyz))
        while pool and pool[0][0] <= t - p["window-s"]:
            pool.popleft()
        p_xyz = np.vstack([q[1] for q in pool] + [xyz[idx]])
        p_vel = np.concatenate([q[2] for q in pool] + [vel[idx]])
        p_sid = np.concatenate([q[3] for q in pool] + [sid[idx]])
        pool.append((t, xyz[idx], vel[idx], sid[idx]))

        dets = detections(p_xyz, cluster(p_xyz, p["eps"], int(p["min-samples"])),
                          min_points=int(p["min-points"]))
        kept = []
        for d in dets:
            if not passes_sensor_gate(d, p_sid, p["min-sensors"]):
                continue
            d["n_moving"] = int(np.sum(np.abs(p_vel[d["idx"]]) >= p["vmove"]))
            kept.append(d)
        confirmed, _ = tracker.update(kept, t)
        if moving_only:
            confirmed = [tr for tr in confirmed if tr.id in tracker.moved]

        if a_start <= t < a_end:
            counts.append(len(confirmed))
            best = max(confirmed, key=lambda tr: tr.hits) if confirmed else None
            series.append((t, best.pos[:2].copy() if best is not None else None))
            for tr in confirmed:
                ids.add(tr.id)
                first_seen.setdefault(tr.id, t)
    minutes = max(a_end - a_start, 1e-9) / 60.0
    counts = np.asarray(counts) if counts else np.zeros(1)
    return {"ticks": len(counts), "per_tick": float(counts.mean()),
            "share": float(np.mean(counts >= 1)), "ids": len(ids), "ids_per_min": len(ids) / minutes,
            "new_per_min": len(first_seen) / minutes if sess["role"] == "quiet" else None,
            "series": series}     # per active tick: (t, xy of the most-hit confirmed track or None)


def row(label, sess, m):
    new = "-" if m["new_per_min"] is None else f"{m['new_per_min']:.2f}"
    return (f"{label:<21}{sess['name']:<30}{sess['role']:<7}{m['ticks']:>6}{m['per_tick']:>9.2f}"
            f"{m['share']:>9.0%}{m['ids']:>6}{m['ids_per_min']:>8.1f}{new:>9}")


HEADER = (f"{'setting':<21}{'session':<30}{'role':<7}{'ticks':>6}{'trk/tick':>9}"
          f"{'>=1 trk':>9}{'IDs':>6}{'IDs/min':>8}{'new/min':>9}")


def main():
    ap = argparse.ArgumentParser(description="Replay sessions through the detection pipeline.")
    ap.add_argument("sessions", nargs="+", help="data/<session> or data/<session>:LEARN_S")
    ap.add_argument("--poses", default=None, help="sensors JSON with pose blocks (overrides meta.json T)")
    ap.add_argument("--learn-s", dest="learn_s", type=float, default=30.0,
                    help="default learning time for sessions without :LEARN_S")
    for name, value in DEFAULTS.items():
        ap.add_argument(f"--{name}", dest=name.replace("-", "_"),
                        type=int if name in INT_PARAMS else float, default=value)
    ap.add_argument("--moving-tracks-only", action="store_true",
                    help="count a track only once it has been given a detection with a moving point")
    ap.add_argument("--sweep", action="append", default=[], metavar="PARAM=v1,v2,...",
                    help="vary one parameter around the defaults; may be given more than once")
    args = ap.parse_args()
    base = {name: getattr(args, name.replace("-", "_")) for name in DEFAULTS}

    sweeps = []
    for spec in args.sweep:
        name, _, values = spec.partition("=")
        if name not in DEFAULTS or not values:
            raise SystemExit(f"bad --sweep {spec!r}; parameters: {', '.join(DEFAULTS)}")
        cast = int if name in INT_PARAMS else float
        sweeps.append((name, [cast(v) for v in values.split(",") if v.strip()]))

    sessions = []
    for arg in args.sessions:
        path, learn_s = parse_session(arg, args.learn_s)
        sess = prepare(path, learn_s, args.poses)
        if sess["role"] is None:
            print(f"skipped {sess['name']}: scenario {sess['scenario']!r} has no role")
            continue
        a0, a1 = sess["active"]
        print(f"{sess['name']}: scenario {sess['scenario']}, role {sess['role']}, learn {learn_s:g} s, "
              f"background {len(sess['bg'].occupied)} voxels, active "
              f"{a0 - sess['start']:.1f}-{a1 - sess['start']:.1f} s ({sess['how']})")
        sessions.append(sess)
    if not sessions:
        raise SystemExit("no sessions to evaluate")

    print("\nbaseline: " + ", ".join(f"{k}={v:g}" for k, v in base.items()) +
          f" | poses {args.poses or 'meta.json'}" +
          (" | counting only tracks that had moving points" if args.moving_tracks_only else ""))
    print(HEADER)
    for sess in sessions:
        print(row("baseline", sess, replay(sess, base, args.moving_tracks_only)))

    for name, values in sweeps:
        print(f"\nsweep {name} (other parameters at baseline)")
        print(HEADER)
        for sess in sessions:
            print(row("baseline", sess, replay(sess, base, args.moving_tracks_only)))
        for value in values:
            p = dict(base, **{name: value})
            for sess in sessions:
                print(row(f"{name}={value:g}", sess, replay(sess, p, args.moving_tracks_only)))


if __name__ == "__main__":
    main()

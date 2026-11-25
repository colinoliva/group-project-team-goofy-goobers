#!/usr/bin/env python3
"""Presence detection and coarse (~1 m) localization for sparse radar points, replayed from
recorded sessions. Scores presence, not track identities.

usage: python3 presence.py SESSION[:LEARN_S] ... [--poses sensors.mac.json] [--learn-s 30]
           [--sigma 0.7] [--tau 1.0] [--threshold 2.0] [--min-sensors 2] [--static-weight 1.0]
           [--sweep PARAM=v1,v2,...] [--pick NAME,NAME ...] [--no-compare]

Points, ticks (0.1 s), background (learned from |Doppler| < 0.05 for LEARN_S s, then frozen),
the 0.3-9 m range filter and the active window are exactly as in eval_detection.py.
moving = |Doppler| >= 0.1; static residual = not moving and not background.

Heat-map detector (per tick):
  * each sensor has its own xy grid (0.2 m cells, +-6 m); its moving points add a Gaussian
    (sigma) to it; all grids decay with exp(-dt/tau). Grid values are scaled by (1 - decay)/dt,
    so they read as Gaussian-weighted moving points per second near the cell, whatever tau is.
  * static residual points within 0.8 m (xy) of a moving point of the last 1.0 s add, with
    weight static-weight, to one more grid (same sigma and tau). It only moves the peak.
  * detection: the peak of (sum of sensor grids + static grid) over the cells where at least
    min-sensors sensor grids exceed threshold. One peak position per tick, or none.
Window rule (for comparison): 1.0 s windows every 0.5 s; flagged when >= N sensors have >= 2
  moving points in the window; position = median xy of its moving points.
DBSCAN (for comparison): eval_detection.replay; flagged when >= 1 confirmed track; position
  = the confirmed track with the most hits.

Presence metrics over the active window, per 1 s bins (a second is flagged when >= half of
its ticks are flagged):
  person sessions (walk, two_people, pose_check): share of seconds flagged (recall)
  quiet sessions (empty, fan, chair): share of seconds flagged, and false alarms per minute
    (runs of flagged ticks, runs less than 1 s apart merged into one)
  jitter: median (and 90th percentile) step in m of the position between consecutive
    flagged ticks
--pick NAME,NAME: choose the setting on those sessions only (highest mean recall of their
  person sessions, with quiet seconds flagged <= 2% and <= 0.5 false alarms/min in each of
  their quiet sessions; ties: lower jitter) and report it on the other sessions.
Reads data/ only; writes nothing.
"""
import argparse
from collections import deque

import numpy as np

from eval_detection import DEFAULTS as DBSCAN_DEFAULTS
from eval_detection import TICK_S, parse_session, prepare, replay

VMOVE = 0.1
WIN_S, HOP_S = 1.0, 0.5
STATIC_R = 0.8          # static residual counts when this close (xy) to a moving point
RECENT_S = 1.0          # ... of the last RECENT_S seconds (heat map)
CELL, EXTENT = 0.2, 6.0
SENSORS = ("S1", "S2", "S3")
WINDOW_FEATURES = ([f"mov_per_frame_{s}" for s in SENSORS] +
                   ["n_sensors_mov2", "mov_rate", "dop_abs_mean", "dop_abs_std", "dop_abs_max",
                    "snr_mean", "z_mean", "z_std", "xy_std", "centroid_dist", "static_near"])
HEAT_DEFAULTS = {"sigma": 0.7, "tau": 1.0, "threshold": 2.0, "min-sensors": 2, "static-weight": 1.0}
INT_PARAMS = {"min-sensors"}


def masks(sess):
    """Per tick (moving, static residual) masks; cached in the session."""
    if "_masks" not in sess:
        out = []
        for xyz, vel, _ in sess["ticks"]:
            mv = np.abs(vel) >= VMOVE
            out.append((mv, ~mv & ~sess["bg"].is_background(xyz)))
        sess["_masks"] = out
    return sess["_masks"]


# --------------------------------------------------------------------------- window features
def window_features(sess, k0, k1):
    """Features of ticks k0..k1-1. Undefined statistics are 0 (centroid_dist: -1)."""
    mk = masks(sess)
    ticks = sess["ticks"][k0:k1]
    xyz = np.vstack([t[0] for t in ticks])
    vel = np.concatenate([t[1] for t in ticks])
    sid = np.concatenate([t[2] for t in ticks])
    snr = np.concatenate(sess["snr"][k0:k1])
    mv = np.concatenate([m[0] for m in mk[k0:k1]])
    st = np.concatenate([m[1] for m in mk[k0:k1]])
    frames = sess["n_frames"][k0:k1].sum(axis=0)
    names = sess["sensor_names"]
    f = {}
    for name in SENSORS:
        idx = [i for i, n in names.items() if n == name]
        n_fr = int(frames[idx[0]]) if idx else 0
        n_mv = int(np.sum(mv & (sid == idx[0]))) if idx else 0
        f[f"mov_per_frame_{name}"] = n_mv / n_fr if n_fr else 0.0
    M, V, S = xyz[mv], np.abs(vel[mv]), sid[mv]
    counts = {s: int(np.sum(S == s)) for s in np.unique(S)}
    f["n_sensors_mov2"] = sum(c >= 2 for c in counts.values())
    f["mov_rate"] = len(M) / max((k1 - k0) * TICK_S, 1e-9)
    f["dop_abs_mean"] = float(V.mean()) if len(V) else 0.0
    f["dop_abs_std"] = float(V.std()) if len(V) else 0.0
    f["dop_abs_max"] = float(V.max()) if len(V) else 0.0
    sn = snr[mv]
    sn = sn[np.isfinite(sn)]
    f["snr_mean"] = float(sn.mean()) if len(sn) else 0.0
    f["z_mean"] = float(M[:, 2].mean()) if len(M) else 0.0
    f["z_std"] = float(M[:, 2].std()) if len(M) else 0.0
    f["xy_std"] = float(np.sqrt(M[:, 0].var() + M[:, 1].var())) if len(M) >= 2 else 0.0
    cents = [M[S == s, :2].mean(axis=0) for s, c in counts.items() if c >= 2]
    if len(cents) >= 2:
        f["centroid_dist"] = float(np.mean([np.linalg.norm(a - b) for i, a in enumerate(cents)
                                            for b in cents[i + 1:]]))
    else:
        f["centroid_dist"] = -1.0
    Q = xyz[st]
    if len(Q) and len(M):
        d = np.linalg.norm(Q[:, None, :2] - M[None, :, :2], axis=2).min(axis=1)
        f["static_near"] = int(np.sum(d < STATIC_R))
    else:
        f["static_near"] = 0
    f["_pos"] = np.median(M[:, :2], axis=0) if len(M) else None
    return f


def window_ranges(sess, t0, t1):
    """(k0, k1) of the 1.0 s windows, 0.5 s apart, that lie inside [t0, t1)."""
    per_w, per_hop = int(round(WIN_S / TICK_S)), int(round(HOP_S / TICK_S))
    k = max(int(np.ceil((t0 - sess["start"]) / TICK_S - 1e-9)), sess["n_learn"])
    k_end = min(int((t1 - sess["start"]) / TICK_S + 1e-9), len(sess["ticks"]))
    while k + per_w <= k_end:
        yield k, k + per_w
        k += per_hop


# --------------------------------------------------------------------------- heat map
class HeatMap:
    def __init__(self, n_sensors, sigma, tau, threshold, min_sensors, static_weight):
        self.c = np.arange(-EXTENT + CELL / 2, EXTENT, CELL)
        n = len(self.c)
        self.G = np.zeros((n_sensors, n, n))
        self.Gs = np.zeros((n, n))
        self.sigma, self.thr, self.min_s, self.w = sigma, threshold, min_sensors, static_weight
        self.decay = np.exp(-TICK_S / tau)
        self.scale = (1.0 - self.decay) / TICK_S       # -> weighted points per second
        self.recent = deque()

    def _splat(self, xy):
        keep = np.all(np.abs(xy) <= EXTENT + 3 * self.sigma, axis=1)
        xy = xy[keep]
        if not len(xy):
            return 0.0
        k = -0.5 / self.sigma ** 2
        gx = np.exp(k * (self.c[None, :] - xy[:, :1]) ** 2)
        gy = np.exp(k * (self.c[None, :] - xy[:, 1:2]) ** 2)
        return gx.T @ gy

    def update(self, t, xyz, sid, mv, st):
        self.G *= self.decay
        self.Gs *= self.decay
        for s in range(len(self.G)):
            m = mv & (sid == s)
            if m.any():
                self.G[s] += self._splat(xyz[m, :2])
        self.recent.append((t, xyz[mv, :2]))
        while self.recent and self.recent[0][0] <= t - RECENT_S:
            self.recent.popleft()
        R = np.vstack([r[1] for r in self.recent])
        if st.any() and len(R) and self.w > 0:
            Q = xyz[st, :2]
            near = np.linalg.norm(Q[:, None] - R[None], axis=2).min(axis=1) < STATIC_R
            if near.any():
                self.Gs += self._splat(Q[near])
        rate = self.G * self.scale
        ok = (rate > self.thr).sum(axis=0) >= self.min_s
        if not ok.any():
            return None
        total = rate.sum(axis=0) + self.w * self.Gs * self.scale
        i, j = np.unravel_index(np.argmax(np.where(ok, total, -np.inf)), total.shape)
        return np.array([self.c[i], self.c[j]])

    def peaks(self, k=3, min_sep=1.5, min_sensors=2):
        """Up to k distinct motion sources in the current grids: [(x, y, score)], strongest first.

        Reads the grids as left by the last update() and changes nothing (update() and its
        single peak are unaffected). Candidate cells need >= min_sensors sensor grids above the
        threshold and must be a local maximum of the summed grid among such cells (3x3
        neighbourhood), so a peak's shoulder is not a second source. Non-maximum suppression:
        a peak closer than min_sep metres to a stronger one is dropped. score is the summed
        grid value (weighted moving points per second near the cell, plus the static channel).
        With min_sensors = self.min_s, the first peak is the cell update() returns.
        """
        rate = self.G * self.scale
        ok = (rate > self.thr).sum(axis=0) >= min_sensors
        if not ok.any():
            return []
        total = rate.sum(axis=0) + self.w * self.Gs * self.scale
        masked = np.where(ok, total, -np.inf)
        padded = np.pad(masked, 1, constant_values=-np.inf)
        n = masked.shape[0]
        neigh = np.max([padded[1 + di:1 + di + n, 1 + dj:1 + dj + n]
                        for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)], axis=0)
        cand = np.argwhere(ok & (masked >= neigh))
        order = np.argsort(-masked[cand[:, 0], cand[:, 1]], kind="stable")
        out = []
        for i, j in cand[order]:
            x, y = float(self.c[i]), float(self.c[j])
            if all((x - px) ** 2 + (y - py) ** 2 >= min_sep ** 2 for px, py, _ in out):
                out.append((x, y, float(total[i, j])))
                if len(out) >= k:
                    break
        return out


def replay_heatmap(sess, p):
    a0, a1 = sess["active"]
    hm = HeatMap(sess["n_frames"].shape[1], p["sigma"], p["tau"], p["threshold"],
                 int(p["min-sensors"]), p["static-weight"])
    mk, series = masks(sess), []
    for k in range(sess["n_learn"], len(sess["ticks"])):
        t = sess["start"] + k * TICK_S
        xyz, _, sid = sess["ticks"][k]
        pos = hm.update(t, xyz, sid, *mk[k])
        if a0 <= t < a1:
            series.append((t, pos))
    return series


def replay_window_rule(sess, min_sensors):
    """Decision at each tick from the latest 1.0 s window that ended at or before the tick."""
    a0, a1 = sess["active"]
    end = sess["start"] + len(sess["ticks"]) * TICK_S
    decided = {}
    for k0, k1 in window_ranges(sess, sess["start"], end):
        f = window_features(sess, k0, k1)
        decided[k1] = f["_pos"] if f["n_sensors_mov2"] >= min_sensors else None
    series, current = [], None
    for k in range(sess["n_learn"], len(sess["ticks"])):
        if k + 1 in decided:
            current = decided[k + 1]
        t = sess["start"] + k * TICK_S
        if a0 <= t < a1:
            series.append((t, current))
    return series


# --------------------------------------------------------------------------- metrics
def presence_metrics(sess, series):
    a0, a1 = sess["active"]
    if not series:
        return {"secs": 0, "flagged": float("nan"), "fa_per_min": None, "jitter": float("nan"),
                "jitter_p90": float("nan")}
    t = np.array([s[0] for s in series])
    flag = np.array([s[1] is not None for s in series])
    sec = ((t - a0) // 1.0).astype(int)
    per_sec = np.array([flag[sec == s].mean() >= 0.5 for s in np.unique(sec)])
    steps = [np.linalg.norm(series[i + 1][1] - series[i][1]) for i in range(len(series) - 1)
             if flag[i] and flag[i + 1]]
    fa = None
    if sess["role"] == "quiet":
        runs, last_on = 0, None
        gap = int(round(1.0 / TICK_S))
        for i in np.flatnonzero(flag):
            if last_on is None or i - last_on > gap:
                runs += 1
            last_on = i
        fa = runs / max((a1 - a0) / 60.0, 1e-9)
    return {"secs": len(per_sec), "flagged": float(per_sec.mean()), "fa_per_min": fa,
            "jitter": float(np.median(steps)) if steps else float("nan"),
            "jitter_p90": float(np.percentile(steps, 90)) if steps else float("nan")}


HEADER = (f"{'setting':<30}{'session':<24}{'role':<7}{'secs':>6}{'flagged':>9}{'FA/min':>8}"
          f"{'jitter m':>10}{'p90 m':>7}")


def row(label, sess, m):
    fa = "-" if m["fa_per_min"] is None else f"{m['fa_per_min']:.2f}"
    jit = "-" if not np.isfinite(m["jitter"]) else f"{m['jitter']:.2f}"
    j90 = "-" if not np.isfinite(m["jitter_p90"]) else f"{m['jitter_p90']:.2f}"
    return (f"{label:<30}{sess['name'][:23]:<24}{sess['role']:<7}{m['secs']:>6}"
            f"{m['flagged']:>9.0%}{fa:>8}{jit:>10}{j90:>7}")


def pick(results, names):
    """results: [(label, {session name: metrics})]; choose on the sessions in names."""
    best = None
    for label, per in results:
        sel = {n: m for n, m in per.items() if n in names}
        quiet = [m for n, m in sel.items() if m["fa_per_min"] is not None]
        person = [m for n, m in sel.items() if m["fa_per_min"] is None]
        if not person or not quiet:
            continue
        if any(m["flagged"] > 0.02 or m["fa_per_min"] > 0.5 for m in quiet):
            continue
        recall = float(np.mean([m["flagged"] for m in person]))
        jit = float(np.nanmean([m["jitter"] for m in person])) if any(
            np.isfinite(m["jitter"]) for m in person) else np.inf
        key = (recall, -jit)
        if best is None or key > best[0]:
            best = (key, label, per)
    return best


def main():
    ap = argparse.ArgumentParser(description="Presence detection replayed from sessions.")
    ap.add_argument("sessions", nargs="+", help="data/<session> or data/<session>:LEARN_S")
    ap.add_argument("--poses", default=None, help="sensors JSON with pose blocks (overrides meta.json T)")
    ap.add_argument("--learn-s", dest="learn_s", type=float, default=30.0)
    for name, value in HEAT_DEFAULTS.items():
        ap.add_argument(f"--{name}", dest=name.replace("-", "_"),
                        type=int if name in INT_PARAMS else float, default=value)
    ap.add_argument("--sweep", action="append", default=[], metavar="PARAM=v1,v2,...",
                    help="vary one heat-map parameter around the defaults; may repeat")
    ap.add_argument("--pick", action="append", default=[], metavar="NAME,NAME",
                    help="choose a heat-map setting on these sessions, report it on the others")
    ap.add_argument("--no-compare", action="store_true", help="skip the window-rule and DBSCAN rows")
    args = ap.parse_args()
    base = {n: getattr(args, n.replace("-", "_")) for n in HEAT_DEFAULTS}

    sessions = []
    for arg in args.sessions:
        path, learn_s = parse_session(arg, args.learn_s)
        s = prepare(path, learn_s, args.poses)
        if s["role"] is None:
            print(f"skipped {s['name']}: scenario {s['scenario']!r} has no role")
            continue
        a0, a1 = s["active"]
        print(f"{s['name']}: {s['scenario']} ({s['role']}), learn {learn_s:g} s, active "
              f"{a0 - s['start']:.1f}-{a1 - s['start']:.1f} s ({s['how']})")
        sessions.append(s)

    results = []

    def run(label, fn):
        per = {}
        for s in sessions:
            per[s["name"]] = m = presence_metrics(s, fn(s))
            print(row(label, s, m))
        return per

    print("\nheat map baseline: " + ", ".join(f"{k}={v:g}" for k, v in base.items()))
    print(HEADER)
    results.append(("heat baseline", run("heat baseline", lambda s: replay_heatmap(s, base))))
    for spec in args.sweep:
        name, _, values = spec.partition("=")
        if name not in HEAT_DEFAULTS or not values:
            raise SystemExit(f"bad --sweep {spec!r}; parameters: {', '.join(HEAT_DEFAULTS)}")
        cast = int if name in INT_PARAMS else float
        print(f"\nsweep {name} (other heat-map parameters at baseline)")
        print(HEADER)
        for v in [cast(x) for x in values.split(",") if x.strip()]:
            p = dict(base, **{name: v})
            label = f"heat {name}={v:g}"
            results.append((label, run(label, lambda s, p=p: replay_heatmap(s, p))))

    if not args.no_compare:
        print("\ncomparison, same metrics")
        print(HEADER)
        for n in (2, 3):
            run(f"window rule >={n} sensors", lambda s, n=n: replay_window_rule(s, n))
        for label, extra in (("dbscan viewer (window 0)", {}),
                             ("dbscan window 0.5, min-sens 2", {"window-s": 0.5, "min-sensors": 2})):
            p = dict(DBSCAN_DEFAULTS, **extra)
            run(label, lambda s, p=p: replay(s, p)["series"])

    for spec in args.pick:
        names = {n.strip() for n in spec.split(",") if n.strip()}
        unknown = names - {s["name"] for s in sessions}
        if unknown:
            raise SystemExit(f"--pick: unknown sessions {sorted(unknown)}")
        best = pick(results, names)
        print(f"\npick on {sorted(names)} (rule: max mean recall, quiet <= 2% flagged and "
              f"<= 0.5 FA/min; ties: lower jitter):")
        if best is None:
            print("  no heat-map setting meets the quiet-session limits on these sessions")
            continue
        _, label, per = best
        print(f"  chosen: {label}")
        print("  " + HEADER)
        for s in sessions:
            tag = "picked on" if s["name"] in names else "held out"
            print("  " + row(f"{label} [{tag}]", s, per[s["name"]]))


if __name__ == "__main__":
    main()

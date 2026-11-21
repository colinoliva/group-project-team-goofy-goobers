"""
radar_dash_viewer.py - web viewer (Dash/Plotly) for the radar array: sensor positions and
fields of view, static room map, live points, clusters with bounding boxes, tracked people
with IDs/trails, and a "spectate" mode that follows one person.

Run it on the Raspberry Pi (or any PC) and open http://<address>:8050 in a browser.

    python radar_dash_viewer.py                                   # mock data, no hardware
    python radar_dash_viewer.py --mock-dense                      # old dense mock data
    python radar_dash_viewer.py --live sensors.json --freerun --sparse   # sparse-data mode
    python radar_dash_viewer.py --live sensors.json --only S1     # one real sensor at 10 Hz
    python radar_dash_viewer.py --live sensors.json               # 3 sensors, time-division
    python radar_dash_viewer.py --live sensors.json --freerun     # 3 sensors, all at once

Live mode first LEARNS the empty room for --learn seconds (stay out of the sensors' view),
then freezes it as background. Anything moving, or not part of the background, is clustered
and tracked. The "Re-learn background" button repeats this.

Spectate: pick a person in the dropdown. The 3D window re-centres on them each update, and
you can still rotate the view with the mouse.
"""
import argparse
import csv
import logging
import threading
from collections import deque
import time

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update

from radar_core import PALETTE, apply_T, filter_points, fov_lines, grid_lines, pose_to_T
from features import extract_features, to_vector
from rule_baseline import is_human_rule
from tracking import Background, Tracker, cluster, detections

TRACK_COLORS = ["#00d4ff", "#ff6b35", "#a855f7", "#44cf6c", "#f59e0b", "#ef4444"]
SENSOR_HEX = ["#%02x%02x%02x" % tuple(int(255 * c) for c in rgb) for rgb in PALETTE]
Z_LO, Z_HI = -0.05, 3.0

# fixed trace slots - the callback only patches these, never rebuilds the figure
I_MAP, I_LIVE, I_CLUS, I_BOX, I_LABEL, I_TRAIL = range(6)
ZONE_LABEL = "motion zone, about 1 m"
SHOW_MAX_S = 2.0            # longest "Show last (s)" setting; live points kept this long for drawing
HEALTH_WINDOW_S = 5.0       # health row: fps / points per frame over this many seconds
HEALTH_STALE_S = 2.0        # health row: red after this long without a frame


class SensorHealth:
    """Per-sensor frame log for the UI health row. Display only; no effect on processing."""

    def __init__(self, n_sensors):
        self.log = [deque() for _ in range(n_sensors)]     # (arrival time, points) per frame
        self.last = [None] * n_sensors
        self.t_start = time.time()
        self.slot_start, self.prev_active = {}, None
        self.lock = threading.Lock()

    def add(self, frames):
        with self.lock:
            for i, t, n in frames:
                if 0 <= i < len(self.log):
                    self.log[i].append((t, n))
                    self.last[i] = t

    def snapshot(self, now, active, continuous):
        """[(fps, points/frame, seconds since last frame or None, state)], state ok/stale/idle."""
        with self.lock:
            if active != self.prev_active:
                if active is not None:
                    self.slot_start[active] = now
                self.prev_active = active
            span = min(HEALTH_WINDOW_S, max(now - self.t_start, 1e-6))
            out = []
            for i, log in enumerate(self.log):
                while log and log[0][0] < now - HEALTH_WINDOW_S:
                    log.popleft()
                fps = len(log) / span
                ppf = float(np.mean([n for _, n in log])) if log else float("nan")
                age = now - self.last[i] if self.last[i] is not None else None
                if continuous:
                    quiet = (now - self.t_start) if age is None else age
                    state = "stale" if quiet > HEALTH_STALE_S else "ok"
                elif active == i:                  # time-division: judge only the active slot
                    since = max(self.last[i] or 0.0, self.slot_start.get(i, now))
                    state = "stale" if now - since > HEALTH_STALE_S else "ok"
                else:
                    state = "idle"
                out.append((fps, ppf, age, state))
            return out


# ---------------------------------------------------------------------------
# Data sources: both return {"xyz": Nx3 world, "vel": N, "sid": N sensor index}
# ---------------------------------------------------------------------------
def default_sensors(side=2.54, height=1.37):      # rig v2 layout
    rc, out = side / np.sqrt(3), []
    for n, phi in enumerate((90, 210, 330), start=1):
        c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
        out.append({"name": f"S{n}", "T": pose_to_T(rc * c, rc * s, height, np.degrees(np.arctan2(c, -s)))})
    return out


class MockSource:
    """Mock data. Default: sparse points matching the measured live statistics; dense=True is
    the original mock (45 tight points per person per tick), kept for --mock-dense.

    Sparse model, per sensor and 0.1 s tick (a frame arrives with probability 0.95, ~9.5 fps):
      * per person: Poisson(1.0) moving points, xy scattered N(0, 0.68 m) per axis around the
        person (median distance from the centre ~0.8 m), z ~ N(1.1, 0.45) in 0-2.2 m, Doppler
        = the person's radial speed toward the sensor + N(0, 0.3), in 0.19 m/s steps, within
        +-3.05 m/s, never 0; plus Poisson(0.5) zero-Doppler points with the same scatter
      * ghosts: Poisson(0.18) moving points anywhere in the room, and two fixed unlearned
        static spots per sensor with Poisson(0.3) points each
      * static room: ~6 points from the walls (the learned background)
    """
    live = False
    DOPPLER_STEP, V_MAX = 0.19, 3.05

    def __init__(self, dense=False):
        self.sensors = default_sensors()
        self.dense = dense
        self.label = "MOCK DATA (dense)" if dense else "MOCK DATA"
        w = []
        for z in (0.2, 0.7, 1.2, 1.7, 2.2):
            for x in np.arange(-3, 3.01, 0.2):
                w += [[x, -2.5, z], [x, 3.5, z]]
            for y in np.arange(-2.5, 3.51, 0.2):
                w += [[-3, y, z], [3, y, z]]
        self.static = np.array(w)
        self.rng = np.random.default_rng()
        spots = np.random.default_rng(7)                  # same ghost spots on every run
        self.ghost_spots = [spots.uniform([-3, -2.5, 0.2], [3, 3.5, 2.2], (2, 3))
                            for _ in self.sensors]

    def background_points(self):
        return self.static

    @staticmethod
    def people(t):
        """Mock people's xy positions at time t."""
        return [(1.5 * np.sin(0.30 * t), 1.2 * np.cos(0.22 * t) + 0.8),
                (-1.2 + 0.7 * np.cos(0.6 * t), 1.5 + 0.9 * np.sin(0.45 * t))]

    def next_frame(self):
        if self.dense:
            return self._dense_frame()
        return self.sparse_frame(time.time())

    def sparse_frame(self, t):
        r = self.rng
        now = np.array(self.people(t))
        vxy = (np.array(self.people(t + 0.05)) - np.array(self.people(t - 0.05))) / 0.1
        xyz, vel, snr, sid, frames = [], [], [], [], []
        for i, sensor in enumerate(self.sensors):
            if r.random() > 0.95:
                continue
            pos = sensor["T"][:3, 3]
            parts = []                                   # (points, doppler, snr)
            for c, v in zip(now, vxy):
                n = r.poisson(1.0)
                if n:
                    p = np.column_stack([c + r.normal(0, 0.68, (n, 2)),
                                         np.clip(r.normal(1.1, 0.45, n), 0.0, 2.2)])
                    u = p[:, :2] - pos[:2]
                    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
                    dop = u @ v + r.normal(0, 0.3, n)
                    dop = np.clip(np.round(dop / self.DOPPLER_STEP) * self.DOPPLER_STEP,
                                  -self.V_MAX, self.V_MAX)
                    dop[dop == 0] = self.DOPPLER_STEP * np.where(r.random(int(np.sum(dop == 0))) < 0.5, -1, 1)
                    parts.append((p, dop, np.clip(r.normal(24, 7, n), 10, None)))
                n = r.poisson(0.5)
                if n:
                    p = np.column_stack([c + r.normal(0, 0.68, (n, 2)),
                                         np.clip(r.normal(1.1, 0.45, n), 0.0, 2.2)])
                    parts.append((p, np.zeros(n), np.clip(r.normal(18, 5, n), 10, None)))
            n = r.poisson(0.18)
            if n:
                dop = np.round(r.normal(0, 0.8, n) / self.DOPPLER_STEP) * self.DOPPLER_STEP
                dop[dop == 0] = self.DOPPLER_STEP
                parts.append((r.uniform([-3, -2.5, 0], [3, 3.5, 2.5], (n, 3)), dop,
                              np.clip(r.normal(22, 6, n), 10, None)))
            for spot in self.ghost_spots[i]:
                n = r.poisson(0.3)
                if n:
                    parts.append((spot + r.normal(0, 0.05, (n, 3)), np.zeros(n),
                                  np.clip(r.normal(16, 4, n), 10, None)))
            n = r.poisson(6)
            parts.append((self.static[r.integers(0, len(self.static), n)] + r.normal(0, 0.03, (n, 3)),
                          np.zeros(n), np.clip(r.normal(16, 4, n), 10, None)))
            count = sum(len(p[0]) for p in parts)
            frames.append((i, t, count))
            for p, d, s in parts:
                xyz.append(p)
                vel.append(d)
                snr.append(s)
                sid.append(np.full(len(p), i))
        if not xyz:
            return {"xyz": np.empty((0, 3)), "vel": np.empty(0), "snr": np.empty(0),
                    "sid": np.empty(0, int), "frames": frames}
        return {"xyz": np.vstack(xyz), "vel": np.concatenate(vel), "snr": np.concatenate(snr),
                "sid": np.concatenate(sid), "frames": frames}

    def _dense_frame(self):
        t, r = time.time(), self.rng
        people = self.people(t)
        xyz, vel, snr = [], [], []
        for cx, cy in people:
            xyz.append(r.normal(size=(45, 3)) * [0.12, 0.12, 0.30] + [cx, cy, 1.0])
            vel.append(r.normal(0.6, 0.3, 45))
            snr.append(r.normal(18, 4, 45))
        xyz.append(self.static[r.integers(0, len(self.static), 60)] + r.normal(0, 0.03, (60, 3)))
        vel.append(np.zeros(60))
        snr.append(r.normal(12, 3, 60))
        xyz.append(r.uniform([-3, -2.5, 0], [3, 3.5, 2.5], (6, 3)))
        vel.append(r.normal(0, 0.05, 6))
        snr.append(np.full(6, np.nan))
        xyz, vel, snr = np.vstack(xyz), np.concatenate(vel), np.concatenate(snr)
        return {"xyz": xyz, "vel": vel, "snr": snr, "sid": r.integers(0, 3, len(xyz))}


class LiveSource:
    live = True

    def __init__(self, sensors_json, only, freerun, slot, recorder=None):
        from radar_core import RadarLink, Scheduler, load_sensors
        _, sensors = load_sensors(sensors_json)
        if only:
            sensors = [s for s in sensors if s["name"] in only]
        self.sensors, self.links = sensors, []
        for s in sensors:
            link = RadarLink(s["name"], s["cli"], s["data"], s["cfg"])
            self.links.append(link)
            link.configure()
        self.sched = Scheduler(self.links, slot=slot, freerun=freerun)
        self.continuous = freerun or len(self.links) == 1     # health row: no time-division slots
        self.recorder = recorder
        self.sched.start()
        self.label = ("LIVE " + ("freerun" if freerun else "time-division") if len(sensors) > 1
                      else f"LIVE {sensors[0]['name']}")

    def next_frame(self):
        xyz, vel, snr, sid = ([np.empty((0, 3))], [np.empty(0)], [np.empty(0)],
                              [np.empty(0, int)])
        frames = []                                           # (sensor index, arrival t, points)
        for i, link in enumerate(self.links):
            for fr in link.drain():
                if self.recorder is not None:
                    self.recorder.add(i, fr)
                pts = fr["pts"]
                frames.append((i, fr["t"], len(pts)))
                if len(pts) == 0:
                    continue
                p, keep = filter_points(pts, fr["snr"], 0.3, 9.0, 0.0)
                xyz.append(apply_T(self.sensors[i]["T"], p[keep]))
                vel.append(pts[keep, 3].astype(float))
                if fr["snr"] is None:
                    snr.append(np.full(int(keep.sum()), np.nan))
                else:
                    snr.append(np.asarray(fr["snr"], dtype=float)[keep])
                sid.append(np.full(int(keep.sum()), i))
        return {"xyz": np.vstack(xyz), "vel": np.concatenate(vel),
                "snr": np.concatenate(snr), "sid": np.concatenate(sid), "frames": frames}

    def close(self):
        self.sched.halt.set()
        self.sched.join(3.0)
        for l in self.links:
            l.close()


# ---------------------------------------------------------------------------
# Processing thread (acquire -> foreground -> cluster -> track), ~10 Hz
# ---------------------------------------------------------------------------
class Engine(threading.Thread):
    SPARSE_MIN_SENSORS = 2                        # --sparse: eval_detection's min-sensors gate

    def __init__(self, source, learn_s, eps, v_move, classify=True, model_path=None, window_s=0.5,
                 sparse=False):
        super().__init__(daemon=True)
        self.src, self.learn_s, self.eps, self.v_move = source, learn_s, eps, v_move
        self.window_s = window_s
        self.sparse = sparse
        if sparse:                                # gate and heat map reused, not reimplemented
            from eval_detection import passes_sensor_gate
            self.passes_sensor_gate = passes_sensor_gate
            self.heat = self.new_heat_map()
        self.pool = deque()                       # (t, xyz, vel, snr, sid) of recent foreground points
        self.history = deque()                    # (t, xyz, sid) of recent live points, drawing only
        self.health = SensorHealth(len(source.sensors)) if source.live else None
        self.classify, self.model = classify, None
        if classify and model_path:
            from xgboost import XGBClassifier
            self.model = XGBClassifier()
            self.model.load_model(model_path)
        self.bg, self.tracker = Background(), Tracker()
        if hasattr(source, "sensors"):
            self.sensor_pos = np.asarray([sensor["T"][:3, 3] for sensor in source.sensors])
        else:
            self.sensor_pos = None
        self.lock, self.halt = threading.Lock(), threading.Event()
        self.t_learn = time.time()
        self.state = {"xyz": np.empty((0, 3)), "sid": np.empty(0, int), "fg": 0, "dets": [],
                      "assign": {}, "tracks": [], "learning": True, "left": learn_s}
        if not source.live:                       # mock: the room is known, skip learning
            for _ in range(5):
                self.bg.learn(source.background_points())
            self.bg.freeze()

    def new_heat_map(self):
        from presence import HEAT_DEFAULTS, HeatMap
        p = HEAT_DEFAULTS
        return HeatMap(len(self.src.sensors), p["sigma"], p["tau"], p["threshold"],
                       int(p["min-sensors"]), p["static-weight"])

    def relearn(self):
        if self.src.live:
            self.bg.reset()
            self.tracker = Tracker()
            self.pool.clear()
            if self.sparse:
                self.heat = self.new_heat_map()
            self.t_learn = time.time()

    def run(self):
        while not self.halt.is_set():
            t0 = time.time()
            fr = self.src.next_frame()
            xyz, vel, snr, sid = fr["xyz"], fr["vel"], fr["snr"], fr["sid"]
            if self.health is not None:
                self.health.add(fr.get("frames", []))
            self.history.append((t0, xyz, sid))
            while self.history and self.history[0][0] < t0 - SHOW_MAX_S:
                self.history.popleft()
            learning = not self.bg.frozen
            dets, assign, tracks, n_fg, zone = [], {}, [], 0, None
            if learning:
                self.pool.clear()
                self.bg.learn(xyz[np.abs(vel) < 0.05])
                if t0 - self.t_learn > self.learn_s:
                    self.bg.freeze()
            else:
                moving = np.abs(vel) >= self.v_move
                is_bg = self.bg.is_background(xyz)
                fg = moving | ~is_bg
                idx = np.flatnonzero(fg)
                n_fg = len(idx)
                if self.sparse:                   # presence heat map: one peak or None
                    zone = self.heat.update(t0, xyz, sid, moving, ~moving & ~is_bg)
                # Pool foreground points over the last window_s seconds (as build_dataset.py
                # does) and cluster the pool. Older points come first; this tick's are last.
                while self.pool and self.pool[0][0] <= t0 - self.window_s:
                    self.pool.popleft()
                p_xyz = np.vstack([p[1] for p in self.pool] + [xyz[idx]])
                p_vel = np.concatenate([p[2] for p in self.pool] + [vel[idx]])
                p_snr = np.concatenate([p[3] for p in self.pool] + [snr[idx]])
                p_sid = np.concatenate([p[4] for p in self.pool] + [sid[idx]])
                n_old = len(p_xyz) - len(idx)
                self.pool.append((t0, xyz[idx], vel[idx], snr[idx], sid[idx]))
                lab = cluster(p_xyz, self.eps)
                dets = detections(p_xyz, lab)
                if self.sparse:
                    dets = [d for d in dets
                            if self.passes_sensor_gate(d, p_sid, self.SPARSE_MIN_SENSORS)]
                for d in dets:
                    d["pool_idx"] = d["idx"]
                    cur = d["idx"][d["idx"] >= n_old] - n_old
                    d["idx"] = idx[cur]               # this tick's points, for rendering
                if self.classify:
                    keep = []
                    for d in dets:
                        cl_xyz = p_xyz[d["pool_idx"]]
                        cl_vel = p_vel[d["pool_idx"]]
                        cl_snr = p_snr[d["pool_idx"]]
                        features = extract_features(cl_xyz, cl_vel, cl_snr, self.sensor_pos)
                        if self.model is None:
                            human = is_human_rule(features)
                            probability = 1.0 if human else 0.0
                        else:
                            probability = float(self.model.predict_proba(
                                to_vector(features).reshape(1, -1))[0, 1])
                            human = probability >= 0.5
                        d["human_probability"] = probability
                        if human:
                            keep.append(d)
                    dets = keep
                else:
                    for d in dets:
                        d["human_probability"] = 1.0
                tracks, assign = self.tracker.update(dets, t0)
            snap = [{"id": tr.id, "pos": tr.pos.copy(), "trail": np.array(tr.trail)} for tr in tracks]
            with self.lock:
                self.state = {"xyz": xyz, "sid": sid, "fg": n_fg, "dets": dets, "assign": assign,
                              "tracks": snap, "learning": learning, "history": list(self.history),
                              "zone": zone,
                              "left": max(0.0, self.learn_s - (t0 - self.t_learn))}
            self.halt.wait(max(0.0, 0.1 - (time.time() - t0)))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def segs(chunks):
    """List of (k,3) polylines -> x, y, z lists with None breaks."""
    xs, ys, zs = [], [], []
    for c in chunks:
        xs += c[:, 0].tolist() + [None]
        ys += c[:, 1].tolist() + [None]
        zs += c[:, 2].tolist() + [None]
    return xs, ys, zs


def box_chunks(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [c[[a, b]] for a, b in edges]


def build_base_figure(sensors, extent, zone=False):
    gp, gl = grid_lines(half=extent, step=1.0)
    grid = segs([gp[l] for l in gl])
    fov = []
    for s in sensors:
        p, l = fov_lines(s["T"], r=4.0)
        fov += [p[e] for e in l]
    fovx = segs(fov)
    pos = np.array([s["T"][:3, 3] for s in sensors])
    scene_axis = dict(backgroundcolor="#111", gridcolor="#333", color="white")
    fig = go.Figure(data=[
        go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Room map", hoverinfo="skip",
                     marker=dict(size=2, color="#6b7280", opacity=0.55)),
        go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Live points", hoverinfo="skip",
                     marker=dict(size=2, color="#888", opacity=0.6)),
        go.Scatter3d(x=[], y=[], z=[], mode="markers", name="People",
                     marker=dict(size=5, color="#00d4ff")),
        go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip", showlegend=False,
                     line=dict(color="#dddddd", width=3)),
        go.Scatter3d(x=[], y=[], z=[], mode="text", hoverinfo="skip", showlegend=False,
                     textfont=dict(color="white", size=15)),
        go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip", showlegend=False,
                     line=dict(color="#888888", width=2)),
        go.Scatter3d(x=grid[0], y=grid[1], z=grid[2], mode="lines", hoverinfo="skip",
                     showlegend=False, line=dict(color="#2a2a33", width=1)),
        go.Scatter3d(x=fovx[0], y=fovx[1], z=fovx[2], mode="lines", hoverinfo="skip",
                     showlegend=False, line=dict(color="#4a5a8a", width=2)),
        go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text", name="Sensors",
                     text=[s["name"] for s in sensors], textposition="top center",
                     textfont=dict(color="white"),
                     marker=dict(size=7, symbol="diamond",
                                 color=[SENSOR_HEX[i % len(SENSOR_HEX)] for i in range(len(sensors))])),
    ])
    fig.update_layout(
        uirevision="keep", paper_bgcolor="#0d0d0d", margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(color="white"), bgcolor="#1a1a1a", bordercolor="#444"),
        scene=dict(xaxis=dict(range=[-extent, extent], title="X (m)", **scene_axis),
                   yaxis=dict(range=[-extent, extent], title="Y (m)", **scene_axis),
                   zaxis=dict(range=[Z_LO, Z_HI], title="Z (m)", **scene_axis),
                   bgcolor="#111111", aspectmode="manual",
                   aspectratio=dict(x=1, y=1, z=(Z_HI - Z_LO) / (2 * extent))),
        scene_camera=dict(eye=dict(x=1.4, y=-1.8, z=1.0), up=dict(x=0, y=0, z=1)))
    if zone:                                      # --sparse: one marker for the heat-map peak
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers+text", name=ZONE_LABEL,
                                   text=[], textposition="top center", hoverinfo="name",
                                   textfont=dict(color="#ffd24d"),
                                   marker=dict(size=22, color="#ffd24d", opacity=0.45,
                                               line=dict(color="#ffd24d", width=2))))
    return fig


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", metavar="SENSORS_JSON", help="use real sensors (default: mock data)")
    ap.add_argument("--sparse", action="store_true",
                    help="sparse-data mode: window-s 0.5, a detection needs >= 2 points from each "
                         "of >= 2 sensors (eval_detection min-sensors gate), plus a PRESENCE "
                         "badge and the presence.py heat-map peak (overrides --window-s)")
    ap.add_argument("--mock-dense", action="store_true",
                    help="mock: the original dense mock (45 tight points per person) instead of "
                         "the sparse mock that matches measured live data")
    ap.add_argument("--only", nargs="+", help="live: run only these sensors, e.g. --only S1")
    ap.add_argument("--freerun", action="store_true", help="live: all sensors transmit continuously")
    ap.add_argument("--slot", type=float, default=3.0, help="live time-division slot, s")
    ap.add_argument("--learn", type=float, default=30.0, help="live: seconds of empty-room learning")
    ap.add_argument("--eps", type=float, default=0.4, help="DBSCAN neighbourhood radius, m")
    ap.add_argument("--vmove", type=float, default=0.1, help="|Doppler| above this counts as moving, m/s")
    ap.add_argument("--window-s", dest="window_s", type=float, default=0.0,
                    help="EXPERIMENTAL: pool foreground points over this many seconds before "
                         "clustering; gave false tracks in an empty-room test "
                         "(default: %(default)s = no pooling, the previous behaviour)")
    ap.add_argument("--extent", type=float, default=4.0, help="half-width of the plotted area, m")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--model", metavar="MODEL_JSON", help="XGBoost model file")
    ap.add_argument("--no-classify", action="store_true", help="keep all detections")
    ap.add_argument("--record", action="store_true", help="record raw frames to a session")
    ap.add_argument("--room", default="", help="recording room label")
    ap.add_argument("--scenario", default="", help="recording scenario label")
    ap.add_argument("--people", default="", help="recording people label")
    ap.add_argument("--notes", default="", help="recording notes")
    args = ap.parse_args()

    recorder = None
    source = None
    try:
        if args.record and args.live:
            from recorder import SessionRecorder
            from radar_core import load_sensors
            _, recording_sensors = load_sensors(args.live)
            if args.only:
                recording_sensors = [s for s in recording_sensors if s["name"] in args.only]
            cfg_path = recording_sensors[0]["cfg"] if recording_sensors else ""
            recorder = SessionRecorder(
                room=args.room, scenario=args.scenario, people=args.people, notes=args.notes,
                cfg_path=cfg_path, sensors_json_path=args.live, sensors=recording_sensors)
        source = (LiveSource(args.live, args.only, args.freerun, args.slot, recorder)
                  if args.live else MockSource(dense=args.mock_dense))
    except Exception:
        if recorder is not None:
            recorder.close()
        raise
    engine = Engine(source, args.learn, args.eps, args.vmove,
                    classify=not args.no_classify, model_path=args.model,
                    window_s=0.5 if args.sparse else args.window_s, sparse=args.sparse)
    engine.start()
    ui = {"mode": "free", "ids": ()}
    E = args.extent

    base_fig = build_base_figure(source.sensors, E, zone=args.sparse)
    i_zone = len(base_fig.data) - 1 if args.sparse else None     # zone marker is the last trace
    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#0d0d0d", "height": "100vh", "fontFamily": "monospace"},
        children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "24px",
                            "padding": "8px 20px", "backgroundColor": "#1a1a1a",
                            "borderBottom": "1px solid #333", "color": "#ddd", "fontSize": "13px"},
                     children=[
                html.B(f"Radar array - {source.label}", style={"color": "#fff"}),
                html.Div(id="stats-bar", style={"flex": "1", "color": "#aaa"}),
                html.Span("Spectate:"),
                dcc.Dropdown(id="follow", clearable=False, value="free", style={"width": "170px", "color": "#000"},
                             options=[{"label": "Free camera", "value": "free"}]),
                html.Span("Window (m):"),
                html.Div(dcc.Slider(id="zoom", min=1.5, max=5, step=0.5, value=2.5,
                                    marks=None, tooltip={"placement": "bottom"}), style={"width": "140px"}),
                html.Button("Re-learn background", id="relearn", n_clicks=0,
                            style={"display": "block" if source.live else "none"}),
                dcc.Input(id="event-text", type="text", placeholder="Event label",
                          style={"width": "180px"}),
                html.Button("Mark event", id="mark-event", n_clicks=0),
            ]),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "18px",
                            "padding": "6px 20px", "backgroundColor": "#141414",
                            "borderBottom": "1px solid #333", "color": "#ddd", "fontSize": "13px"},
                     children=[
                html.Span(id="rec-badge"),
                *([html.Span(id="presence-badge")] if args.sparse else []),
                html.Button("GO", id="mark-go", n_clicks=0, disabled=recorder is None,
                            style={"fontSize": "20px", "fontWeight": "bold", "padding": "6px 26px",
                                   "backgroundColor": "#1f7a1f", "color": "#fff", "border": "none"}),
                html.Button("DONE", id="mark-done", n_clicks=0, disabled=recorder is None,
                            style={"fontSize": "20px", "fontWeight": "bold", "padding": "6px 26px",
                                   "backgroundColor": "#8a1c1c", "color": "#fff", "border": "none"}),
                html.Div(id="last-marks", style={"color": "#aaa", "maxWidth": "420px"}),
                html.Div(id="health-row", style={"flex": "1", "display": "flex", "gap": "14px"}),
                html.Span("Show last (s):"),
                html.Div(dcc.Slider(id="show-s", min=0.1, max=SHOW_MAX_S, step=0.1, value=0.1,
                                    marks=None, tooltip={"placement": "bottom"}), style={"width": "140px"}),
            ]),
            dcc.Graph(id="radar-plot", figure=base_fig,
                      style={"height": "86vh"}, config={"displayModeBar": True}),
            dcc.Interval(id="tick", interval=150),
            html.Div(id="dummy", style={"display": "none"}),
            html.Div(id="mark-ack", style={"display": "none"}),
        ])

    @app.callback(Output("dummy", "children"), Input("relearn", "n_clicks"), prevent_initial_call=True)
    def _relearn(_):
        engine.relearn()
        return ""

    @app.callback(Output("event-text", "value"), Input("mark-event", "n_clicks"),
                  State("event-text", "value"), prevent_initial_call=True)
    def _mark_event(_, text):
        if recorder is not None and text:
            recorder.mark(text)
        return ""

    @app.callback(Output("mark-ack", "children"), Input("mark-go", "n_clicks"),
                  Input("mark-done", "n_clicks"), prevent_initial_call=True)
    def _mark_go_done(_go, _done):
        if recorder is not None and ctx.triggered_id in ("mark-go", "mark-done"):
            recorder.mark("go" if ctx.triggered_id == "mark-go" else "done")
        return ""

    def last_marks(k=5):
        """Last k events as saved by the recorder in events.csv (not the button clicks)."""
        path = recorder.path / "events.csv"
        try:
            with open(path, newline="") as f:
                rows = [r for r in csv.reader(f) if len(r) >= 2]
        except OSError:
            return []
        out = []
        for epoch, label in rows[-k:]:
            try:
                out.append(f"{time.strftime('%H:%M:%S', time.localtime(float(epoch)))} {label}")
            except ValueError:
                continue
        return out

    @app.callback(Output("rec-badge", "children"), Output("rec-badge", "style"),
                  Output("last-marks", "children"), Output("health-row", "children"),
                  Input("tick", "n_intervals"), prevent_initial_call=True)
    def update_status(_n):
        now = time.time()
        if recorder is not None:
            el = int(now - recorder.start_epoch)
            badge = f"\u25cf REC {el // 60:02d}:{el % 60:02d}  {recorder.path.name}"
            badge_style = {"color": "#fff", "backgroundColor": "#c00", "padding": "4px 10px",
                           "fontWeight": "bold", "borderRadius": "4px"}
            marks = last_marks()
            marks_txt = "last marks: " + (" | ".join(marks) if marks else "none yet")
        else:
            badge = "not recording"
            badge_style = {"color": "#888", "border": "1px solid #555", "padding": "4px 10px",
                           "borderRadius": "4px"}
            marks_txt = "last marks: (not recording)"
        if engine.health is None:
            health = [html.Span("health: mock", style={"color": "#888"})]
        else:
            health = []
            rows = engine.health.snapshot(now, source.sched.active, source.continuous)
            for sensor, (fps, ppf, age, state) in zip(source.sensors, rows):
                age_txt = "never" if age is None else f"{age:.1f}s ago"
                ppf_txt = "-" if not np.isfinite(ppf) else f"{ppf:.1f}"
                color = {"ok": "#7c7", "stale": "#fff", "idle": "#888"}[state]
                bg = "#a00" if state == "stale" else "transparent"
                health.append(html.Span(
                    f"{sensor['name']}: {fps:.1f} fps | {ppf_txt} pts/fr | {age_txt}"
                    + (" (idle slot)" if state == "idle" else ""),
                    style={"color": color, "backgroundColor": bg, "padding": "2px 6px"}))
        return badge, badge_style, marks_txt, health

    if args.sparse:
        @app.callback(Output("presence-badge", "children"), Output("presence-badge", "style"),
                      Input("tick", "n_intervals"), prevent_initial_call=True)
        def update_presence(_n):
            with engine.lock:
                s = engine.state
            style = {"fontWeight": "bold", "padding": "4px 10px", "borderRadius": "4px"}
            if s["learning"]:
                return "PRESENCE: learning", {**style, "color": "#888", "border": "1px solid #555"}
            if s.get("zone") is not None:
                return "PRESENCE: YES", {**style, "color": "#000", "backgroundColor": "#ffd24d"}
            return "PRESENCE: NO", {**style, "color": "#aaa", "border": "1px solid #555"}

    @app.callback(Output("radar-plot", "figure"), Output("stats-bar", "children"),
                  Output("follow", "options"),
                  Input("tick", "n_intervals"), State("follow", "value"), State("zoom", "value"),
                  State("show-s", "value"), prevent_initial_call=True)
    def update(n, follow, zoom, show_s):
        with engine.lock:
            s = engine.state
        xyz, sid, tracks = s["xyz"], s["sid"], s["tracks"]
        patch = Patch()

        def put(i, x, y, z):
            patch["data"][i]["x"], patch["data"][i]["y"], patch["data"][i]["z"] = x, y, z

        # live points colored by the sensor that saw them, drawn for the last show_s seconds
        # (drawing only: clustering above uses this tick's points as before)
        hist = s.get("history") or [(0.0, xyz, sid)]
        cutoff = hist[-1][0] - float(show_s or 0.1)
        shown = [h for h in hist if h[0] > cutoff] or hist[-1:]
        lxyz = np.vstack([h[1] for h in shown])
        lsid = np.concatenate([h[2] for h in shown])
        put(I_LIVE, lxyz[:, 0].tolist(), lxyz[:, 1].tolist(), lxyz[:, 2].tolist())
        patch["data"][I_LIVE]["marker"]["color"] = [SENSOR_HEX[i % len(SENSOR_HEX)] for i in lsid.tolist()]

        # clustered points colored by the person (track) they belong to, plus bounding boxes
        cx, cc, boxes = [], [], []
        for di, d in enumerate(s["dets"]):
            tid = s["assign"].get(di)
            if tid is None:
                continue
            cx.append(xyz[d["idx"]])
            cc += [TRACK_COLORS[tid % len(TRACK_COLORS)]] * len(d["idx"])
            boxes += box_chunks(d["lo"], d["hi"])
        cp = np.vstack(cx) if cx else np.empty((0, 3))
        put(I_CLUS, cp[:, 0].tolist(), cp[:, 1].tolist(), cp[:, 2].tolist())
        patch["data"][I_CLUS]["marker"]["color"] = cc
        put(I_BOX, *segs(boxes))

        # labels and trails
        put(I_LABEL, [t["pos"][0] for t in tracks], [t["pos"][1] for t in tracks],
            [t["pos"][2] + 0.5 for t in tracks])
        patch["data"][I_LABEL]["text"] = [f"#{t['id']}" for t in tracks]
        put(I_TRAIL, *segs([t["trail"] for t in tracks if len(t["trail"]) > 1]))
        if args.sparse:
            zone = s.get("zone")
            put(i_zone, *(([zone[0]], [zone[1]], [1.0]) if zone is not None else ([], [], [])))
            patch["data"][i_zone]["text"] = [ZONE_LABEL] if zone is not None else []

        if n % 30 == 0:                                       # static map: refresh every ~4.5 s
            m = source.background_points() if not source.live else engine.bg.points()
            put(I_MAP, m[:, 0].tolist(), m[:, 1].tolist(), m[:, 2].tolist())

        # spectate: slide the plotted window so the followed person stays centred
        target = next((t for t in tracks if str(t["id"]) == str(follow)), None)
        ax = patch["layout"]["scene"]
        if follow != "free" and target is not None:
            w, (px, py, _) = float(zoom), target["pos"]
            ax["xaxis"]["range"], ax["yaxis"]["range"] = [px - w, px + w], [py - w, py + w]
            ax["aspectratio"] = dict(x=1, y=1, z=(Z_HI - Z_LO) / (2 * w))
            ui["mode"] = "follow"
        elif follow == "free" and ui["mode"] != "free":
            ax["xaxis"]["range"], ax["yaxis"]["range"] = [-E, E], [-E, E]
            ax["aspectratio"] = dict(x=1, y=1, z=(Z_HI - Z_LO) / (2 * E))
            ui["mode"] = "free"

        ids = tuple(t["id"] for t in tracks)
        if ids != ui["ids"]:
            ui["ids"] = ids
            opts = [{"label": "Free camera", "value": "free"}] + \
                   [{"label": f"Person #{i}", "value": str(i)} for i in ids]
        else:
            opts = no_update

        status = (f"LEARNING empty room, {s['left']:.0f}s left - stay out of view" if s["learning"]
                  else "tracking")
        txt = (f"points {len(xyz)} | foreground {s['fg']} | people {len(tracks)} | "
               f"map voxels {len(engine.bg.occupied) if engine.bg.frozen else len(engine.bg.counts)} | {status}")
        return patch, txt, opts

    print(f"\n  Viewer running ({source.label}). Open http://<this-machine's-address>:{args.port}\n")
    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # hide per-request log lines
    try:
        app.run(host="0.0.0.0", port=args.port, debug=False)
    finally:
        engine.halt.set()
        if source.live:
            source.close()
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
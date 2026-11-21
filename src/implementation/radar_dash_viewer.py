"""
radar_dash_viewer.py - control-room web app (Dash/Plotly) for the radar array: a top-down
floor plan and a 3D view (sensor positions and fields of view, static room map, live points,
tracked people with trails, or in --sparse mode the presence motion zone), a PRESENCE card,
per-sensor health cards with sparklines, a 60 s presence timeline, and recording marks.
Styles are in assets/style.css (loaded automatically by Dash).

Open http://127.0.0.1:8050 on the same machine. It listens only on 127.0.0.1 unless --lan is
given; --lan listens on all interfaces, so other devices (e.g. a laptop viewing the Raspberry
Pi) can open it, and so can anyone else on the network: the controls have no authentication.
How to use the page: docs/dashboard.md.

    python radar_dash_viewer.py                                   # mock data, no hardware
    python radar_dash_viewer.py --live sensors.json --freerun --lan      # reachable on the LAN
    python radar_dash_viewer.py --mock-dense                      # old dense mock data
    python radar_dash_viewer.py --live sensors.json --freerun --sparse   # sparse-data mode
    python radar_dash_viewer.py --live sensors.json --only S1     # one real sensor at 10 Hz
    python radar_dash_viewer.py --live sensors.json               # 3 sensors, time-division
    python radar_dash_viewer.py --live sensors.json --freerun     # 3 sensors, all at once

Live mode first LEARNS the empty room for --learn seconds (stay out of the sensors' view),
then freezes it as background. Anything moving, or not part of the background, is clustered
and tracked. The "Re-learn background" button repeats this.

Spectate: pick a person in the dropdown. The 3D window re-centres on them each update, and
you can still rotate the view with the mouse. In --sparse mode track IDs are hidden (they churn
on sparse data) and only the motion zone is drawn.
"""
import argparse
import csv
import json
import logging
import signal
import threading
from collections import deque
from pathlib import Path
import time

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update

from radar_core import apply_T, filter_points, fov_lines, grid_lines, pose_to_T
from features import extract_features, to_vector
from rule_baseline import is_human_rule
from tracking import Background, Tracker, cluster, detections

Z_LO, Z_HI = -0.05, 3.0

# figures have fixed trace slots (see build_*_figure); callbacks only patch them
SHOW_MAX_S = 2.0            # longest "Show last (s)" setting; live points kept this long for drawing
HEALTH_WINDOW_S = 5.0       # health row: fps / points per frame over this many seconds
HEALTH_STALE_S = 2.0        # health row: red after this long without a frame
TIMELINE_S = 60             # presence strip and sparklines: seconds shown
ZONES_K, ZONES_MIN_SEP = 3, 1.5     # --sparse: up to 3 motion zones, at least 1.5 m apart


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

    def __init__(self, dense=False, n_people=2):
        self.sensors = default_sensors()
        self.dense = dense
        self.n_people = n_people                  # sparse model only; tests use 0, 1 or 2
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
        now = np.array(self.people(t)[:self.n_people]).reshape(-1, 2)
        vxy = ((np.array(self.people(t + 0.05)[:self.n_people]) -
                np.array(self.people(t - 0.05)[:self.n_people])) / 0.1).reshape(-1, 2)
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
        sid = r.integers(0, 3, len(xyz))
        frames = [(i, t, int(np.sum(sid == i))) for i in range(len(self.sensors))]   # health cards
        return {"xyz": xyz, "vel": vel, "snr": snr, "sid": sid, "frames": frames}


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
        self._rec_lock = threading.Lock()
        self.recorder = recorder
        self.recorded_frames = 0                  # frames handed to the current recorder
        self.sched.start()
        self.label = ("LIVE " + ("freerun" if freerun else "time-division") if len(sensors) > 1
                      else f"LIVE {sensors[0]['name']}")

    def next_frame(self):
        xyz, vel, snr, sid = ([np.empty((0, 3))], [np.empty(0)], [np.empty(0)],
                              [np.empty(0, int)])
        frames = []                                           # (sensor index, arrival t, points)
        for i, link in enumerate(self.links):
            for fr in link.drain():
                with self._rec_lock:
                    if self.recorder is not None:
                        self.recorder.add(i, fr)
                        self.recorded_frames += 1
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

    def set_recorder(self, recorder):
        """Hand a SessionRecorder to the acquisition loop, or detach it (None). Safe to call
        from the UI thread: frames are passed on under the same lock, so after this returns
        no further frame reaches the old recorder."""
        with self._rec_lock:
            self.recorder = recorder
            self.recorded_frames = 0

    def close(self):
        self.sched.halt.set()
        self.sched.join(3.0)
        for l in self.links:
            l.close()


# ---------------------------------------------------------------------------
# Recording from the UI: SessionRecorder lifecycle, activity (go/done) and marks
# ---------------------------------------------------------------------------
SCENARIOS = ["walk", "two_people", "empty", "fan", "chair", "stand", "sit", "pose_check", "other"]
STOP_CONFIRM_S = 5.0            # Stop must be clicked twice within this many seconds
ACTIVITY_DEBOUNCE_S = 1.0       # a second activity click within this time is ignored
MARKS_SHOWN = 8


def rig_geometry(sensors):
    """One line describing the rig from the sensor poses: triangle sides and sensor heights."""
    pos = [(s["name"], np.asarray(s["T"], float)[:3, 3]) for s in sensors]
    sides = [f"{a}-{b} {float(np.linalg.norm(pa[:2] - pb[:2])):.3f} m"
             for k, (a, pa) in enumerate(pos) for b, pb in pos[k + 1:]]
    heights = [f"{n} {p[2]:.2f} m" for n, p in pos]
    return f"rig: sides {', '.join(sides)}; sensor heights {', '.join(heights)}"


def read_marks(path, k=MARKS_SHOWN):
    """Last k (epoch, label) rows of a session's events.csv; [] if none yet."""
    if path is None:
        return []
    try:
        with open(Path(path) / "events.csv", newline="") as f:
            rows = [r for r in csv.reader(f) if len(r) >= 2]
    except OSError:
        return []
    out = []
    for epoch, label in rows[-k:]:
        try:
            out.append((float(epoch), label))
        except ValueError:
            continue
    return out


class RecordingController:
    """Starts and stops SessionRecorder instances at runtime and owns the activity state.

    Activity is a two-state machine, idle -> active (logs "go") -> idle (logs "done"); a
    click within debounce_s of the previous one is ignored, so "done" cannot be logged
    twice by a double click, and the event box refuses "go" and "done". Stopping a
    recording while an activity runs logs "done" first, so every go has its done.
    Only a live source (with set_recorder) can record; mock data is never recorded.
    """

    def __init__(self, source, sensors_json=None, root="data", debounce_s=ACTIVITY_DEBOUNCE_S):
        self.source, self.sensors_json, self.root = source, sensors_json, root
        self.debounce_s = debounce_s
        self.can_record = bool(getattr(source, "live", False)) and hasattr(source, "set_recorder")
        self.lock = threading.RLock()
        self.recorder = None
        self.last_path = None                     # current or most recent session folder
        self.activity_since = None                # epoch of the open "go", or None
        self.last_toggle = -np.inf
        self.stop_armed_until = 0.0

    def start(self, scenario="", people="", room="", notes=""):
        """Create a SessionRecorder and hand it to the source. Returns (recorder, message)."""
        with self.lock:
            if not self.can_record:
                return None, "Recording needs live sensors (--live)."
            if self.recorder is not None:
                return self.recorder, f"Already recording to {self.recorder.path.name}."
            from recorder import SessionRecorder
            sensors = [{"name": s["name"], "cli": s.get("cli", ""), "data": s.get("data", ""),
                        "cfg": s.get("cfg", ""), "T": np.asarray(s["T"], float).tolist()}
                       for s in self.source.sensors]
            geometry = rig_geometry(self.source.sensors)
            notes = f"{notes.strip()} | {geometry}" if (notes or "").strip() else geometry
            rec = SessionRecorder(root=self.root, room=room or "", scenario=scenario or "",
                                  people=people or "", notes=notes,
                                  cfg_path=sensors[0]["cfg"] if sensors else "",
                                  sensors_json_path=self.sensors_json or "", sensors=sensors)
            self.recorder, self.last_path = rec, rec.path
            self.activity_since, self.last_toggle, self.stop_armed_until = None, -np.inf, 0.0
            self.source.set_recorder(rec)
            return rec, f"Recording to {rec.path.name}."

    def stop(self):
        """Detach and close the recorder (ends an open activity first). Returns (recorder, msg)."""
        with self.lock:
            rec = self.recorder
            if rec is None:
                return None, "Not recording."
            if self.activity_since is not None:
                rec.mark("done")
                self.activity_since = None
            self.source.set_recorder(None)        # no frame reaches rec after this
            self.recorder = None
            self.stop_armed_until = 0.0
            rec.close()
            return rec, f"Stopped. Saved {rec.path.name}."

    def request_stop(self, now):
        """Two-step stop: the first click arms, a second within STOP_CONFIRM_S stops."""
        with self.lock:
            if self.recorder is None:
                return "Not recording."
            if now < self.stop_armed_until:
                return self.stop()[1]
            self.stop_armed_until = now + STOP_CONFIRM_S
            return f"Click Stop again within {STOP_CONFIRM_S:.0f} s to stop recording."

    def toggle_activity(self, now):
        with self.lock:
            if self.recorder is None:
                return "Start a recording first."
            if now - self.last_toggle < self.debounce_s:
                return "Ignored a second click."
            self.last_toggle = now
            if self.activity_since is None:
                self.recorder.mark("go")
                self.activity_since = now
                return "Activity started (go)."
            self.recorder.mark("done")
            self.activity_since = None
            return "Activity ended (done)."

    def mark(self, label):
        label = (label or "").strip()
        if not label:
            return ""
        if label.lower() in ("go", "done"):
            return "Use the activity button for go and done."
        with self.lock:
            if self.recorder is None:
                return "Not recording."
            self.recorder.mark(label)
            return f"Marked \"{label}\"."

    def log_settings(self, label):
        """Mark a settings change while recording; returns True if it was written."""
        with self.lock:
            if self.recorder is None or not label:
                return False
            self.recorder.mark(label)
            return True

    def status(self, now):
        with self.lock:
            rec = self.recorder
            st = {"recording": rec is not None, "path": self.last_path,
                  "activity_s": None if self.activity_since is None else now - self.activity_since,
                  "armed": rec is not None and now < self.stop_armed_until}
            if rec is not None:
                st["elapsed"] = now - rec.start_epoch
                st["frames"] = getattr(self.source, "recorded_frames", 0)
        if st["path"] is not None:
            st["chunks"] = len(list(Path(st["path"]).glob("chunk_*.npz")))
        st["marks"] = read_marks(st["path"])
        return st


# ---------------------------------------------------------------------------
# Live settings: one lock-protected object, read by Engine at the start of every tick
# ---------------------------------------------------------------------------
class EngineConfig:
    """Detection settings the UI can change while running. Engine takes a snapshot each tick,
    so a change takes effect on the next tick. Values are validated and clamped here.

    mode "sparse" = what --sparse sets up: the presence heat map and motion zones, and by
    default window_s 0.5 and min_sensors 2. min_sensors >= 2 applies eval_detection's gate
    (>= 2 points from each of >= min_sensors sensors) in either mode; 1 = no gate.
    show_s is display only (how many seconds of points are drawn).
    """
    FIELDS = ("mode", "window_s", "min_sensors", "eps", "vmove", "classify", "show_s")
    PROCESSING = ("mode", "window_s", "min_sensors", "eps", "vmove", "classify")
    LIMITS = {"window_s": (0.0, 2.0), "min_sensors": (1, 3), "eps": (0.1, 2.0),
              "vmove": (0.0, 2.0), "show_s": (0.1, SHOW_MAX_S)}

    def __init__(self, mode="classic", window_s=0.0, min_sensors=1, eps=0.4, vmove=0.1,
                 classify=True, show_s=0.1, classic_window_s=None):
        self._lock = threading.Lock()
        self.classic_window_s = window_s if classic_window_s is None else classic_window_s
        self._v = self.clean(dict(mode=mode, window_s=window_s, min_sensors=min_sensors, eps=eps,
                                  vmove=vmove, classify=classify, show_s=show_s))
        self.defaults = dict(self._v)

    @classmethod
    def from_args(cls, args):
        sparse = bool(getattr(args, "sparse", False))
        return cls(mode="sparse" if sparse else "classic",
                   window_s=0.5 if sparse else args.window_s, min_sensors=2 if sparse else 1,
                   eps=args.eps, vmove=args.vmove, classify=not args.no_classify,
                   classic_window_s=args.window_s)

    def clean(self, values):
        out = {}
        for k, v in values.items():
            if k not in self.FIELDS or v is None:
                continue
            if k == "mode":
                if v in ("classic", "sparse"):
                    out[k] = v
                continue
            if k == "classify":
                out[k] = bool(v)
                continue
            try:
                v = int(v) if k == "min_sensors" else float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            lo, hi = self.LIMITS[k]
            out[k] = min(max(v, lo), hi)
        return out

    def mode_defaults(self, mode):
        """window_s and min_sensors that --sparse (or classic) would use."""
        return ({"window_s": 0.5, "min_sensors": 2} if mode == "sparse"
                else {"window_s": self.classic_window_s, "min_sensors": 1})

    def snapshot(self):
        with self._lock:
            return dict(self._v)

    def update(self, **values):
        """Apply validated values; returns {field: new value} for the fields that changed."""
        new = self.clean(values)
        with self._lock:
            changed = {k: v for k, v in new.items() if self._v.get(k) != v}
            self._v.update(changed)
        return changed

    def reset(self):
        return self.update(**self.defaults)


def settings_label(changes):
    """'settings window_s=0.5 min_sensors=2' for the processing settings that changed."""
    parts = []
    for k in EngineConfig.PROCESSING:
        if k in changes:
            v = changes[k]
            parts.append(f"classifier={'on' if v else 'off'}" if k == "classify" else
                         f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}")
    return "settings " + " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Processing thread (acquire -> foreground -> cluster -> track), ~10 Hz
# ---------------------------------------------------------------------------
class Engine(threading.Thread):
    def __init__(self, source, learn_s, eps, v_move, classify=True, model_path=None, window_s=0.5,
                 sparse=False, config=None):
        """Settings come from config (an EngineConfig, read every tick); without one they are
        built from the arguments, so Engine(...) with arguments behaves as before."""
        super().__init__(daemon=True)
        self.src, self.learn_s = source, learn_s
        self.config = config or EngineConfig(
            mode="sparse" if sparse else "classic", window_s=window_s,
            min_sensors=2 if sparse else 1, eps=eps, vmove=v_move, classify=classify)
        self.heat, self._gate = None, None       # heat map (sparse mode) and gate, made on first use
        self.pool = deque()                       # (t, xyz, vel, snr, sid) of recent foreground points
        self.history = deque()                    # (t, xyz, sid, moving) of recent live points, drawing only
        self.timeline = deque()                   # (t, moving points per sensor, presence or None), UI only
        self.health = SensorHealth(len(source.sensors))   # display only; mock sources report frames too
        self.model = None
        if model_path:                            # loaded even if the classifier starts off
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

    def passes_sensor_gate(self, d, sid, min_sensors):
        if self._gate is None:                    # eval_detection's gate, reused, not reimplemented
            from eval_detection import passes_sensor_gate
            self._gate = passes_sensor_gate
        return self._gate(d, sid, min_sensors)

    def learn_progress(self):
        """0..1 while learning the background, 1.0 when frozen."""
        if self.bg.frozen:
            return 1.0
        return min(max((time.time() - self.t_learn) / max(self.learn_s, 1e-9), 0.0), 1.0)

    def relearn(self):
        if self.src.live:
            self.bg.reset()
            self.tracker = Tracker()
            self.pool.clear()
            self.heat = None
            self.t_learn = time.time()

    def run(self):
        while not self.halt.is_set():
            t0 = time.time()
            c = self.config.snapshot()            # live settings, fixed for this tick
            sparse = c["mode"] == "sparse"
            fr = self.src.next_frame()
            xyz, vel, snr, sid = fr["xyz"], fr["vel"], fr["snr"], fr["sid"]
            if self.health is not None:
                self.health.add(fr.get("frames", []))
            self.history.append((t0, xyz, sid, np.abs(vel) >= c["vmove"]))
            while self.history and self.history[0][0] < t0 - SHOW_MAX_S:
                self.history.popleft()
            learning = not self.bg.frozen
            dets, assign, tracks, n_fg, zone = [], {}, [], 0, None
            zones, gated = [], []                 # --sparse display: motion zones, gate boxes
            if learning:
                self.pool.clear()
                self.bg.learn(xyz[np.abs(vel) < 0.05])
                if t0 - self.t_learn > self.learn_s:
                    self.bg.freeze()
            else:
                moving = np.abs(vel) >= c["vmove"]
                is_bg = self.bg.is_background(xyz)
                fg = moving | ~is_bg
                idx = np.flatnonzero(fg)
                n_fg = len(idx)
                if sparse:                        # presence heat map: one peak or None
                    if self.heat is None:
                        self.heat = self.new_heat_map()
                    zone = self.heat.update(t0, xyz, sid, moving, ~moving & ~is_bg)
                    zones = self.heat.peaks(k=ZONES_K, min_sep=ZONES_MIN_SEP)   # display only
                else:
                    self.heat = None              # a later switch to sparse starts a fresh map
                # Pool foreground points over the last window_s seconds (as build_dataset.py
                # does) and cluster the pool. Older points come first; this tick's are last.
                while self.pool and self.pool[0][0] <= t0 - c["window_s"]:
                    self.pool.popleft()
                p_xyz = np.vstack([p[1] for p in self.pool] + [xyz[idx]])
                p_vel = np.concatenate([p[2] for p in self.pool] + [vel[idx]])
                p_snr = np.concatenate([p[3] for p in self.pool] + [snr[idx]])
                p_sid = np.concatenate([p[4] for p in self.pool] + [sid[idx]])
                n_old = len(p_xyz) - len(idx)
                self.pool.append((t0, xyz[idx], vel[idx], snr[idx], sid[idx]))
                lab = cluster(p_xyz, c["eps"])
                dets = detections(p_xyz, lab)
                if c["min_sensors"] >= 2:         # eval_detection's min-sensors gate
                    dets = [d for d in dets
                            if self.passes_sensor_gate(d, p_sid, c["min_sensors"])]
                    for d in dets:                # display only: box, points, sensors with >= 2 pts
                        _, cnt = np.unique(p_sid[d["idx"]], return_counts=True)
                        gated.append((d["lo"].copy(), d["hi"].copy(), d["n"], int(np.sum(cnt >= 2))))
                for d in dets:
                    d["pool_idx"] = d["idx"]
                    cur = d["idx"][d["idx"] >= n_old] - n_old
                    d["idx"] = idx[cur]               # this tick's points, for rendering
                if c["classify"]:
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
            # display-only timeline for the sparklines and the presence strip
            mv = np.abs(vel) >= c["vmove"]
            per_sensor = [int(np.sum(mv & (sid == i))) for i in range(len(self.src.sensors))]
            present = None if learning else (zone is not None if sparse else len(snap) > 0)
            self.timeline.append((t0, per_sensor, present))
            while self.timeline and self.timeline[0][0] < t0 - TIMELINE_S - 2.0:
                self.timeline.popleft()
            with self.lock:
                self.state = {"xyz": xyz, "sid": sid, "fg": n_fg, "dets": dets, "assign": assign,
                              "tracks": snap, "learning": learning, "history": list(self.history),
                              "zone": zone, "zones": zones, "gated": gated, "mode": c["mode"],
                              "left": max(0.0, self.learn_s - (t0 - self.t_learn))}
            self.halt.wait(max(0.0, 0.1 - (time.time() - t0)))


# ---------------------------------------------------------------------------
# Look: palette, fonts, sensor colours
# ---------------------------------------------------------------------------
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
COL = {"bg": "#0B0F1A", "surface": "#121826", "border": "#26304A", "text": "#E6EAF2",
       "muted": "#8B95AD", "accent": "#38BDF8", "ok": "#34D399", "warn": "#FBBF24",
       "danger": "#F43F5E"}
SENSOR_COLORS = ["#E69F00", "#56B4E9", "#CC79A7"]          # S1, S2, S3 (colour-blind safe)
TRACK_COLORS = ["#38BDF8", "#FBBF24", "#A78BFA", "#34D399", "#F472B6", "#F43F5E"]
ZONE_R = 0.5                    # zones drawn as 1 m circles (their resolution)
ZONE_LABEL_Z = 2.1              # 3D: zone labels float above the sensor labels (sensors at ~1.4 m)
ZONES_NAME = "motion zones, about 1 m"
ZONES_MERGE_NOTE = ("Zones merge when sources are less than about 2.5 m apart; "
                    "the count is a lower bound.")
FOV_R, FOV_HALF_AZ = 4.0, 60.0  # drawn field of view, as radar_core.fov_lines
CAMERAS = {"iso": dict(eye=dict(x=1.4, y=-1.8, z=1.0), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
           "top": dict(eye=dict(x=0, y=0, z=2.6), up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0))}


def sensor_color(i):
    return SENSOR_COLORS[i % len(SENSOR_COLORS)]


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


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


def fov_wedge(T, r=FOV_R, half_az=FOV_HALF_AZ):
    """Closed xy polygon of a sensor's horizontal field of view."""
    p = T[:3, 3]
    fwd = T[:3, 1]                                 # sensor +y (boresight) in the world
    a0 = np.arctan2(fwd[1], fwd[0])
    a = a0 + np.radians(np.linspace(-half_az, half_az, 25))
    xs = [p[0]] + (p[0] + r * np.cos(a)).tolist() + [p[0]]
    ys = [p[1]] + (p[1] + r * np.sin(a)).tolist() + [p[1]]
    return xs, ys


def circle(c, r, n=40):
    a = np.linspace(0, 2 * np.pi, n)
    return (c[0] + r * np.cos(a)).tolist(), (c[1] + r * np.sin(a)).tolist()


def zone_shapes(zones, r=ZONE_R, z=0.02, n=32):
    """Motion zones -> floor-plan circles (x, y with None breaks), 3D disks (mesh) and rings."""
    fx, fy = [], []
    mx, my, mz, mi, mj, mk = [], [], [], [], [], []
    rx, ry, rz = [], [], []
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for x, y, _ in zones:
        cx, cy = circle((x, y), r)
        fx += cx + [None]
        fy += cy + [None]
        base = len(mx)
        mx += [x] + (x + r * np.cos(a)).tolist()
        my += [y] + (y + r * np.sin(a)).tolist()
        mz += [z] * (n + 1)
        for q in range(n):
            mi.append(base)
            mj.append(base + 1 + q)
            mk.append(base + 1 + (q + 1) % n)
        rx += cx + [None]
        ry += cy + [None]
        rz += [z] * len(cx) + [None]
    return (fx, fy), (mx, my, mz, mi, mj, mk), (rx, ry, rz)


def gate_boxes(gated):
    """Gate detections -> floor-plan rectangles (x, y with None breaks) and labels."""
    bx, by, lx, ly, lt = [], [], [], [], []
    for lo, hi, n_pts, n_sens in gated:
        bx += [lo[0], hi[0], hi[0], lo[0], lo[0], None]
        by += [lo[1], lo[1], hi[1], hi[1], lo[1], None]
        lx.append(lo[0])
        ly.append(lo[1])
        lt.append(f"{n_pts} pts, {n_sens} sensors")
    return bx, by, lx, ly, lt


def build_base_figure(sensors, extent, zone=False):
    """3D view. Returns (figure, {slot name: trace index})."""
    gp, gl = grid_lines(half=extent, step=1.0)
    grid = segs([gp[l] for l in gl])
    fov = []
    for s in sensors:
        p, l = fov_lines(s["T"], r=FOV_R)
        fov += [p[e] for e in l]
    fovx = segs(fov)
    pos = np.array([s["T"][:3, 3] for s in sensors])
    slots = {}
    traces = []

    def add(name, trace):
        slots[name] = len(traces)
        traces.append(trace)

    add("map", go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Room map", hoverinfo="skip",
                            marker=dict(size=1.6, color=COL["muted"], opacity=0.25)))
    add("static", go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Static points", hoverinfo="skip",
                               marker=dict(size=2, color=COL["muted"], opacity=0.35)))
    add("moving", go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Moving points", hoverinfo="skip",
                               marker=dict(size=4.5, color=COL["accent"], opacity=0.95)))
    add("clus", go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Tracked clusters",
                             marker=dict(size=5, color=COL["accent"])))
    add("box", go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip", showlegend=False,
                            line=dict(color=rgba(COL["text"], 0.55), width=2)))
    add("label", go.Scatter3d(x=[], y=[], z=[], mode="text", hoverinfo="skip", showlegend=False,
                              textfont=dict(color=COL["text"], size=13, family=FONT)))
    add("trail", go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip", showlegend=False,
                              line=dict(color=rgba(COL["accent"], 0.6), width=2)))
    add("grid", go.Scatter3d(x=grid[0], y=grid[1], z=grid[2], mode="lines", hoverinfo="skip",
                             showlegend=False, line=dict(color=rgba(COL["border"], 0.6), width=1)))
    add("fov", go.Scatter3d(x=fovx[0], y=fovx[1], z=fovx[2], mode="lines", hoverinfo="skip",
                            showlegend=False, line=dict(color=rgba(COL["muted"], 0.35), width=1.5)))
    add("sensors", go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text", name="Sensors",
                                text=[s["name"] for s in sensors], textposition="top center",
                                textfont=dict(color=COL["text"], family=FONT),
                                marker=dict(size=7, symbol="diamond",
                                            color=[sensor_color(i) for i in range(len(sensors))])))
    if zone:                                      # --sparse: one 1 m disk per motion zone
        add("zone_disk", go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], name=ZONES_NAME,
                                   color=COL["warn"], opacity=0.28, hoverinfo="skip",
                                   flatshading=True, showlegend=False))
        add("zone_ring", go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip",
                                      showlegend=False, line=dict(color=COL["warn"], width=3)))
        add("zone_label", go.Scatter3d(x=[], y=[], z=[], mode="text", hoverinfo="skip",
                                       showlegend=False,
                                       textfont=dict(color=COL["warn"], size=13, family=FONT)))
    axis = dict(backgroundcolor="rgba(0,0,0,0)", showbackground=False, gridcolor=rgba(COL["border"], 0.7),
                zerolinecolor=COL["border"], color=COL["muted"], title_font=dict(size=11))
    fig = go.Figure(data=traces)
    fig.update_layout(
        uirevision="keep", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=COL["text"], size=12), margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", x=0, y=0, font=dict(color=COL["muted"], size=11),
                    bgcolor="rgba(0,0,0,0)"),
        scene=dict(xaxis=dict(range=[-extent, extent], title="x (m)", **axis),
                   yaxis=dict(range=[-extent, extent], title="y (m)", **axis),
                   zaxis=dict(range=[Z_LO, Z_HI], title="z (m)", **axis),
                   bgcolor="rgba(0,0,0,0)", aspectmode="manual",
                   aspectratio=dict(x=1, y=1, z=(Z_HI - Z_LO) / (2 * extent))),
        scene_camera=CAMERAS["iso"])
    return fig, slots


def build_floor_figure(sensors, extent, zone=False):
    """Top-down floor plan. Returns (figure, {slot name: trace index})."""
    slots, traces = {}, []

    def add(name, trace):
        slots[name] = len(traces)
        traces.append(trace)

    for i, s in enumerate(sensors):
        xs, ys = fov_wedge(s["T"])
        add(f"wedge{i}", go.Scatter(x=xs, y=ys, mode="lines", fill="toself", hoverinfo="skip",
                                    showlegend=False, fillcolor=rgba(sensor_color(i), 0.09),
                                    line=dict(color=rgba(sensor_color(i), 0.45), width=1)))
    add("map", go.Scatter(x=[], y=[], mode="markers", name="Room map", hoverinfo="skip",
                          marker=dict(size=3, color=COL["muted"], opacity=0.25)))
    add("static", go.Scatter(x=[], y=[], mode="markers", name="Static points", hoverinfo="skip",
                             marker=dict(size=4, color=COL["muted"], opacity=0.35)))
    add("moving", go.Scatter(x=[], y=[], mode="markers", name="Moving points", hoverinfo="skip",
                             marker=dict(size=8, color=COL["accent"], opacity=0.95,
                                         line=dict(width=0))))
    add("trail", go.Scatter(x=[], y=[], mode="lines", hoverinfo="skip", showlegend=False,
                            line=dict(color=rgba(COL["accent"], 0.55), width=2)))
    add("label", go.Scatter(x=[], y=[], mode="text", hoverinfo="skip", showlegend=False,
                            textfont=dict(color=COL["text"], size=12, family=FONT)))
    if zone:
        add("gbox", go.Scatter(x=[], y=[], mode="lines", hoverinfo="skip", showlegend=False,
                               line=dict(color=rgba(COL["text"], 0.55), width=1, dash="dot")))
        add("gbox_label", go.Scatter(x=[], y=[], mode="text", hoverinfo="skip", showlegend=False,
                                     textposition="bottom right",
                                     textfont=dict(color=COL["muted"], size=10.5, family=FONT)))
        add("zones", go.Scatter(x=[], y=[], mode="lines", fill="toself", name=ZONES_NAME,
                                hoverinfo="skip", fillcolor=rgba(COL["warn"], 0.22),
                                line=dict(color=COL["warn"], width=2)))
        add("zone_label", go.Scatter(x=[], y=[], mode="text", hoverinfo="skip", showlegend=False,
                                     textfont=dict(color=COL["warn"], size=12, family=FONT)))
    pos = np.array([s["T"][:3, 3] for s in sensors])
    add("sensors", go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", name="Sensors",
                              text=[s["name"] for s in sensors], textposition="bottom center",
                              textfont=dict(color=COL["text"], family=FONT),
                              marker=dict(size=13, symbol="diamond", line=dict(color=COL["bg"], width=1.5),
                                          color=[sensor_color(i) for i in range(len(sensors))])))
    axis = dict(range=[-extent, extent], gridcolor=rgba(COL["border"], 0.55), zerolinecolor=COL["border"],
                color=COL["muted"], dtick=1, ticksuffix=" m", showline=False)
    fig = go.Figure(data=traces)
    fig.update_layout(
        uirevision="keep", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=COL["text"], size=12), margin=dict(l=44, r=10, t=8, b=34),
        showlegend=False, xaxis=axis, yaxis=dict(axis, scaleanchor="x", scaleratio=1))
    return fig, slots


# ---------------------------------------------------------------------------
# Display-only summaries of Engine state (no effect on processing)
# ---------------------------------------------------------------------------
def timeline_bins(entries, now, n_sensors, seconds=TIMELINE_S):
    """Per 1 s bin, oldest first: moving points per second per sensor (n_sensors x seconds)
    and a presence state per second: 'yes', 'no', 'learn' or 'none' (no data)."""
    rates = np.zeros((n_sensors, seconds))
    flags = [[] for _ in range(seconds)]
    for t, per_sensor, present in entries:
        b = seconds - 1 - int((now - t) // 1.0)
        if 0 <= b < seconds:
            rates[:, b] += np.asarray(per_sensor[:n_sensors], float)
            flags[b].append(present)
    states = []
    for f in flags:
        if not f:
            states.append("none")
        elif sum(x is None for x in f) * 2 >= len(f):
            states.append("learn")
        else:
            states.append("yes" if sum(bool(x) for x in f) * 2 >= len(f) else "no")
    return rates, states


def health_rows(engine, source, now):
    """[(fps, points/frame, age or None, state)] per sensor, from Engine's display-only log."""
    if getattr(source, "live", False):
        return engine.health.snapshot(now, source.sched.active, source.continuous)
    return engine.health.snapshot(now, None, True)


APP_TITLE = "Radar Array"


def silent_sensor_alert(engine, source, rows, now):
    """Alert strip text for live sensors silent > HEALTH_STALE_S, with the time of their last
    frame; ("", hidden class) when all are fine or the source is a mock."""
    if not getattr(source, "live", False):
        return "", "alert-strip hidden"
    parts = []
    for s, (_fps, _ppf, age, state) in zip(source.sensors, rows):
        if state != "stale":
            continue
        if age is None:
            since = time.strftime("%H:%M:%S", time.localtime(engine.health.t_start))
            parts.append(f"{s['name']}: no frames at all since {since}")
        else:
            since = time.strftime("%H:%M:%S", time.localtime(now - age))
            parts.append(f"{s['name']}: silent for {age:.0f} s (last frame {since})")
    if not parts:
        return "", "alert-strip hidden"
    return "\u26a0 Sensor silent \u2014 " + " | ".join(parts), "alert-strip"


def sparkline(values, color):
    vmax = max(float(np.max(values)) if len(values) else 0.0, 1.0)
    return {"data": [{"type": "scatter", "x": list(range(len(values))), "y": [float(v) for v in values],
                      "mode": "lines", "line": {"color": color, "width": 1.5, "shape": "spline"},
                      "fill": "tozeroy", "fillcolor": rgba(color, 0.18), "hoverinfo": "skip"}],
            "layout": {"margin": {"l": 0, "r": 0, "t": 2, "b": 0}, "height": 34,
                       "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                       "xaxis": {"visible": False, "range": [0, max(len(values) - 1, 1)]},
                       "yaxis": {"visible": False, "range": [0, vmax * 1.1]}, "showlegend": False}}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def card(title, *children, cls="card", **kw):
    return html.Div(([html.H3(title)] if title else []) + list(children), className=cls, **kw)


def sensor_card(i, sensor):
    c = sensor_color(i)
    return html.Div(className="card sensor-card", children=[
        html.Div(className="sensor-head", children=[
            html.Span(className="swatch", style={"background": c}),
            html.Span(sensor["name"], className="sensor-name"),
            html.Span(id=f"sdot-{i}", className="dot"),
            html.Span("-", id=f"sstate-{i}", className="sensor-state")]),
        html.Div(className="metrics", children=[
            html.Div([html.Div("-", id=f"sfps-{i}", className="v"), html.Div("fps", className="k")],
                     className="metric"),
            html.Div([html.Div("-", id=f"sppf-{i}", className="v"), html.Div("pts / frame", className="k")],
                     className="metric"),
            html.Div([html.Div("-", id=f"sage-{i}", className="v"), html.Div("last frame", className="k")],
                     className="metric")]),
        dcc.Graph(id=f"spark-{i}", className="spark", figure=sparkline(np.zeros(TIMELINE_S), c),
                  config={"staticPlot": True, "displayModeBar": False}),
        html.Div("moving points / s, last 60 s", className="spark-caption"),
    ])


def recording_card(args, can_record):
    off = not can_record

    def field(label, comp, cls="field"):
        return html.Div(className=cls, children=[html.Label(label), comp])

    return card(
        "Recording",
        html.Div(className="row", children=[html.Span("not recording", id="rec-badge",
                                                      className="rec-badge off")]),
        html.Div(id="rec-info", className="hint"),
        field("Scenario", dcc.Dropdown(id="rec-scenario", clearable=False, disabled=off,
                                       options=[{"label": s, "value": s} for s in SCENARIOS],
                                       value=args.scenario if args.scenario in SCENARIOS else None,
                                       placeholder="choose a scenario")),
        html.Div(className="row", children=[
            field("People", dcc.Input(id="rec-people", type="text", value=args.people,
                                      placeholder="e.g. raghav", disabled=off, className="text-input"),
                  cls="field grow"),
            field("Room", dcc.Input(id="rec-room", type="text", value=args.room, placeholder="e.g. lab",
                                    disabled=off, className="text-input"), cls="field grow")]),
        field("Notes", dcc.Input(id="rec-notes", type="text", value=args.notes, disabled=off,
                                 placeholder="rig geometry is added automatically", className="text-input")),
        html.Div(className="row field", children=[
            html.Button("Start recording", id="rec-start", n_clicks=0, disabled=off, className="btn go seg"),
            html.Button("Stop", id="rec-stop", n_clicks=0, disabled=True, className="btn done seg")]),
        html.Div(id="rec-msg", className="hint"),
        html.Div(className="divider"),
        html.Div(className="row field", children=[
            html.Button("Start activity", id="activity-btn", n_clicks=0, disabled=True,
                        className="btn big go")]),
        html.Div(className="row field", children=[
            dcc.Input(id="event-text", type="text", placeholder="Event label", disabled=True,
                      className="text-input"),
            html.Button("Mark event", id="mark-event", n_clicks=0, disabled=True, className="btn")]),
        html.Div(id="last-marks", className="marks"),
        html.Div("Recording needs live sensors: run with --live <sensors.json>. Mock data is never "
                 "recorded." if off else "", className="hint"))


PRESETS_DIR = Path(__file__).resolve().parent / "presets"      # git-ignored
PRESET_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def list_presets(folder):
    return sorted(p.stem for p in Path(folder).glob("*.json")) if Path(folder).is_dir() else []


def preset_name_ok(name):
    return bool(name) and len(name) <= 40 and set(name) <= PRESET_NAME_CHARS


def save_preset(folder, name, values):
    Path(folder).mkdir(parents=True, exist_ok=True)
    data = {k: values[k] for k in EngineConfig.FIELDS if k in values}
    (Path(folder) / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")


def load_preset(folder, name):
    """Settings dict from presets/<name>.json, or raises ValueError with a readable reason."""
    if not preset_name_ok(name or ""):
        raise ValueError("choose a preset")
    try:
        data = json.loads((Path(folder) / f"{name}.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read preset {name!r}: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"preset {name!r} is not a settings object")
    return {k: data[k] for k in EngineConfig.FIELDS if k in data}


def detection_card(cfg, live, presets):
    def slider(cid, lo, hi, step, value):
        return dcc.Slider(id=cid, min=lo, max=hi, step=step, value=value, marks=None,
                          tooltip={"placement": "bottom", "always_visible": False})

    def field(label, comp):
        return html.Div(className="field", children=[html.Label(label), comp])

    return card(
        "Detection",
        dcc.RadioItems(id="cfg-mode", value=cfg["mode"], className="radio-row",
                       options=[{"label": "Classic", "value": "classic"},
                                {"label": "Sparse", "value": "sparse"}]),
        html.Div("Switching mode also sets its window and min sensors, as --sparse does.",
                 className="hint"),
        field("Window (s): pool points before clustering", slider("cfg-window", 0.0, 2.0, 0.1, cfg["window_s"])),
        field("Min sensors (each with >= 2 points)",
              dcc.RadioItems(id="cfg-minsens", value=cfg["min_sensors"], className="radio-row",
                             options=[{"label": str(n), "value": n} for n in (1, 2, 3)])),
        field("DBSCAN eps (m)", slider("cfg-eps", 0.1, 1.5, 0.05, cfg["eps"])),
        field("Moving above |Doppler| (m/s)", slider("cfg-vmove", 0.05, 1.0, 0.05, cfg["vmove"])),
        html.Div(className="field", children=[dcc.Checklist(
            id="cfg-classify", className="switch", value=["on"] if cfg["classify"] else [],
            options=[{"label": "Classifier filter", "value": "on"}])]),
        field("Show last (s)", slider("show-s", 0.1, SHOW_MAX_S, 0.1, cfg["show_s"])),
        html.Div(className="row field", children=[
            html.Button("Reset to defaults", id="cfg-reset", n_clicks=0, className="btn seg")]),
        html.Div(id="cfg-msg", className="hint"),
        html.Div(className="divider"),
        html.Div(className="row field", children=[
            html.Button("Re-learn background", id="relearn", n_clicks=0, disabled=not live,
                        className="btn seg")]),
        html.Div(className="progress", children=[html.Div(id="learn-bar", className="progress-bar")]),
        html.Div(id="learn-text", className="hint"),
        html.Div(className="divider"),
        field("Presets", dcc.Dropdown(id="cfg-preset", options=[{"label": p, "value": p} for p in presets],
                                      placeholder="choose a preset")),
        html.Div(className="row field", children=[
            html.Button("Load", id="cfg-load", n_clicks=0, className="btn seg"),
            dcc.Input(id="cfg-preset-name", type="text", placeholder="name", className="text-input"),
            html.Button("Save", id="cfg-save", n_clicks=0, className="btn")]),
        html.Div(id="cfg-preset-msg", className="hint"))


def build_layout(source, args, can_record, fig3d, figfloor, cfg=None, presets=()):
    live = getattr(source, "live", False)
    cfg = cfg or EngineConfig.from_args(args).snapshot()
    sparse = cfg["mode"] == "sparse"
    chips = [html.Span(source.label, className="chip live" if live else "chip"),
             html.Span(cfg["mode"], id="mode-chip", className="chip sparse" if sparse else "chip")]
    header = html.Div(className="header", children=[
        html.Div(["Radar Array", html.Span("presence monitor", className="sub")], className="brand"),
        *chips,
        html.Span(id="conn-state", className="chip"),
        html.Div(className="spacer"),
        html.Span(id="clock", className="clock"),
    ])
    left = html.Div(className="col left", children=[
        card("View",
             dcc.RadioItems(id="view-mode", value="both", className="radio-row",
                            options=[{"label": "Floor plan", "value": "floor"},
                                     {"label": "3D", "value": "3d"},
                                     {"label": "Both", "value": "both"}]),
             html.Div(className="field", children=[
                 html.Label("3D camera"),
                 html.Div(className="row", children=[
                     html.Button("Top", id="cam-top", n_clicks=0, className="btn seg"),
                     html.Button("Isometric", id="cam-iso", n_clicks=0, className="btn seg")])]),
             html.Div(className="field", children=[
                 dcc.Checklist(id="show-tracks", className="switch",
                               options=[{"label": "Show tracks (experimental)", "value": "on"}],
                               value=[] if sparse else ["on"]),
                 html.Div("Track IDs churn on sparse data; zones show where motion is.",
                          className="hint")]),
             html.Div(className="field", children=[
                 html.Label("Spectate (3D follows a track)"),
                 dcc.Dropdown(id="follow", clearable=False, value="free",
                              options=[{"label": "Free camera", "value": "free"}])]),
             html.Div(className="field", children=[
                 html.Label("Spectate window (m)"),
                 dcc.Slider(id="zoom", min=1.5, max=5, step=0.5, value=2.5, marks=None,
                            tooltip={"placement": "bottom"})])),
        detection_card(cfg, live, presets),
        recording_card(args, can_record),
    ])
    centre = html.Div(className="col centre", children=[
        html.Div(id="stats-bar", className="toolbar"),
        html.Div(id="views", className="views both", children=[
            html.Div(id="floor-wrap", className="card view-pane", children=[
                html.Div("Floor plan, top-down", className="title"),
                dcc.Graph(id="floor-plot", figure=figfloor, className="view-graph",
                          config={"displayModeBar": False})]),
            html.Div(id="d3-wrap", className="card view-pane", children=[
                html.Div("3D view", className="title"),
                dcc.Graph(id="radar-plot", figure=fig3d, className="view-graph",
                          config={"displayModeBar": True, "displaylogo": False})]),
        ]),
        card("Presence, last 60 s",
             html.Div(id="timeline", className="timeline",
                      children=[html.Div(className="cell") for _ in range(TIMELINE_S)]),
             html.Div([html.Span("-60 s"), html.Span("now")], className="timeline-axis")),
    ])
    right = html.Div(className="col right", children=[
        html.Div(id="presence-card", className="card presence learning", children=[
            html.Div("PRESENCE", className="label"),
            html.Div("-", id="presence-text", className="value"),
            html.Div("", id="presence-sub", className="sub"),
            html.Div(ZONES_MERGE_NOTE, id="presence-note", className="hint" if sparse else "hint hidden")]),
        *[sensor_card(i, s) for i, s in enumerate(source.sensors)],
    ])
    return html.Div(className="app", children=[
        header,
        html.Div(id="alert-strip", className="alert-strip hidden", role="alert"),
        html.Div(className="main", children=[left, centre, right]),
        dcc.Interval(id="tick", interval=150),
        dcc.Interval(id="slow-tick", interval=1000),
        html.Div(id="dummy", style={"display": "none"}),
        dcc.Store(id="rec-ack", data=0),
    ])


# ---------------------------------------------------------------------------
# App: layout + callbacks (UI only; processing lives in Engine)
# ---------------------------------------------------------------------------
def create_app(source, engine, rec_ctl, args, presets_dir=PRESETS_DIR):
    """rec_ctl: a RecordingController (None -> one writing to data/; mock sources never record).
    Live settings go through engine.config; presets are read from and saved to presets_dir."""
    if rec_ctl is None:
        rec_ctl = RecordingController(source, sensors_json=getattr(args, "live", None))
    E = args.extent
    config = engine.config
    n_sensors = len(source.sensors)
    fig3d, s3 = build_base_figure(source.sensors, E, zone=True)      # zone slots stay empty in classic
    figfloor, sf = build_floor_figure(source.sensors, E, zone=True)
    ui = {"mode": "free", "ids": (), "cam_top": 0, "cam_iso": 0, "map3d": False, "mapf": False}
    app = Dash(__name__, title=APP_TITLE, update_title=None)    # favicon: assets/favicon.ico
    app.layout = build_layout(source, args, rec_ctl.can_record, fig3d, figfloor,
                              cfg=config.snapshot(), presets=list_presets(presets_dir))
    app.rec_ctl = rec_ctl                         # for tests and shutdown

    @app.callback(Output("dummy", "children"), Input("relearn", "n_clicks"), prevent_initial_call=True)
    def _relearn(_):
        engine.relearn()
        return ""

    @app.callback(Output("rec-msg", "children"), Output("event-text", "value"), Output("rec-ack", "data"),
                  Input("rec-start", "n_clicks"), Input("rec-stop", "n_clicks"),
                  Input("activity-btn", "n_clicks"), Input("mark-event", "n_clicks"),
                  State("rec-scenario", "value"), State("rec-people", "value"), State("rec-room", "value"),
                  State("rec-notes", "value"), State("event-text", "value"), State("rec-ack", "data"),
                  prevent_initial_call=True)
    def _recording_action(_start, _stop, _act, _mark, scenario, people, room, notes, text, ack):
        now, trig, keep = time.time(), ctx.triggered_id, no_update
        if trig == "rec-start":
            if not scenario:
                msg = "Choose a scenario first."
            else:
                try:
                    msg = rec_ctl.start(scenario, people, room, notes)[1]
                except Exception as e:            # e.g. data/ not writable: show it, do not crash
                    msg = f"Could not start recording: {e}"
        elif trig == "rec-stop":
            msg = rec_ctl.request_stop(now)
        elif trig == "activity-btn":
            msg = rec_ctl.toggle_activity(now)
        elif trig == "mark-event":
            msg = rec_ctl.mark(text)
            keep = "" if msg.startswith("Marked") else no_update
        else:
            return no_update, no_update, no_update
        return msg, keep, (ack or 0) + 1

    @app.callback(Output("rec-badge", "children"), Output("rec-badge", "className"),
                  Output("rec-info", "children"), Output("rec-start", "disabled"),
                  Output("rec-stop", "disabled"), Output("rec-stop", "children"),
                  Output("rec-stop", "className"), Output("rec-scenario", "disabled"),
                  Output("rec-people", "disabled"), Output("rec-room", "disabled"),
                  Output("rec-notes", "disabled"), Output("activity-btn", "children"),
                  Output("activity-btn", "className"), Output("activity-btn", "disabled"),
                  Output("event-text", "disabled"), Output("mark-event", "disabled"),
                  Output("last-marks", "children"),
                  Input("slow-tick", "n_intervals"), Input("rec-ack", "data"))
    def _recording_status(_n, _ack):
        st = rec_ctl.status(time.time())
        rec, can = st["recording"], rec_ctl.can_record
        if rec:
            el = int(st["elapsed"])
            badge, badge_cls = f"\u25cf REC {el // 60:02d}:{el % 60:02d}", "rec-badge on"
            info = (f"{st['path'].name} | {st['frames']} frames recorded | "
                    f"{st.get('chunks', 0)} chunks on disk")
        else:
            badge, badge_cls = "not recording", "rec-badge off"
            info = f"last session: {st['path'].name}" if st["path"] is not None else ""
        if st["activity_s"] is not None:
            a = int(st["activity_s"])
            act, act_cls = f"End activity \u00b7 {a // 60:02d}:{a % 60:02d}", "btn big done"
        else:
            act, act_cls = "Start activity", "btn big go"
        marks = [html.Span(f"{time.strftime('%H:%M:%S', time.localtime(t))} {label}",
                           className="mark " + {"go": "mark-go", "done": "mark-done"}.get(label, "mark-other"))
                 for t, label in st["marks"]]
        if not marks:
            marks = [html.Span("no marks yet" if st["path"] is not None else "", className="mark mark-other")]
        locked = rec or not can
        return (badge, badge_cls, info, locked,
                not rec, "Confirm stop" if st["armed"] else "Stop",
                "btn done seg armed" if st["armed"] else "btn done seg",
                locked, locked, locked, locked,
                act, act_cls, not rec, not rec, not rec, marks)

    @app.callback(Output("cfg-mode", "value"), Output("cfg-window", "value"), Output("cfg-minsens", "value"),
                  Output("cfg-eps", "value"), Output("cfg-vmove", "value"), Output("cfg-classify", "value"),
                  Output("show-s", "value"), Output("show-tracks", "value"), Output("cfg-msg", "children"),
                  Input("cfg-mode", "value"), Input("cfg-window", "value"), Input("cfg-minsens", "value"),
                  Input("cfg-eps", "value"), Input("cfg-vmove", "value"), Input("cfg-classify", "value"),
                  Input("show-s", "value"), Input("cfg-reset", "n_clicks"), Input("cfg-load", "n_clicks"),
                  State("cfg-preset", "value"), prevent_initial_call=True)
    def _settings(mode, window_s, min_sensors, eps, vmove, classify, show_s, _reset, _load, preset):
        """Every Detection control lands here; Engine reads the result on its next tick."""
        trig = ctx.triggered_id
        if trig == "cfg-reset":
            new, how = dict(config.defaults), "Defaults restored"
        elif trig == "cfg-load":
            try:
                new, how = load_preset(presets_dir, preset), f"Preset {preset!r} loaded"
            except ValueError as e:
                new, how = {}, f"Not loaded: {e}"
        elif trig == "cfg-mode":
            new, how = dict(mode=mode, **config.mode_defaults(mode)), f"{mode} mode"
        else:
            field = {"cfg-window": "window_s", "cfg-minsens": "min_sensors", "cfg-eps": "eps",
                     "cfg-vmove": "vmove", "cfg-classify": "classify", "show-s": "show_s"}.get(trig)
            value = {"window_s": window_s, "min_sensors": min_sensors, "eps": eps, "vmove": vmove,
                     "classify": "on" in (classify or []), "show_s": show_s}.get(field)
            new, how = ({field: value} if field else {}), "Applied"
        changes = config.update(**new)
        c = config.snapshot()
        label = settings_label(changes)
        logged = bool(label) and rec_ctl.log_settings(label)
        tracks = ([] if c["mode"] == "sparse" else ["on"]) if "mode" in changes else no_update
        if changes:
            msg = f"{how}: {label[len('settings '):] if label else 'display only'}" + (" (marked)" if logged else "")
        else:
            msg = how if trig in ("cfg-reset", "cfg-load") or how.startswith("Not") else ""
        return (c["mode"], c["window_s"], c["min_sensors"], c["eps"], c["vmove"],
                ["on"] if c["classify"] else [], c["show_s"], tracks, msg)

    @app.callback(Output("cfg-preset", "options"), Output("cfg-preset", "value"),
                  Output("cfg-preset-msg", "children"), Input("cfg-save", "n_clicks"),
                  State("cfg-preset-name", "value"), prevent_initial_call=True)
    def _save_preset(_n, name):
        name = (name or "").strip()
        if not preset_name_ok(name):
            return no_update, no_update, "Name: letters, digits, - and _ only (up to 40)."
        existed = name in list_presets(presets_dir)
        try:
            save_preset(presets_dir, name, config.snapshot())
        except OSError as e:
            return no_update, no_update, f"Could not save: {e}"
        opts = [{"label": p, "value": p} for p in list_presets(presets_dir)]
        return opts, name, f"Saved {name!r}" + (" (replaced)" if existed else "") + "."

    @app.callback(Output("learn-bar", "style"), Output("learn-text", "children"),
                  Input("tick", "n_intervals"))
    def _learn_progress(_n):
        p = engine.learn_progress()
        with engine.lock:
            left = engine.state["left"]
        if p < 1.0:
            txt = f"learning the empty room: {left:.0f} s left, stay out of view"
        elif getattr(source, "live", False):
            txt = f"background learned ({len(engine.bg.occupied)} voxels)"
        else:
            txt = "mock data: the room is known, no learning needed"
        return {"width": f"{p * 100:.0f}%"}, txt

    @app.callback(Output("floor-wrap", "className"), Output("d3-wrap", "className"),
                  Output("views", "className"), Input("view-mode", "value"))
    def _view_mode(view):
        show_f, show_3 = view in ("floor", "both"), view in ("3d", "both")
        return ("card view-pane" + ("" if show_f else " hidden"),
                "card view-pane" + ("" if show_3 else " hidden"),
                "views both" if view == "both" else "views single")

    status_outputs = ([Output("alert-strip", "children"), Output("alert-strip", "className"),
                       Output("clock", "children"),
                       Output("conn-state", "children"), Output("conn-state", "className"),
                       Output("timeline", "children")] +
                      [Output(f"{k}-{i}", p) for i in range(n_sensors)
                       for k, p in (("sdot", "className"), ("sstate", "children"), ("sfps", "children"),
                                    ("sppf", "children"), ("sage", "children"), ("spark", "figure"))])

    @app.callback(*status_outputs, Input("slow-tick", "n_intervals"))
    def update_status(_n):
        now = time.time()
        rows = health_rows(engine, source, now)
        if getattr(source, "live", False):
            silent = [s["name"] for s, r in zip(source.sensors, rows) if r[3] == "stale"]
            if silent:
                conn, conn_cls = f"● {', '.join(silent)} silent", "chip danger"
            else:
                n_ok = sum(r[3] == "ok" for r in rows)
                conn, conn_cls = f"● streaming {n_ok}/{len(rows)}", "chip ok"
        else:
            conn, conn_cls = "", "chip hidden"    # the mode chip already says MOCK DATA
        rates, states = timeline_bins(list(engine.timeline), now, n_sensors)
        cells = [html.Div(className=f"cell {st}", title=f"-{TIMELINE_S - 1 - b} s: {st}")
                 for b, st in enumerate(states)]
        alert, alert_cls = silent_sensor_alert(engine, source, rows, now)
        out = [alert, alert_cls, time.strftime("%H:%M:%S"), conn, conn_cls, cells]
        for i in range(n_sensors):
            fps, ppf, age, state = rows[i]
            out += [f"dot {state}",
                    {"ok": "streaming", "stale": "silent", "idle": "idle slot"}[state],
                    f"{fps:.1f}",
                    "-" if not np.isfinite(ppf) else f"{ppf:.1f}",
                    "never" if age is None else f"{age:.1f} s",
                    sparkline(rates[i], sensor_color(i))]
        return out

    @app.callback(Output("presence-card", "className"), Output("presence-text", "children"),
                  Output("presence-sub", "children"), Output("mode-chip", "children"),
                  Output("mode-chip", "className"), Output("presence-note", "className"),
                  Input("tick", "n_intervals"), prevent_initial_call=True)
    def update_presence(_n):
        with engine.lock:
            s = engine.state
        mode = s.get("mode") or config.snapshot()["mode"]
        extra = (mode, "chip sparse" if mode == "sparse" else "chip",
                 "hint" if mode == "sparse" else "hint hidden")
        return _presence_text(s, mode == "sparse") + extra

    def _presence_text(s, sparse):
        if s["learning"]:
            return "card presence learning", "learning", f"background: {s['left']:.0f} s left"
        if sparse:
            zones = s.get("zones") or []
            if zones:
                where = ", ".join(f"({x:+.1f}, {y:+.1f})" for x, y, _ in zones)
                n = len(zones)
                return ("card presence yes", "YES",
                        f"{n} motion zone{'s' if n > 1 else ''}: {where} m, about 1 m each")
            return "card presence no", "NO", "no motion zones"
        n = len(s["tracks"])
        if n:
            return "card presence yes", "YES", f"{n} tracked"
        return "card presence no", "NO", "no confirmed tracks"

    @app.callback(Output("radar-plot", "figure"), Output("floor-plot", "figure"),
                  Output("stats-bar", "children"), Output("follow", "options"),
                  Input("tick", "n_intervals"), State("follow", "value"), State("zoom", "value"),
                  State("show-s", "value"), State("view-mode", "value"),
                  State("cam-top", "n_clicks"), State("cam-iso", "n_clicks"),
                  State("show-tracks", "value"), prevent_initial_call=True)
    def update(n, follow, zoom, show_s, view, cam_top, cam_iso, show_tracks):
        with engine.lock:
            s = engine.state
        xyz, sid, tracks = s["xyz"], s["sid"], s["tracks"]
        sparse = (s.get("mode") or config.snapshot()["mode"]) == "sparse"
        view = view or "both"
        show3, showf = view in ("3d", "both"), view in ("floor", "both")

        # live points for the last show_s seconds: dim static, bright moving in sensor colours
        hist = s.get("history") or [(0.0, xyz, sid, np.zeros(len(xyz), bool))]
        cutoff = hist[-1][0] - float(show_s or 0.1)
        shown = [h for h in hist if h[0] > cutoff] or hist[-1:]
        lxyz = np.vstack([h[1] for h in shown])
        lsid = np.concatenate([h[2] for h in shown])
        lmov = np.concatenate([h[3] for h in shown]) if len(shown[0]) > 3 else np.zeros(len(lxyz), bool)
        st_xyz, mv_xyz = lxyz[~lmov], lxyz[lmov]
        mv_col = [sensor_color(i) for i in lsid[lmov].tolist()]
        refresh_map = n % 30 == 0
        bg_map = None

        def room_map():
            return source.background_points() if not source.live else engine.bg.points()

        # "Show tracks (experimental)": off by default in --sparse (IDs churn on sparse data)
        tracks_on = "on" in (show_tracks or [])
        vis_tracks = tracks if tracks_on else []
        zones = s.get("zones") or []
        floor_zone, disk, ring = zone_shapes(zones)
        zlabels = [f"zone {q + 1}" for q in range(len(zones))]

        patch3 = no_update
        if show3:
            patch3 = Patch()

            def put(name, x, y, z):
                i = s3[name]
                patch3["data"][i]["x"], patch3["data"][i]["y"], patch3["data"][i]["z"] = x, y, z

            put("static", st_xyz[:, 0].tolist(), st_xyz[:, 1].tolist(), st_xyz[:, 2].tolist())
            put("moving", mv_xyz[:, 0].tolist(), mv_xyz[:, 1].tolist(), mv_xyz[:, 2].tolist())
            patch3["data"][s3["moving"]]["marker"]["color"] = mv_col
            cx, cc, boxes = [], [], []
            if tracks_on:
                for di, d in enumerate(s["dets"]):
                    tid = s["assign"].get(di)
                    if tid is None:
                        continue
                    cx.append(xyz[d["idx"]])
                    cc += [TRACK_COLORS[tid % len(TRACK_COLORS)]] * len(d["idx"])
                    boxes += box_chunks(d["lo"], d["hi"])
            cp = np.vstack(cx) if cx else np.empty((0, 3))
            put("clus", cp[:, 0].tolist(), cp[:, 1].tolist(), cp[:, 2].tolist())
            patch3["data"][s3["clus"]]["marker"]["color"] = cc
            put("box", *segs(boxes))
            put("label", [t["pos"][0] for t in vis_tracks], [t["pos"][1] for t in vis_tracks],
                [t["pos"][2] + 0.5 for t in vis_tracks])
            patch3["data"][s3["label"]]["text"] = [f"#{t['id']}" for t in vis_tracks]
            put("trail", *segs([t["trail"] for t in vis_tracks if len(t["trail"]) > 1]))
            # zone slots: filled in sparse mode, empty in classic (state has no zones there)
            dz = patch3["data"][s3["zone_disk"]]
            dz["x"], dz["y"], dz["z"], dz["i"], dz["j"], dz["k"] = disk
            put("zone_ring", *ring)
            put("zone_label", [x for x, _, _ in zones], [y for _, y, _ in zones],
                [ZONE_LABEL_Z] * len(zones))
            patch3["data"][s3["zone_label"]]["text"] = zlabels
            if refresh_map or not ui["map3d"]:
                bg_map = room_map()
                put("map", bg_map[:, 0].tolist(), bg_map[:, 1].tolist(), bg_map[:, 2].tolist())
                ui["map3d"] = True
            ax = patch3["layout"]["scene"]
            for key, clicks in (("top", cam_top or 0), ("iso", cam_iso or 0)):
                if clicks != ui[f"cam_{key}"]:
                    ui[f"cam_{key}"] = clicks
                    patch3["layout"]["scene"]["camera"] = CAMERAS[key]
            # spectate: slide the plotted window so the followed person stays centred
            target = next((t for t in vis_tracks if str(t["id"]) == str(follow)), None)
            if follow != "free" and target is not None:
                w, (px, py, _) = float(zoom), target["pos"]
                ax["xaxis"]["range"], ax["yaxis"]["range"] = [px - w, px + w], [py - w, py + w]
                ax["aspectratio"] = dict(x=1, y=1, z=(Z_HI - Z_LO) / (2 * w))
                ui["mode"] = "follow"
            elif follow == "free" and ui["mode"] != "free":
                ax["xaxis"]["range"], ax["yaxis"]["range"] = [-E, E], [-E, E]
                ax["aspectratio"] = dict(x=1, y=1, z=(Z_HI - Z_LO) / (2 * E))
                ui["mode"] = "free"
        else:
            ui["map3d"] = False

        patchf = no_update
        if showf:
            patchf = Patch()

            def put2(name, x, y):
                i = sf[name]
                patchf["data"][i]["x"], patchf["data"][i]["y"] = x, y

            put2("static", st_xyz[:, 0].tolist(), st_xyz[:, 1].tolist())
            put2("moving", mv_xyz[:, 0].tolist(), mv_xyz[:, 1].tolist())
            patchf["data"][sf["moving"]]["marker"]["color"] = mv_col
            tx, ty = [], []
            for t in vis_tracks:
                if len(t["trail"]) > 1:
                    tx += t["trail"][:, 0].tolist() + [None]
                    ty += t["trail"][:, 1].tolist() + [None]
            put2("trail", tx, ty)
            put2("label", [t["pos"][0] for t in vis_tracks], [t["pos"][1] + 0.35 for t in vis_tracks])
            patchf["data"][sf["label"]]["text"] = [f"#{t['id']}" for t in vis_tracks]
            # zone and gate-box slots: empty when the state has none (classic mode, min sensors 1)
            put2("zones", *floor_zone)
            put2("zone_label", [x for x, _, _ in zones], [y + ZONE_R + 0.25 for _, y, _ in zones])
            patchf["data"][sf["zone_label"]]["text"] = zlabels
            bx, by, lx, ly, lt = gate_boxes(s.get("gated") or [])
            put2("gbox", bx, by)
            put2("gbox_label", lx, ly)
            patchf["data"][sf["gbox_label"]]["text"] = lt
            if refresh_map or not ui["mapf"]:
                m = bg_map if bg_map is not None else room_map()
                put2("map", m[:, 0].tolist(), m[:, 1].tolist())
                ui["mapf"] = True
        else:
            ui["mapf"] = False

        ids = tuple(t["id"] for t in vis_tracks)
        if ids != ui["ids"]:
            ui["ids"] = ids
            opts = [{"label": "Free camera", "value": "free"}] + \
                   [{"label": f"Person #{i}", "value": str(i)} for i in ids]
        else:
            opts = no_update

        status = (f"LEARNING empty room, {s['left']:.0f}s left - stay out of view" if s["learning"]
                  else "tracking")
        who = f"motion zones {len(zones)}" if sparse else f"people {len(tracks)}"
        txt = (f"points {len(xyz)} | foreground {s['fg']} | {who} | "
               f"map voxels {len(engine.bg.occupied) if engine.bg.frozen else len(engine.bg.counts)} | {status}")
        return patch3, patchf, txt, opts

    return app


# ---------------------------------------------------------------------------
def build_parser():
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
    ap.add_argument("--lan", action="store_true",
                    help="listen on all interfaces (0.0.0.0) instead of 127.0.0.1; the controls "
                         "have no authentication, so anyone on the network can use them")
    ap.add_argument("--model", metavar="MODEL_JSON", help="XGBoost model file")
    ap.add_argument("--no-classify", action="store_true", help="keep all detections")
    ap.add_argument("--record", action="store_true", help="record raw frames to a session")
    ap.add_argument("--room", default="", help="recording room label")
    ap.add_argument("--scenario", default="", help="recording scenario label")
    ap.add_argument("--people", default="", help="recording people label")
    ap.add_argument("--notes", default="", help="recording notes")
    return ap


def server_host(args):
    return "0.0.0.0" if args.lan else "127.0.0.1"


def startup_banner(args, source):
    lines = [f"  Viewer running ({source.label}). Open http://127.0.0.1:{args.port}"]
    if args.lan:
        lines += [f"  Listening on all interfaces: http://<this-machine's-address>:{args.port}",
                  "  WARNING: --lan: the dashboard has no authentication. Anyone who can reach this",
                  "  port can start and stop recordings, change detection settings and re-learn",
                  "  the background. Use it only on a network you trust."]
    else:
        lines.append("  Local only (127.0.0.1). Use --lan to open it from another device.")
    return "\n" + "\n".join(lines) + "\n"


def shutdown(engine, rec_ctl, source):
    """Stop processing, close an open recording (flush, end time), then the sensors."""
    engine.halt.set()
    if engine.is_alive():
        engine.join(3.0)
    rec_ctl.stop()
    if getattr(source, "live", False):
        source.close()


def _exit_on_signal(signum, _frame):
    raise SystemExit(0)                           # runs main()'s finally: clean shutdown


def main():
    args = build_parser().parse_args()

    source = (LiveSource(args.live, args.only, args.freerun, args.slot)
              if args.live else MockSource(dense=args.mock_dense))
    rec_ctl = RecordingController(source, sensors_json=args.live)
    if args.record and args.live:                 # --record: start one right away, as before
        print("  " + rec_ctl.start(args.scenario, args.people, args.room, args.notes)[1])
    engine = Engine(source, args.learn, args.eps, args.vmove, model_path=args.model,
                    config=EngineConfig.from_args(args))
    engine.start()
    app = create_app(source, engine, rec_ctl, args)

    print(startup_banner(args, source))
    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # hide per-request log lines
    for sig in (signal.SIGTERM, signal.SIGHUP):  # kill / closed terminal: shut down cleanly too
        signal.signal(sig, _exit_on_signal)
    # Ctrl-C / kill -INT: Python skips its handler when SIGINT was inherited as ignored (e.g.
    # started in the background by a script), which would make the viewer ignore SIGINT.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        app.run(host=server_host(args), port=args.port, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(engine, rec_ctl, source)
        print("  Stopped." + (f" Last recording: {rec_ctl.last_path}" if rec_ctl.last_path else ""))


if __name__ == "__main__":
    main()

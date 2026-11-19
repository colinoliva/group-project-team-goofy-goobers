"""
radar_dash_viewer.py - web viewer (Dash/Plotly) for the radar array: sensor positions and
fields of view, static room map, live points, clusters with bounding boxes, tracked people
with IDs/trails, and a "spectate" mode that follows one person.

Run it on the Raspberry Pi (or any PC) and open http://<address>:8050 in a browser.

    python radar_dash_viewer.py                                   # mock data, no hardware
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
import threading
import time

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, dcc, html, no_update

from radar_core import PALETTE, apply_T, filter_points, fov_lines, grid_lines, pose_to_T
from features import extract_features, to_vector
from rule_baseline import is_human_rule
from tracking import Background, Tracker, cluster, detections

TRACK_COLORS = ["#00d4ff", "#ff6b35", "#a855f7", "#44cf6c", "#f59e0b", "#ef4444"]
SENSOR_HEX = ["#%02x%02x%02x" % tuple(int(255 * c) for c in rgb) for rgb in PALETTE]
Z_LO, Z_HI = -0.05, 3.0

# fixed trace slots - the callback only patches these, never rebuilds the figure
I_MAP, I_LIVE, I_CLUS, I_BOX, I_LABEL, I_TRAIL = range(6)


# ---------------------------------------------------------------------------
# Data sources: both return {"xyz": Nx3 world, "vel": N, "sid": N sensor index}
# ---------------------------------------------------------------------------
def default_sensors(side=1.5, height=1.3):
    rc, out = side / np.sqrt(3), []
    for n, phi in enumerate((90, 210, 330), start=1):
        c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
        out.append({"name": f"S{n}", "T": pose_to_T(rc * c, rc * s, height, np.degrees(np.arctan2(c, -s)))})
    return out


class MockSource:
    live = False
    label = "MOCK DATA"

    def __init__(self):
        self.sensors = default_sensors()
        w = []
        for z in (0.2, 0.7, 1.2, 1.7, 2.2):
            for x in np.arange(-3, 3.01, 0.2):
                w += [[x, -2.5, z], [x, 3.5, z]]
            for y in np.arange(-2.5, 3.51, 0.2):
                w += [[-3, y, z], [3, y, z]]
        self.static = np.array(w)
        self.rng = np.random.default_rng()

    def background_points(self):
        return self.static

    def next_frame(self):
        t, r = time.time(), self.rng
        people = [(1.5 * np.sin(0.30 * t), 1.2 * np.cos(0.22 * t) + 0.8),
                  (-1.2 + 0.7 * np.cos(0.6 * t), 1.5 + 0.9 * np.sin(0.45 * t))]
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
        self.recorder = recorder
        self.sched.start()
        self.label = ("LIVE " + ("freerun" if freerun else "time-division") if len(sensors) > 1
                      else f"LIVE {sensors[0]['name']}")

    def next_frame(self):
        xyz, vel, snr, sid = ([np.empty((0, 3))], [np.empty(0)], [np.empty(0)],
                              [np.empty(0, int)])
        for i, link in enumerate(self.links):
            for fr in link.drain():
                if self.recorder is not None:
                    self.recorder.add(i, fr)
                pts = fr["pts"]
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
                "snr": np.concatenate(snr), "sid": np.concatenate(sid)}

    def close(self):
        self.sched.halt.set()
        self.sched.join(3.0)
        for l in self.links:
            l.close()


# ---------------------------------------------------------------------------
# Processing thread (acquire -> foreground -> cluster -> track), ~10 Hz
# ---------------------------------------------------------------------------
class Engine(threading.Thread):
    def __init__(self, source, learn_s, eps, v_move, classify=True, model_path=None):
        super().__init__(daemon=True)
        self.src, self.learn_s, self.eps, self.v_move = source, learn_s, eps, v_move
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

    def relearn(self):
        if self.src.live:
            self.bg.reset()
            self.tracker = Tracker()
            self.t_learn = time.time()

    def run(self):
        while not self.halt.is_set():
            t0 = time.time()
            fr = self.src.next_frame()
            xyz, vel, snr, sid = fr["xyz"], fr["vel"], fr["snr"], fr["sid"]
            learning = not self.bg.frozen
            dets, assign, tracks, n_fg = [], {}, [], 0
            if learning:
                self.bg.learn(xyz[np.abs(vel) < 0.05])
                if t0 - self.t_learn > self.learn_s:
                    self.bg.freeze()
            else:
                fg = (np.abs(vel) >= self.v_move) | ~self.bg.is_background(xyz)
                idx = np.flatnonzero(fg)
                n_fg = len(idx)
                lab = cluster(xyz[idx], self.eps)
                dets = detections(xyz[idx], lab)
                for d in dets:
                    d["idx"] = idx[d["idx"]]
                if self.classify:
                    keep = []
                    for d in dets:
                        cl_xyz = xyz[d["idx"]]
                        cl_vel = vel[d["idx"]]
                        cl_snr = snr[d["idx"]]
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
                              "tracks": snap, "learning": learning,
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


def build_base_figure(sensors, extent):
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
    return fig


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", metavar="SENSORS_JSON", help="use real sensors (default: mock data)")
    ap.add_argument("--only", nargs="+", help="live: run only these sensors, e.g. --only S1")
    ap.add_argument("--freerun", action="store_true", help="live: all sensors transmit continuously")
    ap.add_argument("--slot", type=float, default=3.0, help="live time-division slot, s")
    ap.add_argument("--learn", type=float, default=30.0, help="live: seconds of empty-room learning")
    ap.add_argument("--eps", type=float, default=0.4, help="DBSCAN neighbourhood radius, m")
    ap.add_argument("--vmove", type=float, default=0.1, help="|Doppler| above this counts as moving, m/s")
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
                  if args.live else MockSource())
    except Exception:
        if recorder is not None:
            recorder.close()
        raise
    engine = Engine(source, args.learn, args.eps, args.vmove,
                    classify=not args.no_classify, model_path=args.model)
    engine.start()
    ui = {"mode": "free", "ids": ()}
    E = args.extent

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
            dcc.Graph(id="radar-plot", figure=build_base_figure(source.sensors, E),
                      style={"height": "92vh"}, config={"displayModeBar": True}),
            dcc.Interval(id="tick", interval=150),
            html.Div(id="dummy", style={"display": "none"}),
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

    @app.callback(Output("radar-plot", "figure"), Output("stats-bar", "children"),
                  Output("follow", "options"),
                  Input("tick", "n_intervals"), State("follow", "value"), State("zoom", "value"),
                  prevent_initial_call=True)
    def update(n, follow, zoom):
        with engine.lock:
            s = engine.state
        xyz, sid, tracks = s["xyz"], s["sid"], s["tracks"]
        patch = Patch()

        def put(i, x, y, z):
            patch["data"][i]["x"], patch["data"][i]["y"], patch["data"][i]["z"] = x, y, z

        # live points colored by the sensor that saw them
        put(I_LIVE, xyz[:, 0].tolist(), xyz[:, 1].tolist(), xyz[:, 2].tolist())
        patch["data"][I_LIVE]["marker"]["color"] = [SENSOR_HEX[i % len(SENSOR_HEX)] for i in sid.tolist()]

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
"""Headless tests for the dashboard's recording controls (no hardware, temp folders only).

Callbacks are called through the Dash app's Flask test client, exactly as the browser calls
them; LiveSource's real recorder handoff runs with fake serial links.
"""
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

import radar_dash_viewer as V
from load_session import load_session


class FakeLink:
    """Stands in for RadarLink: drain() returns one raw frame per call."""

    def __init__(self, i):
        self.i, self.n = i, 0
        self.rng = np.random.default_rng(i)

    def drain(self):
        self.n += 1
        pts = np.column_stack([self.rng.normal(0, 0.5, (5, 2)) + [0, 1.5],
                               self.rng.normal(0, 0.3, 5), self.rng.normal(0, 0.5, 5)]).astype(np.float32)
        return [{"t": time.time(), "frame": self.n, "pts": pts,
                 "snr": np.full(5, 20.0, np.float32), "noise": np.full(5, 40.0, np.float32)}]


def fake_live_source():
    """A real LiveSource object (its next_frame and set_recorder), with fake links."""
    src = V.LiveSource.__new__(V.LiveSource)
    src.sensors = [dict(s, cli=f"cli{i}", data=f"data{i}", cfg="mapping_v2.cfg")
                   for i, s in enumerate(V.default_sensors())]
    src.links = [FakeLink(i) for i in range(len(src.sensors))]
    src.sched = types.SimpleNamespace(active=None)
    src.continuous = True
    src.label = "LIVE test"
    src._rec_lock = threading.Lock()
    src.recorder = None
    src.recorded_frames = 0
    return src


def mock_args(argv=()):
    return V.build_parser().parse_args(list(argv))


class Client:
    """Posts callback requests the way the browser does."""

    def __init__(self, app):
        self.c = app.server.test_client()
        self.deps = json.loads(self.c.get("/_dash-dependencies").data)

    def call(self, output_part, inputs, state=()):
        dep = next(d for d in self.deps if output_part in d["output"])
        out = dep["output"]
        outs = ([{"id": o.split(".")[0], "property": o.split(".")[1]} for o in out.strip(".").split("...")]
                if out.startswith("..") else {"id": out.split(".")[0], "property": out.split(".")[1]})
        given = dict(inputs, **dict(state))
        body = {"output": out, "outputs": outs,
                "inputs": [{"id": i["id"], "property": i["property"],
                            "value": given.get(f"{i['id']}.{i['property']}")} for i in dep["inputs"]],
                "state": [{"id": i["id"], "property": i["property"],
                           "value": given.get(f"{i['id']}.{i['property']}")} for i in dep["state"]],
                "changedPropIds": list(inputs)}
        r = self.c.post("/_dash-update-component", json=body)
        if r.status_code == 204:
            return None
        assert r.status_code == 200, (r.status_code, r.data[:500])
        return json.loads(r.data)["response"]

    def action(self, trigger, n=1, **fields):
        state = {"rec-scenario.value": fields.get("scenario"), "rec-people.value": fields.get("people"),
                 "rec-room.value": fields.get("room"), "rec-notes.value": fields.get("notes"),
                 "event-text.value": fields.get("text"), "rec-ack.data": 0}
        r = self.call("rec-msg.children", {f"{trigger}.n_clicks": n}, state)
        return r["rec-msg"]["children"]

    def status(self):
        return self.call("rec-badge.children", {"slow-tick.n_intervals": 1, "rec-ack.data": 1})


class StubRecorder:
    def __init__(self):
        self.marks, self.closed = [], False
        self.path = Path("stub_session")
        self.start_epoch = time.time()

    def mark(self, label):
        self.marks.append(label)

    def close(self):
        self.closed = True


class ActivityStateMachineTest(unittest.TestCase):
    def controller(self):
        src = types.SimpleNamespace(live=True, sensors=[], set_recorder=lambda r: None)
        ctl = V.RecordingController(src, root="unused")
        ctl.recorder = StubRecorder()
        return ctl

    def test_go_done_toggle_with_debounce(self):
        ctl = self.controller()
        rec = ctl.recorder
        self.assertIn("go", ctl.toggle_activity(100.0))
        self.assertIn("Ignored", ctl.toggle_activity(100.4))      # double click: no done
        self.assertIn("done", ctl.toggle_activity(101.5))
        self.assertIn("Ignored", ctl.toggle_activity(101.7))      # DONE pressed twice: once only
        self.assertIn("go", ctl.toggle_activity(103.0))
        self.assertEqual(rec.marks, ["go", "done", "go"])

    def test_stop_during_activity_logs_done_and_event_box_refuses_go_done(self):
        ctl = self.controller()
        rec = ctl.recorder
        ctl.toggle_activity(10.0)
        self.assertIn("Use the activity button", ctl.mark("done"))
        self.assertIn("Use the activity button", ctl.mark(" Go "))
        self.assertIn("Marked", ctl.mark("chair moved"))
        ctl.stop()
        self.assertEqual(rec.marks, ["go", "chair moved", "done"])
        self.assertTrue(rec.closed)
        self.assertIsNone(ctl.recorder)
        self.assertIn("Start a recording first", ctl.toggle_activity(20.0))
        self.assertIn("Not recording", ctl.mark("x"))

    def test_two_step_stop(self):
        ctl = self.controller()
        self.assertIn("again", ctl.request_stop(50.0))
        self.assertTrue(ctl.status(51.0)["armed"])
        self.assertIn("again", ctl.request_stop(56.0))             # too late: re-armed, not stopped
        self.assertIsNotNone(ctl.recorder)
        self.assertIn("Stopped", ctl.request_stop(57.0))
        self.assertIsNone(ctl.recorder)


class RecordingCallbacksTest(unittest.TestCase):
    def test_start_stop_recording_from_the_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = fake_live_source()
            eng = V.Engine(src, 1.0, 0.4, 0.1, classify=False)
            eng.start()
            try:
                ctl = V.RecordingController(src, root=tmp)
                app = V.create_app(src, eng, ctl, mock_args(["--sparse"]))
                cl = Client(app)

                st = cl.status()
                self.assertTrue(st["rec-stop"]["disabled"])
                self.assertFalse(st["rec-start"]["disabled"])
                self.assertTrue(st["activity-btn"]["disabled"])
                self.assertIn("Choose a scenario", cl.action("rec-start"))
                self.assertEqual(list(Path(tmp).iterdir()), [])

                msg = cl.action("rec-start", scenario="walk", people="raghav", room="lab", notes="ui test")
                self.assertIn("Recording to", msg)
                sessions = list(Path(tmp).iterdir())
                self.assertEqual(len(sessions), 1)
                meta = json.loads((sessions[0] / "meta.json").read_text())
                self.assertEqual((meta["scenario"], meta["people"], meta["room"]), ("walk", "raghav", "lab"))
                self.assertTrue(meta["notes"].startswith("ui test | rig: sides S1-S2 2.540 m"), meta["notes"])
                self.assertIn("sensor heights S1 1.37 m", meta["notes"])
                self.assertEqual(len(meta["sensor_layout"]), 3)

                st = cl.status()
                self.assertTrue(st["rec-start"]["disabled"])
                self.assertTrue(st["rec-scenario"]["disabled"])
                self.assertFalse(st["activity-btn"]["disabled"])
                self.assertIn("REC", st["rec-badge"]["children"])

                self.assertIn("go", cl.action("activity-btn", n=1))
                self.assertIn("Ignored", cl.action("activity-btn", n=2))
                st = cl.status()
                self.assertTrue(st["activity-btn"]["children"].startswith("End activity"))
                time.sleep(1.1)
                self.assertIn("done", cl.action("activity-btn", n=3))
                self.assertIn("Ignored", cl.action("activity-btn", n=4))   # DONE twice -> once
                self.assertIn("Use the activity button", cl.action("mark-event", text="done"))
                self.assertIn("Marked", cl.action("mark-event", text="chair moved"))

                time.sleep(0.5)                       # let frames flow to the recorder
                st = cl.status()
                frames_shown = int(st["rec-info"]["children"].split("|")[1].split()[0])
                self.assertGreater(frames_shown, 0)
                labels = [m["props"]["children"].split(" ", 1)[1] for m in st["last-marks"]["children"]]
                self.assertEqual(labels, ["go", "done", "chair moved"])
                classes = [m["props"]["className"] for m in st["last-marks"]["children"]]
                self.assertEqual(classes, ["mark mark-go", "mark mark-done", "mark mark-other"])

                self.assertIn("again", cl.action("rec-stop", n=1))
                self.assertEqual(cl.status()["rec-stop"]["children"], "Confirm stop")
                self.assertIsNotNone(ctl.recorder)
                self.assertIn("Stopped", cl.action("rec-stop", n=2))
                self.assertIsNone(ctl.recorder)
                self.assertIsNone(src.recorder)
                n_after_stop = src.recorded_frames
                time.sleep(0.3)
                self.assertEqual(src.recorded_frames, n_after_stop)   # nothing reaches a closed recorder

                meta, frames, events = load_session(sessions[0])
                self.assertIn("end_time", meta)
                self.assertGreater(meta["frame_count"], 0)
                self.assertEqual(len(frames), meta["frame_count"])
                self.assertEqual([e[1] for e in events], ["go", "done", "chair moved"])
                self.assertEqual([e[1] for e in events if e[1] in ("go", "done")], ["go", "done"])
                st = cl.status()
                self.assertFalse(st["rec-start"]["disabled"])
                self.assertIn("last session", st["rec-info"]["children"])
            finally:
                eng.halt.set()
                eng.join(2)

    def test_mock_mode_cannot_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = V.MockSource()
            eng = V.Engine(src, 30.0, 0.4, 0.1, classify=False)
            ctl = V.RecordingController(src, root=tmp)
            self.assertFalse(ctl.can_record)
            app = V.create_app(src, eng, ctl, mock_args())
            cl = Client(app)
            layout = cl.c.get("/_dash-layout").data.decode()
            self.assertIn("Recording needs live sensors", layout)
            st = cl.status()
            for cid in ("rec-start", "rec-stop", "rec-scenario", "rec-people", "activity-btn", "mark-event"):
                self.assertTrue(st[cid]["disabled"], cid)
            self.assertIn("live sensors", cl.action("rec-start", scenario="walk"))
            self.assertEqual(list(Path(tmp).iterdir()), [])


def settings_call(cl, field, value, preset=None):
    """Change one Detection control through the real settings callback."""
    return cl.call("cfg-msg.children", {field: value}, {"cfg-preset.value": preset})


class SettingsTest(unittest.TestCase):
    def test_config_validation_and_labels(self):
        cfg = V.EngineConfig(mode="sparse", window_s=0.5, min_sensors=2)
        self.assertEqual(cfg.update(window_s=5, eps="x", mode="bogus", min_sensors=9),
                         {"window_s": 2.0, "min_sensors": 3})
        self.assertEqual(cfg.update(window_s=2.0), {})              # unchanged -> nothing to log
        self.assertEqual(V.settings_label({"window_s": 0.5, "min_sensors": 2}),
                         "settings window_s=0.5 min_sensors=2")
        self.assertEqual(V.settings_label({"classify": False, "mode": "sparse"}),
                         "settings mode=sparse classifier=off")
        self.assertEqual(V.settings_label({"show_s": 1.0}), "")      # display only: not logged

    def test_settings_change_alters_detections_in_the_sparse_mock(self):
        """Through the callback, on a fake clock: min sensors 2 -> 3 at tick 300, then window
        0.5 -> 0 at tick 600. Each change holds from the very next tick."""
        clock = types.SimpleNamespace(t=8000.0)
        real_time = V.time
        V.time = types.SimpleNamespace(time=lambda: clock.t, strftime=real_time.strftime,
                                       localtime=real_time.localtime)
        try:
            mock = V.MockSource(n_people=2)
            mock.rng = np.random.default_rng(11)
            log = []                                 # (tick, gated sensor counts, n dets)

            class Replay:
                live, sensors, label = False, mock.sensors, "test"

                def background_points(self):
                    return mock.background_points()

                def next_frame(self):
                    k = len(log)
                    if eng.state.get("history") is not None:
                        log.append((k, [g[3] for g in eng.state["gated"]], len(eng.state["dets"])))
                    if k == 300:
                        settings_call(cl, "cfg-minsens.value", 3)
                    if k == 600:
                        settings_call(cl, "cfg-window.value", 0.0)
                    if k >= 900:
                        eng.halt.set()
                    clock.t += 0.1
                    return mock.sparse_frame(clock.t)

            args = mock_args(["--sparse"])
            eng = V.Engine(Replay(), 30.0, 0.4, 0.1, config=V.EngineConfig.from_args(
                V.build_parser().parse_args(["--sparse", "--no-classify"])))
            cl = Client(V.create_app(Replay(), eng, V.RecordingController(Replay()), args))
            eng.run()
        finally:
            V.time = real_time
        before = [g for k, gs, _ in log if 30 <= k <= 300 for g in gs]
        mid = [g for k, gs, _ in log if 301 < k <= 600 for g in gs]
        self.assertIn(2, before)                     # 2-sensor detections exist before
        self.assertTrue(mid and min(mid) >= 3)       # none from the first tick after the change
        dets_mid = np.mean([n for k, _, n in log if 301 < k <= 600])
        dets_after = np.mean([n for k, _, n in log if 601 < k])
        self.assertLess(dets_after, dets_mid / 5)    # no pooling: sparse points rarely cluster
        self.assertEqual(eng.config.snapshot()["window_s"], 0.0)

    def test_mode_switch_reset_and_presets(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = V.MockSource()
            args = mock_args()
            eng = V.Engine(src, 30.0, 0.4, 0.1, config=V.EngineConfig.from_args(args))
            cl = Client(V.create_app(src, eng, V.RecordingController(src), args, presets_dir=tmp))
            r = settings_call(cl, "cfg-mode.value", "sparse")
            self.assertEqual((r["cfg-mode"]["value"], r["cfg-window"]["value"], r["cfg-minsens"]["value"]),
                             ("sparse", 0.5, 2))
            self.assertEqual(r["show-tracks"]["value"], [])
            self.assertEqual(eng.config.snapshot()["mode"], "sparse")
            settings_call(cl, "cfg-eps.value", 0.8)
            save = cl.call("cfg-preset-msg.children", {"cfg-save.n_clicks": 1}, {"cfg-preset-name.value": "wide_eps"})
            self.assertIn("Saved", save["cfg-preset-msg"]["children"])
            self.assertEqual(json.loads((Path(tmp) / "wide_eps.json").read_text())["eps"], 0.8)
            bad = cl.call("cfg-preset-msg.children", {"cfg-save.n_clicks": 2}, {"cfg-preset-name.value": "../x"})
            self.assertIn("letters, digits", bad["cfg-preset-msg"]["children"])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["wide_eps.json"])

            r = settings_call(cl, "cfg-reset.n_clicks", 1)
            self.assertEqual(eng.config.snapshot(), eng.config.defaults)
            self.assertEqual((r["cfg-mode"]["value"], r["cfg-eps"]["value"], r["show-tracks"]["value"]),
                             ("classic", 0.4, ["on"]))
            r = settings_call(cl, "cfg-load.n_clicks", 1, preset="wide_eps")
            self.assertEqual((eng.config.snapshot()["mode"], eng.config.snapshot()["eps"]), ("sparse", 0.8))
            self.assertIn("loaded", r["cfg-msg"]["children"])
            (Path(tmp) / "broken.json").write_text("{not json")
            r = settings_call(cl, "cfg-load.n_clicks", 2, preset="broken")
            self.assertIn("Not loaded", r["cfg-msg"]["children"])
            self.assertEqual(eng.config.snapshot()["eps"], 0.8)       # unchanged by a bad preset

    def test_settings_changes_are_marked_while_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = fake_live_source()
            args = mock_args(["--sparse"])
            eng = V.Engine(src, 1.0, 0.4, 0.1, config=V.EngineConfig.from_args(args))
            ctl = V.RecordingController(src, root=tmp)
            cl = Client(V.create_app(src, eng, ctl, args, presets_dir=tmp))
            r = settings_call(cl, "cfg-window.value", 1.0)            # not recording: no mark
            self.assertNotIn("marked", r["cfg-msg"]["children"])
            cl.action("rec-start", scenario="walk", people="raghav", room="lab")
            r = settings_call(cl, "cfg-window.value", 0.5)
            self.assertIn("(marked)", r["cfg-msg"]["children"])
            settings_call(cl, "cfg-minsens.value", 3)
            settings_call(cl, "cfg-classify.value", ["on"])               # already on: no change, no mark
            settings_call(cl, "cfg-classify.value", [])
            settings_call(cl, "show-s.value", 1.0)                    # display only: not marked
            settings_call(cl, "cfg-mode.value", "classic")
            path = ctl.recorder.path
            ctl.stop()
            _, _, events = load_session(path)
            self.assertEqual([e[1] for e in events],
                             ["settings window_s=0.5", "settings min_sensors=3", "settings classifier=off",
                              "settings mode=classic window_s=0 min_sensors=1"])

    def test_learning_progress(self):
        src = fake_live_source()
        args = mock_args()
        eng = V.Engine(src, 0.6, 0.4, 0.1, config=V.EngineConfig.from_args(args))
        cl = Client(V.create_app(src, eng, V.RecordingController(src), args))
        eng.start()
        try:
            time.sleep(0.2)
            r = cl.call("learn-bar.style", {"tick.n_intervals": 1})
            self.assertNotEqual(r["learn-bar"]["style"]["width"], "100%")
            self.assertIn("learning", r["learn-text"]["children"])
            time.sleep(1.0)
            r = cl.call("learn-bar.style", {"tick.n_intervals": 2})
            self.assertEqual(r["learn-bar"]["style"]["width"], "100%")
            self.assertIn("background learned", r["learn-text"]["children"])
        finally:
            eng.halt.set()
            eng.join(2)


class ProductBasicsTest(unittest.TestCase):
    def test_local_by_default_lan_warns(self):
        src = V.MockSource()
        args = mock_args()
        self.assertEqual(V.server_host(args), "127.0.0.1")
        self.assertIn("Local only", V.startup_banner(args, src))
        lan = mock_args(["--lan"])
        self.assertEqual(V.server_host(lan), "0.0.0.0")
        banner = V.startup_banner(lan, src)
        self.assertIn("WARNING", banner)
        self.assertIn("no authentication", banner)

    def test_alert_strip_for_a_silent_sensor(self):
        src = fake_live_source()
        args = mock_args()
        eng = V.Engine(src, 30.0, 0.4, 0.1, config=V.EngineConfig.from_args(args))
        cl = Client(V.create_app(src, eng, V.RecordingController(src), args))
        now = time.time()
        eng.health.t_start = now - 10
        eng.health.add([(0, now, 5), (1, now, 5), (2, now - 5.0, 5)])
        r = cl.call("alert-strip.children", {"slow-tick.n_intervals": 1})
        self.assertEqual(r["alert-strip"]["className"], "alert-strip")
        text = r["alert-strip"]["children"]
        self.assertIn("S3: silent for 5 s", text)
        self.assertIn(time.strftime("%H:%M:%S", time.localtime(now - 5.0)), text)
        self.assertNotIn("S1", text)
        eng.health.add([(2, time.time(), 5)])                   # S3 is back
        r = cl.call("alert-strip.children", {"slow-tick.n_intervals": 2})
        self.assertEqual(r["alert-strip"]["className"], "alert-strip hidden")

        msrc = V.MockSource()                                   # mock: never an alert
        meng = V.Engine(msrc, 30.0, 0.4, 0.1)
        mcl = Client(V.create_app(msrc, meng, V.RecordingController(msrc), args))
        r = mcl.call("alert-strip.children", {"slow-tick.n_intervals": 1})
        self.assertEqual(r["alert-strip"]["className"], "alert-strip hidden")

    def test_title_and_favicon(self):
        src = V.MockSource()
        app = V.create_app(src, V.Engine(src, 30.0, 0.4, 0.1), V.RecordingController(src), mock_args())
        c = app.server.test_client()
        html_page = c.get("/").data.decode()
        self.assertIn("<title>Radar Array</title>", html_page)
        self.assertIn("assets/favicon.ico", html_page)
        icon = c.get("/assets/favicon.ico")
        try:
            self.assertEqual(icon.status_code, 200)
            self.assertEqual(icon.data[:4], b"\x00\x00\x01\x00")   # ICO header
        finally:
            icon.close()

    def test_shutdown_closes_the_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = fake_live_source()
            src.close = lambda: setattr(src, "closed", True)
            eng = V.Engine(src, 0.5, 0.4, 0.1)
            ctl = V.RecordingController(src, root=tmp)
            ctl.start("walk", "raghav", "lab", "")
            eng.start()
            time.sleep(0.4)
            ctl.toggle_activity(time.time())                    # activity still open at shutdown
            V.shutdown(eng, ctl, src)
            self.assertFalse(eng.is_alive())
            self.assertIsNone(ctl.recorder)
            self.assertTrue(getattr(src, "closed", False))
            meta, frames, events = load_session(ctl.last_path)
            self.assertIn("end_time", meta)
            self.assertEqual(len(frames), meta["frame_count"])
            self.assertGreater(meta["frame_count"], 0)
            self.assertEqual([e[1] for e in events], ["go", "done"])


class MockModeUnchangedTest(unittest.TestCase):
    def test_dense_mock_still_tracks_two_people(self):
        """Dense mock on a fake clock (fixed times, seeded): the classic pipeline keeps 2 tracks."""
        clock = types.SimpleNamespace(t=7000.0)
        real_time = V.time
        V.time = types.SimpleNamespace(time=lambda: clock.t, strftime=real_time.strftime,
                                       localtime=real_time.localtime)
        try:
            mock = V.MockSource(dense=True)
            mock.rng = np.random.default_rng(3)
            counts = []

            class Replay:                         # mock frames, advancing the fake clock 0.1 s
                live, sensors, label = False, mock.sensors, "test"

                def background_points(self):
                    return mock.background_points()

                def next_frame(self):
                    if eng.state.get("history") is not None:
                        counts.append(len(eng.state["tracks"]))
                    if len(counts) >= 300:
                        eng.halt.set()
                    clock.t += 0.1
                    return mock.next_frame()

            eng = V.Engine(Replay(), 30.0, 0.4, 0.1, classify=True)
            eng.run()                             # in this thread; stops itself after 300 ticks
        finally:
            V.time = real_time
        self.assertGreaterEqual(float(np.mean(counts[20:])), 1.9)

    def test_all_callbacks_answer_in_mock_modes(self):
        for argv, dense in (([], False), (["--mock-dense"], True), (["--sparse"], False)):
            src = V.MockSource(dense=dense)
            eng = V.Engine(src, 30.0, 0.4, 0.1, classify=False, sparse="--sparse" in argv)
            eng.start()
            try:
                time.sleep(0.6)
                app = V.create_app(src, eng, V.RecordingController(src, root="/nonexistent"), mock_args(argv))
                cl = Client(app)
                self.assertEqual(cl.c.get("/").status_code, 200)
                self.assertEqual(cl.c.get("/_dash-layout").status_code, 200)
                st = {"follow.value": "free", "zoom.value": 2.5, "show-s.value": 0.1, "view-mode.value": "both",
                      "cam-top.n_clicks": 0, "cam-iso.n_clicks": 0,
                      "show-tracks.value": [] if "--sparse" in argv else ["on"]}
                self.assertIsNotNone(cl.call("floor-plot.figure", {"tick.n_intervals": 30}, st))
                self.assertIsNotNone(cl.call("presence-card.className", {"tick.n_intervals": 3}))
                self.assertIsNotNone(cl.call("clock.children", {"slow-tick.n_intervals": 3}))
                self.assertIsNotNone(cl.call("floor-wrap.className", {"view-mode.value": "floor"}))
                self.assertIsNotNone(cl.status())
            finally:
                eng.halt.set()
                eng.join(2)


if __name__ == "__main__":
    unittest.main()

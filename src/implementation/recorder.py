"""Crash-safe incremental recording for radar sessions."""

import csv
import datetime
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


class SessionRecorder:
    """Buffer raw radar frames and write compressed NPZ chunks."""

    def __init__(self, root="data", room="", scenario="", people="", notes="",
                 cfg_path="", sensors_json_path="", sensors=None):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_scenario = "".join(c if c.isalnum() or c in "-_" else "_" for c in scenario)
        name = stamp + (("_" + safe_scenario) if safe_scenario else "")
        self.path = Path(root) / name
        self.path.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._threads = []
        self._buffer = []
        self._frame_count = 0
        self._frames_by_sensor = {}
        self._chunk = 0
        self._last_flush = time.monotonic()
        self.start_epoch = time.time()
        self.start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        cfg_text = ""
        cfg_file = ""
        if cfg_path:
            p = Path(cfg_path)
            cfg_file = p.name
            try:
                cfg_text = p.read_text()
            except OSError:
                cfg_text = ""

        sensors_copy = None
        if sensors_json_path:
            try:
                sensors_copy = json.loads(Path(sensors_json_path).read_text())
            except (OSError, json.JSONDecodeError):
                sensors_copy = None
        if sensors_copy is None and sensors is not None:
            sensors_copy = sensors if isinstance(sensors, (dict, list)) else None

        sensor_layout = []
        sensor_source = sensors_copy.get("sensors", []) if isinstance(sensors_copy, dict) else (
            sensors_copy if isinstance(sensors_copy, list) else [])
        if sensors is not None and isinstance(sensors, (list, tuple)):
            sensor_source = sensors
        for sensor in sensor_source:
            if not isinstance(sensor, dict):
                continue
            name = sensor.get("name", "")
            T = sensor.get("T")
            if T is None and isinstance(sensor.get("pose"), dict):
                T = sensor["pose"]
            if hasattr(T, "tolist"):
                T = T.tolist()
            sensor_layout.append({"name": name, "T": T})

        git_commit = None
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            git_commit = None

        self.meta = {
            "root": str(root),
            "room": room,
            "scenario": scenario,
            "people": people,
            "notes": notes,
            "cfg_path": cfg_path,
            "sensors_json_path": sensors_json_path,
            "sensors": sensors_copy,
            "sensor_layout": sensor_layout,
            "cfg_file": cfg_file,
            "cfg_text": cfg_text,
            "start_time": self.start_iso,
            "start_epoch": self.start_epoch,
            "git_commit": git_commit,
        }
        self._write_meta()

    def _write_meta(self):
        """Write metadata atomically enough to survive an interrupted session."""
        tmp = self.path / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2))
        tmp.replace(self.path / "meta.json")

    def add(self, sensor_idx, fr):
        """Append one raw RadarLink frame without filtering or transforming it."""
        pts = np.asarray(fr.get("pts", np.empty((0, 4))), dtype=np.float32)
        snr = fr.get("snr")
        noise = fr.get("noise")
        n = len(pts)
        snr_values = np.full(n, np.nan, dtype=np.float32)
        noise_values = np.full(n, np.nan, dtype=np.float32)
        if snr is not None:
            snr_values[:] = np.asarray(snr, dtype=np.float32).reshape(-1)[:n]
        if noise is not None:
            noise_values[:] = np.asarray(noise, dtype=np.float32).reshape(-1)[:n]
        combined = np.zeros((n, 6), dtype=np.float32)
        if n:
            combined[:, :4] = pts[:, :4]
            combined[:, 4] = snr_values
            combined[:, 5] = noise_values

        flush_now = False
        with self._lock:
            if self._closed:
                return
            self._buffer.append({
                "t": np.float64(fr["t"]),
                "sensor": np.int32(sensor_idx),
                "frame": np.int64(fr["frame"]),
                "pts": combined,
            })
            self._frame_count += 1
            self._frames_by_sensor[sensor_idx] = self._frames_by_sensor.get(sensor_idx, 0) + 1
            flush_now = len(self._buffer) >= 500 or (
                time.monotonic() - self._last_flush >= 10.0)
        if flush_now:
            self._flush_async()

    def _snapshot(self):
        """Remove and return the current buffered frames."""
        with self._lock:
            if not self._buffer:
                return None
            snapshot = self._buffer
            self._buffer = []
            self._last_flush = time.monotonic()
            return snapshot

    def _flush_async(self):
        """Flush a snapshot in a worker so the viewer thread is not blocked."""
        snapshot = self._snapshot()
        if snapshot is None:
            return
        thread = threading.Thread(target=self._write_chunk, args=(snapshot,), daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def _write_chunk(self, frames):
        """Write one ordered chunk using its own sequence number."""
        with self._flush_lock:
            with self._lock:
                self._chunk += 1
                chunk_number = self._chunk
            t = np.asarray([f["t"] for f in frames], dtype=np.float64)
            sensor = np.asarray([f["sensor"] for f in frames], dtype=np.int32)
            frame = np.asarray([f["frame"] for f in frames], dtype=np.int64)
            counts = np.asarray([len(f["pts"]) for f in frames], dtype=np.int32)
            pts = np.vstack([f["pts"] for f in frames]) if frames else np.empty((0, 6), np.float32)
            filename = self.path / ("chunk_%04d.npz" % chunk_number)
            np.savez_compressed(filename, t=t, sensor=sensor, frame=frame, counts=counts, pts=pts)

    def mark(self, label):
        """Append a timestamped event label."""
        epoch = time.time()
        with self._lock:
            if self._closed:
                return
            with open(self.path / "events.csv", "a", newline="") as f:
                csv.writer(f).writerow([epoch, str(label)])

    def close(self):
        """Flush remaining frames and finish metadata; safe to call twice."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        snapshot = self._snapshot()
        if snapshot:
            thread = threading.Thread(target=self._write_chunk, args=(snapshot,), daemon=True)
            thread.start()
            thread.join()
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join()
        self.meta["end_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.meta["end_epoch"] = time.time()
        self.meta["frame_count"] = self._frame_count
        self.meta["frames_by_sensor"] = {str(k): v for k, v in self._frames_by_sensor.items()}
        self._write_meta()

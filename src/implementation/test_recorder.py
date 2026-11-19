"""Hardware-free tests for SessionRecorder and load_session."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from load_session import load_session
from recorder import SessionRecorder


class SessionRecorderTest(unittest.TestCase):
    def test_session_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "test.cfg"
            cfg_text = "% test config\nsensorStop\nsensorStart\n"
            cfg_path.write_text(cfg_text)
            sensors_path = tmp_path / "sensors.json"
            sensors = {
                "sensors": [
                    {"name": "S1", "T": [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]},
                    {"name": "S2", "T": [[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]]},
                    {"name": "S3", "T": [[1, 0, 0, 7], [0, 1, 0, 8], [0, 0, 1, 9], [0, 0, 0, 1]]},
                ]
            }
            sensors_path.write_text(json.dumps(sensors))

            recorder = SessionRecorder(
                root=str(tmp_path / "data"), room="lab", scenario="walk",
                people="2", notes="round trip", cfg_path=str(cfg_path),
                sensors_json_path=str(sensors_path), sensors=sensors)
            expected_points = {}
            for n in range(1200):
                sensor_idx = n % 3
                count = (n * 17) % 61
                pts = np.zeros((count, 4), dtype=np.float32)
                pts[:, 0] = np.arange(count, dtype=np.float32) + n
                pts[:, 1] = np.arange(count, dtype=np.float32) - n
                pts[:, 2] = np.arange(count, dtype=np.float32) * 0.01
                pts[:, 3] = np.linspace(-2.0, 2.0, count, dtype=np.float32) if count else []
                snr = None
                noise = None
                if n % 5 != 0:
                    snr = (np.arange(count, dtype=np.float32) + 10.0)
                    noise = (np.arange(count, dtype=np.float32) + 3.0)
                fr = {
                    "t": 1000.0 + n * 0.01,
                    "frame": n // 3 + 1,
                    "pts": pts,
                    "snr": snr,
                    "noise": noise,
                }
                expected_points[(n, sensor_idx)] = (pts.copy(), snr, noise)
                recorder.add(sensor_idx, fr)
            recorder.mark("start")
            recorder.mark("finish")
            recorder.close()
            recorder.close()

            meta, frames, events = load_session(recorder.path)
            self.assertEqual(len(frames), 1200)
            self.assertEqual(meta["cfg_text"], cfg_text)
            self.assertEqual(meta["cfg_file"], "test.cfg")
            self.assertEqual(meta["frame_count"], 1200)
            self.assertEqual(meta["frames_by_sensor"], {"0": 400, "1": 400, "2": 400})
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0][1], "start")
            self.assertEqual(events[1][1], "finish")

            chunk_files = sorted(recorder.path.glob("chunk_*.npz"))
            self.assertGreaterEqual(len(chunk_files), 3)
            total_chunk_frames = 0
            for chunk_file in chunk_files:
                with np.load(chunk_file, allow_pickle=False) as chunk:
                    self.assertEqual(set(chunk.files), {"t", "sensor", "frame", "counts", "pts"})
                    self.assertEqual(chunk["pts"].shape[1], 6)
                    total_chunk_frames += len(chunk["counts"])
            self.assertEqual(total_chunk_frames, 1200)

            for fr in frames:
                n = int(round((fr["t"] - 1000.0) / 0.01))
                sensor_idx = fr["sensor"]
                pts, snr, noise = expected_points[(n, sensor_idx)]
                np.testing.assert_array_equal(fr["pts"][:, :4], pts)
                if snr is None:
                    self.assertTrue(np.isnan(fr["snr"]).all())
                    self.assertTrue(np.isnan(fr["noise"]).all())
                else:
                    np.testing.assert_array_equal(fr["snr"], snr)
                    np.testing.assert_array_equal(fr["noise"], noise)


if __name__ == "__main__":
    unittest.main()

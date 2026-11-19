"""Load and summarize crash-safe radar session recordings."""

import csv
import glob
import json
from pathlib import Path

import numpy as np


def load_session(path):
    """Load chunked session data as one frame list and event list."""
    session = Path(path)
    meta = json.loads((session / "meta.json").read_text())
    frames = []
    for chunk_path in sorted(glob.glob(str(session / "chunk_*.npz"))):
        with np.load(chunk_path, allow_pickle=False) as chunk:
            t = chunk["t"]
            sensor = chunk["sensor"]
            frame_numbers = chunk["frame"]
            counts = chunk["counts"]
            pts = chunk["pts"]
            offset = 0
            for i, count in enumerate(counts):
                n = int(count)
                point_array = pts[offset:offset + n].astype(np.float32, copy=True)
                offset += n
                snr = point_array[:, 4].copy() if point_array.shape[1] >= 5 else np.full(n, np.nan)
                noise = point_array[:, 5].copy() if point_array.shape[1] >= 6 else np.full(n, np.nan)
                frames.append({
                    "t": float(t[i]),
                    "sensor": int(sensor[i]),
                    "frame": int(frame_numbers[i]),
                    "pts": point_array,
                    "snr": snr,
                    "noise": noise,
                })
    frames.sort(key=lambda fr: fr["t"])

    events = []
    events_path = session / "events.csv"
    if events_path.exists():
        with open(events_path, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    events.append([float(row[0]), row[1]])
    return meta, frames, events


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    meta, frames, events = load_session(args.path)

    sensor_counts = {}
    frame_numbers = {}
    all_points = []
    for fr in frames:
        sensor_counts[fr["sensor"]] = sensor_counts.get(fr["sensor"], 0) + 1
        frame_numbers.setdefault(fr["sensor"], []).append(fr["frame"])
        all_points.append(len(fr["pts"]))
    dropped = 0
    for numbers in frame_numbers.values():
        if len(numbers) > 1:
            unique = sorted(set(numbers))
            dropped += sum(b - a - 1 for a, b in zip(unique, unique[1:]) if b > a + 1)

    times = [fr["t"] for fr in frames]
    duration = max(times) - min(times) if len(times) > 1 else 0.0
    mean_points = float(np.mean(all_points)) if all_points else 0.0
    max_doppler = 0.0
    snr_count = 0
    snr_total = 0
    dopplers = []
    for fr in frames:
        if len(fr["pts"]):
            dopplers.append(np.abs(fr["pts"][:, 3]))
        snr_count += int(np.count_nonzero(~np.isnan(fr["snr"])))
        snr_total += len(fr["snr"])
    if dopplers:
        max_doppler = float(max(np.max(v) for v in dopplers))
    snr_fraction = snr_count / snr_total if snr_total else 0.0

    print("Frames per sensor:", dict(sorted(sensor_counts.items())))
    print("Mean points per frame:", mean_points)
    print("Duration:", duration)
    print("Dropped frames:", dropped)
    print("Max |doppler|:", max_doppler)
    print("Non-NaN SNR fraction:", snr_fraction)


if __name__ == "__main__":
    main()

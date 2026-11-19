#!/usr/bin/env python3
"""Guided end-to-end hardware check for one radar sensor."""

import argparse
import re
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

from radar_core import RadarLink, load_sensors


PHASES = (
    ("A", "EMPTY", "Leave the sensor's view."),
    ("B", "STANDING", "Stand still about 2 m directly in front of the sensor."),
    ("C", "WALKING", "Walk toward and away from the sensor at normal speed."),
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a guided one-sensor hardware check.")
    parser.add_argument("--sensors", default="sensors.json", help="Sensor JSON file")
    parser.add_argument("--only", required=True, help="Sensor name")
    parser.add_argument("--cfg", default=None, help="Radar config file")
    parser.add_argument("--phase-s", dest="phase_s", type=float, default=15.0)
    parser.add_argument("--verbose", action="store_true", help="Echo every config line")
    return parser.parse_args()


def range_points(frame, minimum=0.3, maximum=9.0):
    """Return points in the requested range and their Euclidean ranges."""
    pts = frame["pts"]
    if len(pts) == 0:
        return pts, np.empty(0, dtype=np.float64)
    ranges = np.linalg.norm(pts[:, :3].astype(np.float64), axis=1)
    keep = (ranges >= minimum) & (ranges <= maximum)
    return pts[keep], ranges[keep]


def phase_stats(frames):
    """Compute frame, range, SNR, and point statistics for a phase."""
    if not frames:
        return {"frames": 0, "fps": 0.0, "mean": 0.0, "max": 0.0,
                "snr_fraction": 0.0, "snr_median": np.nan, "snr_mismatch": 0,
                "elapsed": 0.0}
    counts = []
    snr_values = []
    snr_mismatch = 0
    elapsed = 0.0
    for frame in frames:
        elapsed = max(elapsed, float(frame.get("_phase_elapsed", 0.0)))
        raw_pts = frame["pts"]
        n = len(raw_pts)
        range_pts, _ = range_points(frame)
        counts.append(len(range_pts))
        if n:
            raw_ranges = np.linalg.norm(
                raw_pts[:, :3].astype(np.float64), axis=1)
            range_keep = (raw_ranges >= 0.3) & (raw_ranges <= 9.0)
        else:
            range_keep = np.zeros(0, dtype=bool)
        snr = frame["snr"]
        if snr is None:
            continue
        snr = np.asarray(snr).reshape(-1)
        if len(snr) != n:
            snr_mismatch += 1
            continue
        snr_values.extend(snr[range_keep].tolist())
        elapsed = max(elapsed, float(frame.get("_phase_elapsed", 0.0)))
    all_snr = np.asarray(snr_values, dtype=np.float64)
    all_snr = all_snr[np.isfinite(all_snr)]
    total_points = sum(counts)
    fraction = len(all_snr) / float(total_points) if total_points else 0.0
    median = float(np.median(all_snr)) if len(all_snr) else np.nan
    fps = len(frames) / elapsed if elapsed > 0 else 0.0
    return {"frames": len(frames), "fps": fps, "mean": float(np.mean(counts)),
            "max": float(np.max(counts)), "snr_fraction": fraction,
            "snr_median": median, "snr_mismatch": snr_mismatch,
            "elapsed": elapsed}


def empty_background(frames):
    """Find 0.15 m voxels occupied in at least 20 percent of empty frames."""
    if not frames:
        return set()
    voxel_counts = {}
    for frame in frames:
        pts, _ = range_points(frame)
        if len(pts):
            voxels = np.floor(pts[:, :3].astype(np.float64) / 0.15).astype(np.int64)
            for voxel in set(map(tuple, voxels)):
                voxel_counts[voxel] = voxel_counts.get(voxel, 0) + 1
    threshold = 0.20 * len(frames)
    return {voxel for voxel, count in voxel_counts.items() if count >= threshold}


def foreground_stats(frames, background):
    """Calculate foreground ranges and points per standing frame."""
    ranges = []
    counts = []
    for frame in frames:
        pts, point_ranges = range_points(frame)
        if len(pts):
            voxels = np.floor(pts[:, :3].astype(np.float64) / 0.15).astype(np.int64)
            foreground = np.asarray([tuple(v) not in background for v in voxels])
        else:
            foreground = np.zeros(0, dtype=bool)
        ranges.extend(point_ranges[foreground].tolist())
        counts.append(int(np.count_nonzero(foreground)))
    median_range = float(np.median(ranges)) if ranges else np.nan
    mean_foreground = float(np.mean(counts)) if counts else 0.0
    return median_range, mean_foreground


def walking_stats(frames):
    """Calculate Doppler extrema and distinct rounded Doppler values."""
    abs_doppler = []
    doppler = []
    for frame in frames:
        pts, _ = range_points(frame)
        if len(pts):
            doppler.extend(pts[:, 3].astype(np.float64).tolist())
            abs_doppler.extend(np.abs(pts[:, 3]).astype(np.float64).tolist())
    if not doppler:
        return 0.0, 0.0, []
    unique = sorted(set(round(float(value), 1) for value in doppler))
    absolute = np.asarray(abs_doppler, dtype=np.float64)
    return float(np.max(absolute)), float(np.percentile(absolute, 95)), unique[:12]


def cfg_max_velocity(path):
    """Read maximum radial velocity from the config header, if present."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as cfg:
            for line in cfg:
                match = re.search(r"Maximum Radial Velocity\(m/s\):([0-9.]+)", line)
                if match:
                    return float(match.group(1))
    except OSError:
        pass
    return None


def collect_phase(link, letter, name, instruction, phase_s):
    """Print a countdown and collect one labeled phase."""
    del name
    print(f"\n{letter}: {instruction}")
    print("Starting in", flush=True)
    for remaining in range(5, 0, -1):
        print(f"  {remaining}...")
        time.sleep(1)
    link.drain()
    print(f"{letter} collecting for {phase_s:g} seconds...")
    collected = []
    collection_start = time.monotonic()
    deadline = collection_start + phase_s
    while time.monotonic() < deadline:
        for frame in link.drain():
            frame["phase"] = letter
            collected.append(frame)
        time.sleep(0.02)
    elapsed = time.monotonic() - collection_start
    for frame in collected:
        frame["_phase_elapsed"] = elapsed
    print(f"{letter}: received {len(collected)} frames")
    return collected


def save_frames(sensor_name, phases):
    """Save concatenated raw frames to a timestamped NPZ file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"check_{sensor_name}_{timestamp}.npz"
    times = []
    labels = []
    counts = []
    point_parts = []
    snr_parts = []
    for letter, frames in phases.items():
        for frame in frames:
            times.append(frame["t"])
            labels.append(letter)
            pts = frame["pts"]
            snr = frame["snr"]
            counts.append(len(pts))
            point_parts.append(pts)
            if snr is None:
                snr_parts.append(np.full(len(pts), np.nan, dtype=np.float32))
            else:
                snr_parts.append(np.asarray(snr, dtype=np.float32).reshape(-1))
    points = np.concatenate(point_parts) if point_parts and sum(counts) else np.empty((0, 4), np.float32)
    snrs = np.concatenate(snr_parts) if snr_parts and sum(counts) else np.empty(0, dtype=np.float32)
    np.savez(filename, t=np.asarray(times, dtype=np.float64),
             phase=np.asarray(labels), counts=np.asarray(counts, dtype=np.int64),
             pts=points, snr=snrs)
    print(f"Saved raw frames to {filename}")
    return filename


def print_report(stats, foreground_range, foreground_mean, walking):
    """Print the report and PASS/WARN/FAIL lines."""
    for letter in ("A", "B", "C"):
        item = stats[letter]
        median_text = "n/a" if np.isnan(item["snr_median"]) else f"{item['snr_median']:.1f}"
        print(f"{letter}: frames={item['frames']} fps={item['fps']:.1f} "
              f"mean_pts={item['mean']:.1f}")
        print(f"{letter}: snr_fraction={item['snr_fraction']:.3f} snr_median={median_text}")
        print(f"{letter}: snr_mismatch_frames={item['snr_mismatch']}")
    print(f"A: mean/max points per frame = {stats['A']['mean']:.1f}/{stats['A']['max']:.0f}")
    print(f"B: median foreground range = {foreground_range:.2f} m"
          if np.isfinite(foreground_range) else "B: median foreground range = n/a")
    print(f"B: mean foreground points per frame = {foreground_mean:.1f}")
    max_doppler, percentile95, distinct = walking
    print(f"C: max |Doppler| = {max_doppler:.2f} m/s")
    print(f"C: 95th percentile |Doppler| = {percentile95:.2f} m/s")
    print("C: distinct rounded Doppler values = " + ", ".join(str(value) for value in distinct))


def main():
    """Run the guided check."""
    args = parse_args()
    if args.phase_s <= 0:
        print("FAIL: --phase-s must be greater than zero", file=sys.stderr)
        return 2
    try:
        _, sensors = load_sensors(args.sensors)
    except Exception as exc:
        print(f"FAIL: could not load sensors: {exc}", file=sys.stderr)
        return 2
    matches = [sensor for sensor in sensors if sensor["name"] == args.only]
    if not matches:
        print(f"FAIL: sensor {args.only} was not found in {args.sensors}", file=sys.stderr)
        return 2
    sensor = matches[0]
    cfg_path = args.cfg or sensor["cfg"]

    link = None
    collection_start = time.time()
    phases = {"A": [], "B": [], "C": []}
    try:
        link = RadarLink(sensor["name"], sensor["cli"], sensor["data"], cfg_path,
                         verbose=args.verbose)
        try:
            link.configure()
            link.start()
        except Exception as exc:
            print(f"CONFIG ERROR: {exc}", file=sys.stderr)
            return 1

        collection_start = time.time()
        for letter, name, instruction in PHASES:
            frames = collect_phase(link, letter, name, instruction, args.phase_s)
            phases[letter] = frames
        first_five = [frame for frame in phases["A"]
                      if frame["t"] - collection_start <= 5.0]
    except KeyboardInterrupt:
        print("\nCtrl+C received; stopping collection.", file=sys.stderr)
        first_five = [frame for frame in phases["A"]
                      if frame["t"] - collection_start <= 5.0]
    finally:
        if link is not None:
            link.close()

    save_frames(args.only, phases)

    stats = {
        letter: {"frames": 0, "fps": 0.0, "mean": 0.0, "max": 0.0,
                 "snr_fraction": 0.0, "snr_median": np.nan, "snr_mismatch": 0,
                 "elapsed": 0.0}
        for letter in ("A", "B", "C")
    }
    try:
        stats = {letter: phase_stats(frames) for letter, frames in phases.items()}
        background = empty_background(phases["A"])
        foreground_range, foreground_mean = foreground_stats(phases["B"], background)
        walking = walking_stats(phases["C"])
        print_report(stats, foreground_range, foreground_mean, walking)

        print("\nStatus:")
        print("PASS: received frames" if first_five else
              "FAIL: no frames arrived in the first 5 s (check COM ports/SOP switches/that the TI visualizer is closed)")
        for letter in ("A", "B", "C"):
            fps = stats[letter]["fps"]
            if fps < 7 or fps > 12:
                print(f"FAIL: phase {letter} fps={fps:.1f} (expected 7-12)")
        if stats["A"]["snr_fraction"] < 0.5:
            print("WARN: SNR is missing for most points")
        if foreground_mean < 3.0 * stats["A"]["mean"]:
            print("WARN: phase B foreground points per frame is less than 3x phase A")
        if np.isfinite(foreground_range) and not 1.2 <= foreground_range <= 3.0:
            print("WARN: phase B median foreground range is outside 1.2 to 3.0 m")
        max_doppler = walking[0]
        if max_doppler < 1.0:
            print("WARN: phase C max |Doppler| < 1.0 m/s (person not walking fast enough or velocity range wrong)")
        max_velocity = cfg_max_velocity(cfg_path)
        if max_velocity is not None and 0.95 * max_velocity <= max_doppler <= 1.05 * max_velocity:
            print("WARN: phase C max |Doppler| is within 5% of cfg maximum velocity (likely aliasing)")
    except Exception:
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

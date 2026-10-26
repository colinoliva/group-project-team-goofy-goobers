#!/usr/bin/env python3
"""
radar_map.py - one IWR6843AOP -> live static 3D occupancy map in Open3D.
Keys in the viewer window:
    R = reset the map     S = save the map to a .ply file     P = pause/resume accumulation
Sensor frame (TI demo): x = right, y = forward (boresight), z = up. Units: meters.
World frame used here: same axes, origin on the floor directly below the sensor.
"""
import argparse
import datetime
import queue
import struct
import threading
import time

import numpy as np
import serial
import open3d as o3d

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HDR = struct.Struct("<8s8I")   # magic, version, totalLen, platform, frameNum,
                               # cpuCycles, numObj, numTLVs, subFrame  -> 40 bytes
TLV = struct.Struct("<2I")     # type, payload length
TLV_POINTS = 1                 # x, y, z, doppler (4 x float32 per point)
TLV_SIDE_INFO = 7              # snr, noise (2 x uint16 per point, 0.1 dB units)
MAX_FRAME_BYTES = 200_000
PROMPT = b"mmwDemo:/>"


# ----------------------------------------------------------------------------
# Radar configuration over the CLI port
# ----------------------------------------------------------------------------
def send_config(cli, path):
    cli.reset_input_buffer()
    cli.write(b"\n")
    cli.read_until(PROMPT)
    cli.write(b"sensorStop\n")          # harmless if already stopped
    cli.read_until(PROMPT)
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("%") or line == "sensorStop":
            continue
        cli.write((line + "\n").encode())
        resp = cli.read_until(PROMPT).decode(errors="ignore")
        short = " | ".join(s.strip() for s in resp.splitlines() if s.strip())
        print(f">> {line}\n   {short}")
        if "Error" in resp or "not recognized" in resp:
            raise RuntimeError(f"Radar rejected config line: {line}")
        if not resp.strip():
            print("   (no reply - check the CLI port and that the firmware is running)")


# ----------------------------------------------------------------------------
# Frame parser (runs in its own thread)
# ----------------------------------------------------------------------------
def reader(port, q, stop):
    buf = bytearray()
    while not stop.is_set():
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            buf += chunk
        while True:
            i = buf.find(MAGIC)
            if i < 0:
                if len(buf) > 7:
                    del buf[:-7]            # keep a tail in case the magic word is split
                break
            if i:
                del buf[:i]
            if len(buf) < HDR.size:
                break
            _, _, total, _, frame, _, n_obj, n_tlv, _ = HDR.unpack_from(buf)
            if total < HDR.size or total > MAX_FRAME_BYTES:
                del buf[:8]                 # false magic match, resync
                continue
            if len(buf) < total:
                break                       # wait for the rest of the frame

            pts = np.empty((0, 4), np.float32)
            snr = None
            off, ok = HDR.size, True
            for _ in range(n_tlv):
                if off + TLV.size > total:
                    ok = False
                    break
                t, ln = TLV.unpack_from(buf, off)
                off += TLV.size
                if off + ln > total:
                    ok = False
                    break
                payload = bytes(buf[off:off + ln])
                if t == TLV_POINTS and ln >= 16:
                    pts = np.frombuffer(payload[:ln // 16 * 16], np.float32).reshape(-1, 4).copy()
                elif t == TLV_SIDE_INFO and ln >= 4:
                    si = np.frombuffer(payload[:ln // 4 * 4], np.uint16).reshape(-1, 2)
                    snr = si[:, 0].astype(np.float32) * 0.1
                off += ln
            del buf[:total]
            if not ok:
                continue
            if snr is not None and len(snr) != len(pts):
                snr = None
            item = {"t": time.time(), "frame": frame, "pts": pts, "snr": snr}
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put(item)


# ----------------------------------------------------------------------------
# Mapping helpers
# ----------------------------------------------------------------------------
class VoxelMap:
    """Counts in how many frames each voxel was hit. Real surfaces get hit repeatedly."""

    def __init__(self, voxel):
        self.v = voxel
        self.hits = {}

    def add(self, xyz):
        if len(xyz) == 0:
            return
        keys = np.unique(np.floor(xyz / self.v).astype(np.int32), axis=0)
        h = self.hits
        for k in map(tuple, keys.tolist()):
            h[k] = h.get(k, 0) + 1

    def occupied(self, min_hits):
        ks = [k for k, c in self.hits.items() if c >= min_hits]
        if not ks:
            return np.empty((0, 3))
        return (np.array(ks, dtype=np.float64) + 0.5) * self.v

    def clear(self):
        self.hits.clear()


def make_to_world(height, pitch_deg):
    """Sensor frame -> world frame. pitch_deg > 0 means the sensor is tilted DOWN."""
    a = np.radians(-pitch_deg)
    R = np.array([[1, 0, 0],
                  [0, np.cos(a), -np.sin(a)],
                  [0, np.sin(a), np.cos(a)]])
    t = np.array([0.0, 0.0, height])
    return lambda p: p @ R.T + t


def height_colors(xyz, zmax=3.0):
    if len(xyz) == 0:
        return np.empty((0, 3))
    t = np.clip(xyz[:, 2] / zmax, 0, 1)
    return np.c_[t, 1 - np.abs(2 * t - 1), 1 - t]      # blue (low) -> green -> red (high)


def floor_grid(half=5.0, step=1.0):
    xs = np.arange(-half, half + 1e-6, step)
    ys = np.arange(0, 2 * half + 1e-6, step)
    pts, lines = [], []
    for x in xs:
        pts += [[x, 0, 0], [x, ys[-1], 0]]
        lines.append([len(pts) - 2, len(pts) - 1])
    for y in ys:
        pts += [[xs[0], y, 0], [xs[-1], y, 0]]
        lines.append([len(pts) - 2, len(pts) - 1])
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.array(pts, float)),
                              lines=o3d.utility.Vector2iVector(np.array(lines, np.int32)))
    ls.paint_uniform_color([0.25, 0.25, 0.3])
    return ls


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, help="CLI port, e.g. COM5")
    ap.add_argument("--data", required=True, help="Data port, e.g. COM6")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--voxel", type=float, default=0.10, help="voxel size in m")
    ap.add_argument("--min-hits", type=int, default=5, help="frames a voxel must be hit to display")
    ap.add_argument("--vthresh", type=float, default=0.05, help="max |Doppler| (m/s) to count as static")
    ap.add_argument("--min-range", type=float, default=0.3)
    ap.add_argument("--max-range", type=float, default=10.0)
    ap.add_argument("--min-snr", type=float, default=0.0, help="dB; 0 disables the SNR filter")
    ap.add_argument("--height", type=float, default=1.0, help="sensor height above floor, m")
    ap.add_argument("--pitch", type=float, default=0.0, help="degrees the sensor is tilted down")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    cli = serial.Serial(args.cli, 115200, timeout=1.0)
    data = serial.Serial(args.data, 921600, timeout=0.1)
    send_config(cli, args.cfg)          # the cfg's last line must be sensorStart

    q = queue.Queue(maxsize=64)
    stop = threading.Event()
    threading.Thread(target=reader, args=(data, q, stop), daemon=True).start()

    to_world = make_to_world(args.height, args.pitch)
    vmap = VoxelMap(args.voxel)
    state = {"paused": False, "reset": False, "save": False}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("IWR6843AOP map  [F fit all | R reset | S save | P pause]", 1280, 800)
    opt = vis.get_render_option()
    opt.point_size = 5.0
    opt.background_color = np.array([0.04, 0.04, 0.07])

    seed = np.array([[0, 0.5, 0.0], [0, 0.6, 0.0]])
    map_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seed))
    live_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seed))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
    axes.translate((0, 0, args.height))
    for g in (floor_grid(), axes, map_pcd, live_pcd):
        vis.add_geometry(g)

    def on_fit_all(_):
        m = vmap.occupied(args.min_hits)
        if len(m):
            map_pcd.points = o3d.utility.Vector3dVector(m)
            map_pcd.colors = o3d.utility.Vector3dVector(height_colors(m))
        vis.update_geometry(map_pcd)
        vis.update_geometry(live_pcd)
        vis.reset_view_point(True)
        return False

    def on_reset(_):
        state["reset"] = True
        return False

    def on_save(_):
        state["save"] = True
        return False

    def on_pause(_):
        state["paused"] = not state["paused"]
        print("Accumulation", "PAUSED" if state["paused"] else "resumed")
        return False

    vis.register_key_callback(ord("F"), on_fit_all)
    vis.register_key_callback(ord("R"), on_reset)
    vis.register_key_callback(ord("S"), on_save)
    vis.register_key_callback(ord("P"), on_pause)

    recorded = []
    n_frames, n_pts_total = 0, 0
    t_start = last_map = last_stat = time.time()
    warned = False

    try:
        while vis.poll_events():
            got = False
            while True:
                try:
                    fr = q.get_nowait()
                except queue.Empty:
                    break
                got = True
                n_frames += 1
                pts, snr = fr["pts"], fr["snr"]
                n_pts_total += len(pts)
                if not args.no_record:
                    recorded.append((fr["t"], fr["frame"], pts))
                if len(pts) == 0:
                    live_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    continue

                xyz = pts[:, :3].astype(np.float64)
                rng = np.linalg.norm(xyz, axis=1)
                keep = (rng >= args.min_range) & (rng <= args.max_range)
                if snr is not None and args.min_snr > 0:
                    keep &= snr >= args.min_snr
                static = keep & (np.abs(pts[:, 3]) <= args.vthresh)

                if not state["paused"]:
                    vmap.add(to_world(xyz[static]))

                live = to_world(xyz[keep])
                live_pcd.points = o3d.utility.Vector3dVector(live)
                live_pcd.paint_uniform_color([1.0, 0.6, 0.1])      # orange = current frame
            if got:
                vis.update_geometry(live_pcd)

            if state["reset"]:
                vmap.clear()
                state["reset"] = False
                print("Map reset")
            if state["save"]:
                fn = datetime.datetime.now().strftime("map_%Y%m%d_%H%M%S.ply")
                m = vmap.occupied(args.min_hits)
                tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(m))
                tmp.colors = o3d.utility.Vector3dVector(height_colors(m))
                o3d.io.write_point_cloud(fn, tmp)
                print(f"Saved {len(m)} voxels to {fn}")
                state["save"] = False

            now = time.time()
            if now - last_map > 1.0:                     # refresh the map ~1x per second
                m = vmap.occupied(args.min_hits)
                if len(m):
                    map_pcd.points = o3d.utility.Vector3dVector(m)
                    map_pcd.colors = o3d.utility.Vector3dVector(height_colors(m))
                else:
                    map_pcd.points = o3d.utility.Vector3dVector(seed)
                    map_pcd.paint_uniform_color([0.2, 0.2, 0.2])
                vis.update_geometry(map_pcd)
                last_map = now

            if now - last_stat > 5.0:
                dt = now - last_stat
                print(f"[{now - t_start:6.0f}s] {n_frames / dt:4.1f} fps, "
                      f"{n_pts_total / max(n_frames, 1):5.1f} pts/frame, "
                      f"{len(vmap.hits)} voxels touched")
                n_frames, n_pts_total, last_stat = 0, 0, now
            if not warned and n_frames == 0 and now - t_start > 4.0 and not recorded:
                print("WARNING: no frames received. Check COM port order, that the cfg ends "
                      "with sensorStart, and that the TI visualizer is closed.")
                warned = True

            vis.update_renderer()
            time.sleep(0.01)
    finally:
        stop.set()
        try:
            cli.write(b"sensorStop\n")
        except Exception:
            pass
        cli.close()
        data.close()
        vis.destroy_window()
        if recorded:
            fn = datetime.datetime.now().strftime("frames_%Y%m%d_%H%M%S.npz")
            np.savez_compressed(
                fn,
                t=np.array([r[0] for r in recorded]),
                frame=np.array([r[1] for r in recorded]),
                counts=np.array([len(r[2]) for r in recorded]),
                pts=np.concatenate([r[2] for r in recorded]),
            )
            print(f"Recorded {len(recorded)} frames to {fn}")


if __name__ == "__main__":
    main()
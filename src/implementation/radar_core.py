"""
Shared helpers for the 3x IWR6843AOP mapping scripts.

Conventions
-----------
Sensor frame (TI demo): x = right, y = forward (boresight), z = up. Meters.
World frame: z = up from the floor; you choose the origin (the layout helper puts it
on the floor at the centroid of the sensor triangle).
Pose of a sensor = 4x4 matrix T that maps SENSOR coordinates -> WORLD coordinates.
Pose angles (degrees), applied as  R = Rz(yaw) @ Rx(-pitch) @ Ry(roll):
    yaw   > 0 : turn left (counter-clockwise seen from above). yaw 0 looks along world +y.
    pitch > 0 : tilt the sensor DOWN.
    roll  > 0 : right side dips (right-hand rotation about the boresight).
"""
import json
import math
import queue
import struct
import threading
import time

import numpy as np
import serial

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HDR = struct.Struct("<8s8I")   # magic, version, totalLen, platform, frameNum,
                               # cpuCycles, numObj, numTLVs, subFrame -> 40 bytes
TLV = struct.Struct("<2I")     # type, payload length
TLV_POINTS = 1                 # x, y, z, doppler (4 x float32 per point)
TLV_SIDE_INFO = 7              # snr, noise (2 x uint16 per point, 0.1 dB units)
MAX_FRAME_BYTES = 200_000
PROMPT = b"mmwDemo:/>"
PALETTE = [(0.95, 0.25, 0.25), (0.25, 0.85, 0.35), (0.30, 0.50, 1.00),
           (0.95, 0.85, 0.20), (0.80, 0.30, 0.90)]


# ----------------------------------------------------------------------------
# Serial ports
# ----------------------------------------------------------------------------
def list_ports():
    from serial.tools import list_ports as lp
    print("Ports from the same EVM usually share the same SER= value.")
    for p in sorted(lp.comports(), key=lambda p: p.device):
        print(f"{p.device:8s} {p.description:58s} SER={p.serial_number}  {p.location or ''}")


def send_config(cli, path, name="", start=False, verbose=False):
    tag = f"[{name}] " if name else ""
    cli.reset_input_buffer()
    cli.write(b"\n")
    cli.read_until(PROMPT)
    cli.write(b"sensorStop\n")
    cli.read_until(PROMPT)
    n = 0
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("%") or line == "sensorStop":
            continue
        if line.startswith("sensorStart") and not start:
            continue
        cli.write((line + "\n").encode())
        resp = cli.read_until(PROMPT).decode(errors="ignore")
        short = " | ".join(s.strip() for s in resp.splitlines() if s.strip())
        bad = "Error" in resp or "not recognized" in resp
        if verbose or bad:
            print(f"{tag}>> {line}\n{tag}   {short}")
        if bad:
            raise RuntimeError(f"{name}: radar rejected config line: {line}")
        if not resp.strip():
            print(f"{tag}no reply to '{line}' - check CLI port / firmware / SOP switches")
        n += 1
    print(f"{tag}config OK ({n} commands sent from {path})")


# ----------------------------------------------------------------------------
# Frame parser (one thread per sensor)
# ----------------------------------------------------------------------------
def reader(port, q, stop):
    buf = bytearray()
    while not stop.is_set():
        try:
            chunk = port.read(port.in_waiting or 1)
        except (serial.SerialException, OSError):
            return
        if chunk:
            buf += chunk
        while True:
            i = buf.find(MAGIC)
            if i < 0:
                if len(buf) > 7:
                    del buf[:-7]
                break
            if i:
                del buf[:i]
            if len(buf) < HDR.size:
                break
            _, _, total, _, frame, _, n_obj, n_tlv, _ = HDR.unpack_from(buf)
            if total < HDR.size or total > MAX_FRAME_BYTES:
                del buf[:8]
                continue
            if len(buf) < total:
                break
            pts = np.empty((0, 4), np.float32)
            snr = None
            noise = None
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
                    noise = si[:, 1].astype(np.float32) * 0.1
                off += ln
            del buf[:total]
            if not ok:
                continue
            if snr is not None and len(snr) != len(pts):
                snr = None
            if noise is not None and len(noise) != len(pts):
                noise = None
            item = {"t": time.time(), "frame": frame, "pts": pts, "snr": snr, "noise": noise}
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put(item)


class RadarLink:
    """One EVM: CLI port + data port + reader thread."""

    def __init__(self, name, cli_port, data_port, cfg_path, verbose=False):
        self.name, self.cfg_path, self.verbose = name, cfg_path, verbose
        self.cli = serial.Serial(cli_port, 115200, timeout=1.0)
        self.data = serial.Serial(data_port, 921600, timeout=0.1)
        self.q = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._thread = None
        # "sensorStart" re-applies the stored config each time (safe). Some SDK builds also
        # accept "sensorStart 0" (skip reconfig, faster restart) - try it only after it works.
        self.restart_cmd = "sensorStart"
        self.running = False

    def configure(self):
        send_config(self.cli, self.cfg_path, self.name, start=False, verbose=self.verbose)
        if self._thread is None:
            self._thread = threading.Thread(target=reader, args=(self.data, self.q, self._stop),
                                            daemon=True)
            self._thread.start()

    def _cmd(self, text):
        self.cli.write((text + "\n").encode())
        return self.cli.read_until(PROMPT).decode(errors="ignore")

    def start(self):
        resp = self._cmd(self.restart_cmd)
        if "Error" in resp:      # fall back to a full re-config, like TI's visualizer does
            send_config(self.cli, self.cfg_path, self.name, start=False, verbose=self.verbose)
            resp = self._cmd("sensorStart")
            if "Error" in resp:
                raise RuntimeError(f"{self.name}: sensorStart failed: {resp.strip()}")
        self.running = True

    def stop(self):
        self._cmd("sensorStop")
        self.running = False

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self._stop.set()
        try:
            self.stop()
        except Exception:
            pass
        time.sleep(0.25)
        for p in (self.cli, self.data):
            try:
                p.close()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Poses and geometry
# ----------------------------------------------------------------------------
def pose_to_T(x, y, z, yaw=0.0, pitch=0.0, roll=0.0):
    ya, pt, ro = math.radians(yaw), math.radians(-pitch), math.radians(roll)
    cz, sz = math.cos(ya), math.sin(ya)
    cx, sx = math.cos(pt), math.sin(pt)
    cy, sy = math.cos(ro), math.sin(ro)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Rx @ Ry
    T[:3, 3] = [x, y, z]
    return T


def apply_T(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def world_to_sensor(T, p):
    return (p - T[:3, 3]) @ T[:3, :3]          # R^T (p - t) for row vectors


def rigid_fit(A, B):
    """R, t such that B ~= A @ R.T + t  (A: sensor-frame points, B: world points, Nx3)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def load_sensors(path):
    cfg = json.load(open(path))
    out = []
    for s in cfg["sensors"]:
        for k in ("cli", "data"):
            if "?" in s[k]:
                raise SystemExit(f"Edit {path}: sensor {s['name']} still has a placeholder {k} port. "
                                 f"Run with --list-ports to find them.")
        if "T" in s:
            T = np.array(s["T"], float)
        else:
            p = s["pose"]
            T = pose_to_T(p["x"], p["y"], p["z"], p.get("yaw", 0), p.get("pitch", 0), p.get("roll", 0))
        out.append({"name": s["name"], "cli": s["cli"], "data": s["data"],
                    "cfg": s.get("cfg", cfg.get("cfg", "mapping.cfg")), "T": T})
    return cfg, out


def filter_points(pts, snr, min_range, max_range, min_snr):
    xyz = pts[:, :3].astype(np.float64)
    rng = np.linalg.norm(xyz, axis=1)
    keep = (rng >= min_range) & (rng <= max_range)
    if snr is not None and min_snr > 0:
        keep &= snr >= min_snr
    return xyz, keep


# ----------------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------------
class MultiVoxelMap:
    """Per-voxel hit counts, kept separately for each sensor."""

    def __init__(self, voxel, n_sensors):
        self.v, self.n, self.hits = voxel, n_sensors, {}

    def add(self, xyz, sensor_idx):
        if len(xyz) == 0:
            return
        keys = np.unique(np.floor(xyz / self.v).astype(np.int32), axis=0)
        for k in map(tuple, keys.tolist()):
            h = self.hits.get(k)
            if h is None:
                h = self.hits[k] = [0] * self.n
            h[sensor_idx] += 1

    def occupied(self, min_hits, min_sensors=1):
        ks, owner = [], []
        for k, h in self.hits.items():
            if sum(h) >= min_hits and sum(1 for c in h if c >= 2) >= min_sensors:
                ks.append(k)
                owner.append(int(np.argmax(h)))
        if not ks:
            return np.empty((0, 3)), np.empty(0, int)
        return (np.array(ks, float) + 0.5) * self.v, np.array(owner)

    def clear(self):
        self.hits.clear()


def height_colors(xyz, zmax=3.0):
    if len(xyz) == 0:
        return np.empty((0, 3))
    t = np.clip(xyz[:, 2] / zmax, 0, 1)
    return np.c_[t, 1 - np.abs(2 * t - 1), 1 - t]


def grid_lines(half=6.0, step=1.0):
    xs = np.arange(-half, half + 1e-6, step)
    pts, lines = [], []
    for c in xs:
        pts += [[c, -half, 0], [c, half, 0]]
        lines.append([len(pts) - 2, len(pts) - 1])
        pts += [[-half, c, 0], [half, c, 0]]
        lines.append([len(pts) - 2, len(pts) - 1])
    return np.array(pts, float), np.array(lines, np.int32)


def fov_lines(T, r=4.0, half_az=60.0, half_el=60.0):
    corners = []
    for az, el in [(-half_az, -half_el), (half_az, -half_el), (half_az, half_el), (-half_az, half_el)]:
        a, e = math.radians(az), math.radians(el)
        corners.append(r * np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)]))
    pts = apply_T(T, np.vstack([[0, 0, 0], *corners]))
    lines = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], np.int32)
    return pts, lines

# ----------------------------------------------------------------------------
# Sensor scheduling that is shared by the Open3D mapper and the web viewer
# ----------------------------------------------------------------------------
class Scheduler(threading.Thread):
    """Starts/stops sensors from a background thread so a viewer never freezes.
 
    default  : time-division - one sensor transmits at a time, each for `slot` seconds
    freerun  : every sensor runs continuously (10 Hz each, but they may interfere)
    1 sensor : just runs continuously
    """
 
    def __init__(self, links, slot=3.0, gap=0.3, freerun=False):
        super().__init__(daemon=True)
        self.links, self.slot, self.gap, self.freerun = links, slot, gap, freerun
        self.halt = threading.Event()
        self.active = None
        self.error = None
 
    def run(self):
        try:
            if self.freerun or len(self.links) == 1:
                for link in self.links:
                    link.start()
                self.active = 0 if len(self.links) == 1 else None
                self.halt.wait()
                return
            while not self.halt.is_set():
                for i, link in enumerate(self.links):
                    if self.halt.is_set():
                        break
                    link.start()
                    self.active = i
                    self.halt.wait(self.slot)
                    link.stop()
                    self.active = None
                    self.halt.wait(self.gap)
        except Exception as e:
            self.error = e
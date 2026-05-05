"""
IWR6843AOPEVM 3D Radar Mapping Simulation
==========================================
Simulates 3 TI IWR6843AOPEVM mmWave radar sensors performing real-time 3D mapping
of an object (a person/box) inside a room using Open3D for visualization.

Sensor layout:
  - Sensor 0: Front wall  (facing +Z into room)
  - Sensor 1: Left wall   (facing +X into room)
  - Sensor 2: Ceiling     (facing -Y downward)
"""

import numpy as np
import open3d as o3d
import time
import threading
import copy

# =================================== Room & sensor configuration ==================================================

ROOM_W, ROOM_H, ROOM_D = 6.0, 3.0, 6.0   # In meters (X, Y, Z)

# Each sensor: position (x,y,z) + pointing direction unit vector
SENSORS = [
    {
        "name": "Front",
        "pos": np.array([ROOM_W / 2, ROOM_H / 2, 0.0]),
        "dir": np.array([0.0, 0.0, 1.0]),
        "color": [1.0, 0.2, 0.2],   # red
    },
    {
        "name": "Left",
        "pos": np.array([0.0, ROOM_H / 2, ROOM_D / 2]),
        "dir": np.array([1.0, 0.0, 0.0]),
        "color": [0.2, 1.0, 0.2],   # green
    },
    {
        "name": "Ceiling",
        "pos": np.array([ROOM_W / 2, ROOM_H, ROOM_D / 2]),
        "dir": np.array([0.0, -1.0, 0.0]),
        "color": [0.2, 0.5, 1.0],   # blue
    },
]

# IWR6843AOP approximate field-of-view (degrees)
FOV_AZ   = 120   # azimuth
FOV_ELEV =  40   # elevation

# Radar noise / detection parameters
POINTS_PER_SENSOR  = 80    # detections per frame per sensor
RANGE_NOISE_STD    = 0.04  # metres
ANGULAR_NOISE_STD  = 0.015 # radians
UPDATE_HZ          = 10    # frames per second
HISTORY_FRAMES     = 60    # how many frames of point cloud to keep

# ---------------------------------------------------------------------------
# Target object definition  (a standing person / box in the room centre)
# ---------------------------------------------------------------------------

def make_target_surface(t: float):
    """Return XYZ points on the surface of a box (person) that drifts slightly."""
    cx = ROOM_W / 2 + 0.3 * np.sin(0.4 * t)
    cy = ROOM_H / 2 - 0.5
    cz = ROOM_D / 2 + 0.3 * np.cos(0.3 * t)

    # Box: 0.5 m wide, 1.7 m tall, 0.3 m deep
    hw, hh, hd = 0.25, 0.85, 0.15
    n = 400
    pts = []

    # Front & back faces
    for _ in range(n // 4):
        y = cy + np.random.uniform(-hh, hh)
        x = cx + np.random.uniform(-hw, hw)
        pts.append([x, y, cz + hd])
        pts.append([x, y, cz - hd])

    # Left & right faces
    for _ in range(n // 4):
        y = cy + np.random.uniform(-hh, hh)
        z = cz + np.random.uniform(-hd, hd)
        pts.append([cx - hw, y, z])
        pts.append([cx + hw, y, z])

    # Top
    for _ in range(n // 4):
        x = cx + np.random.uniform(-hw, hw)
        z = cz + np.random.uniform(-hd, hd)
        pts.append([x, cy + hh, z])

    return np.array(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Mock radar sensor
# ---------------------------------------------------------------------------

def simulate_sensor(sensor: dict, target_pts: np.ndarray) -> np.ndarray:
    """
    Given a sensor configuration and the true target surface points,
    return a noisy point-cloud as the sensor would detect.

    Steps to do this:
      1. Filter target points visible from this sensor (rough cone check).
      2. Sample POINTS_PER_SENSOR of them.
      3. Add range & angular noise (mimicking IWR6843 typical accuracy).
      4. Return in world coordinates.
    """
    spos = sensor["pos"]
    sdir = sensor["dir"]

    # Vectors from sensor to each surface point
    vecs = target_pts - spos                          # (N, 3)
    ranges = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit_vecs = vecs / (ranges + 1e-9)

    # Angle from sensor boresight
    cos_angle = np.clip(unit_vecs @ sdir, -1, 1)
    angle = np.arccos(cos_angle)

    # Rough half-FOV cone
    half_fov = np.radians(min(FOV_AZ, FOV_ELEV) / 2)
    visible = angle < half_fov

    vis_pts  = target_pts[visible]
    vis_rngs = ranges[visible, 0]

    if len(vis_pts) < 4:
        return np.empty((0, 3))

    # Sample
    idx = np.random.choice(len(vis_pts),
                           size=min(POINTS_PER_SENSOR, len(vis_pts)),
                           replace=False)
    sampled    = vis_pts[idx]
    sampled_r  = vis_rngs[idx]

    # Add noise
    range_noise   = np.random.normal(0, RANGE_NOISE_STD,   size=len(idx))
    az_noise      = np.random.normal(0, ANGULAR_NOISE_STD, size=len(idx))
    el_noise      = np.random.normal(0, ANGULAR_NOISE_STD, size=len(idx))

    # Perturb along radial direction
    true_dirs  = (sampled - spos)
    true_norms = np.linalg.norm(true_dirs, axis=1, keepdims=True)
    true_unit  = true_dirs / (true_norms + 1e-9)

    noisy_range = sampled_r + range_noise
    noisy_pts   = spos + true_unit * noisy_range[:, None]

    # Add small lateral jitter (angular noise)
    lateral = np.cross(true_unit, sdir)
    lateral_norm = np.linalg.norm(lateral, axis=1, keepdims=True)
    lateral = np.where(lateral_norm > 1e-9, lateral / (lateral_norm + 1e-9), lateral)
    noisy_pts += lateral * az_noise[:, None] * sampled_r[:, None]

    return noisy_pts.astype(np.float64)


# ---------------------------------------------------------------------------
# Room wireframe geometry
# ---------------------------------------------------------------------------

def make_room_lines():
    """Return an Open3D LineSet representing the room bounding box."""
    pts = np.array([
        [0,       0,       0      ],
        [ROOM_W,  0,       0      ],
        [ROOM_W,  ROOM_H,  0      ],
        [0,       ROOM_H,  0      ],
        [0,       0,       ROOM_D ],
        [ROOM_W,  0,       ROOM_D ],
        [ROOM_W,  ROOM_H,  ROOM_D ],
        [0,       ROOM_H,  ROOM_D ],
    ])
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color([0.6, 0.6, 0.6])
    return ls


def make_sensor_markers():
    """Small spheres at each sensor location."""
    spheres = []
    for s in SENSORS:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        mesh.compute_vertex_normals()
        mesh.translate(s["pos"])
        mesh.paint_uniform_color(s["color"])
        spheres.append(mesh)
    return spheres


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

class RadarSimulation:
    def __init__(self):
        self.frame      = 0
        self.history    = []      # list of (pts, colors) arrays
        self.lock       = threading.Lock()
        self.running    = True
        self.pcd        = o3d.geometry.PointCloud()

    # ---- data generation thread ----------------------------------------
    def _generate(self):
        while self.running:
            t = self.frame / UPDATE_HZ
            target_surface = make_target_surface(t)

            all_pts    = []
            all_colors = []

            for sensor in SENSORS:
                pts = simulate_sensor(sensor, target_surface)
                if len(pts) > 0:
                    cols = np.tile(sensor["color"], (len(pts), 1))
                    all_pts.append(pts)
                    all_colors.append(cols)

            if all_pts:
                pts_combined  = np.vstack(all_pts)
                cols_combined = np.vstack(all_colors)

                with self.lock:
                    self.history.append((pts_combined, cols_combined))
                    if len(self.history) > HISTORY_FRAMES:
                        self.history.pop(0)

            self.frame += 1
            time.sleep(1.0 / UPDATE_HZ)

    # ---- build merged point cloud from history -------------------------
    def _build_pcd(self):
        with self.lock:
            if not self.history:
                return

            # Fade older frames
            all_pts  = []
            all_cols = []
            n = len(self.history)
            for i, (pts, cols) in enumerate(self.history):
                alpha = (i + 1) / n          # 0→1 (oldest→newest)
                faded = cols * alpha + (1 - alpha) * 0.05
                all_pts.append(pts)
                all_cols.append(faded)

            pts_all  = np.vstack(all_pts)
            cols_all = np.vstack(all_cols)

        self.pcd.points = o3d.utility.Vector3dVector(pts_all)
        self.pcd.colors = o3d.utility.Vector3dVector(np.clip(cols_all, 0, 1))

    # ---- run -----------------------------------------------------------
    def run(self):
        print("=" * 60)
        print("  IWR6843AOPEVM  –  3-Sensor mmWave Radar Simulation")
        print("=" * 60)
        print(f"  Room : {ROOM_W} m × {ROOM_H} m × {ROOM_D} m")
        print(f"  Sensors  : {len(SENSORS)}")
        for s in SENSORS:
            col = ["R","G","B"][SENSORS.index(s)]
            print(f"    [{col}] {s['name']:8s}  pos={s['pos']}")
        print()
        print("  Controls (Open3D viewer):")
        print("    Left-drag   – rotate")
        print("    Right-drag  – pan")
        print("    Scroll      – zoom")
        print("    Q / Esc     – quit")
        print("=" * 60)

        # Start data thread
        t = threading.Thread(target=self._generate, daemon=True)
        t.start()

        # Wait for first frame
        while not self.history:
            time.sleep(0.05)

        # Build initial geometry
        room_lines = make_room_lines()
        markers    = make_sensor_markers()
        self._build_pcd()

        # Open3D visualiser
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="IWR6843AOPEVM – 3D Radar Mapping Simulation",
            width=1280,
            height=800,
        )

        vis.add_geometry(room_lines)
        for m in markers:
            vis.add_geometry(m)
        vis.add_geometry(self.pcd)

        # Coordinate frame at origin
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(frame_mesh)

        # Rendering options
        opt = vis.get_render_option()
        opt.background_color    = np.array([0.05, 0.05, 0.12])
        opt.point_size          = 3.0
        opt.show_coordinate_frame = False

        # Camera: isometric-ish view
        ctr = vis.get_view_control()
        ctr.set_lookat([ROOM_W/2, ROOM_H/2, ROOM_D/2])
        ctr.set_front([-0.6, -0.4, -0.7])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.45)

        last_update = time.time()

        while vis.poll_events():
            now = time.time()
            if now - last_update > 1.0 / UPDATE_HZ:
                self._build_pcd()
                vis.update_geometry(self.pcd)
                last_update = now
            vis.update_renderer()

        self.running = False
        vis.destroy_window()
        print("\nSimulation ended.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = RadarSimulation()
    sim.run()
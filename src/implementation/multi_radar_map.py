"""
3x IWR6843AOP into one merged static 3D occupancy map in Open3D.

The sensors take turns (time-division) so they never chirp at the same time:
S1 runs for --slot seconds, stops, then S2, then S3, and so on.

    python multi_radar_map.py --list-ports
    python multi_radar_map.py --sensors sensors.json --only S1        # test one sensor
    python multi_radar_map.py --sensors sensors.json                  # use all three

Keys in the viewer:  R reset map | S save .ply | P pause accumulation
                     C color by height / by sensor | V reset camera
"""
import argparse
import datetime
import threading
import time

import numpy as np
import open3d as o3d

from radar_core import (PALETTE, RadarLink, MultiVoxelMap, Scheduler, apply_T, filter_points, fov_lines,
                        grid_lines, height_colors, list_ports, load_sensors)


def lineset(pts, lines, color):
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts),
                              lines=o3d.utility.Vector2iVector(lines))
    ls.paint_uniform_color(list(color))
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--sensors", default="sensors.json")
    ap.add_argument("--only", nargs="+", help="run only these sensor names, e.g. --only S1")
    ap.add_argument("--slot", type=float, default=3.0, help="seconds each sensor runs per turn")
    ap.add_argument("--gap", type=float, default=0.3, help="idle seconds between turns")
    ap.add_argument("--freerun", action="store_true",
                    help="run all sensors continuously (no time-division) to test interference")
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--min-hits", type=int, default=6, help="total hits (all sensors) to show a voxel")
    ap.add_argument("--min-sensors", type=int, default=1,
                    help="2 = only show voxels seen by >=2 sensors (kills ghosts, also thins the map)")
    ap.add_argument("--vthresh", type=float, default=0.05)
    ap.add_argument("--min-range", type=float, default=0.3)
    ap.add_argument("--max-range", type=float, default=9.0)
    ap.add_argument("--min-snr", type=float, default=0.0)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="echo every config line")
    args = ap.parse_args()

    if args.list_ports:
        list_ports()
        return

    _, sensors = load_sensors(args.sensors)
    if args.only:
        sensors = [s for s in sensors if s["name"] in args.only]
        if not sensors:
            raise SystemExit("--only matched no sensor names")
    n = len(sensors)
    Ts = [s["T"] for s in sensors]
    palette = np.array([PALETTE[i % len(PALETTE)] for i in range(n)])

    links = []
    sched = None
    try:
        for s in sensors:
            link = RadarLink(s["name"], s["cli"], s["data"], s["cfg"], args.verbose)
            links.append(link)
            link.configure()

        vmap = MultiVoxelMap(args.voxel, n)
        st = {"paused": False, "reset": False, "save": False, "by_sensor": False}

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("3x IWR6843AOP  [R reset | S save | P pause | C color | V view]", 1400, 850)
        opt = vis.get_render_option()
        opt.point_size = 5.0
        opt.background_color = np.array([0.04, 0.04, 0.07])

        seed = np.array([[0, 0, 0.01], [0, 0.01, 0.01]])
        map_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seed))
        live_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seed))
        gp, gl = grid_lines()
        vis.add_geometry(lineset(gp, gl, (0.22, 0.22, 0.28)))
        for i, T in enumerate(Ts):
            fp, fl = fov_lines(T)
            vis.add_geometry(lineset(fp, fl, palette[i] * 0.6))
            ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
            ball.paint_uniform_color(list(palette[i]))
            ball.translate(T[:3, 3])
            vis.add_geometry(ball)
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            axes.transform(T)
            vis.add_geometry(axes)
        vis.add_geometry(map_pcd)
        vis.add_geometry(live_pcd)

        ctr = vis.get_view_control()

        def set_view():
            ctr.set_lookat([0, 0, 1.0])
            ctr.set_front([0.0, -1.0, 0.75])       # camera south of the origin, looking north
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.4)

        set_view()

        def flag(key):
            def cb(_):
                st[key] = True
                return False
            return cb

        def on_pause(_):
            st["paused"] = not st["paused"]
            print("Accumulation", "PAUSED" if st["paused"] else "resumed")
            return False

        def on_color(_):
            st["by_sensor"] = not st["by_sensor"]
            print("Map colors:", "by sensor" if st["by_sensor"] else "by height")
            return False

        def on_view(_):
            set_view()
            return False

        vis.register_key_callback(ord("R"), flag("reset"))
        vis.register_key_callback(ord("S"), flag("save"))
        vis.register_key_callback(ord("P"), on_pause)
        vis.register_key_callback(ord("C"), on_color)
        vis.register_key_callback(ord("V"), on_view)

        def map_colors(pts, owner):
            if st["by_sensor"] and len(pts):
                return palette[owner] * 0.85
            return height_colors(pts)

        sched = Scheduler(links, slot=args.slot, gap=args.gap, freerun=args.freerun)
        sched.start()

        recorded = []
        live = {}
        n_fr, n_pt = [0] * n, [0] * n
        t0 = last_map = last_live = last_stat = time.time()

        while vis.poll_events():
            now = time.time()
            if sched.error:
                print("Scheduler stopped:", sched.error)
                break

            for i, link in enumerate(links):
                for fr in link.drain():
                    pts, snr = fr["pts"], fr["snr"]
                    n_fr[i] += 1
                    n_pt[i] += len(pts)
                    if not args.no_record:
                        recorded.append((fr["t"], i, fr["frame"], pts))
                    if len(pts) == 0:
                        continue
                    xyz, keep = filter_points(pts, snr, args.min_range, args.max_range, args.min_snr)
                    static = keep & (np.abs(pts[:, 3]) <= args.vthresh)
                    if not st["paused"] and static.any():
                        vmap.add(apply_T(Ts[i], xyz[static]), i)
                    if keep.any():
                        live[i] = (now, apply_T(Ts[i], xyz[keep]))

            if st["reset"]:
                vmap.clear()
                st["reset"] = False
                print("Map reset")
            if st["save"]:
                fn = datetime.datetime.now().strftime("map3_%Y%m%d_%H%M%S.ply")
                p, o = vmap.occupied(args.min_hits, args.min_sensors)
                tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
                tmp.colors = o3d.utility.Vector3dVector(map_colors(p, o))
                o3d.io.write_point_cloud(fn, tmp)
                print(f"Saved {len(p)} voxels to {fn}")
                st["save"] = False

            if now - last_live > 0.2:
                chunks, cols = [], []
                for i, (t, p) in live.items():
                    if now - t < 1.5 and len(p):
                        chunks.append(p)
                        cols.append(np.tile(palette[i], (len(p), 1)))
                if chunks:
                    live_pcd.points = o3d.utility.Vector3dVector(np.vstack(chunks))
                    live_pcd.colors = o3d.utility.Vector3dVector(np.vstack(cols))
                else:
                    live_pcd.points = o3d.utility.Vector3dVector(seed)
                    live_pcd.paint_uniform_color([0.04, 0.04, 0.07])
                vis.update_geometry(live_pcd)
                last_live = now

            if now - last_map > 1.0:
                p, o = vmap.occupied(args.min_hits, args.min_sensors)
                if len(p):
                    map_pcd.points = o3d.utility.Vector3dVector(p)
                    map_pcd.colors = o3d.utility.Vector3dVector(map_colors(p, o))
                else:
                    map_pcd.points = o3d.utility.Vector3dVector(seed)
                    map_pcd.paint_uniform_color([0.04, 0.04, 0.07])
                vis.update_geometry(map_pcd)
                last_map = now

            if now - last_stat > 5.0:
                dt = now - last_stat
                msg = " | ".join(f"{links[i].name} {n_fr[i] / dt:4.1f}fps {n_pt[i] / max(n_fr[i], 1):5.1f}pts"
                                 for i in range(n))
                act = links[sched.active].name if sched.active is not None else "-"
                print(f"[{now - t0:5.0f}s] {msg} | voxels {len(vmap.hits)} | active {act}")
                n_fr, n_pt, last_stat = [0] * n, [0] * n, now

            vis.update_renderer()
            time.sleep(0.01)
    finally:
        if sched:
            sched.halt.set()
            sched.join(3.0)
        for l in links:
            l.close()
        try:
            vis.destroy_window()
        except Exception:
            pass
        if links and not args.no_record and recorded:
            fn = datetime.datetime.now().strftime("frames3_%Y%m%d_%H%M%S.npz")
            np.savez_compressed(
                fn,
                t=np.array([r[0] for r in recorded]),
                sensor=np.array([r[1] for r in recorded]),
                frame=np.array([r[2] for r in recorded]),
                counts=np.array([len(r[3]) for r in recorded]),
                pts=np.concatenate([r[3] for r in recorded]),
                T=np.array(Ts),
            )
            print(f"Recorded {len(recorded)} frames to {fn}")


if __name__ == "__main__":
    main()
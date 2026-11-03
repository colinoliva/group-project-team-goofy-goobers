"""
Write starting sensors.json for three sensors on an equilateral triangle.

The world origin is on the floor at the triangle's centroid. World +y is "north".
Vertices are at 90, 210 and 330 degrees (S1 at +y, S2 at back-left, S3 at back-right).

  --facing inward   : every sensor looks at the centroid (watch a central zone, max overlap)
  --facing outward  : every sensor looks away from the centroid (~360 degree coverage of walls)
  --facing forward  : all sensors look along +y (wall-mounted cluster, max overlap in front)

Ex:
    python make_layout.py --side 1.5 --height 1.3 --facing inward
Then edit the COM ports in sensors.json (python multi_radar_map.py --list-ports).
"""
import argparse
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("--side", type=float, default=1.5, help="triangle side length, m")
ap.add_argument("--height", type=float, default=1.3, help="sensor height above floor, m")
ap.add_argument("--facing", choices=["inward", "outward", "forward"], default="inward")
ap.add_argument("--yaw-offset", type=float, default=0.0, help="extra yaw applied to all, deg")
ap.add_argument("--pitch", type=float, default=0.0, help="tilt down, deg")
ap.add_argument("--cfg", default="mapping.cfg")
ap.add_argument("--out", default="sensors.json")
a = ap.parse_args()

rc = a.side / math.sqrt(3)
sensors = []
for n, phi in enumerate([90, 210, 330], start=1):
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    d = {"inward": (-c, -s), "outward": (c, s), "forward": (0.0, 1.0)}[a.facing]
    yaw = math.degrees(math.atan2(-d[0], d[1])) + a.yaw_offset
    sensors.append({
        "name": f"S{n}", "cli": "COM?", "data": "COM?",
        "pose": {"x": round(rc * c, 3), "y": round(rc * s, 3), "z": a.height,
                 "yaw": round(yaw, 1), "pitch": a.pitch, "roll": 0.0},
    })
json.dump({"cfg": a.cfg, "sensors": sensors}, open(a.out, "w"), indent=2)
print(f"Wrote {a.out}: side {a.side} m, circumradius {rc:.3f} m, facing {a.facing}")
for s in sensors:
    p = s["pose"]
    print(f"  {s['name']}: x={p['x']:+.3f} y={p['y']:+.3f} z={p['z']:.2f} yaw={p['yaw']:+.1f}")
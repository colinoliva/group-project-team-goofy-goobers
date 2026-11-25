"""Generate labeled synthetic clusters for pipeline testing only.

Results on synthetic data prove nothing about real-world classifier accuracy.
"""

import argparse
import csv
import os

import numpy as np

from features import FEATURE_NAMES, extract_features


def generate_synthetic(count=200, seed=42):
    """Generate synthetic human and non-human cluster records."""
    rng = np.random.default_rng(seed)
    records = []
    kinds = ("human", "tall_static", "fan", "chair", "rolling_chair",
             "swinging_curtain", "plant", "noise")
    density_scales = {
        session: float(rng.uniform(0.65, 1.35))
        for session in range(10)
    }
    for kind in kinds:
        kind_count = count if kind == "human" else count // 5
        for sample in range(kind_count):
            cx, cy = rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)
            session_number = sample // 20
            session = f"synthetic_{session_number:03d}"
            density_scale = density_scales[session_number]
            if kind == "human":
                n = int(np.clip(round(rng.integers(15, 81) * density_scale), 15, 80))
                seated = rng.random() < 0.20
                z_center = 0.6 if seated else rng.uniform(0.9, 1.1)
                z_limit = 1.3 if seated else 1.85
                xy_std = rng.uniform(0.10, 0.20)
                xyz = rng.normal(
                    [cx, cy, z_center],
                    [xy_std * density_scale, xy_std * density_scale, 0.22],
                    (n, 3),
                )
                xyz[:, 2] = np.clip(xyz[:, 2], 0.05, z_limit)
                if rng.random() < 0.35:
                    vel = rng.normal(
                        rng.normal(0.0, 0.02),
                        rng.uniform(0.03, 0.10),
                        n,
                    )
                else:
                    vel = rng.normal(rng.uniform(-1.5, 1.5),
                                     rng.uniform(0.35, 1.10), n)
                label = "human"
            elif kind == "tall_static":
                n = int(np.clip(round(rng.integers(15, 81) * density_scale), 15, 80))
                xyz = rng.normal(
                    [cx, cy, rng.uniform(0.72, 0.82)],
                    [rng.uniform(0.09, 0.16) * density_scale,
                     rng.uniform(0.09, 0.16) * density_scale,
                     rng.uniform(0.55, 0.75)],
                    (n, 3),
                )
                xyz[:, 2] = np.clip(xyz[:, 2], 0.05, 1.55)
                vel = rng.normal(0.0, rng.uniform(0.02, 0.09), n)
                label = "nonhuman"
            elif kind == "fan":
                n = int(np.clip(round(rng.integers(15, 81) * density_scale), 15, 80))
                xyz = rng.normal(
                    [cx, cy, rng.uniform(0.25, 0.55)],
                    [rng.uniform(0.12, 0.24) * density_scale,
                     rng.uniform(0.12, 0.24) * density_scale,
                     rng.uniform(0.08, 0.18)],
                    (n, 3),
                )
                xyz[:, 2] = np.clip(xyz[:, 2], 0.05, 1.20)
                vel = rng.normal(0.0, rng.uniform(1.5, 3.0), n)
                label = "nonhuman"
            elif kind == "chair":
                n = int(np.clip(round(rng.integers(15, 81) * density_scale), 15, 80))
                xyz = rng.normal(
                    [cx, cy, rng.uniform(0.45, 0.65)],
                    [rng.uniform(0.25, 0.40) * density_scale,
                     rng.uniform(0.25, 0.40) * density_scale,
                     rng.uniform(0.18, 0.28)],
                    (n, 3),
                )
                xyz[:, 2] = np.clip(xyz[:, 2], 0.05, 1.30)
                vel = rng.normal(0.0, 0.02, n)
                label = "nonhuman"
            elif kind == "rolling_chair":
                n = int(np.clip(round(rng.integers(15, 81) * density_scale), 15, 80))
                xyz = rng.normal(
                    [cx, cy, rng.uniform(0.45, 0.65)],
                    [rng.uniform(0.25, 0.45) * density_scale,
                     rng.uniform(0.25, 0.45) * density_scale,
                     rng.uniform(0.15, 0.25)],
                    (n, 3),
                )
                xyz[:, 2] = np.clip(xyz[:, 2], 0.05, 1.30)
                moving_mean = rng.uniform(0.2, 0.6) * rng.choice((-1.0, 1.0))
                vel = rng.normal(moving_mean, rng.uniform(0.04, 0.08), n)
                label = "nonhuman"
            elif kind == "swinging_curtain":
                n = int(np.clip(round(rng.integers(15, 81) * density_scale), 15, 80))
                xyz = rng.normal(
                    [cx, cy, rng.uniform(0.65, 0.85)],
                    [rng.uniform(0.35, 0.55) * density_scale,
                     rng.uniform(0.08, 0.14) * density_scale,
                     rng.uniform(0.35, 0.50)],
                    (n, 3),
                )
                xyz[:, 2] = np.clip(xyz[:, 2], 0.05, 1.50)
                moving_mean = rng.uniform(0.2, 0.6) * rng.choice((-1.0, 1.0))
                vel = rng.normal(moving_mean, rng.uniform(0.04, 0.08), n)
                label = "nonhuman"
            elif kind == "plant":
                n = int(np.clip(round(rng.integers(15, 81) * density_scale), 15, 80))
                xyz = rng.normal(
                    [cx, cy, rng.uniform(0.65, 0.85)],
                    [rng.uniform(0.20, 0.32) * density_scale,
                     rng.uniform(0.20, 0.32) * density_scale,
                     rng.uniform(0.35, 0.55)],
                    (n, 3),
                )
                xyz[:, 2] = np.clip(xyz[:, 2], 0.0, 1.5)
                vel = rng.normal(0.0, rng.uniform(0.02, 0.12), n)
                label = "nonhuman"
            else:
                n = int(np.clip(round(rng.integers(15, 81) * density_scale), 15, 80))
                xyz = rng.normal(
                    [cx, cy, rng.uniform(0.20, 1.80)],
                    [rng.uniform(0.12, 0.30) * density_scale,
                     rng.uniform(0.12, 0.30) * density_scale,
                     rng.uniform(0.12, 0.30)],
                    (n, 3),
                )
                xyz[:, 2] = np.clip(xyz[:, 2], 0.05, 1.90)
                vel = rng.normal(0.0, rng.uniform(0.05, 0.35), n)
                label = "nonhuman"
            snr = rng.normal(18.0, 6.0, n)
            features = extract_features(xyz, vel, snr)
            records.append((features, label, session))
    return records


def write_synthetic(path, count=200, seed=42):
    """Write synthetic records to a feature CSV."""
    records = generate_synthetic(count, seed)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FEATURE_NAMES + ["label", "session_id"])
        writer.writeheader()
        for features, label, session in records:
            row = {name: features[name] for name in FEATURE_NAMES}
            row.update(label=label, session_id=session)
            writer.writerow(row)
    print(f"Wrote {len(records)} synthetic records to {path}")


def main():
    """Parse synthetic dataset arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic cluster features.")
    parser.add_argument("--output", default="datasets/synthetic.csv")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_synthetic(args.output, args.count, args.seed)


if __name__ == "__main__":
    main()

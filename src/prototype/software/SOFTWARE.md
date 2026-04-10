**Overview**

This folder contains small simulation and visualization scripts for the TI mmWave prototype. The two main Python programs provide live, browser-based 3D views of simulated radar point clouds and basic processing (clustering + bounding boxes).

**Files**

- **`src/prototype/software/ti_sim.py`**: [src/prototype/software/ti_sim.py](src/prototype/software/ti_sim.py#L1)
  - Full pipeline using Open3D for point-cloud processing:
    - voxel downsampling, statistical outlier removal, DBSCAN clustering
    - computes axis-aligned bounding boxes per cluster and draws them
    - Dash + Plotly front-end for live 3D visualization and a small stats bar
  - Heavier dependencies (notably `open3d`) — best used on a machine with enough RAM/CPU.

- **`src/prototype/software/simple_ti_sim.py`**: [src/prototype/software/simple_ti_sim.py](src/prototype/software/simple_ti_sim.py#L1)
  - Lightweight simulator that generates colored point groups and serves a 3D Plotly view via Dash.
  - No Open3D processing; lower CPU/memory overhead — useful for quick tests or on resource-constrained devices.

**Running (quick start)**

- From the `src/prototype/software` folder run either script:
  - `python simple_ti_sim.py` (lighter; recommended for Raspberry Pi or quick checks)
  - `python ti_sim.py` (full processing pipeline; requires Open3D)

- Both start a Dash app listening on port `8050` and bind to `0.0.0.0` in the scripts, so:
  - Local browser: `http://localhost:8050`
  - From another device on the same network: `http://<device-ip>:8050`

**Dependencies**

- Minimal: `numpy`, `dash`, `plotly`.
- Additional (for `ti_sim.py`): `open3d` (and its native dependencies).
- On Raspberry Pi install with `pip` (or a virtualenv). Example:
  - `pip install numpy dash plotly`
  - `pip install open3d` # only if running `ti_sim.py`

**Raspberry Pi notes**

- These scripts have been run on a Raspberry Pi where the required packages were already installed. `simple_ti_sim.py` is the safer choice on Pi because it avoids `open3d`.
- If you are currently working on a laptop, you can still run either script locally; be aware that `open3d` may be slow to install or run on low-resource systems.
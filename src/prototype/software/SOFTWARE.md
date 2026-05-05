**Overview**

This folder contains small simulation and visualization scripts for the TI prototype. The existing Python programs provide live browser-based 3D mock views, while the live 2D mapper consumes UART range lines from the TI board and trilaterates them into live positions.

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

- **`src/prototype/software/live_2d_mapper.py`**: [src/prototype/software/live_2d_mapper.py](src/prototype/software/live_2d_mapper.py#L1)
  - Reads UART lines such as `S1,dist=13.2` from a serial port and turns each complete 3-sensor frame into a 2D trilaterated point.
  - Shows the live estimate, sensor locations, and a short position trail in a Dash + Plotly map.
  - Falls back to mock data when `--serial-port` is not provided, so the map can be tested without hardware.

**Running (quick start)**

- From the `src/prototype/software` folder run either script:
  - `python simple_ti_sim.py` (lighter; recommended for Raspberry Pi or quick checks)
  - `python ti_sim.py` (full processing pipeline; requires Open3D)

- Both start a Dash app listening on port `8050` and bind to `0.0.0.0` in the scripts, so:
  - Local browser: `http://localhost:8050`
  - From another device on the same network: `http://<device-ip>:8050`

**Dependencies**

- Minimal: `numpy`, `dash`, `plotly`.
- For live serial input: `pyserial`.
- Additional (for `ti_sim.py`): `open3d` (and its native dependencies).
- On Raspberry Pi install with `pip` (or a virtualenv). Example:
  - `pip install numpy dash plotly`
  - `pip install pyserial` # only if using `live_2d_mapper.py` with hardware
  - `pip install open3d` # only if running `ti_sim.py`

**Raspberry Pi notes**

- These scripts have been run on a Raspberry Pi where the required packages were already installed. `simple_ti_sim.py` is the safer choice on Pi because it avoids `open3d`.
- If you are currently working on a laptop, you can still run either script locally; be aware that `open3d` may be slow to install or run on low-resource systems.
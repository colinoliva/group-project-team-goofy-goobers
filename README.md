# Texas Instruments mmWave Radar Sensor

Single-sensor 3D occupancy mapping using the TI IWR6843AOP radar platform.

## Team

| Name                 | GitHub                                           | Email                        |
| -------------------- | ------------------------------------------------ | ---------------------------- |
| Paul Brandon Estigoy | [@Brannflakess](https://github.com/Brannflakess) | paulbrandon.estigoy@sjsu.edu |
| Raghav Gautam        | [@RaghavG17](https://github.com/RaghavG17)       | raghav.gautam@sjsu.edu       |
| Colin Oliva          | [@colinoliva](https://github.com/colinoliva)     | colin.oliva@sjsu.edu         |
| Ernest Ma            | [@maernest04](https://github.com/maernest04)     | ernest.ma@sjsu.edu           |

**Advisor:** Dr. Charan Bhaskar

---

## Problem Statement

Affordable 3D environment mapping is difficult to achieve with low-cost sensing hardware. Commercial LiDAR systems provide high spatial fidelity, but they are expensive and often impractical for embedded or consumer-oriented deployments. A single low-cost radar sensor can provide range and Doppler information in a compact form factor, but the raw data must be filtered and accumulated carefully to produce a reliable static occupancy map rather than a noisy point cloud.

## Solution

This project implements a live static 3D occupancy mapping pipeline using a Texas Instruments IWR6843AOP radar sensor. The radar streams point cloud data over its UART data port, the Python application parses the TI frame format, filters points by range, Doppler, and SNR, and accumulates only static detections into a voxel map. The result is rendered in real time in Open3D, with the ability to reset, pause, save the map, and record the raw frames for later analysis.

### Key Features

#### Hardware Implementation

The active system uses the TI IWR6843AOP radar evaluation module connected over the host's serial UART interfaces. The CLI port is used to send the radar configuration, and the data port receives the point-cloud frames produced by the sensor.

Current setup includes:

- 1x TI IWR6843AOP mmWave radar sensor
- USB serial connection to the host computer or Raspberry Pi
- `src/implementation/mapping.cfg` radar profile generated for the TI sensor

The radar is configured through the CLI port with commands ending in `sensorStart`, while the resulting point cloud frames are read from the data port. The implementation expects the radar frame format described by TI's demo output, including point data and optional side information such as SNR values.

The application treats the sensor frame as:

- x = right
- y = forward
- z = up

The world frame is aligned to the same axes, with the origin placed on the floor directly below the sensor. The sensor height and pitch can be adjusted with command-line arguments to match the physical setup.

#### Software Implementation

The active software is located in:

- `src/implementation/single_sensor_implementation.py` — live radar parsing and occupancy mapping
- `src/implementation/mapping.cfg` — TI radar configuration file

The Python application:

- Opens the CLI and data serial ports
- Sends the radar configuration file over the CLI port
- Reads TI frames from the data stream
- Filters by minimum range, maximum range, SNR threshold, and Doppler threshold
- Only counts static points in the map accumulation process
- Renders the live scene and accumulated static map with Open3D
- Supports hotkeys for fit-all, reset, save, and pause/resume

Useful key bindings in the viewer window:

- `F` — fit all
- `R` — reset the map
- `S` — save the current map to a `.ply` file
- `P` — pause/resume accumulation

---

## Demo

https://drive.google.com/drive/u/1/folders/1BuQGlUXtVkGyHvaYbINmNZ1HAqGcTPhS

---

## Screenshots

Processed Python Data:
![alt text](<images/Single_Sensor_Processed_Data.png>)

Sensor Setup:
![alt text](<images/Single_Sensor_Setup.png>)

Open3D Results:
![alt text](<images/Single_Sensor_Open3D.png>)


---

## Tech Stack

| Category | Technology |
| -------- | ---------- |
| Frontend / Visualization | Open3D |
| Backend / Processing | Python, NumPy, pySerial |
| Hardware | TI IWR6843AOP mmWave radar, serial UART interface |
| Data Recording | `.npz` raw frame capture, `.ply` saved map exports |
| CI/CD | GitHub Actions checks and tag-triggered GitHub Releases |

---

## Getting Started

### Prerequisites

- Python 3.9+ (3.10/3.11 recommended)
- `pip` or another Python package manager
- packages listed in `requirements.txt`
- A TI IWR6843AOP sensor connected over serial UART ports for live hardware mode

### Installation

```bash
# Clone the repository
git clone https://github.com/SJSU-CMPE-195/group-project-team-goofy-goobers.git
cd group-project-team-goofy-goobers
```

Install the required Python dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The repository's `sensors.json` contains local serial-port settings. For a new
setup, copy `sensors.example.json` to `src/implementation/sensors.json` and set
the CLI/data port names and sensor poses for your hardware.

### Running Locally

The active implementation requires both the radar CLI port and the data port. The exact COM names or device paths depend on the host machine and USB configuration.

Example on Windows:

```bash
python src\implementation\single_sensor_implementation.py --cli COM5 --data COM6 --cfg src\implementation\mapping.cfg --voxel 0.10 --min-hits 5 --height 1.0 --pitch 0.0
```

### Mock Mode (No Radar Hardware)

The prototype simulator generates synthetic radar frames and runs them through
voxel downsampling, statistical outlier removal, DBSCAN clustering, and
axis-aligned bounding-box rendering. It displays the live point cloud and frame
statistics in a browser, so it can be used without connecting a radar:

```bash
python src\prototype\software\ti_sim.py
```

Open [http://localhost:8050](http://localhost:8050) while the simulator is
running.

### Hardware Mapper Options

Common arguments:

- `--cli` — UART CLI port used to send the radar configuration
- `--data` — data port used to receive mmWave frame packets
- `--cfg` — TI configuration file, such as `src/implementation/mapping.cfg`
- `--voxel` — voxel size used for occupancy accumulation (default `0.10` m)
- `--min-hits` — minimum hits per voxel before being displayed
- `--vthresh` — Doppler threshold for static filtering
- `--min-range` / `--max-range` — range limits in meters
- `--min-snr` — minimum SNR required to keep a point
- `--height` — sensor mounting height above floor in meters
- `--pitch` — sensor tilt in degrees (positive values tilt the sensor down)
- `--no-record` — disable saving raw frame data to disk

To stop the app, close the Open3D window or terminate the Python process. The script automatically sends `sensorStop` before shutdown.

## Project Structure

Current active implementation:

- `src/implementation/`
  - `single_sensor_implementation.py` — real TI radar mapping application
  - `mapping.cfg` — TI radar configuration used at startup

The prototype simulator under `src/prototype/software/` provides a hardware-free
mock/demo path; the active hardware implementation remains the single-sensor
mmWave pipeline described above.

## CI/CD

GitHub Actions runs the `Python checks` job on pushes and pull requests. It
compiles the implementation and prototype Python files and runs the
hardware-free tests with `requirements-ci.txt`. These automated checks do not
validate a physical radar connection.

Run the same checks locally with:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q src/implementation src/prototype/software
python -m pytest -q
```

To publish a downloadable release, push a version tag matching `v*` (for
example, `v0.1.0`). After the checks pass, the workflow creates a GitHub Release
with a ZIP containing the source, runtime dependencies, setup instructions, and
a placeholder `sensors.json`. The release does not include the workstation's
local serial-port settings. Run a live radar smoke test on the target computer
before treating a release as hardware-validated.

After merging the workflow to the branch you want to release, create and push a
tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release job uses the repository's built-in `GITHUB_TOKEN`; no additional
secret is needed. If repository or organization policy restricts workflow
permissions, allow the release job to write repository contents. To block
merges when checks fail, configure the `Python checks` status check as required
in the repository's branch protection rule or ruleset.

### Installing a release bundle

Unzip the release archive, create a virtual environment, and install its
dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The bundled `src/implementation/sensors.json` uses `COM?` placeholders. Replace
them with the CLI and data port names for the connected radar before using the
multi-radar viewer. For the single-sensor mapper, pass the actual ports and
configuration path as shown in [Running Locally](#running-locally).

## Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request


## What's Next

- Validate the live sensor setup on the physical TI IWR6843AOP hardware.
- Tune the radar configuration and filtering thresholds for the target environment.
- Extend the current single-sensor mapping logic toward a synchronized 3 sensor configuration when needed.
- Continue refining the occupancy map for robust 3D scene reconstruction in real-world conditions.

---

_CMPE 195A/B - Senior Design Project | San Jose State University | Spring 2026_

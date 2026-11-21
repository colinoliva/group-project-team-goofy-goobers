# Using the dashboard

`src/implementation/radar_dash_viewer.py` is the web dashboard for the three-radar rig. It shows
where motion is, records sessions for the classifier, and lets you change detection settings
while it runs. Run it from `src/implementation/` (config paths are relative), inside the venv.

## Starting it

| Command | What you get |
|---|---|
| `python3 radar_dash_viewer.py` | Mock data (no hardware). Good for trying the page. |
| `python3 radar_dash_viewer.py --live sensors.json --freerun --sparse --no-classify` | All three radars, sparse mode. The usual setup. |
| `... --record --scenario walk --people NAME --room lab` | Same, and starts a recording immediately. |
| `... --lan` | Reachable from other devices (see the warning below). |

Then open **http://127.0.0.1:8050**. Live mode first learns the empty room for 30 s
(`--learn`): stay out of the sensors' view until the PRESENCE card stops saying "learning".

By default the page is only reachable from the same machine. `--lan` makes it reachable on
the network, for example from a laptop when the dashboard runs on the Raspberry Pi. **The
dashboard has no login**: anyone who can reach the port can start and stop recordings, change
settings and re-learn the background. Only use `--lan` on a network you trust.

Stop it with Ctrl-C. An open recording is closed properly (also on `kill`).

## The page

- **Header**: mode (MOCK DATA or LIVE), classic or sparse, sensor connection, clock.
- **Red strip under the header**: a sensor has sent nothing for more than 2 s, with the time of
  its last frame. Check its USB cable and power; the other sensors keep working.
- **Floor plan and 3D view** (centre; switch in the View card): sensors with their fields of
  view, dim static points, bright moving points in the sensor's colour (S1 orange, S2 blue,
  S3 pink).
  - Sparse mode draws **motion zones**: 1 m circles where motion is.
  - Dotted boxes are clusters that at least 2 sensors agree on.
- **PRESENCE card**: YES (green), NO (grey) or learning (blue), with the zone positions.
  - Two people closer than about 2.5 m show as one zone, so the zone count is a lower bound.
  - Zones are not people and have no IDs.
- **Sensor cards**: frames per second, points per frame, seconds since the last frame, and
  moving points per second over the last minute.
- **Presence strip**: one cell per second for the last 60 s.

"Show tracks (experimental)" draws the tracker's boxes, IDs and trails. On our sparse radar
data the IDs change often (one person can get dozens per minute), so it is off in sparse mode.

## Recording a session

Recording needs live sensors; mock data is never recorded.

1. Wait until learning is done, with the room in the state the scenario needs.
2. In the Recording card choose the **scenario** (walk, two_people, empty, fan, chair, stand,
   sit, pose_check, other) and fill in **people** and **room**. Notes are optional; the rig
   geometry (triangle sides, sensor heights) is added to them automatically.
3. Click **Start recording**. The card shows the folder name (`data/<date>_<time>_<scenario>`),
   elapsed time and frames recorded.
4. Click **Start activity** when the scenario begins (logs `go`) and **End activity** when it
   ends (logs `done`). A second click within 1 s is ignored, so a double click cannot log
   `done` twice. The analysis scripts use the go-done span as the labelled part of the session.
5. Use **Mark event** for anything else worth noting ("chair moved", "door opened"). `go` and
   `done` can only come from the activity button.
6. Click **Stop**, then **Confirm stop** within 5 s. If an activity is still running, `done` is
   logged first.

The last 8 marks are listed in the card (go green, done red, others grey).

Before recording on a changed rig, update the poses in your sensors JSON. Each session stores
the poses it was recorded with. Analysis scripts given `--poses` warn if those differ by more
than 5 cm, because `--poses` must match the rig the session was recorded on.

## Detection settings

The Detection card changes settings on the next frame, without a restart:

- **Mode**: classic (clustering and tracking) or sparse (motion zones; made for our sparse
  data). Switching mode also sets its usual window and min-sensors values.
- **Window (s)**: how long points are pooled before clustering. 0 means no pooling.
- **Min sensors**: a cluster needs at least 2 points from each of this many sensors. 2 removes
  most empty-room false alarms.
- **DBSCAN eps**, **moving threshold**, **classifier filter**, **show last (s)** (display only).
- **Reset to defaults**: back to what the command line set.
- **Presets**: save the current settings under a name and load them later. They are stored in
  `src/implementation/presets/`, which is not committed.
- **Re-learn background** (live only): learns the empty room again, with a progress bar.

While recording, every settings change is saved as a mark such as
`settings window_s=0.5 min_sensors=2`, so a session's settings can be reconstructed later.

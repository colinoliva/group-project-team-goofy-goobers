"""Live 2D mapper for TI ultrasonic or mmWave-style range data.

This script consumes UART lines in the format:

    S1,dist=13.2
    S2,dist=25.7
    S3,dist=42.1

It can read the data from a serial port or generate mock frames when no port
is supplied. Once a full frame is available, it trilaterates the target
position and renders a live 2D map in Dash.
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from collections import deque
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np

from mapping_mock_data import S1, S2, S3, trilaterate

SENSORS = {
    "S1": S1,
    "S2": S2,
    "S3": S3,
}

LINE_PATTERN = re.compile(
    r"^\s*(?P<sensor>S[123])\s*,\s*dist\s*=\s*(?P<distance>-?\d+(?:\.\d+)?)\s*$"
)

COLORS = {
    "S1": "#00d4ff",
    "S2": "#ff6b35",
    "S3": "#44cf6c",
}


def parse_distance_line(line: str) -> Optional[Tuple[str, float]]:
    """Parse a UART range line.

    Returns None for malformed input so the caller can safely ignore noise.
    """

    match = LINE_PATTERN.match(line)
    if not match:
        return None

    return match.group("sensor"), float(match.group("distance"))


def complete_frame(distances: Dict[str, float]) -> bool:
    return all(sensor in distances for sensor in SENSORS)


def estimate_position(distances: Dict[str, float]) -> Optional[np.ndarray]:
    if not complete_frame(distances):
        return None

    return trilaterate(
        SENSORS["S1"],
        SENSORS["S2"],
        SENSORS["S3"],
        distances["S1"],
        distances["S2"],
        distances["S3"],
    )


def generate_mock_uart_lines() -> Iterable[str]:
    """Yield a moving target as UART lines using the same protocol as the MCU."""

    t = time.time()
    target = np.array([
        10.0 + 4.0 * np.sin(t * 0.5),
        8.0 + 3.0 * np.cos(t * 0.35),
    ])

    for sensor_name, sensor_position in SENSORS.items():
        distance = float(np.linalg.norm(target - sensor_position))
        yield f"{sensor_name},dist={distance:.2f}"


def build_figure(trail: Deque[Tuple[float, float]], last_position: Optional[np.ndarray]):
    import plotly.graph_objects as go

    traces = []

    for sensor_name, sensor_position in SENSORS.items():
        traces.append(
            go.Scatter(
                x=[float(sensor_position[0])],
                y=[float(sensor_position[1])],
                mode="markers+text",
                text=[sensor_name],
                textposition="top center",
                name=sensor_name,
                marker=dict(size=14, color=COLORS[sensor_name], line=dict(width=0)),
                hovertemplate=f"{sensor_name}<br>x:%{{x:.2f}} y:%{{y:.2f}}<extra></extra>",
            )
        )

    if trail:
        trail_x = [point[0] for point in trail]
        trail_y = [point[1] for point in trail]
        traces.append(
            go.Scatter(
                x=trail_x,
                y=trail_y,
                mode="lines+markers",
                name="Estimated path",
                line=dict(color="#f59e0b", width=3),
                marker=dict(size=6, color="#f59e0b"),
                hovertemplate="x:%{x:.2f} y:%{y:.2f}<extra></extra>",
            )
        )

    if last_position is not None:
        traces.append(
            go.Scatter(
                x=[float(last_position[0])],
                y=[float(last_position[1])],
                mode="markers",
                name="Current estimate",
                marker=dict(size=18, color="#ef4444"),
                hovertemplate="Estimate<br>x:%{x:.2f} y:%{y:.2f}<extra></extra>",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        uirevision="constant",
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(color="white"), bgcolor="#111827", bordercolor="#374151"),
        xaxis=dict(
            title="X (cm)",
            range=[-5, 25],
            gridcolor="#1f2937",
            zerolinecolor="#374151",
            color="white",
        ),
        yaxis=dict(
            title="Y (cm)",
            range=[-5, 25],
            gridcolor="#1f2937",
            zerolinecolor="#374151",
            color="white",
            scaleanchor="x",
            scaleratio=1,
        ),
        template="plotly_dark",
    )
    return fig


def start_serial_thread(port: str, baud_rate: int, latest_state: Dict[str, object]) -> threading.Thread:
    def loop() -> None:
        try:
            import serial  # type: ignore
        except ImportError:
            latest_state["status"] = "pyserial is not installed"
            return

        try:
            connection = serial.Serial(port, baud_rate, timeout=1)
        except Exception as exc:  # pragma: no cover - hardware-specific failure path
            latest_state["status"] = f"Unable to open {port}: {exc}"
            return

        distances: Dict[str, float] = {}
        latest_state["status"] = f"Listening on {port} at {baud_rate} baud"

        with connection:
            while True:
                raw = connection.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                parsed = parse_distance_line(raw)
                if parsed is None:
                    latest_state["status"] = f"Ignored line: {raw}"
                    continue

                sensor_name, distance = parsed
                distances[sensor_name] = distance
                latest_state["distances"] = dict(distances)

                position = estimate_position(distances)
                if position is not None:
                    latest_state["position"] = position
                    trail = latest_state["trail"]
                    trail.append((float(position[0]), float(position[1])))
                    latest_state["frame_count"] = int(latest_state["frame_count"]) + 1
                    latest_state["status"] = f"Frame {latest_state['frame_count']} received from {port}"

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def start_mock_thread(latest_state: Dict[str, object], sample_delay: float) -> threading.Thread:
    def loop() -> None:
        latest_state["status"] = "Running in mock mode"

        while True:
            distances: Dict[str, float] = {}
            for line in generate_mock_uart_lines():
                parsed = parse_distance_line(line)
                if parsed is None:
                    continue
                sensor_name, distance = parsed
                distances[sensor_name] = distance
                latest_state["distances"] = dict(distances)

            position = estimate_position(distances)
            if position is not None:
                latest_state["position"] = position
                trail = latest_state["trail"]
                trail.append((float(position[0]), float(position[1])))
                latest_state["frame_count"] = int(latest_state["frame_count"]) + 1
                latest_state["status"] = f"Mock frame {latest_state['frame_count']} generated"

            time.sleep(sample_delay)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def make_app(initial_state: Dict[str, object]):
    from dash import Dash, Input, Output, dcc, html

    app = Dash(__name__)

    app.layout = html.Div(
        style={
            "minHeight": "100vh",
            "background": "linear-gradient(180deg, #050816 0%, #0b1020 100%)",
            "fontFamily": "Inter, system-ui, sans-serif",
            "color": "white",
        },
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "padding": "16px 24px",
                    "borderBottom": "1px solid #1f2937",
                    "background": "rgba(10, 15, 30, 0.85)",
                    "backdropFilter": "blur(12px)",
                },
                children=[
                    html.Div(
                        children=[
                            html.Div("Live 2D Mapper", style={"fontSize": "22px", "fontWeight": "700"}),
                            html.Div(
                                "UART ranges from the TI board are trilaterated into a live map.",
                                style={"color": "#94a3b8", "fontSize": "13px"},
                            ),
                        ]
                    ),
                    html.Div(id="status-bar", style={"textAlign": "right", "fontSize": "13px", "color": "#cbd5e1"}),
                ],
            ),
            html.Div(
                style={"padding": "20px"},
                children=[
                    dcc.Graph(
                        id="map-plot",
                        style={"height": "78vh"},
                        config={"displayModeBar": False},
                    ),
                ],
            ),
            dcc.Interval(id="refresh", interval=150, n_intervals=0),
        ],
    )

    @app.callback(
        Output("map-plot", "figure"),
        Output("status-bar", "children"),
        Input("refresh", "n_intervals"),
    )
    def update_view(_: int):
        trail = initial_state["trail"]
        position = initial_state["position"]
        distances = initial_state["distances"]
        status = initial_state["status"]

        figure = build_figure(trail, position)

        if distances:
            distance_text = " | ".join(
                f"{sensor}: {distance:.2f} cm" for sensor, distance in sorted(distances.items())
            )
        else:
            distance_text = "Waiting for sensor data..."

        status_text = html.Div([
            html.Div(status),
            html.Div(distance_text, style={"color": "#94a3b8", "marginTop": "4px"}),
        ])

        return figure, status_text

    return app


def build_initial_state(history_size: int) -> Dict[str, object]:
    return {
        "distances": {},
        "position": None,
        "trail": deque(maxlen=history_size),
        "frame_count": 0,
        "status": "Starting...",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live 2D mapper for TI distance data.")
    parser.add_argument("--serial-port", help="Serial port path, for example /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--history-size", type=int, default=30, help="Number of mapped positions to keep")
    parser.add_argument("--sample-delay", type=float, default=0.1, help="Mock mode update interval in seconds")
    parser.add_argument("--host", default="0.0.0.0", help="Dash host")
    parser.add_argument("--port", type=int, default=8050, help="Dash port")
    parser.add_argument("--mock", action="store_true", help="Force mock data instead of serial input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latest_state = build_initial_state(args.history_size)

    if args.serial_port and not args.mock:
        start_serial_thread(args.serial_port, args.baud, latest_state)
    else:
        start_mock_thread(latest_state, args.sample_delay)

    app = make_app(latest_state)
    print("\nLive 2D mapper running")
    if args.serial_port and not args.mock:
        print(f"Reading serial data from {args.serial_port} at {args.baud} baud")
    else:
        print("Running in mock mode")
    print(f"Open http://localhost:{args.port} in your browser\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
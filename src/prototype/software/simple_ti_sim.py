import numpy as np
import time
import threading
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input

# ── Mock data generator ───────────────────────────────────────────────────────

def generate_mock_ti_frame():
    t = time.time()

    obj1_center = [1.0 + 0.3 * np.sin(t * 0.5),  0.5 * np.cos(t * 0.3), 0.5]
    obj2_center = [-0.8 + 0.5 * np.cos(t * 0.8), 1.0 + 0.3 * np.sin(t), 0.4]

    obj1   = np.random.randn(60, 3) * [0.08, 0.08, 0.04] + obj1_center
    obj2   = np.random.randn(60, 3) * [0.08, 0.08, 0.04] + obj2_center
    ground = np.random.uniform(-2, 2, (20, 3))
    ground[:, 2] = np.random.uniform(0, 0.05, 20)
    noise  = np.random.uniform(-2, 2, (15, 3))

    points = np.vstack([obj1, obj2, ground, noise])

    # Label each group so we can color by object
    labels = (
        ['Object 1'] * 60 +
        ['Object 2'] * 60 +
        ['Ground']   * 20 +
        ['Noise']    * 15
    )
    return points, labels

# ── Shared state (thread-safe enough for this use case) ───────────────────────

latest = {"points": None, "labels": None}

def data_loop():
    while True:
        points, labels = generate_mock_ti_frame()
        latest["points"] = points
        latest["labels"] = labels
        time.sleep(0.1)   # 10 Hz update rate

threading.Thread(target=data_loop, daemon=True).start()

# ── Dash app ──────────────────────────────────────────────────────────────────

app = Dash(__name__)

COLOR_MAP = {
    "Object 1": "#00d4ff",
    "Object 2": "#ff6b35",
    "Ground":   "#44cf6c",
    "Noise":    "#888888",
}

app.layout = html.Div(
    style={"backgroundColor": "#0d0d0d", "height": "100vh", "padding": "0"},
    children=[
        html.H3(
            "TI mmWave Radar — Live Simulation",
            style={"color": "#ffffff", "textAlign": "center",
                   "margin": "0", "padding": "12px",
                   "fontFamily": "monospace", "fontSize": "16px"}
        ),
        dcc.Graph(
            id="radar-plot",
            style={"height": "90vh"},
            config={"displayModeBar": True}
        ),
        dcc.Interval(
            id="interval",
            interval=150,    # ms — browser polls every 150 ms
            n_intervals=0
        ),
    ]
)

@app.callback(
    Output("radar-plot", "figure"),
    Input("interval", "n_intervals")
)
def update_plot(_):
    points = latest["points"]
    labels = latest["labels"]

    if points is None:
        return go.Figure()

    # One trace per group so legend works cleanly
    traces = []
    for group, color in COLOR_MAP.items():
        mask = np.array(labels) == group
        pts  = points[mask]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            name=group,
            marker=dict(
                size=6 if group in ("Object 1", "Object 2") else 4,
                color=color,
                opacity=1.0 if group != "Noise" else 0.6,
                line=dict(width=0), # removes outline that muddies small points
            )
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        uirevision="constant",      # keeps camera angle between updates
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        legend=dict(
            font=dict(color="white"),
            bgcolor="#1a1a1a",
            bordercolor="#444"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(range=[-2, 2], title="X (m)",
                       backgroundcolor="#111", gridcolor="#333",
                       color="white"),
            yaxis=dict(range=[-2, 2], title="Y (m)",
                       backgroundcolor="#111", gridcolor="#333",
                       color="white"),
            zaxis=dict(range=[-0.1, 1.5], title="Z (m)",
                       backgroundcolor="#111", gridcolor="#333",
                       color="white"),
            bgcolor="#111111",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),  # flatten Z so points aren't stretched
        ),
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.0),   # closer starting viewpoint
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
    )
    return fig

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Simulation running!")
    print("  Open your browser to:  http://localhost:8050")
    print("  On another device:     http://<your-pi-ip>:8050")
    print("  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=8050, debug=False)
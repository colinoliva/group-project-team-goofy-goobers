import numpy as np
import time
import threading
import open3d as o3d
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input

# ── Mock TI data generator ────────────────────────────────────────────────────

def generate_mock_ti_frame():
    t = time.time()

    obj1_center = [1.0 + 0.3 * np.sin(t * 0.5),  0.5 * np.cos(t * 0.3), 0.5]
    obj2_center = [-0.8 + 0.5 * np.cos(t * 0.8), 1.0 + 0.3 * np.sin(t), 0.4]

    obj1   = np.random.randn(60, 3) * [0.08, 0.08, 0.04] + obj1_center
    obj2   = np.random.randn(60, 3) * [0.08, 0.08, 0.04] + obj2_center
    ground = np.random.uniform(-2, 2, (20, 3))
    ground[:, 2] = np.random.uniform(0, 0.05, 20)
    noise  = np.random.uniform(-2, 2, (15, 3))

    return np.vstack([obj1, obj2, ground, noise])

# ── Open3D processing pipeline ────────────────────────────────────────────────

def process_frame(raw_points):
    """
    Run raw points through Open3D:
      1. Voxel downsample      — reduce noise density
      2. Statistical outlier removal — strip extreme noise
      3. DBSCAN clustering     — label each detected object
      4. Bounding boxes        — one per cluster
    Returns processed points, cluster labels, and bounding box corners.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points)

    # 1 — Voxel downsample (merges points within 5 cm cubes)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # 2 — Statistical outlier removal
    #     nb_neighbors=20: looks at 20 nearest neighbours
    #     std_ratio=1.5:   removes points > 1.5 std devs from mean distance
    pcd, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=1.5
    )

    points = np.asarray(pcd.points)
    if len(points) == 0:
        return points, np.array([]), []

    # 3 — DBSCAN clustering
    #     eps=0.15:        points within 15 cm are neighbours
    #     min_points=5:    need at least 5 points to form a cluster
    labels = np.array(
        pcd.cluster_dbscan(eps=0.15, min_points=5, print_progress=False)
    )

    # 4 — Bounding boxes (one per valid cluster, label >= 0)
    boxes = []
    n_clusters = labels.max() + 1 if len(labels) > 0 and labels.max() >= 0 else 0

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if mask.sum() < 5:
            continue

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(points[mask])
        bbox = cluster_pcd.get_axis_aligned_bounding_box()

        min_b = np.asarray(bbox.min_bound)
        max_b = np.asarray(bbox.max_bound)
        boxes.append((cluster_id, min_b, max_b))

    return points, labels, boxes

def bbox_to_plotly_lines(min_b, max_b, color):
    """
    Convert an axis-aligned bounding box into 12 edge lines for Plotly.
    Returns a single Scatter3d trace drawing the wireframe box.
    """
    x0, y0, z0 = min_b
    x1, y1, z1 = max_b

    # 8 corners of the box
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])

    # 12 edges: pairs of corner indices
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # bottom face
        (4,5),(5,6),(6,7),(7,4),   # top face
        (0,4),(1,5),(2,6),(3,7),   # vertical edges
    ]

    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=3),
        showlegend=False,
        hoverinfo="skip",
    )

# ── Cluster color palette ─────────────────────────────────────────────────────

CLUSTER_COLORS = [
    "#00d4ff",  # cyan
    "#ff6b35",  # orange
    "#a855f7",  # purple
    "#44cf6c",  # green
    "#f59e0b",  # amber
    "#ef4444",  # red
]

def get_color(cluster_id):
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

# ── Shared state ──────────────────────────────────────────────────────────────

latest = {
    "points": None,
    "labels": None,
    "boxes":  None,
    "stats":  {"raw": 0, "filtered": 0, "clusters": 0},
}

def data_loop():
    while True:
        raw   = generate_mock_ti_frame()
        pts, labels, boxes = process_frame(raw)
        n_clusters = int(labels.max()) + 1 if len(labels) and labels.max() >= 0 else 0

        latest["points"] = pts
        latest["labels"] = labels
        latest["boxes"]  = boxes
        latest["stats"]  = {
            "raw":      len(raw),
            "filtered": len(pts),
            "clusters": n_clusters,
        }
        time.sleep(0.1)

threading.Thread(target=data_loop, daemon=True).start()

# ── Dash app ──────────────────────────────────────────────────────────────────

app = Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": "#0d0d0d", "height": "100vh",
           "fontFamily": "monospace"},
    children=[

        # Header + live stats bar
        html.Div(
            style={"display": "flex", "alignItems": "center",
                   "justifyContent": "space-between",
                   "padding": "10px 20px", "backgroundColor": "#1a1a1a",
                   "borderBottom": "1px solid #333"},
            children=[
                html.H3("TI mmWave Radar — Live Simulation",
                        style={"color": "#ffffff", "margin": "0",
                               "fontSize": "15px"}),
                html.Div(id="stats-bar",
                         style={"color": "#aaaaaa", "fontSize": "13px"}),
            ]
        ),

        # 3D plot
        dcc.Graph(
            id="radar-plot",
            style={"height": "88vh"},
            config={"displayModeBar": True}
        ),

        dcc.Interval(id="interval", interval=150, n_intervals=0),
    ]
)

@app.callback(
    Output("radar-plot", "figure"),
    Output("stats-bar", "children"),
    Input("interval", "n_intervals")
)
def update_plot(_):
    points = latest["points"]
    labels = latest["labels"]
    boxes  = latest["boxes"]
    stats  = latest["stats"]

    stats_text = (
        f"Raw points: {stats['raw']}  |  "
        f"After filtering: {stats['filtered']}  |  "
        f"Clusters detected: {stats['clusters']}"
    )

    if points is None or len(points) == 0:
        return go.Figure(), stats_text

    traces = []

    # ── Point cloud — colored by cluster ─────────────────────────────────────
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        pts  = points[mask]

        if lbl == -1:
            # Unclustered noise — gray, small, faint
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                name="Unclassified",
                marker=dict(size=2, color="#555555", opacity=0.4),
                hovertemplate="Noise<br>x:%{x:.2f} y:%{y:.2f} z:%{z:.2f}<extra></extra>",
            ))
        else:
            color = get_color(lbl)
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                name=f"Cluster {lbl}",
                marker=dict(size=6, color=color, opacity=1.0,
                            line=dict(width=0)),
                hovertemplate=(
                    f"Cluster {lbl}<br>"
                    "x:%{x:.2f} y:%{y:.2f} z:%{z:.2f}<extra></extra>"
                ),
            ))

    # ── Bounding boxes ────────────────────────────────────────────────────────
    if boxes:
        for cluster_id, min_b, max_b in boxes:
            color = get_color(cluster_id)
            traces.append(bbox_to_plotly_lines(min_b, max_b, color))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = go.Figure(data=traces)
    fig.update_layout(
        uirevision="constant",
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        legend=dict(font=dict(color="white"), bgcolor="#1a1a1a",
                    bordercolor="#444"),
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
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
    )
    return fig, stats_text

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Simulation running!")
    print("  Open your browser to: http://localhost:8050\n")
    app.run(host="0.0.0.0", port=8050, debug=False)
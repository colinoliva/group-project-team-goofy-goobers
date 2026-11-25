"""Simple rule-based cluster classifier."""

HEIGHT_MIN = 0.8
HEIGHT_MAX = 2.1
XY_MAX = 1.2
MIN_POINTS = 8


def is_human_rule(features):
    """Classify a cluster using fixed human-shape thresholds."""
    height = features["height_extent"]
    width = features["xy_extent"]
    points = features["n_points"]
    return HEIGHT_MIN <= height <= HEIGHT_MAX and width < XY_MAX and points >= MIN_POINTS

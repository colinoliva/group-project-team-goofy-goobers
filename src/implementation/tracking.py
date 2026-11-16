#!/usr/bin/env python3
"""
Background subtraction, clustering and multi-person tracker

Pipeline per frame (world-frame points in):
    foreground = moving points  OR  points that are not part of the learned static scene
    DBSCAN clusters the foreground -> one detection per cluster
    Tracker gives each person a persistent ID (nearest-neighbour association, xy only,
    because elevation is the AOP's weakest axis)
"""
import itertools
from collections import deque

import numpy as np
from sklearn.cluster import DBSCAN

_OFFS = list(itertools.product((-1, 0, 1), repeat=3))


class Background:
    """Voxel model of the static room. learn() while the room is empty, then freeze()."""

    def __init__(self, voxel=0.15, min_hits=4):
        self.v, self.min_hits = voxel, min_hits
        self.reset()

    def reset(self):
        self.counts, self.occupied, self.frozen = {}, set(), False

    def learn(self, xyz):
        if self.frozen or len(xyz) == 0:
            return
        for k in map(tuple, np.unique(np.floor(xyz / self.v).astype(np.int32), axis=0).tolist()):
            self.counts[k] = self.counts.get(k, 0) + 1

    def freeze(self):
        self.occupied = {k for k, c in self.counts.items() if c >= self.min_hits}
        self.frozen = True

    def is_background(self, xyz):
        """True for points inside (or touching) a learned static voxel."""
        out = np.zeros(len(xyz), bool)
        if not self.frozen or len(xyz) == 0:
            return out
        occ = self.occupied
        keys = np.floor(xyz / self.v).astype(np.int32).tolist()
        for i, (kx, ky, kz) in enumerate(keys):
            out[i] = any((kx + dx, ky + dy, kz + dz) in occ for dx, dy, dz in _OFFS)
        return out

    def points(self):
        src = self.occupied if self.frozen else [k for k, c in list(self.counts.items()) if c >= self.min_hits]
        if not len(src):
            return np.empty((0, 3))
        return (np.array(list(src), float) + 0.5) * self.v


def cluster(xyz, eps=0.4, min_samples=4):
    if len(xyz) < min_samples:
        return np.full(len(xyz), -1, int)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz)


def detections(xyz, labels, min_points=5):
    dets = []
    for lbl in sorted(set(labels.tolist()) - {-1}):
        m = labels == lbl
        if m.sum() < min_points:
            continue
        p = xyz[m]
        dets.append({"idx": np.flatnonzero(m), "c": p.mean(0), "n": int(m.sum()),
                     "lo": p.min(0), "hi": p.max(0)})
    return dets


class Track:
    def __init__(self, tid, pos, t):
        self.id, self.pos, self.t = tid, np.array(pos, float), t
        self.vel = np.zeros(3)
        self.hits, self.misses = 1, 0
        self.trail = deque([self.pos.copy()], maxlen=40)


class Tracker:
    def __init__(self, gate=0.9, max_misses=25, confirm_hits=3, alpha=0.6, min_new_points=6):
        self.gate, self.max_misses, self.confirm_hits = gate, max_misses, confirm_hits
        self.alpha, self.min_new = alpha, min_new_points
        self.tracks, self.next_id = [], 1

    def update(self, dets, t):
        """Returns (confirmed_tracks, {detection_index: track_id}) for confirmed tracks."""
        pairs = []
        for ti, tr in enumerate(self.tracks):
            pred = tr.pos + tr.vel * min(max(t - tr.t, 0.0), 0.5)
            for di, d in enumerate(dets):
                dist = float(np.linalg.norm(d["c"][:2] - pred[:2]))
                if dist < min(self.gate + 0.1 * tr.misses, 2.0):   # widen while coasting
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d, assign = set(), set(), {}
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr, d = self.tracks[ti], dets[di]
            dt = max(t - tr.t, 1e-3)
            new = tr.pos + self.alpha * (d["c"] - tr.pos)
            tr.vel = 0.6 * tr.vel + 0.4 * (new - tr.pos) / dt
            tr.pos, tr.t = new, t
            tr.hits += 1
            tr.misses = 0
            tr.trail.append(new.copy())
            assign[di] = tr
        for ti, tr in enumerate(self.tracks):            # coast unmatched tracks briefly
            if ti not in used_t:
                tr.misses += 1
                tr.pos = tr.pos + tr.vel * min(max(t - tr.t, 0.0), 0.3) * 0.5
                tr.vel *= 0.8
                tr.t = t
        for di, d in enumerate(dets):
            if di not in used_d and d["n"] >= self.min_new:
                tr = Track(self.next_id, d["c"], t)
                self.next_id += 1
                self.tracks.append(tr)
                assign[di] = tr
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        confirmed = [tr for tr in self.tracks if tr.hits >= self.confirm_hits]
        ids = {tr.id for tr in confirmed}
        return confirmed, {di: tr.id for di, tr in assign.items() if tr.id in ids}
    
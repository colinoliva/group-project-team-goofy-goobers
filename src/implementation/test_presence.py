"""Unit tests for presence.HeatMap.peaks (synthetic points, deterministic)."""
import unittest

import numpy as np

from presence import HEAT_DEFAULTS, HeatMap, TICK_S


def heat_map():
    p = HEAT_DEFAULTS
    return HeatMap(3, p["sigma"], p["tau"], p["threshold"], int(p["min-sensors"]), p["static-weight"])


def feed(hm, centres, ticks=20, per_sensor=2, sensors=(0, 1, 2), scatter=0.4, seed=0):
    """Each tick, every sensor in `sensors` sees per_sensor moving points around each centre."""
    rng = np.random.default_rng(seed)
    out = None
    for k in range(ticks):
        xyz, sid = [np.empty((0, 3))], [np.empty(0, int)]
        for c in centres:
            for s in sensors:
                xy = np.asarray(c) + rng.normal(0, scatter, (per_sensor, 2))
                xyz.append(np.column_stack([xy, np.full(per_sensor, 1.0)]))
                sid.append(np.full(per_sensor, s))
        xyz, sid = np.vstack(xyz), np.concatenate(sid)
        mv = np.ones(len(xyz), bool)
        out = hm.update(k * TICK_S, xyz, sid, mv, ~mv)
    return out


class PeaksTest(unittest.TestCase):
    def test_no_motion_gives_no_peaks(self):
        hm = heat_map()
        self.assertIsNone(feed(hm, []))
        self.assertEqual(hm.peaks(), [])

    def test_one_source_gives_one_peak_at_the_single_peak(self):
        hm = heat_map()
        single = feed(hm, [(0.5, -0.5)])
        peaks = hm.peaks()
        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0][0], single[0])
        self.assertAlmostEqual(peaks[0][1], single[1])
        self.assertLess(np.hypot(peaks[0][0] - 0.5, peaks[0][1] + 0.5), 0.5)

    def test_two_sources_give_two_peaks_strongest_first(self):
        hm = heat_map()
        rng = np.random.default_rng(1)
        for k in range(20):            # source A: 3 points per sensor, source B: 1 per sensor
            xyz, sid = [], []
            for c, n in (((-1.5, 0.0), 3), ((1.5, 0.5), 1)):
                for s in range(3):
                    xy = np.asarray(c) + rng.normal(0, 0.3, (n, 2))
                    xyz.append(np.column_stack([xy, np.ones(n)]))
                    sid.append(np.full(n, s))
            xyz, sid = np.vstack(xyz), np.concatenate(sid)
            mv = np.ones(len(xyz), bool)
            hm.update(k * TICK_S, xyz, sid, mv, ~mv)
        peaks = hm.peaks()
        self.assertEqual(len(peaks), 2)
        self.assertGreater(peaks[0][2], peaks[1][2])
        self.assertLess(np.hypot(peaks[0][0] + 1.5, peaks[0][1]), 0.5)
        self.assertLess(np.hypot(peaks[1][0] - 1.5, peaks[1][1] - 0.5), 0.5)

    def test_one_sensor_only_gives_no_peak(self):
        hm = heat_map()
        feed(hm, [(0.0, 0.0)], sensors=(1,))
        self.assertEqual(hm.peaks(), [])

    def test_close_sources_are_suppressed_and_k_limits(self):
        hm = heat_map()
        feed(hm, [(0.0, 0.0), (0.6, 0.0)])
        self.assertEqual(len(hm.peaks(min_sep=1.5)), 1)
        hm = heat_map()
        feed(hm, [(-3.0, 0.0), (0.0, 0.0), (3.0, 0.0), (0.0, 3.0)])
        self.assertEqual(len(hm.peaks(k=3)), 3)
        self.assertEqual(len(hm.peaks(k=5)), 4)

    def test_peaks_does_not_change_the_grids(self):
        hm = heat_map()
        feed(hm, [(0.0, 0.0)])
        G, Gs = hm.G.copy(), hm.Gs.copy()
        hm.peaks()
        np.testing.assert_array_equal(G, hm.G)
        np.testing.assert_array_equal(Gs, hm.Gs)


class MockZonesByDistanceTest(unittest.TestCase):
    """2-person sparse mock through the viewer's zone settings, judged by the people's distance.

    Zones merge when sources are closer than about 2.5 m (point scatter + smoothing), so the
    zone count is only checked where it is well defined: >= 3 m apart -> 2 zones, < 1.5 m -> 1.
    """
    BINS = [(0.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 99.0)]

    def test_zone_count_by_distance(self):
        from radar_dash_viewer import ZONES_K, ZONES_MIN_SEP, MockSource
        from tracking import Background
        mock = MockSource(n_people=2)
        mock.rng = np.random.default_rng(12345)       # repeatable run; the mock itself unchanged
        hm = heat_map()
        bg = Background()
        for _ in range(5):                             # as Engine does for mock sources
            bg.learn(mock.background_points())
        bg.freeze()
        rows = []
        for k in range(3000):                          # 300 s at 0.1 s ticks
            t = 5000.0 + k * TICK_S
            fr = mock.sparse_frame(t)
            mv = np.abs(fr["vel"]) >= 0.1
            hm.update(t, fr["xyz"], fr["sid"], mv, ~mv & ~bg.is_background(fr["xyz"]))
            if k < 30:                                 # heat map warming up
                continue
            a, b = np.array(mock.people(t))
            rows.append((float(np.linalg.norm(a - b)), len(hm.peaks(k=ZONES_K, min_sep=ZONES_MIN_SEP))))
        dist = np.array([r[0] for r in rows])
        zones = np.array([r[1] for r in rows])

        print(f"\n2-person sparse mock, {len(rows)} ticks: zones per tick by distance between the people")
        print(f"{'distance':>12}{'ticks':>7}{'0 zones':>9}{'1 zone':>8}{'2 zones':>9}{'3 zones':>9}")
        share = {}
        for lo, hi in self.BINS:
            sel = (dist >= lo) & (dist < hi)
            n = int(sel.sum())
            fr_ = [float(np.mean(zones[sel] == z)) if n else float("nan") for z in range(4)]
            share[(lo, hi)] = fr_
            label = f"{lo:.1f}-{hi:.1f} m" if hi < 99 else f">= {lo:.1f} m"
            print(f"{label:>12}{n:>7}" + "".join(f"{v:>9.0%}" if z != 1 else f"{v:>8.0%}"
                                                for z, v in enumerate(fr_)))

        far, near = share[(3.0, 99.0)], share[(0.0, 1.5)]
        self.assertGreater(int(np.sum(dist >= 3.0)), 100, "too few ticks with the people >= 3 m apart")
        self.assertGreater(int(np.sum(dist < 1.5)), 100, "too few ticks with the people < 1.5 m apart")
        self.assertGreaterEqual(far[2], 0.90, f">= 3 m apart: 2 zones on only {far[2]:.0%} of ticks")
        self.assertGreaterEqual(near[1], 0.90, f"< 1.5 m apart: 1 zone on only {near[1]:.0%} of ticks")


if __name__ == "__main__":
    unittest.main()

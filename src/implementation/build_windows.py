#!/usr/bin/env python3
"""Turn recorded sessions into labeled window-feature rows (presence.WINDOW_FEATURES).

usage: python3 build_windows.py SESSION[:LEARN_S] ... --out datasets/windows.csv
           [--poses sensors.mac.json] [--learn-s 30]
then:  python3 train_classifier.py --csv datasets/windows.csv --group-by session

Per session: points, background and active window exactly as in eval_detection.py /
presence.py; one row per 1.0 s window, 0.5 s apart, lying inside the active window
(go -> done; for person sessions without both, inferred from moving points; for quiet
sessions, everything after learning).
Labels from meta scenario: walk, two_people -> human; empty, fan, chair -> nonhuman;
stand, sit, pose_check and anything else are skipped.
Rows are appended (the file is never overwritten). A session whose session_id is already
in the output file is skipped, so rerunning cannot duplicate rows.
"""
import argparse
import csv
import json
from pathlib import Path

from eval_detection import parse_session, prepare
from presence import WINDOW_FEATURES, window_features, window_ranges

LABELS = {"walk": "human", "two_people": "human",
          "empty": "nonhuman", "fan": "nonhuman", "chair": "nonhuman"}
COLUMNS = WINDOW_FEATURES + ["label", "session_id", "person"]


def existing_sessions(path):
    """Session ids already in the output CSV; refuses a file with a different header."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, newline="") as f:
        reader = csv.reader(f)
        if next(reader, []) != COLUMNS:
            raise SystemExit(f"{path} has a different header; refusing to append")
        i = COLUMNS.index("session_id")
        return {row[i] for row in reader if len(row) > i}


def main():
    ap = argparse.ArgumentParser(description="Build labeled window-feature rows from sessions.")
    ap.add_argument("sessions", nargs="+", help="data/<session> or data/<session>:LEARN_S")
    ap.add_argument("--out", required=True, help="CSV to append to")
    ap.add_argument("--poses", default=None, help="sensors JSON with pose blocks (overrides meta.json T)")
    ap.add_argument("--learn-s", dest="learn_s", type=float, default=30.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_sessions(out)
    new_file = not out.exists() or out.stat().st_size == 0
    total = 0
    with open(out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(COLUMNS)
        for arg in args.sessions:
            path, learn_s = parse_session(arg, args.learn_s)
            sid = Path(path).name
            if sid in done:
                print(f"{sid}: skipped, already in {out}")
                continue
            sess = prepare(path, learn_s, args.poses)
            label = LABELS.get(sess["scenario"])
            if label is None:
                print(f"{sid}: skipped, scenario {sess['scenario']!r} has no label")
                continue
            people = ""
            meta_path = Path(path) / "meta.json"
            if meta_path.exists():
                people = json.loads(meta_path.read_text()).get("people", "")
            a0, a1 = sess["active"]
            n = 0
            for k0, k1 in window_ranges(sess, a0, a1):
                feats = window_features(sess, k0, k1)
                writer.writerow([repr(float(feats[c])) for c in WINDOW_FEATURES] + [label, sid, people])
                n += 1
            total += n
            print(f"{sid}: {sess['scenario']} -> {label}, people {people!r}, active "
                  f"{a0 - sess['start']:.1f}-{a1 - sess['start']:.1f} s ({sess['how']}), {n} windows")
    print(f"{'created' if new_file else 'appended to'} {out}: {total} new rows")


if __name__ == "__main__":
    main()

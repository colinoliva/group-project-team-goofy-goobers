"""Train and evaluate a session-grouped human cluster classifier."""

import argparse
import csv
import json
import os
import re

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from features import FEATURE_NAMES
from make_synthetic import write_synthetic
import rule_baseline
from rule_baseline import is_human_rule


def load_dataset(path):
    """Load features, labels, session IDs and people (None if no person column) from CSV."""
    features, labels, sessions, people = [], [], [], []
    with open(path, "r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        has_person = "person" in (reader.fieldnames or [])
        for row in reader:
            vector = [float(row[name]) for name in FEATURE_NAMES]
            if np.isfinite(vector).all():
                features.append(vector)
                labels.append(1 if row["label"].lower() == "human" else 0)
                sessions.append(row["session_id"])
                people.append(row["person"] if has_person else "")
    return (np.asarray(features), np.asarray(labels), np.asarray(sessions),
            np.asarray(people) if has_person else None)


def split_people(value):
    """Names in a person field such as "raghav" or "raghav,alex"; blank/none -> []."""
    names = [n.strip().lower() for n in re.split(r"[,+;]", value or "")]
    return [n for n in names if n and n not in ("none", "nobody", "-")]


def make_groups(sessions, people):
    """Group ids such that no session and no person appears on both sides of a split.

    Sessions that share a person are merged into one group (union-find), so a person's
    rows are never split between train and test, and neither are a session's rows.
    """
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for i, session in enumerate(sessions):
        find(("s", session))
        if people is not None:
            for name in split_people(people[i]):
                union(("s", session), ("p", name))
    roots = [find(("s", session)) for session in sessions]
    ids = {root: k for k, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([ids[root] for root in roots])


def new_model(y_train):
    """XGBoost with the same settings as before, weighted for class balance."""
    pos = int(np.count_nonzero(y_train == 1))
    neg = int(np.count_nonzero(y_train == 0))
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=neg / pos if pos else 1.0,
        eval_metric="logloss",
    )


def rule_predict(x):
    """Rule baseline prediction for each feature row."""
    return np.asarray([int(is_human_rule(dict(zip(FEATURE_NAMES, row.tolist())))) for row in x])


METRICS = ("precision", "recall", "f1", "FPR(nonhuman)")


def fold_scores(truth, predicted):
    """precision, recall, f1 and false-positive rate on the nonhuman rows.

    NaN (printed n/a, left out of mean/std) where the test fold lacks a needed class:
    precision and f1 need both classes, recall needs human rows, FPR needs nonhuman rows.
    """
    has_human = bool(np.any(truth == 1))
    has_nonhuman = bool(np.any(truth == 0))
    both = has_human and has_nonhuman
    tn, fp, _, _ = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return (precision_score(truth, predicted, zero_division=0) if both else np.nan,
            recall_score(truth, predicted, zero_division=0) if has_human else np.nan,
            f1_score(truth, predicted, zero_division=0) if both else np.nan,
            fp / (fp + tn) if has_nonhuman else np.nan)


def format_scores(scores):
    """One fold's metrics as text, n/a where undefined."""
    return " ".join(f"{label}={v:.3f}" if np.isfinite(v) else f"{label}=n/a"
                    for label, v in zip(METRICS, scores))


def print_summary(name, scores, truth, predicted):
    """Mean +/- std across folds, plus the confusion matrix pooled over all folds."""
    scores = np.asarray(scores, dtype=float)
    parts = []
    for k, label in enumerate(METRICS):
        col = scores[:, k][np.isfinite(scores[:, k])]
        parts.append(f"{label}={np.mean(col):.3f}±{np.std(col):.3f} ({len(col)}/{len(scores)} folds)"
                     if len(col) else f"{label}=n/a (0/{len(scores)} folds)")
    print(f"{name}: " + " ".join(parts))
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    print(f"{name}: confusion matrix over all folds (rows = true, cols = predicted)")
    print(f"{'':16}{'pred nonhuman':>14}{'pred human':>12}")
    print(f"{'true nonhuman':16}{tn:>14}{fp:>12}")
    print(f"{'true human':16}{fn:>14}{tp:>12}")


def train(path, save=False, group_by="session+person"):
    """Grouped cross-validation of XGBoost and the rule baseline; optionally save."""
    x, y, sessions, people = load_dataset(path)
    if len(x) == 0:
        raise SystemExit("No finite feature rows found")
    n_human = int(np.count_nonzero(y == 1))
    n_nonhuman = int(np.count_nonzero(y == 0))
    print(f"data: {path} | rows {len(y)} | human {n_human} | nonhuman {n_nonhuman}")
    for label, count in (("human", n_human), ("nonhuman", n_nonhuman)):
        if count < 30:
            print(f"WARNING: only {count} {label} rows (< 30); metrics will be unreliable")

    use_people = people if group_by == "session+person" else None
    groups = make_groups(sessions, use_people)
    n_groups = len(np.unique(groups))
    how = "session_id and person" if use_people is not None else "session_id"
    print(f"split by {how}: {len(np.unique(sessions))} sessions -> {n_groups} independent groups")
    if n_groups < 2:
        raise SystemExit("Need at least two independent groups for a held-out split "
                         "(sessions that share a person are merged into one group)")

    n_splits = min(5, n_groups)
    xgb_scores, rule_scores = [], []
    truth_all, xgb_all, rule_all = [], [], []
    for fold, (train_index, test_index) in enumerate(
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0).split(x, y, groups),
            start=1):
        y_train, y_test = y[train_index], y[test_index]
        print(f"fold {fold}: test rows {len(y_test)} (human {int(np.sum(y_test == 1))}, "
              f"nonhuman {int(np.sum(y_test == 0))}), test groups {len(np.unique(groups[test_index]))}")
        if len(np.unique(y_train)) < 2:
            print("  skipped (training rows have only one class)")
            continue
        model = new_model(y_train)
        model.fit(x[train_index], y_train)
        xgb_pred = model.predict(x[test_index])
        rule_pred = rule_predict(x[test_index])
        xgb_scores.append(fold_scores(y_test, xgb_pred))
        rule_scores.append(fold_scores(y_test, rule_pred))
        truth_all.append(y_test)
        xgb_all.append(xgb_pred)
        rule_all.append(rule_pred)
        print(f"  XGBoost       {format_scores(xgb_scores[-1])}")
        print(f"  Rule baseline {format_scores(rule_scores[-1])}")
    if not xgb_scores:
        raise SystemExit("No usable folds")

    print(f"\n{len(xgb_scores)} folds trained; metrics as mean ± std over the folds where "
          f"each metric is defined (count in brackets)")
    print(f"rule baseline thresholds: height {rule_baseline.HEIGHT_MIN:g}-{rule_baseline.HEIGHT_MAX:g} m, "
          f"xy_extent < {rule_baseline.XY_MAX:g} m, n_points >= {rule_baseline.MIN_POINTS}")
    truth_all = np.concatenate(truth_all)
    print_summary("XGBoost", xgb_scores, truth_all, np.concatenate(xgb_all))
    print_summary("Rule baseline", rule_scores, truth_all, np.concatenate(rule_all))

    model = new_model(y)
    model.fit(x, y)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    print("XGBoost feature importances (model fit on all rows):")
    for index in order:
        print(f"  {FEATURE_NAMES[index]}={importances[index]:.4f}")
    print("Rule baseline feature importances:")
    for name in ("height_extent", "xy_extent", "n_points"):
        print(f"  {name}=n/a")

    if save:
        model.save_model("model.json")
        with open("feature_names.json", "w", encoding="utf-8") as output:
            json.dump(FEATURE_NAMES, output)
        print("Saved model.json and feature_names.json (fit on all rows)")
    else:
        print("Not saved (pass --save to write model.json)")


def main():
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train the radar cluster classifier.")
    parser.add_argument("csv", nargs="?", default="datasets/radar.csv")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--csv", dest="csv_path", metavar="PATH",
                        help="CSV from build_dataset.py (overrides the positional csv and --synthetic)")
    parser.add_argument("--rule-height-min", dest="rule_height_min", type=float, default=0.3,
                        help="rule baseline minimum height_extent, m")
    parser.add_argument("--save", action="store_true", help="write model.json and feature_names.json")
    parser.add_argument("--group-by", dest="group_by", choices=("session", "session+person"),
                        default="session+person",
                        help="split groups: session_id only, or session_id merged with person")
    args = parser.parse_args()
    rule_baseline.HEIGHT_MIN = args.rule_height_min      # is_human_rule reads it at call time
    if args.csv_path:
        path = args.csv_path
    elif args.synthetic:
        os.makedirs("datasets", exist_ok=True)
        path = "datasets/synthetic.csv"
        write_synthetic(path)
    else:
        path = args.csv
    train(path, save=args.save, group_by=args.group_by)


if __name__ == "__main__":
    main()

"""Train and evaluate a session-grouped human cluster classifier."""

import argparse
import csv
import os

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from features import FEATURE_NAMES
from make_synthetic import write_synthetic
from rule_baseline import is_human_rule


def load_dataset(path):
    """Load features, labels, and session IDs from CSV."""
    features, labels, sessions = [], [], []
    with open(path, "r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        for row in reader:
            vector = [float(row[name]) for name in FEATURE_NAMES]
            if np.isfinite(vector).all():
                features.append(vector)
                labels.append(1 if row["label"].lower() == "human" else 0)
                sessions.append(row["session_id"])
    return np.asarray(features), np.asarray(labels), np.asarray(sessions)


def print_metrics(name, truth, predicted):
    """Print one classifier's evaluation metrics."""
    print(f"{name}: precision={precision_score(truth, predicted, zero_division=0):.3f} "
          f"recall={recall_score(truth, predicted, zero_division=0):.3f} "
          f"f1={f1_score(truth, predicted, zero_division=0):.3f}")
    print(f"{name}: confusion_matrix=\n{confusion_matrix(truth, predicted).tolist()}")


def train(path):
    """Split by session, train XGBoost, and evaluate both classifiers."""
    x, y, groups = load_dataset(path)
    if len(x) == 0:
        raise SystemExit("No finite feature rows found")
    sessions = np.unique(groups)
    if len(sessions) < 2:
        raise SystemExit("At least two sessions are required for a held-out test split")
    n_splits = min(5, len(sessions))
    train_index, test_index = next(iter(GroupKFold(n_splits=n_splits).split(x, y, groups)))
    pos = int(np.count_nonzero(y[train_index] == 1))
    neg = int(np.count_nonzero(y[train_index] == 0))
    weight = neg / pos if pos else 1.0
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=weight,
        eval_metric="logloss",
    )
    model.fit(x[train_index], y[train_index])
    predicted = model.predict(x[test_index])
    print_metrics("XGBoost", y[test_index], predicted)

    rules = []
    for row in x[test_index]:
        features = dict(zip(FEATURE_NAMES, row.tolist()))
        rules.append(int(is_human_rule(features)))
    print_metrics("Rule baseline", y[test_index], np.asarray(rules))

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    print("XGBoost feature importances:")
    for index in order:
        print(f"  {FEATURE_NAMES[index]}={importances[index]:.4f}")
    print("Rule baseline feature importances:")
    for name in ("height_extent", "xy_extent", "n_points"):
        print(f"  {name}=n/a")

    model.save_model("model.json")
    with open("feature_names.json", "w", encoding="utf-8") as output:
        import json
        json.dump(FEATURE_NAMES, output)
    print("Saved model.json and feature_names.json")


def main():
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train the radar cluster classifier.")
    parser.add_argument("csv", nargs="?", default="datasets/radar.csv")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        os.makedirs("datasets", exist_ok=True)
        path = "datasets/synthetic.csv"
        write_synthetic(path)
    else:
        path = args.csv
    train(path)


if __name__ == "__main__":
    main()

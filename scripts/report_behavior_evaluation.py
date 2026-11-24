"""Validate clip coverage and report metrics for a frozen binary evaluation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--merged", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_rows(args.manifest)
    expected = {row["clip"]: row["label"] for row in manifest}
    rows = [row for path in args.predictions for row in read_rows(path)]
    observed = {row["clip"]: row["label"] for row in rows}
    if not expected or observed != expected or len(rows) != len(manifest) or len(rows) != len(observed):
        parser.error("Prediction rows must match the manifest exactly, including labels")

    valid = [row for row in rows if not row["error"]]
    if not valid:
        parser.error("No successful predictions")
    actual = [int(row["label"]) for row in valid]
    predicted = [int(row["prediction"]) for row in valid]
    names = ["Non-fight", "Fight"]
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    per_class = {}
    for index, name in enumerate(names):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        per_class[name] = {
            "precision": tp / max(1, predicted_count),
            "recall": tp / max(1, support),
            "f1": 2 * tp / max(1, support + predicted_count),
            "support": support,
        }
    summary = {
        "expected": len(expected),
        "scored": len(valid),
        "errors_by_class": {
            name: sum(bool(row["error"]) and row["label"] == str(label) for row in rows)
            for label, name in enumerate(names)
        },
        "accuracy": float(matrix.trace() / len(valid)),
        "per_class": per_class,
        "confusion_matrix_labels": names,
        "confusion_matrix": matrix.tolist(),
        "latency_seconds": dict(zip(("median", "p95"), np.percentile(
            [float(row["seconds"]) for row in valid], [50, 95]
        ))),
    }
    with args.merged.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["clip"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

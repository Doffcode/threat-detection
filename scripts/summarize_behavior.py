"""Compare behavior predictions from separate model workers."""

import argparse
import csv
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("predictions", nargs="+", type=Path)
args = parser.parse_args()
reference = None
ranked = []
for source in args.predictions:
    with source.open(newline="") as file:
        rows = list(csv.DictReader(file))
    clips = {(row["clip"], row["label"]) for row in rows}
    if len(clips) != len(rows) or not rows or any(row["error"] for row in rows):
        print(f"{source.stem}: incomplete or invalid run")
        continue
    if reference is None:
        reference = clips
    if clips != reference:
        raise SystemExit(f"{source}: clip set differs from other runs")
    tp = sum(row["label"] == row["prediction"] == "1" for row in rows)
    fp = sum(row["label"] == "0" and row["prediction"] == "1" for row in rows)
    fn = sum(row["label"] == "1" and row["prediction"] == "0" for row in rows)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    latency = [float(row["seconds"]) for row in rows]
    median, p95 = np.percentile(latency, [50, 95])
    realtime = np.mean([seconds <= float(row["duration"]) for seconds, row in zip(latency, rows)])
    print(f"{source.stem}: n={len(rows)} F1={f1:.4f} FN={fn} FP={fp} "
          f"median={median:.3f}s p95={p95:.3f}s within playback={realtime:.1%}")
    if realtime >= 0.95:
        ranked.append((-f1, fp, p95, source.stem))

print("Selected:", min(ranked)[-1] if ranked else "none")

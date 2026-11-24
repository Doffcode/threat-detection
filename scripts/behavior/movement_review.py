"""Blind review and scoring for the frozen movement cue sample."""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/eval/movement_review"
DATASET = Path(os.environ.get("SCFD_ROOT", "/tmp/threat-behavior-baseline/scfd"))


def load_queue(output):
    with (output / "manifest.csv").open(newline="") as file:
        queue = list(csv.DictReader(file))
    path = output / "labels.json"
    return queue, json.loads(path.read_text()) if path.exists() else {}


def clip_bytes(row):
    data = (DATASET / row["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Clip hash differs from the frozen manifest.")
    return data


def intervals_from_text(text):
    ranges = []
    for line in text.splitlines():
        if line.strip():
            start, end = map(float, line.split(","))
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError("Each interval needs an end time after its start.")
            ranges.append([start, end])
    ranges.sort()
    if not ranges or any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Enter at least one interval. Intervals cannot overlap.")
    return ranges


def review_app():
    import streamlit as st

    output = Path(os.environ.get("MOVEMENT_REVIEW_DIR", str(OUTPUT)))
    queue, labels = load_queue(output)
    row = next((item for item in queue if item["review_id"] not in labels), None)
    st.title("Movement cue review")
    st.progress(len(labels) / len(queue), text=f"{len(labels)} of {len(queue)} clips saved")
    if row is None:
        st.success("Review complete. All labels are saved.")
        if st.button("Score reviewed clips"):
            with st.spinner("Running the movement rule on the reviewed clips…"):
                score(output)
        summary = output / "summary.json"
        if summary.exists():
            st.json(json.loads(summary.read_text()))
        return
    try:
        data = clip_bytes(row)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return

    st.caption("Look for one person moving toward another while the other moves away and the gap narrows.")
    st.video(data, format="video/mp4")
    review_id = row["review_id"]
    no_cue, unclear = st.columns(2)
    choice = None
    intervals = []
    if no_cue.button("No cue", key=f"no-{review_id}"):
        choice = "No cue"
    if unclear.button("Unclear", key=f"unclear-{review_id}"):
        choice = "Unclear"
    with st.expander("Cue present"):
        text = st.text_area("Start and end in seconds, one pair per line", placeholder="0.4, 1.2",
                            key=f"times-{review_id}")
        if st.button("Save cue", key=f"yes-{review_id}"):
            try:
                intervals = intervals_from_text(text)
                choice = "Cue present"
            except ValueError as error:
                st.error(f"Use start, end in seconds. {error}")
    if choice is not None:
        labels[review_id] = {"label": choice, "intervals": intervals}
        path = output / "labels.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(labels, indent=2) + "\n")
        temp.replace(path)
        st.rerun()


def summarize(queue, labels, alerts, errors):
    counts = Counter(matched=0, missed=0, unmatched=0, cue_free=0, cue_free_flagged=0,
                     unclear=0, processing_errors=0)
    for row in queue:
        key = row["review_id"]
        label = labels[key]
        if key in errors:
            counts["processing_errors"] += 1
        if label["label"] == "Unclear":
            counts["unclear"] += 1
        if key in errors or label["label"] == "Unclear":
            continue
        remaining = list(label["intervals"])
        if label["label"] == "No cue":
            counts["cue_free"] += 1
            counts["cue_free_flagged"] += bool(alerts[key])
        for alert in sorted(alerts[key], key=lambda event: event["timestamp_s"]):
            t = alert["timestamp_s"]
            index = next((i for i, (start, end) in enumerate(remaining) if start <= t <= end), None)
            if index is None:
                counts["unmatched"] += 1
            else:
                remaining.pop(index)
                counts["matched"] += 1
        counts["missed"] += len(remaining)
    total = counts["matched"] + counts["missed"]
    return {**counts, "reviewed_clips": len(queue), "recall": counts["matched"] / total if total else None}


def score(output):
    queue, labels = load_queue(output)
    if set(labels) != {row["review_id"] for row in queue}:
        raise SystemExit("Finish the human review before scoring.")
    import sys

    import cv2
    import torch

    sys.path.insert(0, str(ROOT))
    from src.behavior.video_review import ApproachRetreat
    from src.pipeline.tracker import PersonTracker

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tracker = PersonTracker({"weights": str(ROOT / "yolo11m.pt")}, device=device)
    alerts, errors = {}, {}
    for row in queue:
        key = row["review_id"]
        alerts[key] = []
        capture = None
        try:
            clip_bytes(row)
            capture = cv2.VideoCapture(str(DATASET / row["path"]))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Unreadable frame rate.")
            tracker.reset()
            rule = ApproachRetreat(fps)
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                alerts[key].extend(dict(timestamp_s=index / fps, **event)
                                   for event in rule.update(tracker.track(frame)))
                index += 1
            if not index:
                raise ValueError("No decodable frames.")
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            errors[key] = str(error)
        finally:
            if capture is not None:
                capture.release()

    with (output / "alerts.csv").open("w", newline="") as file:
        fields = ("review_id", "timestamp_s", "approaching", "retreating", "approach",
                  "retreat", "separation_start", "separation_end")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, events in alerts.items():
            writer.writerows(dict(review_id=key, **event) for event in events)
    (output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    report_saved(output)


def report_saved(output):
    queue, labels = load_queue(output)
    if set(labels) != {row["review_id"] for row in queue}:
        raise SystemExit("Finish the human review before reporting.")
    alerts = {row["review_id"]: [] for row in queue}
    with (output / "alerts.csv").open(newline="") as file:
        for alert in csv.DictReader(file):
            alert["timestamp_s"] = float(alert["timestamp_s"])
            alerts[alert["review_id"]].append(alert)
    errors = json.loads((output / "errors.json").read_text())
    result = json.dumps(summarize(queue, labels, alerts, errors), indent=2)
    (output / "summary.json").write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["review", "score", "report"], default="review")
    args = parser.parse_args()
    if args.action == "review":
        review_app()
    elif args.action == "score":
        score(OUTPUT)
    else:
        report_saved(OUTPUT)

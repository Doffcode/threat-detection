"""Single-clip workflow for the presentation dashboard."""

import csv
import hashlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch
import yaml

from scripts.compare_behavior import load_behavior_model
from src.behavior.video_review import encode_cue_replay, review_video
from src.pipeline.detector import WeaponDetector
from src.pipeline.tracker import PersonTracker

MAX_BYTES = 100_000_000
MAX_SECONDS = 60
TYPES = ("Weapon", "Fight", "Movement")
ERRORS = (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError, cv2.error)
cached_behavior_model = st.cache_resource(load_behavior_model)


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, _ = cap.read()
        if not ok or not all(math.isfinite(n) and n > 0 for n in (fps, frames)):
            raise ValueError("Choose a decodable MP4 with a readable duration.")
        if frames / fps > MAX_SECONDS:
            raise ValueError("Choose a video no longer than 60 seconds.")
        return fps
    finally:
        cap.release()


def weapon_example(root, path):
    manifest = root / "configs/detection/cctv_test_v2_occlusion_template.csv"
    with manifest.open(newline="") as stream:
        files = [row["image_filename"] for row in csv.DictReader(stream)
                 if row["sequence_id"] == "innovatus_cam_b_seq02"]
    if not files:
        raise ValueError("Weapon example frames are unavailable.")
    image_dir = root / "data/cctv_test_v2_candidates/images"
    joined = path.with_suffix(".mjpg")
    with joined.open("wb") as stream:
        for name in files:
            stream.write((image_dir / name).read_bytes())
    subprocess.run(["ffmpeg", "-v", "error", "-f", "mjpeg", "-framerate", "4",
                    "-i", str(joined), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(path)], check=True, capture_output=True, timeout=30)


def weapon_events(path, root, fps, device):
    config = yaml.safe_load((root / "src/pipeline/config.yaml").read_text())
    detector = WeaponDetector(config["detector"], device=device)
    groups = {}
    capture = cv2.VideoCapture(str(path))
    try:
        for index in range(math.ceil(MAX_SECONDS * fps)):
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / fps
            for detection in detector.detect(frame):
                key = (detection.cls_name, int(timestamp))
                if key in groups and groups[key][0]["score"] >= detection.conf:
                    continue
                still = frame.copy()
                x1, y1, x2, y2 = map(int, detection.box)
                cv2.rectangle(still, (x1, y1), (x2, y2), (0, 220, 255), 2)
                label = f"{detection.cls_name} {detection.conf:.2f}"
                cv2.putText(still, label, (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                encoded, data = cv2.imencode(".jpg", still)
                groups[key] = (dict(type="Weapon", timestamp_s=timestamp, label=detection.cls_name,
                                    score=detection.conf), data.tobytes() if encoded else None)
    finally:
        capture.release()
    return list(groups.values())


def analyze(path, root, runtime, status):
    fps = video_metadata(path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    events, errors, windows = [], {}, []
    for branch in TYPES:
        status.write(f"Running {branch.lower()} analysis")
        try:
            if branch == "Weapon":
                events.extend(weapon_events(path, root, fps, device))
            elif branch == "Fight":
                deps = str(runtime / "deps")
                if deps not in sys.path:
                    sys.path.append(deps)
                model = cached_behavior_model("x3d", runtime / "source", runtime / "final_x3d_realtime.pt", device)
                windows, _ = review_video(path, model, device)
                events.extend((dict(type=branch, timestamp_s=w["start_s"], **w), None)
                              for w in windows if w["label"] == "Fight detected")
            else:
                tracker = PersonTracker({"weights": str(root / "yolo11m.pt")}, device=device)
                _, cues = review_video(path, None, device, tracker)
                events.extend((dict(type=branch, label="Approach and retreat", **row), (still, replay))
                              for row, still, replay in cues)
        except ERRORS as error:
            errors[branch] = str(error)
    return sorted(events, key=lambda event: event[0]["timestamp_s"]), errors, windows


def render_unified_video(root):
    st.header("Analyze Video")
    runtime = root / "outputs/runs/behavior_runtime"
    samples = {"Weapon example (frame sequence)": None, "Fight example": "fight.mp4",
               "Non-fight example": "non_fight.mp4", "Movement cue example": "movement.mp4"}
    source = st.selectbox("Choose a video", [*samples, "Upload video"], key="unified_source")
    uploaded = st.file_uploader("Upload MP4", type=["mp4"], key="unified_upload") if source == "Upload video" else None
    st.caption("MP4 up to 60 seconds and 100 MB.")
    key = (source, hashlib.sha256(uploaded.getbuffer()).hexdigest() if uploaded else None)
    if st.session_state.get("unified_key") != key:
        st.session_state["unified_key"] = key
        st.session_state.pop("unified_result", None)
        st.session_state.pop("unified_event", None)
    if st.button("Analyze video", type="primary", disabled=source == "Upload video" and uploaded is None):
        st.session_state.pop("unified_result", None)
        st.session_state.pop("unified_event", None)
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "clip.mp4"
                if uploaded:
                    if uploaded.size > MAX_BYTES:
                        raise ValueError("Choose an MP4 no larger than 100 MB.")
                    path.write_bytes(uploaded.getbuffer())
                elif samples[source]:
                    path = runtime / samples[source]
                else:
                    weapon_example(root, path)
                with st.status("Analyzing the selected video") as status:
                    events, errors, windows = analyze(path, root, runtime, status)
                    status.update(label="Analysis finished" if not errors else "Analysis finished with errors",
                                  state="complete" if not errors else "error")
                st.session_state["unified_result"] = (path.read_bytes(), events, errors, windows, {})
        except ERRORS as error:
            st.error(f"Video could not be analyzed. {error}")
    result = st.session_state.get("unified_result")
    if result is None:
        return
    video, events, errors, windows, replays = result
    player, details = st.columns([3, 2])
    selected = None
    with details:
        st.subheader("Analysis results")
        for branch in TYPES:
            if branch in errors:
                st.error(f"{branch}: {errors[branch]}")
            elif branch == "Fight":
                peak = max(windows, key=lambda row: row["score"])
                st.write(f"**Fight:** {peak['label']} · peak score {peak['score']:.3f}")
            else:
                count = sum(row["type"] == branch for row, _ in events)
                st.write(f"**{branch}:** {count} {'event' if count == 1 else 'events'}")
        if events:
            st.subheader("Timeline")
            types = st.multiselect("Event types", TYPES, default=TYPES)
            indexes = [i for i, (row, _) in enumerate(events) if row["type"] in types]
            if indexes:
                index = st.selectbox("Inspect event", indexes, key="unified_event",
                                     format_func=lambda i: f"{events[i][0]['timestamp_s']:.2f} s · "
                                                           f"{events[i][0]['type']} · {events[i][0]['label']}")
                selected = events[index]
            else:
                st.info("No events match the selected filters.")
            st.download_button("Download timeline CSV", pd.DataFrame([row for row, _ in events]).to_csv(index=False),
                               "timeline.csv", "text/csv")
        elif not errors:
            st.info("Analysis completed. No events detected.")
    with player:
        st.subheader("Video evidence")
        if selected is None:
            st.video(video)
            return
        row, evidence = selected
        if row["type"] == "Movement":
            still, frames = evidence
            if index not in replays:
                try:
                    replays[index] = encode_cue_replay(video, row, frames)
                except ERRORS:
                    replays[index] = None
            if replays[index]:
                st.video(replays[index])
                st.download_button("Download movement replay", replays[index], "movement-cue.mp4", "video/mp4")
            else:
                st.info("Annotated replay unavailable. Showing the original video.")
                st.video(video, start_time=max(0, int(row["timestamp_s"] - 1)))
                if still:
                    st.image(still)
            st.caption(f"Track {row['approaching']} approaches as track {row['retreating']} moves away. "
                       f"Gap: {row['separation_start']:.2f} → {row['separation_end']:.2f} box heights.")
        else:
            st.video(video, start_time=int(row["timestamp_s"]))
            if row["type"] == "Weapon" and evidence:
                st.image(evidence, caption=f"{row['label']} · confidence {row['score']:.2f}")
            elif row["type"] == "Fight":
                st.caption(f"{row['start_s']:.2f}–{row['end_s']:.2f} s · Fight score "
                           f"{row['score']:.3f} · threshold 0.40")

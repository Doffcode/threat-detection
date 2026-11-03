"""SentinelGuard Streamlit dashboard (port of src/dashboard/app.py).

Run:  streamlit run work/app/dashboard.py
Env:  WEAPON_MODEL_DIR=/path/to/runs/<run>  (weights + thresholds + imgsz + verifier from pipeline_thresholds.json)
      WEAPON_WEIGHTS=/path/to/best.pt   VERIFIER_WEIGHTS=/path/to/cls.pt   DEMO_VIDEO_DIR=/path/to/videos

Two tabs: still images (main use case, image mode, no temporal smoothing) and
CCTV video (video mode, 2-of-5 frame smoothing).
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from action_intent_engine import (
    ARMED,
    COMBINE_MODES,
    DEFAULT_NMS_IOU,
    DEFAULT_PERSON_WEIGHTS,
    LEVEL_CRITICAL,
    LEVEL_SUSPICIOUS,
    ContourEdgeKnifePreprocessor,  # noqa: F401  (kept for API parity with app.py)
    SuspiciousBehaviorEngine,
    WeightsError,
    find_default_weights,
    load_pipeline_config,
    resolve_model_dir,
)

APP_DIR = Path(__file__).resolve().parent
DEMO_VIDEO_DIR = os.environ.get("DEMO_VIDEO_DIR", str(APP_DIR / "videos"))
COMBINE_FROM_FILE = "(from pipeline_thresholds.json / prod)"

st.set_page_config(
    page_title="SentinelGuard - Weapon & Suspicious Behavior Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 15px;
    }
    .threat-critical {
        color: #ff4d4d;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .threat-suspicious {
        color: #ffaa00;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .threat-normal {
        color: #00e676;
        font-weight: bold;
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.title("🛡️ SentinelGuard AI: Action & Harm-Intended Behavior Detection")
st.caption("Two-stage weapon detection (detector + crop verifier) | Derived armed-person | Image & CCTV video analysis")

st.markdown("---")


@st.cache_resource(show_spinner="Loading models...")
def get_engine(model_dir, weights, verifier_weights, person_weights, pose_weights, combine, class_conf_items,
               enable_contour_edge, tile, tta_image, tta_video, imgsz, nms_iou):
    return SuspiciousBehaviorEngine(
        model_dir=model_dir or None,
        weights=weights or None,
        verifier_weights=verifier_weights or ("auto" if model_dir else None),
        person_weights=person_weights or None,
        pose_weights=pose_weights or None,
        combine=combine,
        class_conf=dict(class_conf_items) or None,
        enable_contour_edge=enable_contour_edge,
        imgsz=imgsz,
        tile=tile,
        tta_image=tta_image,
        tta_video=tta_video,
        nms_iou=nms_iou,
    )


def status_card(score, level):
    if score >= LEVEL_CRITICAL:
        return f"<div class='metric-card'><span class='threat-critical'>🚨 CRITICAL THREAT DETECTED</span><br>Status: {level}</div>"
    if score >= LEVEL_SUSPICIOUS:
        return f"<div class='metric-card'><span class='threat-suspicious'>⚠️ SUSPICIOUS BEHAVIOR</span><br>Status: {level}</div>"
    return f"<div class='metric-card'><span class='threat-normal'>✅ NORMAL / MONITORED</span><br>Status: {level}</div>"


def detections_table(meta):
    rows = [{"Class": d["name"], "Score": round(d["conf"], 3), "Detector conf": round(d["det_conf"], 3),
             "Verifier p": None if d["verifier_p"] is None else round(d["verifier_p"], 3),
             "Box": [round(v) for v in d["box"]]} for d in meta["weapons"]]
    rows += [{"Class": ARMED, "Score": None, "Detector conf": None, "Verifier p": None,
              "Box": [round(v) for v in b]} for b in meta["armed_persons"]]
    return pd.DataFrame(rows)


def load_engine():
    try:
        return get_engine(model_dir_path.strip(), weights_path.strip(), verifier_path.strip(), person_path.strip(),
                          pose_path.strip(), None if combine_choice == COMBINE_FROM_FILE else combine_choice,
                          tuple(sorted(class_conf.items())), enable_contour_edge, enable_tiling,
                          enable_tta_image, enable_tta_video, None if imgsz == "auto" else int(imgsz), nms_iou)
    except (WeightsError, ValueError, FileNotFoundError) as e:
        st.error(str(e))
        st.stop()


# Sidebar Configuration
st.sidebar.header("⚙️ Models")

model_dir_path = st.sidebar.text_input(
    "Model run directory", value=os.environ.get("WEAPON_MODEL_DIR", ""),
    help="e.g. work/runs/indian_strict_m960: weights/best.pt, thresholds (pipeline[_tta]/pipeline_thresholds.json "
         "or thresholds.json), imgsz, and the verifier named in pipeline_thresholds.json are picked up.",
)
default_weights = ""
if not model_dir_path.strip():
    try:
        default_weights = os.environ.get("WEAPON_WEIGHTS") or find_default_weights()
    except WeightsError:
        default_weights = ""
weights_path = st.sidebar.text_input(
    "Weapon detector weights (.pt; overrides the run dir)", value=default_weights,
    help="gun / knife / handheld_object (old armed person / gun / knife also accepted). "
         "Default: model run dir, $WEAPON_WEIGHTS or newest work/runs/*/weights/best.pt",
)
verifier_path = st.sidebar.text_input(
    "Verifier weights (optional, classification .pt; 'none' = off)", value=os.environ.get("VERIFIER_WEIGHTS", ""),
    help="YOLO classifier (gun / knife / not_weapon) run on padded square crops of each gun/knife box. "
         "Empty with a run dir = the verifier named in its pipeline_thresholds.json.",
)
person_path = st.sidebar.text_input(
    "Person detector (for derived 'armed person'; empty = off)",
    value=DEFAULT_PERSON_WEIGHTS if os.path.isfile(DEFAULT_PERSON_WEIGHTS) else "",
    help="COCO model, person class 0. Not used with old 3-class models.",
)
pose_path = st.sidebar.text_input("Pose model weights (optional, e.g. yolo11n-pose.pt)", value="")

anchor = weights_path.strip()
if not anchor and model_dir_path.strip():
    try:
        anchor = resolve_model_dir(model_dir_path.strip())["weights"]
    except WeightsError as e:
        st.sidebar.error(str(e))
has_verifier = verifier_path.strip().lower() not in ("none", "off") and bool(verifier_path.strip() or model_dir_path.strip())

st.sidebar.header("🎛️ Inference Settings")
combine_choice = st.sidebar.selectbox(
    "Score combination (with verifier)", [COMBINE_FROM_FILE, *COMBINE_MODES],
    help="det = d, cls = p, prod = d x p, geo = sqrt(d x p), wgeo25 = d^0.75 p^0.25, wgeo75 = d^0.25 p^0.75. "
         "Thresholds apply to this score.",
)
enable_tta_image = st.sidebar.toggle("TTA for images (test-time augmentation; recommended)", value=True)
enable_tta_video = st.sidebar.toggle("TTA for video (slow)", value=False)
nms_iou = st.sidebar.slider("NMS IoU", 0.3, 0.9, float(DEFAULT_NMS_IOU), 0.05)
cfg = load_pipeline_config(anchor if anchor and os.path.isfile(anchor) else None, prefer_pipeline=has_verifier,
                           tta=enable_tta_image,
                           combine=None if combine_choice == COMBINE_FROM_FILE else combine_choice)
defaults = cfg["thresholds"]
if cfg["source"]:
    st.sidebar.caption(f"Thresholds from {cfg['source']}: gun {defaults.get('gun', 0):.3f}, "
                       f"knife {defaults.get('knife', 0):.3f}")
enable_contour_edge = st.sidebar.toggle(
    "Enable Contour-Edge Feature Extraction (Concealed/Hidden Knives)", value=False,
    help="Only enable if the model was trained on contour-edge preprocessed images.",
)
enable_tiling = st.sidebar.toggle(
    "Tiled inference (full frame + 2x2 tiles; better for small knives, ~5x slower)", value=False,
)
override_thr = st.sidebar.checkbox("Override thresholds", value=False,
                                   help="Off = the tuned per-class thresholds from the run's files.")
class_conf = {}
if override_thr:
    class_conf["gun"] = class_conf[ARMED] = st.sidebar.slider(
        "Gun (and legacy armed person) threshold", 0.01, 0.90, float(round(defaults.get("gun", 0.25), 2)), 0.01)
    class_conf["knife"] = st.sidebar.slider("Knife threshold", 0.01, 0.90,
                                            float(round(defaults.get("knife", 0.20), 2)), 0.01)
imgsz = st.sidebar.select_slider("Inference image size", options=["auto", 640, 800, 960, 1280], value="auto")
threat_alert_thresh = st.sidebar.slider("Threat Alert Threshold Score (video log)", 30, 90, 60, 5)

tab_image, tab_video = st.tabs(["🖼️ Image analysis", "📹 CCTV video analysis"])

# ------------------------------------------------------------------ images
with tab_image:
    uploads = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png", "bmp", "webp"],
                               accept_multiple_files=True, key="image_upload")
    if uploads:
        engine = load_engine()
        with st.expander("Engine settings"):
            st.code(engine.describe())
        summary = []
        for up in uploads:
            data = np.frombuffer(up.getvalue(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                st.warning(f"Cannot decode {up.name}")
                continue
            engine.reset()
            annotated, meta = engine.process_frame(frame, mode="image")
            summary.append({"Image": up.name, "Score": meta["threat_score"], "Level": meta["threat_level"],
                            "Flags": ", ".join(meta["flags"])})
            st.markdown(f"#### {up.name}")
            c_img, c_meta = st.columns([0.65, 0.35])
            with c_img:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
            with c_meta:
                st.markdown(status_card(meta["threat_score"], meta["threat_level"]), unsafe_allow_html=True)
                st.metric("Threat Score", f"{meta['threat_score']:.1f} / 100")
                m1, m2, m3 = st.columns(3)
                m1.metric("Guns", sum(1 for w in meta["weapons"] if w["name"] == "gun"))
                m2.metric("Knives", sum(1 for w in meta["weapons"] if w["name"] == "knife"))
                m3.metric("Armed persons", len(meta["armed_persons"]))
                df = detections_table(meta)
                if len(df):
                    st.dataframe(df, width="stretch", hide_index=True)
                else:
                    st.caption("No weapons detected.")
        if len(summary) > 1:
            st.markdown("#### Summary")
            st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
    else:
        st.info("Upload a photo to scan it for guns and knives.")

# ------------------------------------------------------------------ video
with tab_video:
    source_option = st.radio("Select Video Input Source", ["Demo CCTV Sample Videos", "Upload Recorded CCTV Video"],
                             horizontal=True)
    video_path = None

    if source_option == "Demo CCTV Sample Videos":
        demo_files = []
        if os.path.isdir(DEMO_VIDEO_DIR):
            demo_files = sorted(f for f in os.listdir(DEMO_VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov')))

        # Priority samples
        priority_samples = ['concealed_weapon_cctv_demo.mp4', 'suspicious_activity_cctv_demo.mp4', 'weapon_video.mp4', 'video1.mp4']
        ordered_files = [f for f in priority_samples if f in demo_files] + [f for f in demo_files if f not in priority_samples]

        if ordered_files:
            selected_file = st.selectbox("Choose Recorded CCTV Sample Video", ordered_files)
            video_path = os.path.join(DEMO_VIDEO_DIR, selected_file)
        else:
            st.warning(f"No sample video found in {DEMO_VIDEO_DIR} (set DEMO_VIDEO_DIR). Please upload a video.")
    else:
        uploaded_file = st.file_uploader("Upload CCTV / Surveillance Clip", type=["mp4", "avi", "mov"], key="video_upload")
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1] or '.mp4'
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = tfile.name

    col_video, col_metrics = st.columns([0.65, 0.35])

    with col_metrics:
        st.subheader("📊 Live Security Metrics")
        metric_status = st.empty()
        metric_score = st.empty()
        metric_knife = st.empty()
        metric_gun = st.empty()
        metric_armed = st.empty()

        st.subheader("🚨 Real-Time Alert Log")
        alert_log_placeholder = st.empty()

    with col_video:
        st.subheader("📹 CCTV Feed Analysis")
        video_placeholder = st.empty()

    if video_path and os.path.exists(video_path):
        start_btn = st.button("▶️ Start CCTV Stream Analysis", type="primary")

        if start_btn:
            engine = load_engine()
            engine.reset()  # cached engine: clear temporal history between videos

            cap = cv2.VideoCapture(video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            st.info(f"Processing Video ({total_frames} frames @ {fps:.1f} FPS) with {os.path.basename(str(engine.weights))}...")
            progress_bar = st.progress(0)

            frame_idx = 0
            logs = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                processed_frame, meta = engine.process_frame(frame, mode="video")

                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", width="stretch")

                score = meta['threat_score']
                level = meta['threat_level']

                metric_status.markdown(status_card(score, level), unsafe_allow_html=True)
                metric_score.metric("Dynamic Threat Score", f"{score:.1f} / 100")

                knives = sum(1 for w in meta['weapons'] if w['name'] == 'knife')
                guns = sum(1 for w in meta['weapons'] if w['name'] == 'gun')

                metric_knife.metric("Knives Detected", knives)
                metric_gun.metric("Guns Detected", guns)
                metric_armed.metric("Armed Persons", len(meta['armed_persons']))

                if score >= threat_alert_thresh:
                    timestamp_sec = frame_idx / fps
                    logs.append({
                        'Frame': frame_idx,
                        'Time (s)': f"{timestamp_sec:.1f}s",
                        'Threat Level': level,
                        'Score': f"{score:.1f}",
                        'Flags': ", ".join(meta['flags'])
                    })
                    df_log = pd.DataFrame(logs)
                    alert_log_placeholder.dataframe(df_log.tail(8), width="stretch")

                if total_frames > 0:
                    progress_bar.progress(min(1.0, frame_idx / total_frames))

            cap.release()
            progress_bar.progress(1.0)
            st.success("CCTV Video Analysis Complete!")
    else:
        st.info("Select a demo video or upload a CCTV clip to begin analysis.")

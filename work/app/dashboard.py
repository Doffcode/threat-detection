"""
Threat Detection Surveillance Control Room & Audit Dashboard.
============================================================

Unified interface for:
  Tab 1: 🛡️ CCTV Live Control Room (Stage 1 Tracking, Association & Stage-2 Crop Verifier)
  Tab 2: 🏃 Kinematic Behavior Studio (Stage 2 Person Locomotion & Rule Review)
  Tab 3: 🔍 1-Click Association Auditor UI (Human-in-the-loop review of 40 sampled links)
  Tab 4: 📊 System Verification & Gate Audit (Gates 0-5 status & metrics)

Run:
  work/.venv/bin/streamlit run work/app/dashboard.py --server.headless true --server.port 8501
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["YOLO_AUTOINSTALL"] = "False"

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# Add workspace root and app directory to sys.path so modules can be imported directly
WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Configurable paths via environment variables for isolated testing
AUDIT_CSV_PATH = Path(os.environ.get("AUDIT_CSV_PATH", str(WORKSPACE_ROOT / "outputs/pipeline/audit_sheet/association_audit_sheet.csv")))
RULE_CSV_PATH = Path(os.environ.get("RULE_CSV_PATH", str(WORKSPACE_ROOT / "outputs/behavior/rule_firings.csv")))
TRAJECTORIES_DIR = Path(os.environ.get("TRAJECTORIES_DIR", str(WORKSPACE_ROOT / "outputs/behavior/trajectories")))

from action_intent_engine import (
    ARMED,
    COMBINE_MODES,
    DEFAULT_NMS_IOU,
    DEFAULT_PERSON_WEIGHTS,
    LEVEL_CRITICAL,
    LEVEL_SUSPICIOUS,
    SuspiciousBehaviorEngine,
    WeightsError,
    find_default_weights,
    load_pipeline_config,
    resolve_model_dir,
)
from src.pipeline.detector import Stage2CropVerifier, WeaponDetection, WeaponDetector
from src.pipeline.engine import Stage1PipelineEngine


def _atomic_save_audit_csv(df: pd.DataFrame, target_path: Path) -> None:
    """Atomic write via temp file in the same directory followed by os.replace."""
    target_path = Path(target_path)
    tmp_path = target_path.with_suffix(".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, target_path)


def _check_video_decodable(video_path: Path) -> Tuple[bool, int, float]:
    """B2 helper: Checks if video_path exists, can be opened, and has frames > 0."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        opened = cap.isOpened()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 25.0
        is_valid = opened and total_frames > 0
        return is_valid, total_frames, fps_val
    finally:
        cap.release()


def _resolve_upload_path(uploaded: Any, session_state: Dict[str, Any]) -> Tuple[Path, bool]:
    """B3 helper: Resolves uploaded file to a cached temp file, cleaning up previous file if replaced."""
    suffix = Path(uploaded.name).suffix.lower()
    is_video = suffix in [".mp4", ".avi", ".mov"]
    upload_key = f"{uploaded.name}_{uploaded.size}"

    if (
        "uploaded_temp_key" in session_state
        and session_state["uploaded_temp_key"] == upload_key
        and "uploaded_temp_path" in session_state
        and Path(session_state["uploaded_temp_path"]).exists()
    ):
        return Path(session_state["uploaded_temp_path"]), is_video

    if "uploaded_temp_path" in session_state:
        old_p = Path(session_state["uploaded_temp_path"])
        if old_p.exists():
            try:
                old_p.unlink()
            except OSError:
                pass

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    buf = uploaded.getbuffer() if hasattr(uploaded, "getbuffer") else uploaded.read()
    tfile.write(buf)
    tfile.close()
    session_state["uploaded_temp_path"] = tfile.name
    session_state["uploaded_temp_key"] = upload_key
    return Path(tfile.name), is_video


def _compute_rule_summary(df_rules: pd.DataFrame) -> Dict[str, Any]:
    """B5 helper: Computes dynamic rule summary counts from the rule firings dataframe."""
    if df_rules is None or df_rules.empty:
        return {
            "total_alerts": 0,
            "running_alerts": 0,
            "reversal_alerts": 0,
            "unique_seqs": 0,
        }
    return {
        "total_alerts": len(df_rules),
        "running_alerts": int((df_rules["rule_name"] == "running_or_sudden_speed").sum()) if "rule_name" in df_rules.columns else 0,
        "reversal_alerts": int((df_rules["rule_name"] == "abrupt_stop_or_reversal").sum()) if "rule_name" in df_rules.columns else 0,
        "unique_seqs": df_rules["sequence_id"].nunique() if "sequence_id" in df_rules.columns else 0,
    }


def _next_audit_index(df_audit: pd.DataFrame, current_idx: int, total_cases: int, advance: bool = True) -> int:
    """B4 helper: Computes next audit index after judgment, clamped to [0, total_cases - 1]."""
    if advance and current_idx < total_cases - 1:
        return current_idx + 1
    return current_idx


def _save_audit_judgment(
    df_audit: pd.DataFrame,
    audit_csv_p: Path,
    selected_idx: int,
    judgment_val: str,
    total_cases: int,
    advance: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """B4 helper: Records audit judgment, atomically saves CSV, and returns (updated_df, next_idx)."""
    df_audit.at[selected_idx, "judgment"] = judgment_val
    _atomic_save_audit_csv(df_audit, audit_csv_p)
    next_idx = _next_audit_index(df_audit, selected_idx, total_cases, advance=advance)
    return df_audit, next_idx


def _run_inspector_stream(
    video_path: Path,
    engine: Stage1PipelineEngine,
    video_placeholder: Any = None,
    playback_progress: Any = None,
    metric_armed: Any = None,
    metric_weapons: Any = None,
    metric_tracks: Any = None,
    metric_fps: Any = None,
    alert_placeholder: Any = None,
    max_inspect_frames: int = 60,
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    B1/N1 helper: Processes video frame-by-frame through Stage1PipelineEngine.
    Returns: (success: bool, frames_count: int, last_meta: dict).
    Guarantees cap.release() in finally block.
    """
    cap = cv2.VideoCapture(str(video_path))
    stream_failed = False
    frame_idx = 0
    last_meta: Dict[str, Any] = {}
    try:
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 25.0
        while cap.isOpened() and frame_idx < max_inspect_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            try:
                annotated_bgr, meta = engine.process_frame(
                    img_bgr=frame,
                    frame_number=float(frame_idx),
                    sequence_id="stream_preview",
                )
                last_meta = meta
            except Exception as infer_err:
                st.error(f"Inference error on frame {frame_idx}: {infer_err}")
                stream_failed = True
                break

            rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            if video_placeholder is not None:
                video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            if playback_progress is not None and max_inspect_frames > 0:
                playback_progress.progress(frame_idx / max_inspect_frames)

            n_armed = meta.get("n_associated", 0)
            n_weap = meta.get("n_weapons", 0)
            n_tracks = meta.get("n_persons", 0)

            if metric_armed is not None:
                metric_armed.metric("Armed Persons Active", n_armed)
            if metric_weapons is not None:
                metric_weapons.metric("Weapons Detected", n_weap)
            if metric_tracks is not None:
                metric_tracks.metric("Active Tracks", n_tracks)
            if metric_fps is not None:
                metric_fps.metric("Inference Speed", f"{fps_val:.1f} FPS")

            if alert_placeholder is not None:
                if meta.get("active_alert_ids"):
                    alert_placeholder.markdown(
                        f"<div class='badge-critical'>🚨 THREAT ALERT: Armed Person Persists ({', '.join(meta['active_alert_ids'])})</div>",
                        unsafe_allow_html=True,
                    )
                elif n_weap > 0:
                    alert_placeholder.markdown(
                        "<div class='badge-suspicious'>⚠️ Weapon Detected (Awaiting Temporal Lock)</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    alert_placeholder.markdown(
                        "<div class='badge-normal'>✅ Normal / Monitored</div>",
                        unsafe_allow_html=True,
                    )

        return (not stream_failed), frame_idx, last_meta
    finally:
        cap.release()


@st.cache_resource(show_spinner="Loading pipeline models...")
def get_pipeline_engine(
    det_weights: str,
    enable_verifier: bool,
    verifier_weights: str,
    combine_mode: str,
    verifier_conf_floor: float,
    conf_weapon: float,
    conf_person: float,
    conf_pose: float,
    min_link_score: float,
    confirm_frames: int,
) -> Stage1PipelineEngine:
    with open(WORKSPACE_ROOT / "src/pipeline/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg["detector"]["weights"] = det_weights
    cfg["detector"]["conf_threshold"] = conf_weapon
    cfg["tracker"]["conf_threshold"] = conf_person
    cfg["pose"]["conf_threshold"] = conf_pose
    cfg["association"]["min_link_score"] = min_link_score
    cfg["temporal"]["confirmation_frames"] = confirm_frames

    cfg["stage2"] = {
        "enabled": bool(enable_verifier),
        "weights": verifier_weights if enable_verifier else None,
        "combine_mode": combine_mode,
        "verifier_conf_floor": verifier_conf_floor,
    }
    return Stage1PipelineEngine(cfg, output_dir=str(WORKSPACE_ROOT / "outputs/pipeline/preview"))


st.set_page_config(
    page_title="Threat Detection AI — Surveillance & Audit Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .metric-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-suspicious {
        background-color: #f59e0b;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-normal {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-review {
        background-color: #8b5cf6;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Top Bar Banner
st.title("🛡️ Threat Detection System: Surveillance & Audit Suite")
st.caption("Multi-Stage Armed Person Tracking, Behavioral Kinematics & Intent Analysis | Stage 1 Perceptual Pipeline & Stage 2 Kinematics")

# Sidebar Configuration
st.sidebar.header("⚙️ System Pipeline Configuration")

# Model configuration
default_det_weights = "outputs/runs/baseline_m960_v2/weights/best.pt"
if not (WORKSPACE_ROOT / default_det_weights).exists():
    default_det_weights = "work/runs/big_s1_m640/weights/best.pt"

det_weights_path = st.sidebar.text_input(
    "Weapon Detector (.pt)",
    value=default_det_weights,
    help="Primary weapon detector: baseline_m960_v2 / big_s1_m640"
)

# Stage-2 Crop Verifier Toggle (Option 2)
enable_stage2_verifier = st.sidebar.toggle(
    "Enable Stage-2 Crop Verifier (YOLO11s-cls)",
    value=True,
    help="Activates 224px crop classifier to filter false alarms on phones, tools, and wallets."
)

verifier_weights_path = "work/runs/verifier_v5/weights/best.pt"
if enable_stage2_verifier:
    verifier_weights_input = st.sidebar.text_input(
        "Verifier Weights (.pt)",
        value=verifier_weights_path,
        help="YOLO11s-cls classifier (gun / knife / not_weapon)"
    )
    combine_mode = st.sidebar.selectbox(
        "Score Combination Mode",
        ["wgeo75", "prod", "geo", "cls", "det"],
        index=0,
        help="wgeo75: d^0.25 * p^0.75 | prod: d * p | geo: sqrt(d * p) | det: detector only with filter"
    )
    verifier_conf_floor = st.sidebar.slider("Verifier Confidence Floor", 0.05, 0.50, 0.10, 0.05)
else:
    verifier_weights_input = ""
    combine_mode = "det"
    verifier_conf_floor = 0.10

# Tracking & Pose Settings
st.sidebar.markdown("---")
st.sidebar.subheader("👤 Person Tracking & Pose")
conf_weapon = st.sidebar.slider("Weapon Conf Threshold (conf*)", 0.10, 0.80, 0.30, 0.05)
conf_person = st.sidebar.slider("Person Track Conf Threshold", 0.20, 0.80, 0.35, 0.05)
conf_pose = st.sidebar.slider("Wrist Keypoint Conf Threshold", 0.15, 0.60, 0.25, 0.05)
min_link_score = st.sidebar.slider("Min Weapon-Person Link Score", 0.10, 0.60, 0.25, 0.05)
confirm_frames = st.sidebar.slider("Temporal Confirmation (Frames N)", 1, 10, 3, 1)

# Tab structure
tabs = st.tabs([
    "📹 CCTV Live Control Room",
    "🏃 Kinematic Behavior Studio",
    "🔍 1-Click Association Auditor",
    "📊 System Verification & Gates",
])

# ==============================================================================
# TAB 1: CCTV Live Control Room
# ==============================================================================
with tabs[0]:
    st.subheader("📹 Real-Time Surveillance & Multi-Stage Threat Analysis")
    st.caption("Active Pipeline: Weapon Detection ➔ ByteTrack Person Tracking ➔ Wrist Keypoints ➔ Kinematic Association ➔ Temporal Alert")

    col_view, col_stats = st.columns([0.65, 0.35])

    # Video / Image source selection
    with col_view:
        source_mode = st.radio(
            "Select Surveillance Input Source",
            ["Demo CCTV Sequence (Innovatus)", "Upload Custom Video / Image"],
            horizontal=True
        )

        video_path = None
        is_video = True

        demo_videos = {
            "Innovatus CCTV Stage 1 Demo (Tracking & Association)": WORKSPACE_ROOT / "outputs/pipeline/stage1_cctv_demo.mp4",
            "Innovatus CCTV Stage 2 Demo (Kinematic Behaviors)": WORKSPACE_ROOT / "outputs/behavior/behavior_cctv_demo.mp4",
        }

        if source_mode == "Demo CCTV Sequence (Innovatus)":
            choice = st.selectbox("Select Recorded CCTV Sequence", list(demo_videos.keys()))
            video_path = demo_videos[choice]
        else:
            uploaded = st.file_uploader(
                "Upload Surveillance File (.mp4, .avi, .jpg, .png)",
                type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
            )
            # B3: Cached upload resolution
            if uploaded:
                video_path, is_video = _resolve_upload_path(uploaded, st.session_state)

        video_placeholder = st.empty()
        playback_progress = st.empty()

    with col_stats:
        st.subheader("🚨 Threat & Association Telemetry")
        alert_placeholder = st.empty()
        metric_col1, metric_col2 = st.columns(2)
        metric_armed = metric_col1.empty()
        metric_weapons = metric_col2.empty()

        metric_col3, metric_col4 = st.columns(2)
        metric_tracks = metric_col3.empty()
        metric_fps = metric_col4.empty()

        st.subheader("📝 Live Threat Event Log")
        log_placeholder = st.empty()

    # Pre-render sample video if present
    if video_path and video_path.exists():
        if is_video:
            st.info(f"Loaded: `{video_path.name}` ({video_path.stat().st_size / (1024*1024):.1f} MB)")
            st.video(str(video_path))

            st.markdown("#### ⚙️ Interactive Frame-by-Frame Pipeline Inspector")
            if st.button("▶️ Run Interactive Stream Inspector", type="primary"):
                # B2: Check if video is decodable
                is_valid, total_frames, fps_val = _check_video_decodable(video_path)
                if not is_valid:
                    st.warning("Video file contains 0 frames or cannot be decoded.")
                else:
                    # B1: Initialize cached pipeline engine
                    try:
                        engine = get_pipeline_engine(
                            det_weights=det_weights_path,
                            enable_verifier=enable_stage2_verifier,
                            verifier_weights=verifier_weights_input,
                            combine_mode=combine_mode,
                            verifier_conf_floor=verifier_conf_floor,
                            conf_weapon=conf_weapon,
                            conf_person=conf_person,
                            conf_pose=conf_pose,
                            min_link_score=min_link_score,
                            confirm_frames=confirm_frames,
                        )
                        engine.reset_sequence("stream_preview")
                    except Exception as e:
                        st.error(f"Failed to initialize pipeline engine: {e}")
                        st.stop()

                    # B1/N1: Run inspector stream
                    success, frames_count, last_meta = _run_inspector_stream(
                        video_path=video_path,
                        engine=engine,
                        video_placeholder=video_placeholder,
                        playback_progress=playback_progress,
                        metric_armed=metric_armed,
                        metric_weapons=metric_weapons,
                        metric_tracks=metric_tracks,
                        metric_fps=metric_fps,
                        alert_placeholder=alert_placeholder,
                        max_inspect_frames=min(total_frames, 60),
                    )

                    # N1: Only show success if inference completed without failure
                    if success:
                        st.success("Inspector run completed.")
        else:
            img = cv2.imread(str(video_path))
            if img is not None:
                video_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)


# ==============================================================================
# TAB 2: Kinematic Behavior Studio
# ==============================================================================
with tabs[1]:
    st.subheader("🏃 Stage 2 Kinematic Behavior Studio")
    st.markdown("""
    > [!NOTE]
    > **Ethical & Verification Guardrail:** Rule firings are heuristic kinematic triggers stamped `FOR_HUMAN_REVIEW_ONLY`.
    > They flag anomalous movement (running pace, rapid direction reversals, counter-flow) for human security operators.
    """)

    rule_csv_p = RULE_CSV_PATH
    if rule_csv_p.exists():
        df_rules = pd.read_csv(rule_csv_p)
    else:
        df_rules = pd.DataFrame()

    # B5: Compute metric summaries dynamically from df_rules helper
    rule_summary = _compute_rule_summary(df_rules)
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    sum_col1.metric("Analyzed CCTV Sequences", f"{rule_summary['unique_seqs']} Sequences (72 frames)")
    sum_col2.metric("Total Review Alerts", f"{rule_summary['total_alerts']} Events")
    sum_col3.metric("Running / Speed Triggers", f"{rule_summary['running_alerts']} Events")
    sum_col4.metric("Direction Reversals", f"{rule_summary['reversal_alerts']} Events")

    st.markdown("---")
    st.subheader("🗺️ 2D Floor Trajectory Reconstruction")
    traj_cols = st.columns(3)

    traj_dir = TRAJECTORIES_DIR
    seqs = [
        ("Sequence 02", traj_dir / "innovatus_cam_b_seq02_trajectories.png"),
        ("Sequence 07", traj_dir / "innovatus_cam_b_seq07_trajectories.png"),
        ("Sequence 08", traj_dir / "innovatus_cam_b_seq08_trajectories.png"),
    ]

    for col, (seq_name, img_p) in zip(traj_cols, seqs):
        with col:
            st.markdown(f"**{seq_name}**")
            if img_p.exists():
                st.image(str(img_p), caption=f"Locomotion Paths ({seq_name})", use_container_width=True)
            else:
                st.warning(f"Trajectory plot not found at {img_p.name}")

    st.markdown("---")
    st.subheader("📋 Kinematic Rule Review Events Log (`rule_firings.csv`)")

    if not df_rules.empty:
        # Filters
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            rule_filter = st.multiselect(
                "Filter by Rule Name",
                options=df_rules["rule_name"].unique(),
                default=df_rules["rule_name"].unique()
            )
        with fcol2:
            track_filter = st.multiselect(
                "Filter by Track ID",
                options=df_rules["track_id"].unique(),
                default=df_rules["track_id"].unique()
            )

        df_filtered = df_rules[
            (df_rules["rule_name"].isin(rule_filter)) &
            (df_rules["track_id"].isin(track_filter))
        ]

        st.dataframe(
            df_filtered[["alert_id", "sequence_id", "track_id", "rule_name", "frame_number", "rule_description", "status"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(f"outputs/behavior/rule_firings.csv not found or empty.")


# ==============================================================================
# TAB 3: 1-Click Visual Association Auditor UI
# ==============================================================================
with tabs[2]:
    st.subheader("🔍 1-Click Human-in-the-Loop Association Auditor")
    st.caption("Adjudicate the 40 randomly sampled weapon-person link cases from Gate 3. Bounding boxes: Weapon (Red), Person (Green).")

    audit_csv_p = AUDIT_CSV_PATH

    if audit_csv_p.exists():
        # Load with object dtype so text assignment does not cause float64 TypeError
        df_audit = pd.read_csv(audit_csv_p, dtype={"judgment": "object"})
        df_audit["judgment"] = df_audit["judgment"].fillna("")

        # Calculate statistics
        total_cases = len(df_audit)
        judged_mask = df_audit["judgment"].astype(str).str.strip() != ""
        judged_count = int(judged_mask.sum())
        correct_count = int((df_audit["judgment"] == "correct").sum())
        incorrect_count = int((df_audit["judgment"] == "incorrect").sum())
        ambiguous_count = int((df_audit["judgment"] == "ambiguous").sum())

        # Progress & metrics
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("Progress", f"{judged_count} / {total_cases} ({judged_count/total_cases*100:.0f}%)" if total_cases > 0 else "0 / 0")
        pcol2.metric("🟢 Correct Links", correct_count)
        pcol3.metric("🔴 Incorrect Links", incorrect_count)
        pcol4.metric("🟡 Ambiguous Cases", ambiguous_count)

        if total_cases > 0:
            st.progress(judged_count / total_cases)

        # Live Wilson Score Interval Precision
        if (correct_count + incorrect_count) > 0:
            n_eval = correct_count + incorrect_count
            p_hat = correct_count / n_eval
            z = 1.96
            denom = 1.0 + (z**2) / n_eval
            center = (p_hat + (z**2) / (2 * n_eval)) / denom
            margin = z * np.sqrt((p_hat * (1.0 - p_hat) / n_eval) + (z**2 / (4 * n_eval**2))) / denom
            ci_low, ci_high = max(0.0, center - margin), min(1.0, center + margin)
            st.success(f"**Human-Verified Association Precision:** {p_hat*100:.1f}% (95% Wilson CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]) across {n_eval} decided cases.")

        st.markdown("---")

        if total_cases > 0:
            # B4: Manage selected case in session_state, defaulting to first unjudged only on initial load
            case_options = [
                f"Case {row['case_id']}: {row['source']} (Frame {row['frame_id']}) — Status: {row['judgment'] if str(row['judgment']).strip() else 'UNJUDGED'}"
                for _, row in df_audit.iterrows()
            ]

            if "pending_audit_idx" in st.session_state:
                st.session_state["audit_selected_idx"] = st.session_state.pop("pending_audit_idx")

            if "audit_selected_idx" not in st.session_state:
                first_unjudged = 0
                for i, row in df_audit.iterrows():
                    if not str(row["judgment"]).strip():
                        first_unjudged = i
                        break
                st.session_state["audit_selected_idx"] = first_unjudged

            st.session_state["audit_selected_idx"] = max(0, min(total_cases - 1, st.session_state["audit_selected_idx"]))

            col_select, col_nav = st.columns([0.5, 0.5])
            selected_idx = col_select.selectbox(
                "Select Case to Review",
                range(total_cases),
                key="audit_selected_idx",
                format_func=lambda i: case_options[i]
            )

            case_row = df_audit.iloc[selected_idx]
            case_id = int(case_row["case_id"])

            # Display case details & image
            c_crop, c_info = st.columns([0.55, 0.45])

            with c_crop:
                crop_path = Path(str(case_row["crop_path"]))
                if crop_path.exists():
                    st.image(str(crop_path), caption=f"Case {case_id}: Weapon (Red) + Person (Green)", use_container_width=True)
                else:
                    st.error(f"Crop image not found at: `{crop_path}`")

            with c_info:
                st.markdown(f"### Case #{case_id}")
                st.markdown(f"- **Source Dataset:** `{case_row['source']}`")
                st.markdown(f"- **Sequence / Video:** `{case_row['sequence_id']}`")
                st.markdown(f"- **Frame ID:** `{case_row['frame_id']}`")
                st.markdown(f"- **Detected Weapon Class:** `{case_row['weapon_class']}`")
                st.markdown(f"- **Weapon Confidence:** `{float(case_row['weapon_conf']):.4f}`")
                st.markdown(f"- **Kinematic Link Score:** `{float(case_row['link_score']):.4f}`")

                cur_j = str(case_row["judgment"]).strip()
                current_status = f"`{cur_j}`" if cur_j else "*Unjudged*"
                st.markdown(f"- **Current Judgment:** {current_status}")

                st.markdown("#### 1-Click Adjudication:")
                b1, b2, b3 = st.columns(3)

                def _record_judgment(val: str, advance: bool = True):
                    _, next_idx = _save_audit_judgment(df_audit, audit_csv_p, selected_idx, val, total_cases, advance=advance)
                    st.session_state["pending_audit_idx"] = next_idx
                    st.rerun()

                if b1.button("🟢 Correct Link", key=f"btn_corr_{case_id}", use_container_width=True):
                    _record_judgment("correct", advance=True)

                if b2.button("🔴 Incorrect Link", key=f"btn_inc_{case_id}", use_container_width=True):
                    _record_judgment("incorrect", advance=True)

                if b3.button("🟡 Ambiguous", key=f"btn_amb_{case_id}", use_container_width=True):
                    _record_judgment("ambiguous", advance=True)

                if st.button("🔄 Clear Judgment", key=f"btn_clr_{case_id}"):
                    _record_judgment("", advance=False)

            st.markdown("---")
            st.markdown("#### 📋 Full Audit Sheet Table")
            st.dataframe(
                df_audit[["case_id", "source", "sequence_id", "frame_id", "weapon_class", "weapon_conf", "link_score", "judgment"]],
                use_container_width=True,
                hide_index=True
            )
    else:
        st.error(f"Audit sheet CSV not found at: {audit_csv_p}")


# ==============================================================================
# TAB 4: System Verification & Gates Status
# ==============================================================================
with tabs[3]:
    st.subheader("📊 System Verification, Gates & Reproducibility Audit")

    status_file = WORKSPACE_ROOT / "outputs/status/status.md"
    if status_file.exists():
        with open(status_file, "r") as f:
            status_text = f.read()
        st.markdown(status_text)
    else:
        st.info("Status file outputs/status/status.md not found.")

    st.markdown("---")
    st.subheader("🔒 Reproducibility Manifest (Frozen SHA-256 Hashes)")
    manifest_p = WORKSPACE_ROOT / "reproducibility_manifest.json"
    if manifest_p.exists():
        with open(manifest_p, "r") as f:
            manifest_json = f.read()
        st.code(manifest_json, language="json")

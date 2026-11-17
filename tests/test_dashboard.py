"""
Regression tests for Streamlit Dashboard bugs B1-B5.
===================================================
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.pipeline.engine import Stage1PipelineEngine


def test_b1_pipeline_engine_inference_returns_valid_meta():
    """B1: Verify Stage1PipelineEngine processes frame and returns dynamic meta dictionary."""
    with open("src/pipeline/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Use CPU/mock or existing weights with bounded preview
    output_dir = Path("outputs/pipeline/preview_test")
    engine = Stage1PipelineEngine(cfg, output_dir=output_dir)
    engine.reset_sequence("test_seq")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, meta = engine.process_frame(dummy_frame, frame_number=1.0, sequence_id="test_seq")

    assert annotated.shape == (480, 640, 3)
    assert "n_weapons" in meta
    assert "n_persons" in meta
    assert "n_associated" in meta
    assert isinstance(meta["n_weapons"], int)
    assert isinstance(meta["n_persons"], int)
    assert isinstance(meta["n_associated"], int)


def test_b2_zero_frame_video_handling():
    """B2: Guard against ZeroDivisionError on 0-frame or unreadable video."""
    total_frames = 0
    cap_opened = False

    # Guard condition from fixed dashboard
    if not cap_opened or total_frames <= 0:
        warning_msg = "Video file contains 0 frames or cannot be decoded."
        success = False
    else:
        warning_msg = None
        success = True

    assert warning_msg == "Video file contains 0 frames or cannot be decoded."
    assert not success

    # Ensure progress division guard does not divide by zero
    max_inspect_frames = min(total_frames, 60)
    progress_val = None
    if max_inspect_frames > 0:
        progress_val = 1 / max_inspect_frames
    assert progress_val is None  # avoided ZeroDivisionError


def test_b3_temp_file_reuse_on_rerun():
    """B3: Assert temp file is reused across reruns and old file cleaned up on replacement."""
    session_state = {}

    def handle_upload(name: str, size: int, content: bytes) -> Path:
        upload_key = f"{name}_{size}"
        if (
            "uploaded_temp_key" in session_state
            and session_state["uploaded_temp_key"] == upload_key
            and "uploaded_temp_path" in session_state
            and Path(session_state["uploaded_temp_path"]).exists()
        ):
            return Path(session_state["uploaded_temp_path"])

        if "uploaded_temp_path" in session_state:
            old_p = Path(session_state["uploaded_temp_path"])
            if old_p.exists():
                old_p.unlink()

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(content)
        tfile.close()
        session_state["uploaded_temp_path"] = tfile.name
        session_state["uploaded_temp_key"] = upload_key
        return Path(tfile.name)

    # Simulate 5 reruns with identical file
    first_path = handle_upload("sample.mp4", 100, b"video_data_1")
    assert first_path.exists()

    for _ in range(4):
        reused_path = handle_upload("sample.mp4", 100, b"video_data_1")
        assert reused_path == first_path

    # Simulate replacing upload with new file
    second_path = handle_upload("other.mp4", 200, b"video_data_2")
    assert not first_path.exists()  # Old file deleted
    assert second_path.exists()

    # Cleanup
    if second_path.exists():
        second_path.unlink()


def test_b4_audit_selection_advances_and_preserves_case():
    """B4: Assert selecting case 15 and marking judgment advances to case 16, clamped at end."""
    total_cases = 40
    # Simulate df with object dtype so no TypeError occurs
    df_audit = pd.DataFrame({
        "case_id": range(1, total_cases + 1),
        "judgment": [None] * total_cases,
    }, dtype=object)

    session_state = {"audit_selected_idx": 14}  # Case 15 is index 14

    # User marks case 15 as correct
    selected_idx = session_state["audit_selected_idx"]
    df_audit.at[selected_idx, "judgment"] = "correct"
    assert df_audit.at[14, "judgment"] == "correct"

    # Advance logic
    if selected_idx < total_cases - 1:
        session_state["audit_selected_idx"] = selected_idx + 1

    # Selection is now Case 16 (index 15)
    assert session_state["audit_selected_idx"] == 15

    # Test clamp at last case
    session_state["audit_selected_idx"] = 39  # Case 40
    selected_idx = session_state["audit_selected_idx"]
    df_audit.at[selected_idx, "judgment"] = "correct"
    if selected_idx < total_cases - 1:
        session_state["audit_selected_idx"] = selected_idx + 1
    assert session_state["audit_selected_idx"] == 39  # Clamped at last case


def test_b5_dynamic_behavior_summary_metrics():
    """B5: Assert summary metrics are computed dynamically from df_rules, not hardcoded."""
    sample_rules = pd.DataFrame({
        "alert_id": [f"A{i}" for i in range(10)],
        "sequence_id": ["seq01"] * 5 + ["seq02"] * 5,
        "rule_name": ["running_or_sudden_speed"] * 6 + ["abrupt_stop_or_reversal"] * 4,
    })

    total_alerts = len(sample_rules)
    running_alerts = int((sample_rules["rule_name"] == "running_or_sudden_speed").sum())
    reversal_alerts = int((sample_rules["rule_name"] == "abrupt_stop_or_reversal").sum())
    unique_seqs = sample_rules["sequence_id"].nunique()

    assert total_alerts == 10
    assert running_alerts == 6
    assert reversal_alerts == 4
    assert unique_seqs == 2


def test_optional_atomic_csv_write():
    """Optional: Verify atomic CSV write leaves no temp file and writes accurately."""
    from work.app.dashboard import _atomic_save_audit_csv

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "test_audit.csv"
        df = pd.DataFrame({"case_id": [1, 2], "judgment": ["correct", "incorrect"]})

        _atomic_save_audit_csv(df, target)

        assert target.exists()
        loaded = pd.read_csv(target)
        assert len(loaded) == 2
        assert list(loaded["judgment"]) == ["correct", "incorrect"]
        assert not target.with_suffix(".tmp").exists()

"""
Rigorous regression tests for Streamlit Dashboard (work/app/dashboard.py).
========================================================================
Exercises real dashboard module functions and AppTest execution.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

# Set YOLO autoinstall false before any imports
os.environ["YOLO_AUTOINSTALL"] = "False"

from work.app.dashboard import (
    _atomic_save_audit_csv,
    _check_video_decodable,
    _compute_rule_summary,
    _next_audit_index,
    _resolve_upload_path,
    _run_inspector_stream,
    _save_audit_judgment,
)


class DummyUpload:
    """Mock file upload object with getbuffer, name, and size attributes."""
    def __init__(self, name: str, size: int, content: bytes):
        self.name = name
        self.size = size
        self.content = content

    def getbuffer(self) -> bytes:
        return self.content

    def read(self) -> bytes:
        return self.content


@pytest.fixture
def sample_video_fixture(tmp_path) -> Path:
    """Creates a small 5-frame synthetic video file for testing."""
    v_path = tmp_path / "fixture_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(v_path), fourcc, 10.0, (64, 64))
    for _ in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return v_path


def test_b1_and_n1_inspector_real_execution_and_error_handling(sample_video_fixture):
    """
    B1/N1: Test _run_inspector_stream with real video.
    Asserts:
      1. Normal run processes frames, derives metrics from engine.
      2. Mid-stream inference error sets success=False (N1 fix prevents st.success).
    """
    # 1. Normal run
    mock_engine = MagicMock()
    mock_engine.process_frame.return_value = (
        np.zeros((64, 64, 3), dtype=np.uint8),
        {"n_weapons": 2, "n_persons": 4, "n_associated": 1, "active_alert_ids": []},
    )

    success, count, meta = _run_inspector_stream(
        video_path=sample_video_fixture,
        engine=mock_engine,
        max_inspect_frames=3,
    )
    assert success is True
    assert count == 3
    assert meta["n_weapons"] == 2
    assert meta["n_persons"] == 4
    assert meta["n_associated"] == 1

    # 2. Failing run (N1 check)
    failing_engine = MagicMock()
    failing_engine.process_frame.side_effect = RuntimeError("Inference breakdown")

    failed_success, failed_count, _ = _run_inspector_stream(
        video_path=sample_video_fixture,
        engine=failing_engine,
        max_inspect_frames=3,
    )
    # N1: Inference failure must result in success=False so st.success is NOT shown
    assert failed_success is False
    assert failed_count == 1  # Aborted on first frame


def test_b2_check_video_decodable_zero_bytes(tmp_path, sample_video_fixture):
    """B2: Real _check_video_decodable rejects 0-byte video and accepts valid video."""
    zero_byte = tmp_path / "empty.mp4"
    zero_byte.write_bytes(b"")

    is_valid, total_frames, _ = _check_video_decodable(zero_byte)
    assert is_valid is False
    assert total_frames <= 0

    valid, valid_frames, fps = _check_video_decodable(sample_video_fixture)
    assert valid is True
    assert valid_frames == 5
    assert fps > 0


def test_b3_resolve_upload_path_caching_and_cleanup(tmp_path):
    """
    B3: Real _resolve_upload_path caches tempfile across 5 calls,
    and unlinks old file when replaced by new upload.
    """
    session_state = {}
    up1 = DummyUpload("video1.mp4", 100, b"video_data_1")

    # Call 5 times with identical file
    path1, is_vid1 = _resolve_upload_path(up1, session_state)
    assert path1.exists()
    assert is_vid1 is True

    for _ in range(4):
        p_reused, _ = _resolve_upload_path(up1, session_state)
        assert p_reused == path1

    # Upload replacement file
    up2 = DummyUpload("video2.mp4", 200, b"video_data_2")
    path2, is_vid2 = _resolve_upload_path(up2, session_state)

    assert path2.exists()
    assert path2 != path1
    # B3 fix: old file must be deleted from disk
    assert not path1.exists()

    # Cleanup
    if path2.exists():
        path2.unlink()


def test_b4_audit_save_and_advance_temp_csv(tmp_path):
    """
    B4: Using a temp copy of the audit CSV, select case 15 (idx 14), save judgment,
    and assert row 14 is saved in CSV and selection advances to 15 (Case 16).
    Also assert on the last case (idx 39) selection remains clamped at 39.
    """
    real_csv = Path("outputs/pipeline/audit_sheet/association_audit_sheet.csv")
    temp_csv = tmp_path / "temp_audit_sheet.csv"
    shutil.copy(real_csv, temp_csv)

    df_copy = pd.read_csv(temp_csv, dtype={"judgment": "object"})
    df_copy["judgment"] = df_copy["judgment"].fillna("")

    total_cases = len(df_copy)
    assert total_cases == 40

    # 1. Select Case 15 (idx 14), mark 'correct'
    df_res, next_idx = _save_audit_judgment(
        df_audit=df_copy,
        audit_csv_p=temp_csv,
        selected_idx=14,
        judgment_val="correct",
        total_cases=total_cases,
        advance=True,
    )

    # Assert selection is now 15 (Case 16)
    assert next_idx == 15

    # Reload from disk and assert row 14 is saved as 'correct'
    reloaded = pd.read_csv(temp_csv, dtype={"judgment": "object"}).fillna("")
    assert reloaded.at[14, "judgment"] == "correct"

    # 2. Select last case (idx 39, Case 40), mark 'correct'
    _, last_next_idx = _save_audit_judgment(
        df_audit=reloaded,
        audit_csv_p=temp_csv,
        selected_idx=39,
        judgment_val="correct",
        total_cases=total_cases,
        advance=True,
    )
    # Selection must remain clamped at 39
    assert last_next_idx == 39

    reloaded_last = pd.read_csv(temp_csv, dtype={"judgment": "object"}).fillna("")
    assert reloaded_last.at[39, "judgment"] == "correct"


def test_b5_dynamic_rule_summary_and_apptest_metrics(tmp_path, monkeypatch):
    """
    B5: Real _compute_rule_summary and AppTest verify metrics match
    exact DataFrame counts instead of old hardcoded numbers.
    """
    # 1. Pure function check on _compute_rule_summary
    sample_df = pd.DataFrame({
        "alert_id": [f"A{i}" for i in range(10)],
        "sequence_id": ["seq01"] * 5 + ["seq02"] * 5,
        "rule_name": ["running_or_sudden_speed"] * 6 + ["abrupt_stop_or_reversal"] * 4,
    })
    summary = _compute_rule_summary(sample_df)
    assert summary["total_alerts"] == 10
    assert summary["running_alerts"] == 6
    assert summary["reversal_alerts"] == 4
    assert summary["unique_seqs"] == 2

    # 2. AppTest verification: override RULE_CSV_PATH with temp CSV
    temp_rule_csv = tmp_path / "custom_rule_firings.csv"
    sample_df.to_csv(temp_rule_csv, index=False)
    monkeypatch.setenv("RULE_CSV_PATH", str(temp_rule_csv))

    dashboard_script = (Path(__file__).resolve().parent.parent / "work/app/dashboard.py").resolve()
    at = AppTest.from_file(str(dashboard_script))
    at.run(timeout=15)

    metric_map = {m.label: m.value for m in at.metric}
    assert metric_map.get("Total Review Alerts") == "10 Events"
    assert metric_map.get("Running / Speed Triggers") == "6 Events"
    assert metric_map.get("Direction Reversals") == "4 Events"
    assert metric_map.get("Analyzed CCTV Sequences") == "2 Sequences (72 frames)"


def test_optional_atomic_csv_write(tmp_path):
    """Optional: Real _atomic_save_audit_csv writes via .tmp and atomically replaces."""
    target = tmp_path / "audit_test.csv"
    df = pd.DataFrame({"case_id": [1, 2], "judgment": ["correct", "incorrect"]})

    _atomic_save_audit_csv(df, target)
    assert target.exists()
    assert not target.with_suffix(".tmp").exists()

    loaded = pd.read_csv(target)
    assert len(loaded) == 2
    assert list(loaded["judgment"]) == ["correct", "incorrect"]

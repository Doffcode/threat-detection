"""
scripts/pipeline/run_stage1_pipeline.py
=======================================
Executes the Stage 1 Pipeline on Innovatus CCTV frames ordered by sequence.
Scores against test_cctv_v2 labels strictly for evaluation.
Produces:
1. Annotated demo video (MP4)
2. JSON event log (events.json)
3. 20 good case overlays & 20 bad case overlays
4. Quantitative metrics report
"""

from __future__ import annotations

import collections
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
os.chdir(WORKSPACE_ROOT)

import sys
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.pipeline.engine import Stage1PipelineEngine


def box_iou(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def main():
    print("=" * 80)
    print("RUNNING STAGE 1 PIPELINE ON INNOVATUS CCTV FRAMES")
    print("=" * 80)

    images_dir = WORKSPACE_ROOT / "data/cctv_test_v2_candidates/images"
    labels_dir = WORKSPACE_ROOT / "data/cctv_test_v2_candidates/labels"
    csv_p = WORKSPACE_ROOT / "configs/detection/cctv_test_v2_occlusion_template.csv"
    output_dir = WORKSPACE_ROOT / "outputs/pipeline"
    
    # Galleries
    good_dir = output_dir / "galleries/good"
    bad_dir = output_dir / "galleries/bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_p)
    inno_df = df[df["source"] == "innovatus_rob"].copy()

    engine = Stage1PipelineEngine(
        config_path="src/pipeline/config.yaml",
        output_dir=str(output_dir),
    )

    # Sort sequences and frames
    sequences = sorted(inno_df["sequence_id"].unique())
    print(f"Loaded {len(inno_df)} frames across sequences: {sequences}")

    # Tracking evaluation structures
    gt_total_weapons = 0
    gt_detected_weapons = 0
    gt_linked_to_person = 0

    frame_eval_records = []
    annotated_frames_list = []

    for seq_id in sequences:
        seq_df = inno_df[inno_df["sequence_id"] == seq_id].sort_values("frame_number").reset_index(drop=True)
        print(f"\nProcessing Sequence: {seq_id} ({len(seq_df)} frames)...")
        engine.reset_sequence(seq_id)

        prev_tids = set()
        seq_id_switches = 0

        for idx, row in seq_df.iterrows():
            fn = row["image_filename"]
            fnum = float(row["frame_number"])
            img_p = images_dir / fn
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Read GT boxes strictly for evaluation
            gt_boxes = []
            lbl_p = labels_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                for l in lbl_p.read_text().splitlines():
                    if l.strip():
                        parts = l.split()
                        cid = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2.0) * w
                        y1 = (cy - bh / 2.0) * h
                        x2 = (cx + bw / 2.0) * w
                        y2 = (cy + bh / 2.0) * h
                        gt_boxes.append({"cls": cid, "box": [x1, y1, x2, y2]})

            gt_total_weapons += len(gt_boxes)

            # Process frame
            annotated_bgr, meta = engine.process_frame(
                img_bgr=img,
                frame_number=fnum,
                sequence_id=seq_id,
                image_name=fn,
            )
            annotated_frames_list.append(annotated_bgr)

            # Track ID stability
            current_tids = {t["track_id"] for t in meta["tracks"] if t["track_id"] > 0}
            # Score GT weapons vs detected & linked
            matched_gt = set()
            for g_idx, g in enumerate(gt_boxes):
                for assoc in meta["associations"]:
                    iou = box_iou(g["box"], assoc["weapon_box"])
                    if iou >= 0.30:  # Valid spatial overlap
                        if g_idx not in matched_gt:
                            matched_gt.add(g_idx)
                            gt_detected_weapons += 1
                            if assoc["is_associated"]:
                                gt_linked_to_person += 1

            frame_eval_records.append({
                "sequence_id": seq_id,
                "frame_number": fnum,
                "filename": fn,
                "img_path": str(img_p),
                "annotated_path": meta["annotated_frame_path"],
                "n_gt": len(gt_boxes),
                "n_matched_gt": len(matched_gt),
                "n_weapons_detected": len(meta["weapons"]),
                "n_persons_tracked": len(meta["tracks"]),
                "active_alerts": meta["active_alert_ids"],
                "associations": meta["associations"],
                "has_alert": len(meta["active_alert_ids"]) > 0,
            })

    # Finalize pipeline and dump events.json
    summary = engine.finalize()
    alerts = summary["alerts"]
    print(f"\nPipeline run completed. Total alerts generated: {len(alerts)}")

    # -------------------------------------------------------------
    # Render Demo Video
    # -------------------------------------------------------------
    demo_video_path = output_dir / "stage1_cctv_demo.mp4"
    if annotated_frames_list:
        h, w = annotated_frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(demo_video_path), fourcc, 6.0, (w, h))
        for f in annotated_frames_list:
            out_writer.write(f)
        out_writer.release()
        print(f"Saved demo video to: {demo_video_path}")

    # -------------------------------------------------------------
    # Curate 20 Good Cases and 20 Bad Cases Galleries
    # -------------------------------------------------------------
    print("\nCurating 20 Good Cases and 20 Bad Cases Galleries...")
    # Good cases: GT weapon present, detected, associated to track, and confirmed/alerted
    good_candidates = []
    bad_candidates = []

    for rec in frame_eval_records:
        n_gt = rec["n_gt"]
        n_det = rec["n_weapons_detected"]
        has_alert = rec["has_alert"]
        n_assoc = sum(1 for a in rec["associations"] if a["is_associated"])

        if n_gt > 0 and n_det > 0 and n_assoc > 0:
            score = 10.0 + (5.0 if has_alert else 0.0) + rec["associations"][0].get("weapon_conf", 0.0)
            good_candidates.append((score, rec))
        elif n_gt > 0 and (n_det == 0 or n_assoc == 0):
            # Missed weapon or failed association
            reason = "Missed weapon detection" if n_det == 0 else "Failed association to person track"
            bad_candidates.append((reason, rec))
        elif n_gt == 0 and n_det > 0:
            # False alarm
            reason = "False weapon detection on background/clutter"
            bad_candidates.append((reason, rec))

    good_candidates.sort(key=lambda x: x[0], reverse=True)
    selected_good = good_candidates[:20]
    selected_bad = bad_candidates[:20]

    good_manifest = []
    for idx, (sc, rec) in enumerate(selected_good):
        fn = f"good_{idx+1:02d}_{rec['sequence_id']}_f{int(rec['frame_number']):04d}.jpg"
        save_p = good_dir / fn
        shutil.copy(rec["annotated_path"], save_p)
        good_manifest.append({
            "case_id": idx + 1,
            "filename": fn,
            "sequence_id": rec["sequence_id"],
            "frame_number": rec["frame_number"],
            "n_gt": rec["n_gt"],
            "n_weapons": rec["n_weapons_detected"],
            "active_alerts": rec["active_alerts"],
            "assessment": "Successful weapon detection, accurate person association, active temporal tracking",
        })

    bad_manifest = []
    for idx, (reason, rec) in enumerate(selected_bad):
        fn = f"bad_{idx+1:02d}_{rec['sequence_id']}_f{int(rec['frame_number']):04d}.jpg"
        save_p = bad_dir / fn
        shutil.copy(rec["annotated_path"], save_p)
        bad_manifest.append({
            "case_id": idx + 1,
            "filename": fn,
            "sequence_id": rec["sequence_id"],
            "frame_number": rec["frame_number"],
            "failure_cause": reason,
            "n_gt": rec["n_gt"],
            "n_weapons": rec["n_weapons_detected"],
            "assessment": reason,
        })

    with open(good_dir / "manifest.json", "w") as f:
        json.dump(good_manifest, f, indent=2)
    with open(bad_dir / "manifest.json", "w") as f:
        json.dump(bad_manifest, f, indent=2)

    # -------------------------------------------------------------
    # Quantitative Metrics
    # -------------------------------------------------------------
    frames_with_gt = sum(1 for r in frame_eval_records if r["n_gt"] > 0)
    frames_with_alert = sum(1 for r in frame_eval_records if r["has_alert"])
    pct_gt_linked = (gt_linked_to_person / gt_total_weapons) * 100.0 if gt_total_weapons > 0 else 0.0
    pct_gt_detected = (gt_detected_weapons / gt_total_weapons) * 100.0 if gt_total_weapons > 0 else 0.0

    # Per-sequence breakdown
    seq_metrics = {}
    for seq_id in sequences:
        s_recs = [r for r in frame_eval_records if r["sequence_id"] == seq_id]
        s_alerts = [a for a in alerts if a["sequence_id"] == seq_id]
        s_gt = sum(r["n_gt"] for r in s_recs)
        s_gt_frames = sum(1 for r in s_recs if r["n_gt"] > 0)
        s_alert_frames = sum(1 for r in s_recs if r["has_alert"])
        seq_metrics[seq_id] = {
            "total_frames": len(s_recs),
            "gt_weapons": s_gt,
            "gt_weapon_frames": s_gt_frames,
            "alert_frames": s_alert_frames,
            "alert_events_count": len(s_alerts),
            "alert_event_ids": [a["alert_id"] for a in s_alerts],
        }

    metrics_report = {
        "dataset": "Innovatus CCTV (test_cctv_v2 clean)",
        "total_frames": len(frame_eval_records),
        "total_gt_weapons": gt_total_weapons,
        "gt_weapons_detected": gt_detected_weapons,
        "gt_weapons_linked_to_person": gt_linked_to_person,
        "share_gt_weapons_linked_pct": round(pct_gt_linked, 2),
        "share_gt_weapons_detected_pct": round(pct_gt_detected, 2),
        "frames_with_gt_weapons": frames_with_gt,
        "frames_with_active_alerts": frames_with_alert,
        "total_alert_events": len(alerts),
        "per_sequence_metrics": seq_metrics,
        "good_cases_count": len(selected_good),
        "bad_cases_count": len(selected_bad),
        "demo_video_path": str(demo_video_path),
    }

    with open(output_dir / "pipeline_metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 1 PIPELINE QUANTITATIVE METRICS SUMMARY")
    print("=" * 80)
    print(f"Total Frames Processed           : {len(frame_eval_records)}")
    print(f"Total Ground Truth Weapons       : {gt_total_weapons}")
    print(f"GT Weapons Detected by v2        : {gt_detected_weapons} ({pct_gt_detected:.1f}%)")
    print(f"GT Weapons Linked to Person      : {gt_linked_to_person} ({pct_gt_linked:.1f}%)")
    print(f"Frames with Ground Truth Weapons : {frames_with_gt}")
    print(f"Frames with Active Alerts        : {frames_with_alert}")
    print(f"Total Alert Events Generated     : {len(alerts)}")
    print("\nPer-Sequence Breakdown:")
    for seq_id, sm in seq_metrics.items():
        print(f"  [{seq_id}]: {sm['total_frames']} frames | GT: {sm['gt_weapons']} boxes in {sm['gt_weapon_frames']} frames | Alert Events: {sm['alert_events_count']} ({sm['alert_frames']} alert frames)")


if __name__ == "__main__":
    main()

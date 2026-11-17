"""
scripts/behavior/run_behavior_analysis.py
=========================================
Runs Stage 2 behavioral kinematics & rule analysis on the Innovatus CCTV stretches.
Treats timings as relative because FPS is unknown.
Produces:
1. 2D Trajectory plots per sequence
2. Annotated behavioral demo video
3. Table of rule firings per person (CSV + JSON)
4. Comprehensive behavior metrics report
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import yaml

WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
os.chdir(WORKSPACE_ROOT)

import sys
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.behavior.kinematics import PersonKinematicsTracker
from src.behavior.rules import BehaviorAlert, BehaviorRuleEngine
from src.behavior.visualizer import BehaviorVisualizer
from src.pipeline.tracker import PersonTracker


def main():
    print("=" * 80)
    print("RUNNING STAGE 2 BEHAVIORAL KINEMATICS & RULES ON INNOVATUS CCTV")
    print("=" * 80)

    cfg_p = WORKSPACE_ROOT / "src/behavior/config.yaml"
    with open(cfg_p) as f:
        cfg = yaml.safe_load(f)

    output_dir = WORKSPACE_ROOT / "outputs/behavior"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize modules
    tracker_model = PersonTracker(
        cfg={"weights": "work/raw/IntProj_Weapon_Dataset_annot/yolo11m.pt", "imgsz": 640, "conf_threshold": 0.35},
        device="cuda:0",
    )
    kin_tracker = PersonKinematicsTracker(zones_cfg=cfg["zones"], default_fps=cfg["pipeline"]["default_fps"])
    rule_engine = BehaviorRuleEngine(thresholds_cfg=cfg["thresholds"])
    visualizer = BehaviorVisualizer(zones_cfg=cfg["zones"], output_dir=output_dir)

    inno_csv = WORKSPACE_ROOT / "configs/detection/cctv_test_v2_occlusion_template.csv"
    df_inno = pd.read_csv(inno_csv)
    inno_subset = df_inno[df_inno["source"] == "innovatus_rob"].copy()
    inno_img_dir = WORKSPACE_ROOT / "data/cctv_test_v2_candidates/images"

    sequences = sorted(inno_subset["sequence_id"].unique())
    all_rule_firings = []
    annotated_frames_list = []
    trajectory_plot_paths = {}

    for seq_id in sequences:
        seq_df = inno_subset[inno_subset["sequence_id"] == seq_id].sort_values("frame_number").reset_index(drop=True)
        print(f"\nProcessing Sequence {seq_id} ({len(seq_df)} frames)...")

        tracker_model.reset()
        kin_tracker.reset()
        rule_engine.reset()

        last_img_bgr = None
        img_w, img_h = 512, 512

        for f_idx, row in seq_df.iterrows():
            fn = row["image_filename"]
            fnum = float(row["frame_number"])
            img_p = inno_img_dir / fn
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            last_img_bgr = img
            img_h, img_w = img.shape[:2]

            # 1. Person tracking
            tracks = tracker_model.track(img)

            # 2. Kinematics
            profiles = []
            for t in tracks:
                prof = kin_tracker.update_track(
                    track_id=t.track_id,
                    box=t.box,
                    frame_idx=f_idx + 1,
                    frame_number=fnum,
                    img_w=img_w,
                    img_h=img_h,
                )
                profiles.append(prof)

            # 3. Rule evaluation
            alerts = rule_engine.evaluate_frame(profiles, frame_idx=f_idx + 1, frame_number=fnum)

            for al in alerts:
                all_rule_firings.append({
                    "alert_id": al.alert_id,
                    "sequence_id": seq_id,
                    "track_id": al.track_id,
                    "rule_name": al.rule_name,
                    "frame_idx": al.frame_idx,
                    "frame_number": al.frame_number,
                    "rule_description": al.rule_description,
                    "trigger_numbers": json.dumps(al.trigger_numbers),
                    "status": al.status,
                })

            # 4. Render visual annotations
            annotated = visualizer.render_frame_annotations(
                img_bgr=img,
                profiles=profiles,
                alerts=alerts,
                track_histories=kin_tracker.track_histories,
                frame_number=fnum,
                sequence_id=seq_id,
            )
            annotated_frames_list.append(annotated)

        # Plot 2D trajectories
        plot_p = visualizer.plot_sequence_trajectories(
            sequence_id=seq_id,
            track_histories=kin_tracker.track_histories,
            img_w=img_w,
            img_h=img_h,
            background_img_bgr=last_img_bgr,
        )
        trajectory_plot_paths[seq_id] = plot_p
        print(f"Saved trajectory plot to: {plot_p}")

    # Export demo video
    demo_video_p = output_dir / "behavior_cctv_demo.mp4"
    if annotated_frames_list:
        vh, vw = annotated_frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vwriter = cv2.VideoWriter(str(demo_video_p), fourcc, 6.0, (vw, vh))
        for f in annotated_frames_list:
            vwriter.write(f)
        vwriter.release()
        print(f"\nSaved behavioral demo video to: {demo_video_p}")

    # Export table of rule firings
    firings_df = pd.DataFrame(all_rule_firings)
    csv_p = output_dir / "rule_firings.csv"
    firings_df.to_csv(csv_p, index=False)
    print(f"Saved table of rule firings to: {csv_p} ({len(all_rule_firings)} events)")

    # Summary JSON
    summary = {
        "dataset": "Innovatus CCTV (72 frames, 3 sequences)",
        "total_behavior_alerts": len(all_rule_firings),
        "rule_counts": collections.Counter([r["rule_name"] for r in all_rule_firings]),
        "demo_video_path": str(demo_video_p),
        "trajectory_plots": trajectory_plot_paths,
        "rule_firings_csv": str(csv_p),
    }
    with open(output_dir / "behavior_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBehavioral Rule Firings Breakdown:")
    for rule, count in summary["rule_counts"].items():
        print(f"  - {rule}: {count} firings")


if __name__ == "__main__":
    main()

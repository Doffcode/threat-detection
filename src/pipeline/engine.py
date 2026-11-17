"""
src/pipeline/engine.py
======================
Master Stage 1 Pipeline Engine.
Coordinates detector, tracker, pose estimator, associator, and temporal alert manager.
Generates annotated frames, video demo, and structured JSON event logs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .associator import WeaponAssociation, WeaponPersonAssociator
from .detector import WeaponDetection, WeaponDetector
from .pose import PoseEstimator
from .temporal import AlertRecord, TemporalAlertManager
from .tracker import PersonTrack, PersonTracker


class Stage1PipelineEngine:
    def __init__(self, config_path: str, output_dir: str = "outputs/pipeline"):
        cfg_p = Path(config_path)
        if not cfg_p.is_absolute():
            cfg_p = Path("/home/shiva/Desktop/projects/akriti/weapon-detection") / cfg_p
        
        with open(cfg_p, "r") as f:
            self.raw_config = yaml.safe_load(f)

        self.device = self.raw_config.get("pipeline", {}).get("device", "cuda:0")
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = Path("/home/shiva/Desktop/projects/akriti/weapon-detection") / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.annotated_frames_dir = self.output_dir / "annotated_frames"
        self.annotated_frames_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-modules
        self.detector = WeaponDetector(self.raw_config["detector"], device=self.device)
        self.tracker = PersonTracker(self.raw_config["tracker"], device=self.device)
        self.pose = PoseEstimator(self.raw_config["pose"], device=self.device)
        self.associator = WeaponPersonAssociator(self.raw_config["association"])
        self.temporal = TemporalAlertManager(self.raw_config["temporal"], output_dir=self.output_dir)

        self.current_sequence_id: Optional[str] = None
        self.frame_idx: int = 0
        self.all_frame_records: List[Dict[str, Any]] = []

    def reset_sequence(self, sequence_id: str):
        """Resets tracker and temporal state when transitioning sequences."""
        self.tracker.reset()
        self.temporal.reset_sequence()
        self.current_sequence_id = sequence_id
        self.frame_idx = 0

    def process_frame(
        self,
        img_bgr: np.ndarray,
        frame_number: float,
        sequence_id: str,
        image_name: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Processes a single frame through the full pipeline."""
        if sequence_id != self.current_sequence_id:
            self.reset_sequence(sequence_id)

        self.frame_idx += 1
        h, w = img_bgr.shape[:2]

        # 1. Weapon Detection
        weapons = self.detector.detect(img_bgr)

        # 2. Person Detection & Tracking
        tracks = self.tracker.track(img_bgr)

        # 3. Pose / Wrist Keypoints
        track_wrists = self.pose.estimate_wrists_for_tracks(img_bgr, tracks)

        # 4. Association
        associations = self.associator.associate(weapons, tracks, track_wrists)

        # 5. Temporal Confirmation & Alerting
        active_alerts = self.temporal.update(
            frame_idx=self.frame_idx,
            frame_number=frame_number,
            sequence_id=sequence_id,
            img_bgr=img_bgr,
            associations=associations,
        )

        # 6. Visual Annotation
        annotated_bgr = self._render_annotations(
            img_bgr=img_bgr,
            tracks=tracks,
            weapons=weapons,
            associations=associations,
            active_alerts=active_alerts,
            frame_number=frame_number,
            sequence_id=sequence_id,
        )

        # Save annotated frame
        frame_fn = f"{sequence_id}_f{int(frame_number):04d}_{image_name}" if image_name else f"{sequence_id}_f{int(frame_number):04d}.jpg"
        frame_save_p = self.annotated_frames_dir / frame_fn
        cv2.imwrite(str(frame_save_p), annotated_bgr)

        # Record metadata
        meta = {
            "sequence_id": sequence_id,
            "frame_idx": self.frame_idx,
            "frame_number": frame_number,
            "image_name": image_name,
            "annotated_frame_path": str(frame_save_p),
            "n_weapons": len(weapons),
            "n_persons": len(tracks),
            "n_associated": sum(1 for a in associations if a.is_associated),
            "active_alert_ids": [a.alert_id for a in active_alerts],
            "weapons": [
                {"box": w.box, "conf": w.conf, "cls": w.cls_name} for w in weapons
            ],
            "tracks": [
                {"box": t.box, "track_id": t.track_id, "conf": t.conf} for t in tracks
            ],
            "associations": [
                {
                    "weapon_box": a.weapon.box,
                    "weapon_cls": a.weapon.cls_name,
                    "weapon_conf": a.weapon.conf,
                    "track_id": a.track_id,
                    "is_associated": a.is_associated,
                    "link_score": a.link_score,
                    "containment_ioa": a.containment_ioa,
                    "center_inside": a.center_inside,
                    "wrist_dist_px": a.wrist_dist_px,
                    "wrist_dist_rel": a.wrist_dist_rel_height,
                }
                for a in associations
            ],
        }
        self.all_frame_records.append(meta)
        return annotated_bgr, meta

    def finalize(self) -> Dict[str, Any]:
        """Finalizes execution and dumps structured event log."""
        alerts = self.temporal.finalize_all()
        summary = {
            "total_frames_processed": len(self.all_frame_records),
            "total_alerts_generated": len(alerts),
            "alerts": alerts,
            "frame_records": self.all_frame_records,
        }
        events_p = self.output_dir / "events.json"
        with open(events_p, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _render_annotations(
        self,
        img_bgr: np.ndarray,
        tracks: List[PersonTrack],
        weapons: List[WeaponDetection],
        associations: List[WeaponAssociation],
        active_alerts: List[AlertRecord],
        frame_number: float,
        sequence_id: str,
    ) -> np.ndarray:
        vis = img_bgr.copy()
        h, w = vis.shape[:2]

        armed_tids = {a.track_id for a in associations if a.is_associated and a.track_id is not None}
        alerted_tids = {al.track_id for al in active_alerts}

        # 1. Draw Persons
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            if t.track_id in alerted_tids:
                color = (0, 0, 255)  # Red for Alerted Armed Person
                label = f"ALERT ARMED Person #{t.track_id}"
                thick = 3
            elif t.track_id in armed_tids:
                color = (0, 140, 255)  # Orange for Associated Armed Person
                label = f"Armed Person #{t.track_id}"
                thick = 2
            else:
                color = (0, 220, 0)  # Green for Unarmed Person
                label = f"Person #{t.track_id}"
                thick = 2

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            cv2.putText(
                vis,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Weapons & Association Links
        for assoc in associations:
            wb = list(map(int, assoc.weapon.box))
            w_cls = assoc.weapon.cls_name
            w_conf = assoc.weapon.conf
            w_color = (0, 69, 255) if w_cls == "handgun" else (0, 255, 255) if w_cls == "knife" else (255, 0, 0)

            # Weapon box
            cv2.rectangle(vis, (wb[0], wb[1]), (wb[2], wb[3]), w_color, 2)
            cv2.putText(
                vis,
                f"{w_cls} {w_conf:.2f}",
                (wb[0], max(15, wb[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                w_color,
                2,
                cv2.LINE_AA,
            )

            # Association link to person
            if assoc.is_associated and assoc.track_id is not None and assoc.person_box is not None:
                w_cx = int((wb[0] + wb[2]) / 2)
                w_cy = int((wb[1] + wb[3]) / 2)
                pb = list(map(int, assoc.person_box))
                p_cx = int((pb[0] + pb[2]) / 2)
                p_cy = int((pb[1] + pb[3]) / 2)

                # Draw dashed / direct line
                cv2.line(vis, (w_cx, w_cy), (p_cx, p_cy), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(vis, (w_cx, w_cy), 4, (0, 255, 255), -1)

        # 3. Top Alert Banner
        if active_alerts:
            banner_h = 36
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
            
            al_txt = f"!! CRITICAL ALERT: {len(active_alerts)} ARMED PERSON(S) DETECTED "
            details = [f"Track #{al.track_id} ({al.weapon_class} {al.max_confidence:.2f})" for al in active_alerts]
            al_txt += " | ".join(details)
            cv2.putText(
                vis,
                al_txt,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            # Info overlay
            cv2.putText(
                vis,
                f"Seq: {sequence_id} | Frame {int(frame_number)} | Weapons: {len(weapons)} | Persons: {len(tracks)}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return vis

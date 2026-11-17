"""
src/behavior/rules.py
=====================
Rule-based behavioral alert engine.
Evaluates four movement heuristics:
1. Loitering (sustained dwell inside zone with low net displacement)
2. Running / Sudden Speed Change (speed or acceleration exceeding walking thresholds)
3. Abrupt Stop or Reversal (sharp direction change >= 120 deg)
4. Moving Against Dominant Flow (heading opposing crowd velocity vector)

MANDATORY SAFETY CONSTRAINT:
Every alert generated explicitly carries:
  status: "FOR_HUMAN_REVIEW_ONLY"
Behavioral cues are contextual indicators for human review, NEVER "threat confirmed".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .kinematics import KinematicProfile, compute_angle_difference


@dataclass
class BehaviorAlert:
    alert_id: str
    track_id: int
    rule_name: str
    frame_idx: int
    frame_number: float
    trigger_numbers: Dict[str, Any]
    rule_description: str
    status: str = "FOR_HUMAN_REVIEW_ONLY"


class BehaviorRuleEngine:
    def __init__(self, thresholds_cfg: Dict[str, Any]):
        self.t_cfg = thresholds_cfg
        self.alert_counter = 0
        self.cooldown_tracker: Dict[Tuple[int, str], int] = {}  # (track_id, rule) -> last_fired_frame

    def reset(self):
        self.alert_counter = 0
        self.cooldown_tracker.clear()

    def evaluate_frame(
        self,
        profiles: List[KinematicProfile],
        frame_idx: int,
        frame_number: float,
    ) -> List[BehaviorAlert]:
        """Evaluates all behavioral rules for all active tracks in the frame."""
        alerts: List[BehaviorAlert] = []

        # 1. Estimate Dominant Crowd Flow (if multiple moving tracks exist)
        moving_headings = [
            p.heading_deg for p in profiles if p.heading_deg is not None and p.speed_h_per_frame > 0.02
        ]
        dominant_crowd_heading = None
        if len(moving_headings) >= self.t_cfg.get("against_dominant_flow", {}).get("min_simultaneous_tracks", 3):
            vx = sum(math.cos(math.radians(h)) for h in moving_headings)
            vy = sum(math.sin(math.radians(h)) for h in moving_headings)
            if math.hypot(vx, vy) > 0.5:
                dominant_crowd_heading = math.degrees(math.atan2(vy, vx))

        # 2. Evaluate Per-Person Rules
        for p in profiles:
            tid = p.track_id

            # Rule A: Loitering
            r_loiter = self._check_loitering(p, frame_idx, frame_number)
            if r_loiter:
                alerts.append(r_loiter)

            # Rule B: Running or Sudden Speed Change
            r_run = self._check_running(p, frame_idx, frame_number)
            if r_run:
                alerts.append(r_run)

            # Rule C: Abrupt Stop or Reversal
            r_rev = self._check_reversal(p, frame_idx, frame_number)
            if r_rev:
                alerts.append(r_rev)

            # Rule D: Moving Against Dominant Flow
            if dominant_crowd_heading is not None and p.heading_deg is not None and p.speed_h_per_frame > 0.02:
                r_flow = self._check_flow(p, dominant_crowd_heading, len(moving_headings), frame_idx, frame_number)
                if r_flow:
                    alerts.append(r_flow)

        return alerts

    def _can_fire(self, track_id: int, rule_name: str, frame_idx: int, cooldown: int = 10) -> bool:
        key = (track_id, rule_name)
        last_frame = self.cooldown_tracker.get(key, -999)
        if frame_idx - last_frame >= cooldown:
            self.cooldown_tracker[key] = frame_idx
            return True
        return False

    def _check_loitering(self, p: KinematicProfile, frame_idx: int, frame_number: float) -> Optional[BehaviorAlert]:
        cfg = self.t_cfg.get("loitering", {})
        min_dwell = cfg.get("min_dwell_frames", 8)
        max_ratio = cfg.get("max_displacement_ratio", 0.25)

        for z_name, dwell_cnt in p.dwell_time_frames_per_zone.items():
            if dwell_cnt >= min_dwell and p.displacement_ratio <= max_ratio and p.total_path_length_px > 10.0:
                if self._can_fire(p.track_id, f"loitering_{z_name}", frame_idx, cooldown=15):
                    self.alert_counter += 1
                    return BehaviorAlert(
                        alert_id=f"BEH-{self.alert_counter:04d}",
                        track_id=p.track_id,
                        rule_name="loitering",
                        frame_idx=frame_idx,
                        frame_number=frame_number,
                        trigger_numbers={
                            "zone": z_name,
                            "dwell_frames": dwell_cnt,
                            "dwell_threshold_frames": min_dwell,
                            "displacement_ratio": round(p.displacement_ratio, 3),
                            "max_displacement_ratio_threshold": max_ratio,
                            "total_path_length_px": round(p.total_path_length_px, 1),
                        },
                        rule_description=f"Person residing in zone '{z_name}' for {dwell_cnt} frames with low net transit (ratio {p.displacement_ratio:.2f})",
                    )
        return None

    def _check_running(self, p: KinematicProfile, frame_idx: int, frame_number: float) -> Optional[BehaviorAlert]:
        cfg = self.t_cfg.get("running_or_sudden_speed", {})
        speed_thresh = cfg.get("speed_h_per_frame", 0.10)
        accel_thresh = cfg.get("acceleration_h_per_frame2", 0.06)

        is_running = p.speed_h_per_frame >= speed_thresh
        is_sudden_accel = p.acceleration_h_per_frame2 >= accel_thresh

        if is_running or is_sudden_accel:
            if self._can_fire(p.track_id, "running_or_sudden_speed", frame_idx, cooldown=10):
                self.alert_counter += 1
                return BehaviorAlert(
                    alert_id=f"BEH-{self.alert_counter:04d}",
                    track_id=p.track_id,
                    rule_name="running_or_sudden_speed",
                    frame_idx=frame_idx,
                    frame_number=frame_number,
                    trigger_numbers={
                        "speed_h_per_frame": round(p.speed_h_per_frame, 3),
                        "speed_threshold": speed_thresh,
                        "acceleration_h_per_frame2": round(p.acceleration_h_per_frame2, 3),
                        "acceleration_threshold": accel_thresh,
                        "person_height_px": round(p.person_height, 1),
                    },
                    rule_description=f"Locomotion pace ({p.speed_h_per_frame:.2f} heights/frame) or acceleration ({p.acceleration_h_per_frame2:.2f}) exceeds normal walking limit",
                )
        return None

    def _check_reversal(self, p: KinematicProfile, frame_idx: int, frame_number: float) -> Optional[BehaviorAlert]:
        cfg = self.t_cfg.get("abrupt_stop_or_reversal", {})
        min_angle = cfg.get("min_heading_change_deg", 120.0)

        if p.last_heading_change_deg >= min_angle:
            if self._can_fire(p.track_id, "abrupt_stop_or_reversal", frame_idx, cooldown=8):
                self.alert_counter += 1
                return BehaviorAlert(
                    alert_id=f"BEH-{self.alert_counter:04d}",
                    track_id=p.track_id,
                    rule_name="abrupt_stop_or_reversal",
                    frame_idx=frame_idx,
                    frame_number=frame_number,
                    trigger_numbers={
                        "heading_change_deg": round(p.last_heading_change_deg, 1),
                        "heading_change_threshold_deg": min_angle,
                        "total_reversals_count": p.direction_reversals_count,
                    },
                    rule_description=f"Sharp direction reversal of {p.last_heading_change_deg:.1f} deg detected",
                )
        return None

    def _check_flow(
        self,
        p: KinematicProfile,
        crowd_heading: float,
        crowd_size: int,
        frame_idx: int,
        frame_number: float,
    ) -> Optional[BehaviorAlert]:
        cfg = self.t_cfg.get("against_dominant_flow", {})
        min_opposing = cfg.get("min_opposing_angle_deg", 120.0)

        diff = compute_angle_difference(p.heading_deg, crowd_heading)
        if diff >= min_opposing:
            if self._can_fire(p.track_id, "against_dominant_flow", frame_idx, cooldown=12):
                self.alert_counter += 1
                return BehaviorAlert(
                    alert_id=f"BEH-{self.alert_counter:04d}",
                    track_id=p.track_id,
                    rule_name="against_dominant_flow",
                    frame_idx=frame_idx,
                    frame_number=frame_number,
                    trigger_numbers={
                        "person_heading_deg": round(p.heading_deg, 1),
                        "crowd_heading_deg": round(crowd_heading, 1),
                        "opposing_angle_deg": round(diff, 1),
                        "opposing_threshold_deg": min_opposing,
                        "active_crowd_tracks_count": crowd_size,
                    },
                    rule_description=f"Person heading ({p.heading_deg:.1f} deg) opposes dominant crowd flow ({crowd_heading:.1f} deg) by {diff:.1f} deg",
                )
        return None

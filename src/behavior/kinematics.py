"""
src/behavior/kinematics.py
==========================
Computes kinematic motion profiles per tracked person:
- Speed (normalized by person height per frame and per second)
- Heading (direction angle in degrees)
- Direction reversals (abrupt turns >= threshold)
- Dwell time inside defined polygon zones
- Total path length
- Displacement / path-length ratio
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def compute_angle_difference(angle1_deg: float, angle2_deg: float) -> float:
    """Computes the shortest angular difference between two headings in [0, 180] degrees."""
    diff = abs(angle1_deg - angle2_deg) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm to determine if a 2D point is inside a polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


@dataclass
class KinematicProfile:
    track_id: int
    frame_idx: int
    frame_number: float
    current_position: Tuple[float, float]  # (x, y) foot point
    current_box: List[float]  # [x1, y1, x2, y2]
    person_height: float
    speed_h_per_frame: float
    speed_h_per_sec: Optional[float]
    acceleration_h_per_frame2: float
    heading_deg: Optional[float]
    direction_reversals_count: int
    last_heading_change_deg: float
    dwell_time_frames_per_zone: Dict[str, int]
    dwell_time_sec_per_zone: Dict[str, Optional[float]]
    total_path_length_px: float
    net_displacement_px: float
    displacement_ratio: float  # net_displacement / total_path_length in [0, 1]


class PersonKinematicsTracker:
    def __init__(self, zones_cfg: List[Dict[str, Any]], default_fps: Optional[float] = None):
        self.zones_cfg = zones_cfg
        self.default_fps = default_fps
        # track_id -> history of observations
        self.track_histories: Dict[int, List[Dict[str, Any]]] = {}
        self.reversal_counts: Dict[int, int] = {}
        self.zone_dwell_counts: Dict[int, Dict[str, int]] = {}

    def reset(self):
        """Reset internal state between video sequences."""
        self.track_histories.clear()
        self.reversal_counts.clear()
        self.zone_dwell_counts.clear()

    def update_track(
        self,
        track_id: int,
        box: List[float],
        frame_idx: int,
        frame_number: float,
        img_w: int,
        img_h: int,
        fps: Optional[float] = None,
    ) -> KinematicProfile:
        effective_fps = fps or self.default_fps

        x1, y1, x2, y2 = box
        p_h = max(1.0, y2 - y1)
        foot_point = ((x1 + x2) / 2.0, y2)

        if track_id not in self.track_histories:
            self.track_histories[track_id] = []
            self.reversal_counts[track_id] = 0
            self.zone_dwell_counts[track_id] = {z["name"]: 0 for z in self.zones_cfg}

        hist = self.track_histories[track_id]
        zone_dwells = self.zone_dwell_counts[track_id]

        # Check zone residency
        for z in self.zones_cfg:
            z_name = z["name"]
            # Convert normalized polygon to pixel coordinates
            poly_px = [(pt[0] * img_w, pt[1] * img_h) for pt in z["polygon_normalized"]]
            if point_in_polygon(foot_point, poly_px):
                zone_dwells[z_name] += 1

        # Motion kinematics
        speed_h_frame = 0.0
        accel_h_frame2 = 0.0
        heading_deg = None
        heading_change = 0.0

        if len(hist) >= 1:
            prev = hist[-1]
            dx = foot_point[0] - prev["pos"][0]
            dy = foot_point[1] - prev["pos"][1]
            dist_px = math.sqrt(dx * dx + dy * dy)
            speed_h_frame = dist_px / p_h

            if dist_px > 1.0:  # Threshold to avoid noise at standstill
                heading_deg = math.degrees(math.atan2(dy, dx))

            # Acceleration
            prev_speed = prev.get("speed_h_frame", 0.0)
            accel_h_frame2 = speed_h_frame - prev_speed

            # Heading change & direction reversal
            prev_heading = prev.get("heading_deg")
            if heading_deg is not None and prev_heading is not None:
                heading_change = compute_angle_difference(heading_deg, prev_heading)
                if heading_change >= 120.0 and prev_speed > 0.03:
                    self.reversal_counts[track_id] += 1

        # Total path length & net displacement
        total_path_px = 0.0
        for i in range(1, len(hist)):
            p0 = hist[i - 1]["pos"]
            p1 = hist[i]["pos"]
            total_path_px += math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)

        if hist:
            start_pos = hist[0]["pos"]
            dx_net = foot_point[0] - start_pos[0]
            dy_net = foot_point[1] - start_pos[1]
            net_disp_px = math.sqrt(dx_net * dx_net + dy_net * dy_net)
        else:
            net_disp_px = 0.0

        if total_path_px > 5.0:
            disp_ratio = min(1.0, max(0.0, net_disp_px / total_path_px))
        else:
            disp_ratio = 1.0

        speed_h_sec = speed_h_frame * effective_fps if effective_fps is not None else None
        dwell_sec = (
            {z: (cnt / effective_fps) for z, cnt in zone_dwells.items()}
            if effective_fps is not None
            else {z: None for z in zone_dwells}
        )

        record = {
            "frame_idx": frame_idx,
            "frame_number": frame_number,
            "pos": foot_point,
            "box": box,
            "height": p_h,
            "speed_h_frame": speed_h_frame,
            "heading_deg": heading_deg,
        }
        hist.append(record)

        return KinematicProfile(
            track_id=track_id,
            frame_idx=frame_idx,
            frame_number=frame_number,
            current_position=foot_point,
            current_box=box,
            person_height=p_h,
            speed_h_per_frame=float(speed_h_frame),
            speed_h_per_sec=float(speed_h_sec) if speed_h_sec is not None else None,
            acceleration_h_per_frame2=float(accel_h_frame2),
            heading_deg=float(heading_deg) if heading_deg is not None else None,
            direction_reversals_count=self.reversal_counts[track_id],
            last_heading_change_deg=float(heading_change),
            dwell_time_frames_per_zone=dict(zone_dwells),
            dwell_time_sec_per_zone=dwell_sec,
            total_path_length_px=float(total_path_px),
            net_displacement_px=float(net_disp_px),
            displacement_ratio=float(disp_ratio),
        )

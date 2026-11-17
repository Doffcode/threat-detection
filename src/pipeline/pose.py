"""
src/pipeline/pose.py
====================
Pose and hand/wrist keypoint estimation using YOLO11-pose.
Extracts wrist keypoints (COCO index 9: left wrist, 10: right wrist)
and maps them to active person tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .tracker import PersonTrack


def box_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


@dataclass
class WristKeypoint:
    wrist_name: str  # "left_wrist" or "right_wrist"
    x: float
    y: float
    conf: float


class PoseEstimator:
    def __init__(self, cfg: Dict[str, Any], device: str = "cuda:0"):
        self.cfg = cfg
        self.device = device
        self.enabled = cfg.get("enabled", True)
        if not self.enabled:
            self.model = None
            return

        weights_p = Path(cfg["weights"])
        if not weights_p.is_absolute():
            weights_p = Path("/home/shiva/Desktop/projects/akriti/weapon-detection") / weights_p
        
        if not weights_p.exists():
            raise FileNotFoundError(f"Pose model weights not found at: {weights_p}")

        self.model = YOLO(str(weights_p))
        self.imgsz = cfg.get("imgsz", 640)
        self.conf = cfg.get("conf_threshold", 0.25)
        self.kpt_conf_thr = cfg.get("keypoint_conf_threshold", 0.25)
        self.wrist_indices = cfg.get("wrist_indices", [9, 10])

    def estimate_wrists_for_tracks(
        self, img_bgr: np.ndarray, tracks: List[PersonTrack]
    ) -> Dict[int, List[WristKeypoint]]:
        """
        Runs pose estimation and assigns wrist keypoints to active tracks
        based on maximum bounding box IoU.
        """
        if not self.enabled or self.model is None or not tracks:
            return {t.track_id: [] for t in tracks}

        results = self.model.predict(
            img_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]

        track_wrists: Dict[int, List[WristKeypoint]] = {t.track_id: [] for t in tracks}
        if results.boxes is None or len(results.boxes) == 0 or results.keypoints is None:
            return track_wrists

        pose_boxes = [b.xyxy[0].tolist() for b in results.boxes]
        pose_kpts = results.keypoints.xy.cpu().numpy()  # [N, 17, 2]
        pose_confs = (
            results.keypoints.conf.cpu().numpy()
            if results.keypoints.conf is not None
            else np.ones((len(pose_boxes), 17))
        )

        for p_idx, p_box in enumerate(pose_boxes):
            # Find matching track with highest IoU >= 0.30
            best_iou = 0.30
            best_tid = None
            for t in tracks:
                iou = box_iou(p_box, t.box)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = t.track_id

            if best_tid is not None:
                wrists = []
                for widx in self.wrist_indices:
                    k_xy = pose_kpts[p_idx, widx]
                    k_conf = float(pose_confs[p_idx, widx])
                    if k_conf >= self.kpt_conf_thr:
                        name = "left_wrist" if widx == 9 else "right_wrist"
                        wrists.append(
                            WristKeypoint(
                                wrist_name=name,
                                x=float(k_xy[0]),
                                y=float(k_xy[1]),
                                conf=k_conf,
                            )
                        )
                track_wrists[best_tid].extend(wrists)

        return track_wrists

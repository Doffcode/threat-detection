"""
src/pipeline/detector.py
========================
Weapon detector wrapper using YOLO11m (baseline_m960_v2).
Supports pluggable Stage 2 verification hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class WeaponDetection:
    box: List[float]  # [x1, y1, x2, y2]
    conf: float
    cls_id: int
    cls_name: str
    det_conf: float
    verifier_p: Optional[float] = None


class WeaponDetector:
    def __init__(self, cfg: Dict[str, Any], device: str = "cuda:0"):
        self.cfg = cfg
        self.device = device
        weights_p = Path(cfg["weights"])
        if not weights_p.is_absolute():
            weights_p = Path("/home/shiva/Desktop/projects/akriti/weapon-detection") / weights_p
        
        if not weights_p.exists():
            raise FileNotFoundError(f"Detector weights not found at: {weights_p}")

        self.model = YOLO(str(weights_p))
        self.imgsz = cfg.get("imgsz", 960)
        self.conf = cfg.get("conf_threshold", 0.30)
        self.iou = cfg.get("iou_threshold", 0.50)
        self.classes = cfg.get("classes", [0, 1, 2])
        self.class_names = cfg.get("class_names", {0: "handgun", 1: "rifle", 2: "knife"})

    def detect(self, img_bgr: np.ndarray) -> List[WeaponDetection]:
        results = self.model.predict(
            img_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]

        detections: List[WeaponDetection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for b in results.boxes:
            box = [float(x) for x in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            cid = int(b.cls[0])
            cname = self.class_names.get(cid, str(cid))
            detections.append(
                WeaponDetection(
                    box=box,
                    conf=conf,
                    cls_id=cid,
                    cls_name=cname,
                    det_conf=conf,
                )
            )
        return detections

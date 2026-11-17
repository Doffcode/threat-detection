"""
src/pipeline/tracker.py
=======================
Person detection and tracking using YOLO11 person class with ByteTrack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class PersonTrack:
    track_id: int
    box: List[float]  # [x1, y1, x2, y2]
    conf: float


class PersonTracker:
    def __init__(self, cfg: Dict[str, Any], device: str = "cuda:0"):
        self.cfg = cfg
        self.device = device
        weights_p = Path(cfg["weights"])
        if not weights_p.is_absolute():
            weights_p = Path("/home/shiva/Desktop/projects/akriti/weapon-detection") / weights_p
        
        if not weights_p.exists():
            raise FileNotFoundError(f"Person detector weights not found at: {weights_p}")

        self.model = YOLO(str(weights_p))
        self.imgsz = cfg.get("imgsz", 640)
        self.conf = cfg.get("conf_threshold", 0.35)
        self.tracker_type = cfg.get("tracker_type", "bytetrack.yaml")
        self.classes = cfg.get("classes", [0])

    def reset(self):
        """Reset internal tracker state for a new video / continuous sequence."""
        if hasattr(self.model, "predictor") and hasattr(self.model.predictor, "trackers"):
            for trk in self.model.predictor.trackers:
                if hasattr(trk, "reset"):
                    trk.reset()

    def track(self, img_bgr: np.ndarray) -> List[PersonTrack]:
        results = self.model.track(
            img_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            tracker=self.tracker_type,
            persist=True,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]

        tracks: List[PersonTrack] = []
        if results.boxes is None or len(results.boxes) == 0:
            return tracks

        for b in results.boxes:
            box = [float(x) for x in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            tid = int(b.id[0]) if b.id is not None else -1
            tracks.append(
                PersonTrack(
                    track_id=tid,
                    box=box,
                    conf=conf,
                )
            )
        return tracks

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


def crop_region(box: List[float], W: int, H: int, pad: float = 0.2) -> Tuple[int, int, int, int]:
    """Pad the box by `pad` of its size on every side, grow the short side to square, clip to frame."""
    x0, y0, x1, y1 = map(float, box)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    side = max(w, h) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sw, sh = min(side, W), min(side, H)
    nx0 = min(max(cx - sw / 2.0, 0), W - sw)
    ny0 = min(max(cy - sh / 2.0, 0), H - sh)
    return int(round(nx0)), int(round(ny0)), int(round(nx0 + sw)), int(round(ny0 + sh))


def square_crop(img: np.ndarray, box: List[float], pad: float = 0.2, size: int = 224, fill: int = 114) -> np.ndarray:
    """BGR crop around box, padded and letterboxed to size x size."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = crop_region(box, W, H, pad)
    c = img[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    if c.size == 0:
        return np.full((size, size, 3), fill, np.uint8)
    s = size / max(c.shape[:2])
    nw, nh = max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))
    c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if nw == size and nh == size:
        return c
    out = np.full((size, size, 3), fill, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = c
    return out


class Stage2CropVerifier:
    """Stage 2 Crop Verifier (e.g. YOLO11s-cls) to filter false alarms on handheld objects."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, device: str = "cuda:0"):
        self.cfg = cfg or {}
        self.device = device
        self.enabled = self.cfg.get("enabled", False)
        self.weights_p = self.cfg.get("weights")
        self.combine_mode = self.cfg.get("combine_mode", "prod")  # "prod", "geo", "wgeo75", "cls", "det"
        self.verifier_conf_floor = self.cfg.get("verifier_conf_floor", 0.10)
        self.pad = self.cfg.get("pad", 0.20)
        self.imgsz = self.cfg.get("imgsz", 224)
        self.model = None

        if self.enabled and self.weights_p:
            p = Path(self.weights_p)
            if not p.is_absolute():
                p = Path("/home/shiva/Desktop/projects/akriti/weapon-detection") / p
            if p.exists():
                self.model = YOLO(str(p))
            else:
                self.enabled = False

    def verify(self, img_bgr: np.ndarray, detections: List[WeaponDetection]) -> List[WeaponDetection]:
        if not self.enabled or self.model is None or not detections:
            return detections

        verified: List[WeaponDetection] = []
        for det in detections:
            crop = square_crop(img_bgr, det.box, pad=self.pad, size=self.imgsz)
            res = self.model.predict(crop, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            probs = res.probs.data.cpu().numpy()
            # verifier_v5 classes: {0: 'gun', 1: 'knife', 2: 'not_weapon'}
            p_gun = float(probs[0])
            p_knife = float(probs[1])
            p_not_weapon = float(probs[2])

            target_p = p_gun if det.cls_name in ("handgun", "rifle") else p_knife
            det.verifier_p = target_p

            # False positive suppression rule:
            if p_not_weapon > 0.50 or target_p < self.verifier_conf_floor:
                continue

            # Score combination modes
            if self.combine_mode == "prod":
                det.conf = float(det.det_conf * target_p)
            elif self.combine_mode == "geo":
                det.conf = float(np.sqrt(det.det_conf * target_p))
            elif self.combine_mode == "wgeo75":
                det.conf = float((det.det_conf ** 0.25) * (target_p ** 0.75))
            elif self.combine_mode == "cls":
                det.conf = float(target_p)
            # if "det", keep det.conf as is (only filter)

            verified.append(det)

        return verified


class WeaponDetector:
    def __init__(self, cfg: Dict[str, Any], device: str = "cuda:0", verifier_cfg: Optional[Dict[str, Any]] = None):
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

        # Optional Stage-2 Crop Verifier
        v_cfg = verifier_cfg or cfg.get("stage2", {})
        self.verifier = Stage2CropVerifier(v_cfg, device=self.device)

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

        if self.verifier.enabled:
            detections = self.verifier.verify(img_bgr, detections)

        return detections


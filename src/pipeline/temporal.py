"""
src/pipeline/temporal.py
========================
Temporal confirmation and alert management.
Enforces that a weapon-person association must persist for at least N frames
(default N=3) on the same track ID before firing an alert.
"""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .associator import WeaponAssociation


@dataclass
class AlertRecord:
    alert_id: str
    track_id: int
    sequence_id: str
    weapon_class: str
    start_frame_idx: int
    end_frame_idx: int
    start_frame_num: float
    end_frame_num: float
    frame_range: List[float]
    max_confidence: float
    mean_confidence: float
    max_link_score: float
    mean_link_score: float
    consecutive_hits: int
    weapon_crop_path: Optional[str] = None
    person_crop_path: Optional[str] = None
    crop_image_path: Optional[str] = None  # Alias for weapon crop
    annotated_frame_path: Optional[str] = None


class TrackHistory:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.consecutive_hits: int = 0
        self.total_hits: int = 0
        self.last_seen_frame_idx: int = -1
        self.last_weapon_class: Optional[str] = None
        self.history_records: List[Dict[str, Any]] = []
        self.is_alerted: bool = False
        self.alert_id: Optional[str] = None
        self.alert_start_frame_idx: Optional[int] = None
        self.alert_start_frame_num: Optional[float] = None


class TemporalAlertManager:
    def __init__(self, cfg: Dict[str, Any], output_dir: Path):
        self.cfg = cfg
        self.confirmation_n = cfg.get("confirmation_frames", 3)
        self.max_gap = cfg.get("max_gap_frames", 2)
        self.cooldown_frames = cfg.get("cooldown_frames", 20)
        self.crop_padding = cfg.get("crop_padding", 0.20)
        self.output_dir = output_dir
        
        self.crops_dir = self.output_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        self.track_histories: Dict[int, TrackHistory] = {}
        self.alert_counter: int = 0
        self.active_alerts: List[AlertRecord] = []
        self.completed_alerts: List[AlertRecord] = []

    def reset_sequence(self):
        """Reset state between distinct camera/video sequences."""
        for hist in self.track_histories.values():
            if hist.is_alerted and hist.alert_id is not None:
                self._finalize_alert(hist)
        self.track_histories.clear()

    def update(
        self,
        frame_idx: int,
        frame_number: float,
        sequence_id: str,
        img_bgr: np.ndarray,
        associations: List[WeaponAssociation],
    ) -> List[AlertRecord]:
        """
        Updates temporal state with associations from current frame.
        Returns any newly triggered or active alerts for this frame.
        """
        current_frame_alerts: List[AlertRecord] = []
        associated_tids = set()

        for assoc in associations:
            if not assoc.is_associated or assoc.track_id is None:
                continue

            tid = assoc.track_id
            associated_tids.add(tid)

            if tid not in self.track_histories:
                self.track_histories[tid] = TrackHistory(track_id=tid)

            hist = self.track_histories[tid]
            gap = frame_idx - hist.last_seen_frame_idx if hist.last_seen_frame_idx >= 0 else 1

            if gap <= (self.max_gap + 1):
                hist.consecutive_hits += 1
            else:
                hist.consecutive_hits = 1

            hist.total_hits += 1
            hist.last_seen_frame_idx = frame_idx
            hist.last_weapon_class = assoc.weapon.cls_name

            # Save snapshot
            hist.history_records.append(
                {
                    "frame_idx": frame_idx,
                    "frame_number": frame_number,
                    "conf": assoc.weapon.conf,
                    "link_score": assoc.link_score,
                    "cls_name": assoc.weapon.cls_name,
                    "w_box": assoc.weapon.box,
                    "p_box": assoc.person_box,
                    "ioa": assoc.containment_ioa,
                    "center_in": assoc.center_inside,
                    "dist_px": assoc.wrist_dist_px,
                    "dist_rel": assoc.wrist_dist_rel_height,
                }
            )

            # Check if threshold reached
            if hist.consecutive_hits >= self.confirmation_n and not hist.is_alerted:
                # Trigger alert!
                self.alert_counter += 1
                hist.is_alerted = True
                hist.alert_id = f"ALERT-{self.alert_counter:04d}"
                hist.alert_start_frame_idx = hist.history_records[0]["frame_idx"]
                hist.alert_start_frame_num = hist.history_records[0]["frame_number"]

                alert_rec = self._create_alert_record(hist, sequence_id, img_bgr)
                current_frame_alerts.append(alert_rec)
                self.active_alerts.append(alert_rec)

            elif hist.is_alerted:
                # Already alerted; update active alert record
                for al in self.active_alerts:
                    if al.alert_id == hist.alert_id:
                        al.end_frame_idx = frame_idx
                        al.end_frame_num = frame_number
                        al.frame_range = [al.start_frame_num, frame_number]
                        confs = [r["conf"] for r in hist.history_records]
                        scores = [r["link_score"] for r in hist.history_records]
                        al.max_confidence = float(max(confs))
                        al.mean_confidence = float(sum(confs) / len(confs))
                        al.max_link_score = float(max(scores))
                        al.mean_link_score = float(sum(scores) / len(scores))
                        al.consecutive_hits = hist.consecutive_hits
                        current_frame_alerts.append(al)

        # Decay consecutive hits for unassociated tracks
        for tid, hist in list(self.track_histories.items()):
            if tid not in associated_tids:
                if frame_idx - hist.last_seen_frame_idx > self.max_gap:
                    if hist.is_alerted:
                        self._finalize_alert(hist)
                    hist.consecutive_hits = 0

        return current_frame_alerts

    def _create_alert_record(
        self, hist: TrackHistory, sequence_id: str, img_bgr: np.ndarray
    ) -> AlertRecord:
        confs = [r["conf"] for r in hist.history_records]
        scores = [r["link_score"] for r in hist.history_records]
        best_rec = max(hist.history_records, key=lambda x: x["conf"])

        # Generate and save crops
        h, w = img_bgr.shape[:2]
        w_crop_p = None
        p_crop_p = None

        if best_rec["w_box"] is not None:
            wb = best_rec["w_box"]
            pad_w = (wb[2] - wb[0]) * self.crop_padding
            pad_h = (wb[3] - wb[1]) * self.crop_padding
            x1 = max(0, int(wb[0] - pad_w))
            y1 = max(0, int(wb[1] - pad_h))
            x2 = min(w, int(wb[2] + pad_w))
            y2 = min(h, int(wb[3] + pad_h))
            if x2 > x1 and y2 > y1:
                w_crop = img_bgr[y1:y2, x1:x2]
                w_crop_fn = f"{hist.alert_id}_weapon_t{hist.track_id}_{best_rec['cls_name']}.jpg"
                w_crop_p = str(self.crops_dir / w_crop_fn)
                cv2.imwrite(w_crop_p, w_crop)

        if best_rec["p_box"] is not None:
            pb = best_rec["p_box"]
            x1 = max(0, int(pb[0]))
            y1 = max(0, int(pb[1]))
            x2 = min(w, int(pb[2]))
            y2 = min(h, int(pb[3]))
            if x2 > x1 and y2 > y1:
                p_crop = img_bgr[y1:y2, x1:x2]
                p_crop_fn = f"{hist.alert_id}_person_t{hist.track_id}.jpg"
                p_crop_p = str(self.crops_dir / p_crop_fn)
                cv2.imwrite(p_crop_p, p_crop)

        return AlertRecord(
            alert_id=hist.alert_id,
            track_id=hist.track_id,
            sequence_id=sequence_id,
            weapon_class=hist.last_weapon_class or "weapon",
            start_frame_idx=hist.alert_start_frame_idx or 0,
            end_frame_idx=best_rec["frame_idx"],
            start_frame_num=hist.alert_start_frame_num or 0.0,
            end_frame_num=best_rec["frame_number"],
            frame_range=[hist.alert_start_frame_num or 0.0, best_rec["frame_number"]],
            max_confidence=float(max(confs)),
            mean_confidence=float(sum(confs) / len(confs)),
            max_link_score=float(max(scores)),
            mean_link_score=float(sum(scores) / len(scores)),
            consecutive_hits=hist.consecutive_hits,
            weapon_crop_path=w_crop_p,
            person_crop_path=p_crop_p,
            crop_image_path=w_crop_p,
        )

    def _finalize_alert(self, hist: TrackHistory):
        for idx, al in enumerate(self.active_alerts):
            if al.alert_id == hist.alert_id:
                self.completed_alerts.append(al)
                self.active_alerts.pop(idx)
                break
        hist.is_alerted = False

    def finalize_all(self) -> List[Dict[str, Any]]:
        for hist in list(self.track_histories.values()):
            if hist.is_alerted:
                self._finalize_alert(hist)
        return [asdict(a) for a in self.completed_alerts]

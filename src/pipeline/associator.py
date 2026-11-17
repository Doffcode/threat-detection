"""
src/pipeline/associator.py
==========================
Associates detected weapons with tracked persons using:
1. Spatial containment (Intersection / Weapon_Area)
2. Wrist keypoint proximity relative to person height (H = y2 - y1)
3. Weapon-center inside person box
4. Configurable assignment (Hungarian or Max Score) with minimum score threshold
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .detector import WeaponDetection
from .pose import WristKeypoint
from .tracker import PersonTrack


def box_intersection_over_weapon(w_box: List[float], p_box: List[float]) -> float:
    """Intersection area divided by weapon box area (IoA)."""
    x1 = max(w_box[0], p_box[0])
    y1 = max(w_box[1], p_box[1])
    x2 = min(w_box[2], p_box[2])
    y2 = min(w_box[3], p_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    w_area = max(0.0, w_box[2] - w_box[0]) * max(0.0, w_box[3] - w_box[1])
    return inter / w_area if w_area > 0 else 0.0


def box_iou(b1: List[float], b2: List[float]) -> float:
    """Standard Intersection over Union (IoU)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def is_center_inside_box(center: Tuple[float, float], box: List[float]) -> bool:
    cx, cy = center
    return (box[0] <= cx <= box[2]) and (box[1] <= cy <= box[3])


@dataclass
class WeaponAssociation:
    weapon: WeaponDetection
    track_id: Optional[int]
    person_box: Optional[List[float]]
    containment_ioa: float
    center_inside: bool
    wrist_dist_px: Optional[float]
    wrist_dist_rel_height: Optional[float]
    link_score: float
    is_associated: bool
    candidate_count: int = 0


class WeaponPersonAssociator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.assignment_method = cfg.get("assignment_method", "max_score")
        self.min_link_score = cfg.get("min_link_score", 0.25)
        self.min_containment_ioa = cfg.get("min_containment_ioa", 0.20)
        self.max_wrist_dist_rel = cfg.get("max_wrist_distance_rel", 0.40)
        self.w_wrist = cfg.get("weight_wrist", 0.45)
        self.w_cont = cfg.get("weight_containment", 0.40)
        self.w_center = cfg.get("weight_center_in_box", 0.15)

    def score_pair(
        self,
        w: WeaponDetection,
        t: PersonTrack,
        wrists: List[WristKeypoint],
    ) -> Tuple[float, float, bool, Optional[float], Optional[float]]:
        """
        Computes link score between weapon w and person track t.
        Returns: (total_score, ioa, center_inside, min_wrist_dist_px, dist_rel_height)
        """
        w_cx = (w.box[0] + w.box[2]) / 2.0
        w_cy = (w.box[1] + w.box[3]) / 2.0

        p_h = max(1.0, t.box[3] - t.box[1])
        ioa = box_intersection_over_weapon(w.box, t.box)
        center_in = is_center_inside_box((w_cx, w_cy), t.box)
        center_score = 1.0 if center_in else 0.0

        min_wdist = None
        dist_rel_h = None
        if wrists:
            dists = [math.sqrt((w_cx - k.x) ** 2 + (w_cy - k.y) ** 2) for k in wrists]
            min_wdist = min(dists)
            dist_rel_h = min_wdist / p_h

        # Fallback if wrists are occluded / absent:
        if dist_rel_h is not None:
            wrist_term = max(0.0, 1.0 - (dist_rel_h / self.max_wrist_dist_rel))
            score = (
                self.w_wrist * wrist_term
                + self.w_cont * ioa
                + self.w_center * center_score
            )
        else:
            # Fallback to person-box containment
            score = (self.w_wrist + self.w_cont) * ioa + self.w_center * center_score

        return float(score), float(ioa), center_in, min_wdist, dist_rel_h

    def associate(
        self,
        weapons: List[WeaponDetection],
        tracks: List[PersonTrack],
        track_wrists: Dict[int, List[WristKeypoint]],
    ) -> List[WeaponAssociation]:
        if not weapons:
            return []

        if not tracks:
            return [
                WeaponAssociation(
                    weapon=w,
                    track_id=None,
                    person_box=None,
                    containment_ioa=0.0,
                    center_inside=False,
                    wrist_dist_px=None,
                    wrist_dist_rel_height=None,
                    link_score=0.0,
                    is_associated=False,
                    candidate_count=0,
                )
                for w in weapons
            ]

        # Compute pairwise score matrix [M weapons x N persons]
        M = len(weapons)
        N = len(tracks)
        score_matrix = np.zeros((M, N), dtype=np.float32)
        details_matrix = [[None for _ in range(N)] for _ in range(M)]

        for i, w in enumerate(weapons):
            for j, t in enumerate(tracks):
                wrists = track_wrists.get(t.track_id, [])
                score, ioa, c_in, dist_px, dist_rel = self.score_pair(w, t, wrists)
                score_matrix[i, j] = score
                details_matrix[i][j] = (ioa, c_in, dist_px, dist_rel)

        associations: List[WeaponAssociation] = []

        if self.assignment_method == "hungarian" and HAS_SCIPY and M <= N:
            # 1-to-1 matching via Hungarian algorithm
            cost_matrix = 1.0 - score_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_pairs = {r: c for r, c in zip(row_ind, col_ind)}
        else:
            # Max-score assignment: for every weapon, assign best person >= min_link_score
            # Naturally supports multiple weapons on one person (e.g. holster + drawn)
            matched_pairs = {}
            for i in range(M):
                best_j = int(np.argmax(score_matrix[i]))
                if score_matrix[i, best_j] >= self.min_link_score:
                    matched_pairs[i] = best_j

        for i, w in enumerate(weapons):
            candidates_above_zero = int(np.sum(score_matrix[i] >= self.min_link_score))
            if i in matched_pairs and score_matrix[i, matched_pairs[i]] >= self.min_link_score:
                j = matched_pairs[i]
                t = tracks[j]
                ioa, c_in, dist_px, dist_rel = details_matrix[i][j]
                associations.append(
                    WeaponAssociation(
                        weapon=w,
                        track_id=t.track_id,
                        person_box=t.box,
                        containment_ioa=ioa,
                        center_inside=c_in,
                        wrist_dist_px=dist_px,
                        wrist_dist_rel_height=dist_rel,
                        link_score=float(score_matrix[i, j]),
                        is_associated=True,
                        candidate_count=candidates_above_zero,
                    )
                )
            else:
                associations.append(
                    WeaponAssociation(
                        weapon=w,
                        track_id=None,
                        person_box=None,
                        containment_ioa=0.0,
                        center_inside=False,
                        wrist_dist_px=None,
                        wrist_dist_rel_height=None,
                        link_score=0.0,
                        is_associated=False,
                        candidate_count=candidates_above_zero,
                    )
                )

        return associations

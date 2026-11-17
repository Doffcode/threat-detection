"""
src/behavior/visualizer.py
==========================
Renders 2D trajectory plots and behavioral alert overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .kinematics import KinematicProfile
from .rules import BehaviorAlert


class BehaviorVisualizer:
    def __init__(self, zones_cfg: List[Dict[str, Any]], output_dir: Path):
        self.zones_cfg = zones_cfg
        self.output_dir = output_dir
        self.traj_dir = self.output_dir / "trajectories"
        self.traj_dir.mkdir(parents=True, exist_ok=True)

    def render_frame_annotations(
        self,
        img_bgr: np.ndarray,
        profiles: List[KinematicProfile],
        alerts: List[BehaviorAlert],
        track_histories: Dict[int, List[Dict[str, Any]]],
        frame_number: float,
        sequence_id: str,
    ) -> np.ndarray:
        vis = img_bgr.copy()
        h, w = vis.shape[:2]

        # 1. Draw Zones (Semi-transparent overlay)
        overlay = vis.copy()
        for z in self.zones_cfg:
            poly_px = np.array(
                [[int(pt[0] * w), int(pt[1] * h)] for pt in z["polygon_normalized"]],
                dtype=np.int32,
            )
            cv2.polylines(vis, [poly_px], isClosed=True, color=(255, 200, 0), thickness=2)
            cv2.fillPoly(overlay, [poly_px], color=(255, 200, 0))
            cv2.putText(
                vis,
                z["name"],
                (poly_px[0][0] + 5, poly_px[0][1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

        # 2. Draw Motion Trails & Person Boxes
        alert_tids = {a.track_id: a for a in alerts}
        for p in profiles:
            tid = p.track_id
            box = list(map(int, p.current_box))
            is_alerted = tid in alert_tids

            # Motion trail
            hist = track_histories.get(tid, [])
            if len(hist) >= 2:
                pts = [np.array([int(h_pt["pos"][0]), int(h_pt["pos"][1])]) for h_pt in hist]
                for k in range(1, len(pts)):
                    cv2.line(vis, tuple(pts[k - 1]), tuple(pts[k]), (0, 255, 255), 2, cv2.LINE_AA)

            # Person Box
            box_color = (0, 100, 255) if is_alerted else (0, 220, 0)
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), box_color, 2)
            label = f"Person #{tid} | v={p.speed_h_per_frame:.2f}h/f"
            cv2.putText(
                vis,
                label,
                (box[0], max(18, box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA,
            )

        # 3. Behavioral Alerts Banner
        if alerts:
            banner_h = 32 * len(alerts)
            banner_overlay = vis.copy()
            cv2.rectangle(banner_overlay, (0, 0), (w, banner_h + 10), (20, 20, 180), -1)
            cv2.addWeighted(banner_overlay, 0.85, vis, 0.15, 0, vis)

            for idx, al in enumerate(alerts):
                txt = f"[HUMAN REVIEW] Person #{al.track_id} - {al.rule_name.upper()}: {al.rule_description}"
                cv2.putText(
                    vis,
                    txt,
                    (10, 22 + idx * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return vis

    def plot_sequence_trajectories(
        self,
        sequence_id: str,
        track_histories: Dict[int, List[Dict[str, Any]]],
        img_w: int,
        img_h: int,
        background_img_bgr: Optional[np.ndarray] = None,
    ) -> str:
        """Exports 2D trajectory plot for all tracks in a sequence."""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

        if background_img_bgr is not None:
            bg_rgb = cv2.cvtColor(background_img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(bg_rgb)
        else:
            ax.set_facecolor("#1e1e1e")
            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)  # Inverted Y for image coordinates

        # Draw Zones
        for z in self.zones_cfg:
            poly_px = [(pt[0] * img_w, pt[1] * img_h) for pt in z["polygon_normalized"]]
            poly_px.append(poly_px[0])
            xs, ys = zip(*poly_px)
            ax.plot(xs, ys, color="cyan", linestyle="--", linewidth=2, label=f"Zone: {z['name']}")
            ax.fill(xs, ys, color="cyan", alpha=0.15)

        # Draw Tracks
        cmap = plt.get_cmap("tab10")
        for idx, (tid, hist) in enumerate(track_histories.items()):
            if len(hist) < 2:
                continue
            xs = [h_pt["pos"][0] for h_pt in hist]
            ys = [h_pt["pos"][1] for h_pt in hist]
            color = cmap(idx % 10)
            ax.plot(xs, ys, "-o", markersize=4, linewidth=2, color=color, label=f"Track #{tid} ({len(hist)} frames)")
            ax.scatter(xs[0], ys[0], color="green", marker="^", s=60, zorder=5)  # Start
            ax.scatter(xs[-1], ys[-1], color="red", marker="s", s=60, zorder=5)   # End

        ax.set_title(f"Person Kinematic Trajectories — {sequence_id}", fontsize=14, pad=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.grid(True, linestyle=":", alpha=0.4)

        save_p = self.traj_dir / f"{sequence_id}_trajectories.png"
        fig.tight_layout()
        fig.savefig(str(save_p))
        plt.close(fig)
        return str(save_p)

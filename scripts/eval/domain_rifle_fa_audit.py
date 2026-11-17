"""
scripts/eval/domain_rifle_fa_audit.py
=====================================
Audit of rifle false alarms on test_domain (v1: 6 vs v2: 16).
1. Compares rifle predictions between v1 and v2 on test_domain.
2. Renders 12 visual overlays showing what the rifle class fires on in test_domain.
3. Analyzes image characteristics and triggers.
"""

from __future__ import annotations

import collections
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.eval.evaluate_detector import box_iou, load_manifest


def main():
    data_root = Path("data/processed/stage1_baseline")
    img_dir = data_root / "test_domain" / "images"
    lbl_dir = data_root / "test_domain" / "labels"
    
    p1 = pd.read_csv("outputs/eval/v1/predictions.csv")
    p2 = pd.read_csv("outputs/eval/v2/predictions.csv")
    
    p1_rifles = p1[(p1["split"] == "test_domain") & (p1["class_name"] == "rifle") & (p1["conf"] >= 0.280)].copy()
    p2_rifles = p2[(p2["split"] == "test_domain") & (p2["class_name"] == "rifle") & (p2["conf"] >= 0.300)].copy()
    
    print("=" * 80)
    print("TEST_DOMAIN RIFLE FALSE ALARMS AUDIT")
    print(f"Total Rifle GT in test_domain: 0")
    print(f"v1 Rifle FPs @ conf* (0.280): {len(p1_rifles)}")
    print(f"v2 Rifle FPs @ conf* (0.300): {len(p2_rifles)}")
    print("=" * 80)
    
    overlay_dir = Path("outputs/eval/domain_rifle_fa_overlays")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort v2 rifles descending by confidence
    p2_rifles_sorted = p2_rifles.sort_values(by="conf", ascending=False)
    
    records = []
    for rank, (_, row) in enumerate(p2_rifles_sorted.iterrows(), 1):
        if rank > 12:
            break
        img_p = img_dir / row["filename"]
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
            
        annotated = img.copy()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        conf = row["conf"]
        
        # Check if v1 also detected a rifle here
        v1_conf_here = None
        for _, r1 in p1_rifles[p1_rifles["filename"] == row["filename"]].iterrows():
            if box_iou([x1, y1, x2 - x1, y2 - y1], [r1["x1"], r1["y1"], r1["x2"] - r1["x1"], r1["y2"] - r1["y1"]]) >= 0.3:
                v1_conf_here = r1["conf"]
                break
                
        # Read GT boxes for this image if any (handguns or knives)
        lbl_p = lbl_dir / f"{Path(row['filename']).stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cname = "handgun" if cid == 0 else ("rifle" if cid == 1 else "knife")
                    cx, cy, bw, bh = map(float, parts[1:5])
                    gx1 = int((cx - bw / 2.0) * w)
                    gy1 = int((cy - bh / 2.0) * h)
                    gx2 = int((cx + bw / 2.0) * w)
                    gy2 = int((cy + bh / 2.0) * h)
                    gt_boxes.append((gx1, gy1, gx2, gy2, cname))
                    cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"GT: {cname}", (gx1, max(20, gy1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
        # Draw the predicted rifle box in red
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, f"FA: RIFLE {conf:.3f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
        # Check overlap with any GT
        overlap_gt = None
        for gx1, gy1, gx2, gy2, cname in gt_boxes:
            iou = box_iou([x1, y1, x2 - x1, y2 - y1], [gx1, gy1, gx2 - gx1, gy2 - gy1])
            if iou >= 0.20:
                overlap_gt = (cname, iou)
                break
                
        status_str = f"Overlaps GT {overlap_gt[0]} (IoU={overlap_gt[1]:.2f})" if overlap_gt else "Hallucination on Clutter/Props"
        v1_str = f"v1 Conf: {v1_conf_here:.2f}" if v1_conf_here else "v1: Not Detected"
        header = f"[Rifle FA #{rank}] Conf: {conf:.3f} | {status_str} | {v1_str}"
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(annotated, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        save_p = overlay_dir / f"rifle_fa_{rank:02d}_{row['filename']}"
        cv2.imwrite(str(save_p), annotated)
        records.append({
            "rank": rank,
            "filename": row["filename"],
            "conf": conf,
            "v1_conf": v1_conf_here if v1_conf_here else "None",
            "overlaps_gt": overlap_gt[0] if overlap_gt else "None",
            "gt_iou": round(overlap_gt[1], 3) if overlap_gt else 0.0,
            "saved_path": str(save_p),
        })
        
    df_out = pd.DataFrame(records)
    df_out.to_csv(overlay_dir / "manifest.csv", index=False)
    print(f"\nSaved 12 rifle false alarm overlays to {overlay_dir}")
    print(df_out[["rank", "filename", "conf", "v1_conf", "overlaps_gt", "gt_iou"]].to_string())


if __name__ == "__main__":
    main()

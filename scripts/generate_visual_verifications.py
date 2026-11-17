"""
scripts/generate_visual_verifications.py
=========================================
Generates visual verification artifacts for Checklist Items 11 & 12:
- Item 11: 16 augmented training samples with bounding boxes overlaid (verifies box alignment under Albumentations).
- Item 12: 24 ground-truth training samples verifying label mapping (especially aruval -> knife, rifles, handguns).
"""

import random
from pathlib import Path
import cv2
import numpy as np

from src.detection.albumentations_cctv import apply_cctv_transform

CLASS_NAMES = {0: "handgun", 1: "rifle", 2: "knife"}
CLASS_COLORS = {
    0: (0, 0, 255),    # Red: Handgun
    1: (255, 0, 0),    # Blue: Rifle
    2: (0, 255, 0),    # Green: Knife
}

def draw_boxes(img: np.ndarray, boxes: list, class_labels: list) -> np.ndarray:
    """Draws bounding boxes and labels on an image (boxes in YOLO format cx, cy, w, h)."""
    h, w = img.shape[:2]
    canvas = img.copy()
    for (cx, cy, bw, bh), cid in zip(boxes, class_labels):
        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)
        
        color = CLASS_COLORS.get(cid, (0, 255, 255))
        cname = CLASS_NAMES.get(cid, str(cid))
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_text = f"{cname}"
        cv2.putText(canvas, label_text, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(canvas, label_text, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas

def main():
    train_dir = Path("data/processed/stage1_baseline/train")
    img_dir = train_dir / "images"
    lbl_dir = train_dir / "labels"
    
    aug_out = Path("outputs/debug/aug_samples")
    label_out = Path("outputs/debug/label_check")
    aug_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # Item 11: 16 Augmented Training Samples
    # -------------------------------------------------------------
    print("Generating 16 augmented training samples for Item 11...")
    all_imgs = sorted(list(img_dir.glob("*.*")))
    random.seed(2026)
    
    # Filter images with at least one box
    valid_pairs = []
    for ip in all_imgs:
        lp = lbl_dir / f"{ip.stem}.txt"
        if lp.exists() and lp.read_text().strip():
            valid_pairs.append((ip, lp))
            
    sampled_16 = random.sample(valid_pairs, 16)
    
    for idx, (ip, lp) in enumerate(sampled_16, 1):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        lines = lp.read_text().strip().splitlines()
        bboxes = []
        labels = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                bboxes.append([cx, cy, bw, bh])
                labels.append(cid)
                
        aug_img, aug_bboxes, aug_labels = apply_cctv_transform(img, bboxes, labels)
        rendered = draw_boxes(aug_img, aug_bboxes, aug_labels)
        out_p = aug_out / f"aug_{idx:02d}_{ip.stem}.jpg"
        cv2.imwrite(str(out_p), rendered)
        
    print(f"[PASS] 16 augmented samples saved to {aug_out}/")
    
    # -------------------------------------------------------------
    # Item 12: 24 Label-Swap Regression Check Samples
    # -------------------------------------------------------------
    print("\nGenerating 24 label check samples for Item 12...")
    # Gather specifically:
    # 6 Indian cinema (including aruval), 6 rifles, 6 handguns, 6 knives
    aruval_pairs = [p for p in valid_pairs if "aruval" in p[0].name.lower() or "cinema" in p[0].name.lower()]
    rifle_pairs = []
    handgun_pairs = []
    knife_pairs = []
    
    for ip, lp in valid_pairs:
        cids = [int(line.split()[0]) for line in lp.read_text().splitlines() if line.strip()]
        if 1 in cids and len(rifle_pairs) < 10:
            rifle_pairs.append((ip, lp))
        elif 0 in cids and len(handgun_pairs) < 10:
            handgun_pairs.append((ip, lp))
        elif 2 in cids and len(knife_pairs) < 10:
            knife_pairs.append((ip, lp))
            
    selected_24 = []
    selected_24.extend(aruval_pairs[:6])
    selected_24.extend(rifle_pairs[:6])
    selected_24.extend(handgun_pairs[:6])
    selected_24.extend(knife_pairs[:6])
    
    # Fill remaining if needed
    if len(selected_24) < 24:
        remaining = [p for p in valid_pairs if p not in selected_24]
        selected_24.extend(remaining[: 24 - len(selected_24)])
        
    for idx, (ip, lp) in enumerate(selected_24[:24], 1):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        lines = lp.read_text().strip().splitlines()
        bboxes = []
        labels = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                bboxes.append([cx, cy, bw, bh])
                labels.append(cid)
                
        rendered = draw_boxes(img, bboxes, labels)
        out_p = label_out / f"label_check_{idx:02d}_{ip.stem}.jpg"
        cv2.imwrite(str(out_p), rendered)
        
    print(f"[PASS] 24 label verification samples saved to {label_out}/")


if __name__ == "__main__":
    main()

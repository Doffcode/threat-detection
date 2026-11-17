"""
scripts/generate_cctv_bits_fp_overlays.py
=========================================
Generates visual overlays for BITS Pilani false alarms at conf* = 0.280.
Draws:
- Green dashed box: Ground Truth (full-scene knife annotation).
- Red box: Model prediction (handgun/knife/rifle false alarm).
- Inset zoom: Magnified crop of the predicted region to visually audit what triggered the detector.
Saves 20 representative overlays to outputs/eval/cctv_bits_fp_overlays/.
"""

from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd

def main():
    workspace = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
    out_dir = workspace / "outputs/eval/cctv_bits_fp_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions
    preds_df = pd.read_csv(workspace / "outputs/eval/predictions.csv")
    bits_preds = preds_df[(preds_df["source"] == "bits_pilani") & (preds_df["conf"] >= 0.28)].copy()

    # 2. Load ground truth
    manifest = pd.read_csv(workspace / "data/processed/stage1_baseline/manifest.csv")
    v1_bits = manifest[(manifest["source"] == "bits_pilani") & (manifest["split"] == "test_cctv_v1")]
    
    # Map image_id or filename to GT label path
    test_cctv_dir = workspace / "data/processed/stage1_baseline/test_cctv_v1"

    print(f"Total BITS predictions at conf >= 0.28: {len(bits_preds)}")
    print(bits_preds["class_name"].value_counts().to_dict())

    # Pick 20 representative predictions:
    # 16 handguns (diverse confidences from 0.30 to 0.75), 2 rifles, 2 knives
    handguns = bits_preds[bits_preds["class_name"] == "handgun"].sort_values("conf", ascending=False)
    rifles = bits_preds[bits_preds["class_name"] == "rifle"].sort_values("conf", ascending=False)
    knives = bits_preds[bits_preds["class_name"] == "knife"].sort_values("conf", ascending=False)

    # Sample uniformly from handguns
    stride_hg = len(handguns) // 16
    sampled_hg = handguns.iloc[::stride_hg].head(16)
    sampled = pd.concat([sampled_hg, rifles.head(2), knives.head(2)]).reset_index(drop=True)

    summary_records = []

    for idx, row in sampled.iterrows():
        fname = row["filename"]
        img_path = test_cctv_dir / "images" / fname
        lbl_path = test_cctv_dir / "labels" / fname.replace(".jpg", ".txt").replace(".png", ".txt")

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Draw GT boxes in green
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    gx1 = int((xc - bw / 2) * w)
                    gy1 = int((yc - bh / 2) * h)
                    gx2 = int((xc + bw / 2) * w)
                    gy2 = int((yc + bh / 2) * h)
                    gt_boxes.append((cls_id, gx1, gy1, gx2, gy2))
                    # Draw GT with green dashed/solid line
                    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)
                    cv2.putText(img, "GT knife (full scene)", (gx1 + 5, gy1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)

        # Draw Pred box in red
        px1, py1, px2, py2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w - 1, px2), min(h - 1, py2)

        pred_label = f"PRED: {row['class_name']} {row['conf']:.2f}"
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.rectangle(img, (px1, py1 - 18), (px1 + len(pred_label) * 8, py1), (0, 0, 255), -1)
        cv2.putText(img, pred_label, (px1 + 2, py1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Inset zoom of predicted region in top-right corner
        crop = img[py1:py2, px1:px2]
        if crop.size > 0:
            inset_size = 120
            crop_resized = cv2.resize(crop, (inset_size, inset_size))
            # Border
            cv2.rectangle(crop_resized, (0, 0), (inset_size - 1, inset_size - 1), (0, 0, 255), 2)
            cv2.putText(crop_resized, "ZOOM", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # Paste into top-right
            img[10:10 + inset_size, w - inset_size - 10:w - 10] = crop_resized

        out_name = f"overlay_{idx:02d}_{row['class_name']}_{row['conf']:.2f}_{fname[:30]}.jpg"
        out_file = out_dir / out_name
        cv2.imwrite(str(out_file), img)

        summary_records.append({
            "idx": idx + 1,
            "filename": fname,
            "predicted_class": row["class_name"],
            "conf": round(row["conf"], 4),
            "pred_box_xyxy": f"[{px1}, {py1}, {px2}, {py2}]",
            "overlay_path": str(out_file)
        })

    df_out = pd.DataFrame(summary_records)
    summary_csv = out_dir / "bits_fp_overlays_summary.csv"
    df_out.to_csv(summary_csv, index=False)
    print(f"[SUCCESS] Saved {len(df_out)} overlays to {out_dir}")
    print(f"[SUCCESS] Wrote summary CSV to {summary_csv}")

if __name__ == "__main__":
    main()

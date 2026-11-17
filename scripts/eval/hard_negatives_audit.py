"""
scripts/eval/hard_negatives_audit.py
===================================
Detailed audit of test_hard_negatives:
1. FPPI with 95% Bootstrap CIs for v1 and v2.
2. FP counts by class and by source.
3. Identify and render 20 overlays of FPs that are NEW in v2 (not in v1).
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.eval.evaluate_detector import box_iou, load_manifest


def bootstrap_fppi(df_preds: pd.DataFrame, manifest_df: pd.DataFrame, conf_thresh: float,
                   n_bootstraps: int = 1000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    
    # All hard negative images from manifest
    hard_neg_imgs = manifest_df[manifest_df["split"] == "test_hard_negatives"].copy()
    groups = hard_neg_imgs.groupby("group_id")["filename"].apply(list).to_dict()
    unique_gids = list(groups.keys())
    n_groups = len(unique_gids)
    
    # Filter preds above conf
    preds_above = df_preds[(df_preds["split"] == "test_hard_negatives") & (df_preds["conf"] >= conf_thresh)]
    fp_counts_by_img = preds_above.groupby("filename").size().to_dict()
    
    fppis = []
    for _ in range(n_bootstraps):
        sampled_gids = rng.choice(unique_gids, size=n_groups, replace=True)
        tot_imgs = 0
        tot_fps = 0
        for gid in sampled_gids:
            imgs = groups[gid]
            tot_imgs += len(imgs)
            for img in imgs:
                tot_fps += fp_counts_by_img.get(img, 0)
        fppis.append(tot_fps / tot_imgs if tot_imgs > 0 else 0.0)
        
    mean_val = float(np.mean(fppis))
    ci_low = float(np.percentile(fppis, 2.5))
    ci_high = float(np.percentile(fppis, 97.5))
    return {
        "mean": round(mean_val, 4),
        "ci_95": (round(ci_low, 4), round(ci_high, 4)),
    }


def main():
    data_root = Path("data/processed/stage1_baseline")
    manifest_df = pd.read_csv(data_root / "manifest.csv")
    manifest_map = manifest_df.set_index("filename").to_dict(orient="index")
    
    p1 = pd.read_csv("outputs/eval/v1/predictions.csv")
    p2 = pd.read_csv("outputs/eval/v2/predictions.csv")
    
    p1_hn = p1[(p1["split"] == "test_hard_negatives") & (p1["conf"] >= 0.280)].copy()
    p2_hn = p2[(p2["split"] == "test_hard_negatives") & (p2["conf"] >= 0.300)].copy()
    
    # Add source column from manifest
    p1_hn["source"] = p1_hn["filename"].map(lambda f: manifest_map.get(f, {}).get("source", "unknown"))
    p2_hn["source"] = p2_hn["filename"].map(lambda f: manifest_map.get(f, {}).get("source", "unknown"))
    
    print("=" * 80)
    print("TEST_HARD_NEGATIVES AUDIT")
    print("=" * 80)
    
    # 1. FPPI with Bootstrap CIs
    ci1 = bootstrap_fppi(p1, manifest_df, conf_thresh=0.280)
    ci2 = bootstrap_fppi(p2, manifest_df, conf_thresh=0.300)
    print(f"\n1. FPPI @ conf*:")
    print(f"   baseline_m960_v1 (conf*=0.280): {len(p1_hn)} / 303 = {len(p1_hn)/303:.4f} | 95% CI: [{ci1['ci_95'][0]:.4f}, {ci1['ci_95'][1]:.4f}]")
    print(f"   baseline_m960_v2 (conf*=0.300): {len(p2_hn)} / 303 = {len(p2_hn)/303:.4f} | 95% CI: [{ci2['ci_95'][0]:.4f}, {ci2['ci_95'][1]:.4f}]")
    
    # At fixed 0.25
    p1_hn_025 = p1[(p1["split"] == "test_hard_negatives") & (p1["conf"] >= 0.250)]
    p2_hn_025 = p2[(p2["split"] == "test_hard_negatives") & (p2["conf"] >= 0.250)]
    ci1_025 = bootstrap_fppi(p1, manifest_df, conf_thresh=0.250)
    ci2_025 = bootstrap_fppi(p2, manifest_df, conf_thresh=0.250)
    print(f"   baseline_m960_v1 (conf=0.250):  {len(p1_hn_025)} / 303 = {len(p1_hn_025)/303:.4f} | 95% CI: [{ci1_025['ci_95'][0]:.4f}, {ci1_025['ci_95'][1]:.4f}]")
    print(f"   baseline_m960_v2 (conf=0.250):  {len(p2_hn_025)} / 303 = {len(p2_hn_025)/303:.4f} | 95% CI: [{ci2_025['ci_95'][0]:.4f}, {ci2_025['ci_95'][1]:.4f}]")
    
    # 2. FP Counts by class and by source
    print(f"\n2. FP Breakdown by Class:")
    cls_df = pd.DataFrame({
        "v1 @ conf* (0.28)": p1_hn["class_name"].value_counts(),
        "v2 @ conf* (0.30)": p2_hn["class_name"].value_counts(),
        "v1 @ 0.25": p1_hn_025["class_name"].value_counts(),
        "v2 @ 0.25": p2_hn_025["class_name"].value_counts(),
    }).fillna(0).astype(int)
    print(cls_df.to_string())
    
    print(f"\n3. FP Breakdown by Source:")
    src_df = pd.DataFrame({
        "v1 @ conf* (0.28)": p1_hn["source"].value_counts(),
        "v2 @ conf* (0.30)": p2_hn["source"].value_counts(),
    }).fillna(0).astype(int)
    print(src_df.to_string())
    
    # Also cross-tabulation source x class for v2
    print(f"\n4. Cross-tabulation: Source x Class (v2 @ conf* 0.300):")
    ct = pd.crosstab(p2_hn["source"], p2_hn["class_name"], margins=True)
    print(ct.to_string())
    
    # 5. Identify FPs new in v2 (not in v1)
    # For each prediction in v2, check if v1 had any prediction on that image of the same or any class with IoU >= 0.3
    p1_by_img = collections.defaultdict(list)
    for _, r in p1_hn.iterrows():
        p1_by_img[r["filename"]].append([r["x1"], r["y1"], r["x2"] - r["x1"], r["y2"] - r["y1"], r["class_name"], r["conf"]])
        
    new_v2_fps = []
    for _, r in p2_hn.iterrows():
        fname = r["filename"]
        b2 = [r["x1"], r["y1"], r["x2"] - r["x1"], r["y2"] - r["y1"]]
        v1_preds = p1_by_img.get(fname, [])
        is_matched = False
        for b1 in v1_preds:
            if box_iou(b2, b1[:4]) >= 0.30:
                is_matched = True
                break
        if not is_matched:
            new_v2_fps.append(r)
            
    df_new = pd.DataFrame(new_v2_fps)
    print(f"\n5. New False Positives in v2 (not matched to any v1 FP with IoU >= 0.3): {len(df_new)}")
    print(df_new[["filename", "source", "class_name", "conf"]].head(10).to_string())
    
    # Render 20 overlays
    overlay_dir = Path("outputs/eval/hard_neg_new_v2_overlays")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    img_dir = data_root / "test_hard_negatives" / "images"
    saved_records = []
    
    df_new_sorted = df_new.sort_values(by="conf", ascending=False)
    for rank, (_, row) in enumerate(df_new_sorted.iterrows(), 1):
        if rank > 20:
            break
        img_p = img_dir / row["filename"]
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
            
        annotated = img.copy()
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cname = row["class_name"]
        conf = row["conf"]
        src = row["source"]
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, f"NEW FP v2: {cname.upper()} {conf:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    
        header = f"[New FP #{rank}] {cname.upper()} {conf:.3f} | Source: {src} | File: {row['filename']}"
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(annotated, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        save_p = overlay_dir / f"new_fp_{rank:02d}_{cname}_{row['filename']}"
        cv2.imwrite(str(save_p), annotated)
        saved_records.append({
            "rank": rank,
            "filename": row["filename"],
            "source": src,
            "class_name": cname,
            "conf": conf,
            "saved_path": str(save_p),
        })
        
    pd.DataFrame(saved_records).to_csv(overlay_dir / "manifest.csv", index=False)
    print(f"Saved 20 new FP overlays to {overlay_dir}")


if __name__ == "__main__":
    main()

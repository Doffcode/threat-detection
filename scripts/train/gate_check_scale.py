"""
scripts/train/gate_check_scale.py
=================================
Gate check for v3 scale parameter:
Samples 2,000 augmented training boxes for scale=0.40, scale=0.80, and scale=0.90,
and compares histograms against Innovatus CCTV ground truth.
Selects the better scale for v3 and verifies the gate condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset


def sample_boxes_for_scale(data_yaml_path: str, scale: float, target_boxes: int = 2000, seed: int = 2026) -> np.ndarray:
    data_info = check_det_dataset(data_yaml_path)
    train_path = data_info["train"]
    
    hyp = get_cfg()
    hyp.mosaic = 1.0
    hyp.scale = scale
    hyp.degrees = 0.0
    hyp.translate = 0.1
    hyp.shear = 0.0
    hyp.perspective = 0.0
    hyp.hsv_h = 0.015
    hyp.hsv_s = 0.6
    hyp.hsv_v = 0.4
    hyp.fliplr = 0.5
    hyp.flipud = 0.0
    hyp.mixup = 0.0
    hyp.copy_paste = 0.0
    
    dataset = YOLODataset(
        img_path=train_path,
        data=data_info,
        imgsz=960,
        augment=True,
        hyp=hyp,
        rect=False,
        stride=32,
    )
    
    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    indices = rng.permutation(n_total)
    
    collected_areas_ratio = []
    for idx in indices:
        item = dataset[int(idx)]
        bboxes = item.get("bboxes")
        if bboxes is not None and len(bboxes) > 0:
            if hasattr(bboxes, "cpu"):
                bboxes = bboxes.cpu().numpy()
            for b in bboxes:
                bw = float(b[2])
                bh = float(b[3])
                area_ratio = bw * bh
                collected_areas_ratio.append(area_ratio)
                if len(collected_areas_ratio) >= target_boxes:
                    break
        if len(collected_areas_ratio) >= target_boxes:
            break
            
    return np.array(collected_areas_ratio[:target_boxes])


def get_innovatus_boxes() -> np.ndarray:
    data_root = Path("data/processed/stage1_baseline")
    manifest = pd.read_csv(data_root / "manifest.csv")
    inno_files = set(manifest[manifest["source"] == "innovatus_rob"]["filename"].tolist())
    
    lbl_dir = data_root / "test_cctv_v1" / "labels"
    areas = []
    for f in inno_files:
        lp = lbl_dir / f"{Path(f).stem}.txt"
        if lp.exists():
            for line in lp.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    bw, bh = float(parts[3]), float(parts[4])
                    areas.append(bw * bh)
    return np.array(areas)


def compute_distribution_stats(areas_ratio: np.ndarray) -> dict:
    pcts = np.percentile(areas_ratio * 100, [10, 25, 50, 75, 90])
    b_sub1 = float(np.mean(areas_ratio < 0.01)) * 100
    b_1_5 = float(np.mean((areas_ratio >= 0.01) & (areas_ratio < 0.05))) * 100
    b_5_25 = float(np.mean((areas_ratio >= 0.05) & (areas_ratio < 0.25))) * 100
    b_gt25 = float(np.mean(areas_ratio >= 0.25)) * 100
    return {
        "count": len(areas_ratio),
        "mean_pct": round(float(np.mean(areas_ratio * 100)), 3),
        "median_pct": round(float(pcts[2]), 3),
        "p10_pct": round(float(pcts[0]), 3),
        "p25_pct": round(float(pcts[1]), 3),
        "p75_pct": round(float(pcts[3]), 3),
        "p90_pct": round(float(pcts[4]), 3),
        "bucket_sub1_pct": round(b_sub1, 2),
        "bucket_1_5_pct": round(b_1_5, 2),
        "combined_sub5_pct": round(b_sub1 + b_1_5, 2),
        "bucket_5_25_pct": round(b_5_25, 2),
        "bucket_gt25_pct": round(b_gt25, 2),
    }


def main():
    data_yaml = "configs/detection/dataset_stage1.yaml"
    
    print("=" * 80)
    print("GATE CHECK: AUGMENTATION BOX-SCALE AUDIT (scale=0.40 vs 0.80 vs 0.90)")
    print("=" * 80)
    
    print("\n1. Sampling Innovatus CCTV GT boxes ...")
    inno_boxes = get_innovatus_boxes()
    inno_stats = compute_distribution_stats(inno_boxes)
    
    print("2. Sampling 2,000 augmented boxes with scale=0.40 (v2 baseline) ...")
    boxes_040 = sample_boxes_for_scale(data_yaml, scale=0.40, target_boxes=2000)
    stats_040 = compute_distribution_stats(boxes_040)
    
    print("3. Sampling 2,000 augmented boxes with scale=0.80 (proposed v3) ...")
    boxes_080 = sample_boxes_for_scale(data_yaml, scale=0.80, target_boxes=2000)
    stats_080 = compute_distribution_stats(boxes_080)
    
    print("4. Sampling 2,000 augmented boxes with scale=0.90 ...")
    boxes_090 = sample_boxes_for_scale(data_yaml, scale=0.90, target_boxes=2000)
    stats_090 = compute_distribution_stats(boxes_090)
    
    df = pd.DataFrame({
        "Innovatus CCTV Ground Truth": [
            f"{inno_stats['count']} boxes",
            f"{inno_stats['median_pct']:.2f}%",
            f"{inno_stats['mean_pct']:.2f}%",
            f"[{inno_stats['p10_pct']:.2f}%, {inno_stats['p90_pct']:.2f}%]",
            f"{inno_stats['bucket_sub1_pct']:.1f}%",
            f"{inno_stats['bucket_1_5_pct']:.1f}%",
            f"{inno_stats['combined_sub5_pct']:.1f}%",
            f"{inno_stats['bucket_5_25_pct']:.1f}%",
            f"{inno_stats['bucket_gt25_pct']:.1f}%",
        ],
        "v2 Augmentation (scale=0.40)": [
            f"{stats_040['count']} boxes",
            f"{stats_040['median_pct']:.2f}%",
            f"{stats_040['mean_pct']:.2f}%",
            f"[{stats_040['p10_pct']:.2f}%, {stats_040['p90_pct']:.2f}%]",
            f"{stats_040['bucket_sub1_pct']:.1f}%",
            f"{stats_040['bucket_1_5_pct']:.1f}%",
            f"{stats_040['combined_sub5_pct']:.1f}%",
            f"{stats_040['bucket_5_25_pct']:.1f}%",
            f"{stats_040['bucket_gt25_pct']:.1f}%",
        ],
        "v3 Candidate (scale=0.80)": [
            f"{stats_080['count']} boxes",
            f"{stats_080['median_pct']:.2f}%",
            f"{stats_080['mean_pct']:.2f}%",
            f"[{stats_080['p10_pct']:.2f}%, {stats_080['p90_pct']:.2f}%]",
            f"{stats_080['bucket_sub1_pct']:.1f}%",
            f"{stats_080['bucket_1_5_pct']:.1f}%",
            f"{stats_080['combined_sub5_pct']:.1f}%",
            f"{stats_080['bucket_5_25_pct']:.1f}%",
            f"{stats_080['bucket_gt25_pct']:.1f}%",
        ],
        "v3 Candidate (scale=0.90)": [
            f"{stats_090['count']} boxes",
            f"{stats_090['median_pct']:.2f}%",
            f"{stats_090['mean_pct']:.2f}%",
            f"[{stats_090['p10_pct']:.2f}%, {stats_090['p90_pct']:.2f}%]",
            f"{stats_090['bucket_sub1_pct']:.1f}%",
            f"{stats_090['bucket_1_5_pct']:.1f}%",
            f"{stats_090['combined_sub5_pct']:.1f}%",
            f"{stats_090['bucket_5_25_pct']:.1f}%",
            f"{stats_090['bucket_gt25_pct']:.1f}%",
        ],
    }, index=[
        "Sample Size",
        "Median Box Area (% Frame)",
        "Mean Box Area (% Frame)",
        "10th–90th Percentile Range",
        "< 1% Frame Area Bucket",
        "1%–5% Frame Area Bucket",
        "Combined CCTV-Scale (< 5%)",
        "5%–25% Frame Area Bucket",
        "> 25% Frame Area Bucket",
    ])
    
    print("\n" + df.to_string())
    print("=" * 80)
    
    # Gate decision logic
    v2_sub5 = stats_040["combined_sub5_pct"]
    v3_080_sub5 = stats_080["combined_sub5_pct"]
    v3_090_sub5 = stats_090["combined_sub5_pct"]
    
    print(f"\nGate Assessment:")
    print(f"  v2 (scale=0.40) <5% frame area share: {v2_sub5:.1f}%")
    print(f"  scale=0.80 <5% frame area share:     {v3_080_sub5:.1f}% (shift: {v3_080_sub5 - v2_sub5:+.1f}%)")
    print(f"  scale=0.90 <5% frame area share:     {v3_090_sub5:.1f}% (shift: {v3_090_sub5 - v2_sub5:+.1f}%)")
    
    best_scale = 0.80
    if v3_080_sub5 > v2_sub5:
        print("  --> Scale 0.80 successfully shifts distribution toward <5% frame area! GATE PASSES.")
        if v3_090_sub5 > v3_080_sub5:
            print(f"  --> Scale 0.90 provides an even stronger shift ({v3_090_sub5:.1f}% vs {v3_080_sub5:.1f}%).")
            # Check if median or 10th percentile collapses too low (<0.1%)
            if stats_090["p10_pct"] >= 0.2:
                best_scale = 0.90
                print("  --> Selected scale 0.90 for v3 training config.")
            else:
                print("  --> Scale 0.90 causes excessive tiny degenerate boxes. Retaining scale 0.80.")
                best_scale = 0.80
        else:
            best_scale = 0.80
            print("  --> Scale 0.80 is optimal. Retaining scale 0.80.")
    else:
        print("  --> Scale 0.80 did not shift distribution. Testing scale 0.90...")
        if v3_090_sub5 > v2_sub5:
            best_scale = 0.90
            print("  --> Scale 0.90 succeeds! GATE PASSES with scale 0.90.")
        else:
            raise RuntimeError("GATE FAILED: Neither scale 0.80 nor 0.90 shifted distribution toward <5% frame area.")
            
    # Update config with best_scale
    v3_yaml_path = Path("configs/detection/yolo11m_960_v3.yaml")
    with open(v3_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["scale"] = best_scale
    with open(v3_yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nUpdated {v3_yaml_path} with scale: {best_scale}")
    
    out_dict = {
        "innovatus": inno_stats,
        "scale_040": stats_040,
        "scale_080": stats_080,
        "scale_090": stats_090,
        "selected_scale": best_scale,
        "gate_status": "PASS",
    }
    with open("outputs/eval/gate_scale_audit.json", "w") as f:
        json.dump(out_dict, f, indent=2)


if __name__ == "__main__":
    main()

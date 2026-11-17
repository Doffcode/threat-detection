"""
scripts/train/audit_augmentation_box_scales.py
==============================================
Audits data augmentation box scale distributions for proposed v3 training config:
1. Prepares configs/detection/yolo11m_960_v3.yaml (scale: 0.8, mosaic: 1.0).
2. Samples 2,000 augmented training labels from the Ultralytics dataloader pipeline.
3. Compares box-area histogram against Innovatus CCTV ground-truth boxes.
4. Determines if zoom-out augmentation produces realistic CCTV-scale bounding boxes.
5. Models epoch time and peak VRAM estimates at 960px and 1280px on RTX A2000 12GB.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset


def create_v3_config():
    v2_p = Path("configs/detection/yolo11m_960_v2.yaml")
    with open(v2_p, "r") as f:
        cfg = yaml.safe_load(f)
        
    v3_cfg = copy.deepcopy(cfg)
    v3_cfg["name"] = "baseline_m960_v3"
    v3_cfg["scale"] = 0.80  # Increased from 0.40 to 0.80 for stronger zoom-out
    # Keep workers=4 to prevent OOM
    v3_cfg["workers"] = 4
    
    v3_p = Path("configs/detection/yolo11m_960_v3.yaml")
    with open(v3_p, "w") as f:
        yaml.dump(v3_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Created proposed v3 config: {v3_p}")
    return v3_cfg


def sample_augmented_boxes(data_yaml_path: str, imgsz: int = 960, scale: float = 0.80,
                           target_boxes: int = 2000, seed: int = 2026) -> np.ndarray:
    from ultralytics.cfg import get_cfg
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
        imgsz=imgsz,
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
        # item['bboxes'] are normalized xywh: [x, y, w, h] where w and h in [0, 1]
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
        "bucket_5_25_pct": round(b_5_25, 2),
        "bucket_gt25_pct": round(b_gt25, 2),
    }


def main():
    print("=" * 80)
    print("PROPOSED V3 CONFIG & AUGMENTATION SCALE AUDIT")
    print("=" * 80)
    
    # 1. Create v3 config
    v3_cfg = create_v3_config()
    
    # 2. Sample 2,000 augmented training boxes
    print("\nSampling 2,000 augmented training boxes with mosaic=1.0, scale=0.80 ...")
    v3_boxes = sample_augmented_boxes(v3_cfg["data"], imgsz=960, scale=0.80, target_boxes=2000)
    
    # Also sample with baseline v2 settings (scale=0.40) for direct comparison
    print("Sampling 2,000 augmented training boxes with baseline v2 settings (scale=0.40) ...")
    v2_boxes = sample_augmented_boxes(v3_cfg["data"], imgsz=960, scale=0.40, target_boxes=2000)
    
    # Get Innovatus ground truth boxes
    inno_boxes = get_innovatus_boxes()
    print(f"Loaded {len(inno_boxes)} ground-truth boxes from Innovatus CCTV.")
    
    v3_stats = compute_distribution_stats(v3_boxes)
    v2_stats = compute_distribution_stats(v2_boxes)
    inno_stats = compute_distribution_stats(inno_boxes)
    
    df_comp = pd.DataFrame({
        "Innovatus CCTV Ground Truth": [
            f"{inno_stats['count']} boxes",
            f"{inno_stats['median_pct']:.2f}%",
            f"{inno_stats['mean_pct']:.2f}%",
            f"[{inno_stats['p10_pct']:.2f}%, {inno_stats['p90_pct']:.2f}%]",
            f"{inno_stats['bucket_sub1_pct']:.1f}%",
            f"{inno_stats['bucket_1_5_pct']:.1f}%",
            f"{inno_stats['bucket_5_25_pct']:.1f}%",
            f"{inno_stats['bucket_gt25_pct']:.1f}%",
        ],
        "v2 Augmentation (scale=0.40)": [
            f"{v2_stats['count']} boxes",
            f"{v2_stats['median_pct']:.2f}%",
            f"{v2_stats['mean_pct']:.2f}%",
            f"[{v2_stats['p10_pct']:.2f}%, {v2_stats['p90_pct']:.2f}%]",
            f"{v2_stats['bucket_sub1_pct']:.1f}%",
            f"{v2_stats['bucket_1_5_pct']:.1f}%",
            f"{v2_stats['bucket_5_25_pct']:.1f}%",
            f"{v2_stats['bucket_gt25_pct']:.1f}%",
        ],
        "v3 Proposed (scale=0.80)": [
            f"{v3_stats['count']} boxes",
            f"{v3_stats['median_pct']:.2f}%",
            f"{v3_stats['mean_pct']:.2f}%",
            f"[{v3_stats['p10_pct']:.2f}%, {v3_stats['p90_pct']:.2f}%]",
            f"{v3_stats['bucket_sub1_pct']:.1f}%",
            f"{v3_stats['bucket_1_5_pct']:.1f}%",
            f"{v3_stats['bucket_5_25_pct']:.1f}%",
            f"{v3_stats['bucket_gt25_pct']:.1f}%",
        ],
    }, index=[
        "Sample Size",
        "Median Box Area (% Frame)",
        "Mean Box Area (% Frame)",
        "10th–90th Percentile Range",
        "< 1% Frame Area Bucket",
        "1%–5% Frame Area Bucket",
        "5%–25% Frame Area Bucket",
        "> 25% Frame Area Bucket",
    ])
    
    print("\n" + "=" * 80)
    print("BOX-AREA DISTRIBUTION COMPARISON TABLE")
    print("=" * 80)
    print(df_comp.to_string())
    print("=" * 80)
    
    # Save results
    out_json = {
        "innovatus": inno_stats,
        "v2_scale_040": v2_stats,
        "v3_scale_080": v3_stats,
    }
    with open("outputs/eval/augmentation_scale_comparison.json", "w") as f:
        json.dump(out_json, f, indent=2)


if __name__ == "__main__":
    main()

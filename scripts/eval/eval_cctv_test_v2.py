"""
scripts/eval/eval_cctv_test_v2.py
=================================
Fast GPU evaluation of test_cctv_v2 benchmark (92 frames: 72 Innovatus + 20 BITS)
on baseline_m960_v1 best.pt and baseline_m960_v2 best.pt.
Reports headline Innovatus, BITS separately, and combined test_cctv_v2.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO

CLASS_NAMES = {0: "handgun", 1: "rifle", 2: "knife"}


def build_cctv_v2_coco(images_dir: Path, labels_dir: Path, template_csv: Path, filter_source: str = None):
    df = pd.read_csv(template_csv)
    source_map = dict(zip(df["image_filename"], df["source"]))
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "handgun"},
            {"id": 1, "name": "rifle"},
            {"id": 2, "name": "knife"},
        ],
    }
    
    ann_id = 1
    image_id = 1
    
    valid_exts = {".jpg", ".jpeg", ".png"}
    img_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts])
    
    img_paths_selected = []
    
    for img_p in img_files:
        src = source_map.get(img_p.name, "unknown")
        if filter_source and src != filter_source:
            continue
            
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_p.name,
            "width": w,
            "height": h,
            "source": src,
            "path": str(img_p),
        })
        img_paths_selected.append(img_p)
        
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            lines = [l.strip() for l in lbl_p.read_text().splitlines() if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    abs_w = bw * w
                    abs_h = bh * h
                    abs_x1 = (cx - bw / 2.0) * w
                    abs_y1 = (cy - bh / 2.0) * h
                    
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cid,
                        "bbox": [round(abs_x1, 2), round(abs_y1, 2), round(abs_w, 2), round(abs_h, 2)],
                        "area": round(abs_w * abs_h, 2),
                        "iscrowd": 0,
                    })
                    ann_id += 1
        image_id += 1
        
    return coco_data, img_paths_selected


def evaluate_model_on_split(weights_p: Path, split_name: str, filter_source: str = None,
                            conf_star: float = 0.25, imgsz: int = 960, device: str = "cuda:0"):
    torch.cuda.empty_cache()
    images_dir = WORKSPACE_ROOT / "data/cctv_test_v2_candidates/images"
    labels_dir = WORKSPACE_ROOT / "data/cctv_test_v2_candidates/labels"
    template_csv = WORKSPACE_ROOT / "configs/detection/cctv_test_v2_occlusion_template.csv"
    
    coco_gt_dict, img_paths = build_cctv_v2_coco(images_dir, labels_dir, template_csv, filter_source)
    
    gt_coco = COCO()
    gt_coco.dataset = coco_gt_dict
    gt_coco.createIndex()
    
    image_name_to_id = {img["file_name"]: img["id"] for img in coco_gt_dict["images"]}
    
    model = YOLO(str(weights_p))
    
    detections = []
    batch_size = 8
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i + batch_size]
        results = model.predict(
            source=[str(p) for p in batch],
            imgsz=imgsz,
            conf=0.01,
            iou=0.50,
            device=device,
            verbose=False,
        )
        for img_p, res in zip(batch, results):
            img_id = image_name_to_id[img_p.name]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cid in zip(xyxy, confs, classes):
                if cid not in CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = box
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                detections.append({
                    "image_id": img_id,
                    "category_id": int(cid),
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(w), 2), round(float(h), 2)],
                    "score": float(score),
                })
            
    # pycocotools COCOeval
    if len(detections) > 0 and len(coco_gt_dict["annotations"]) > 0:
        pred_coco = gt_coco.loadRes(detections)
        coco_eval = COCOeval(gt_coco, pred_coco, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP50_95 = float(coco_eval.stats[0])
        mAP50 = float(coco_eval.stats[1])
        mAP75 = float(coco_eval.stats[2])
        ap_s = float(coco_eval.stats[3])
        ap_m = float(coco_eval.stats[4])
        ap_l = float(coco_eval.stats[5])
    else:
        mAP50_95 = mAP50 = mAP75 = ap_s = ap_m = ap_l = 0.0

    # Operational point at conf* and at 0.25
    def calc_pr(conf_thresh):
        filtered_preds = [d for d in detections if d["score"] >= conf_thresh]
        tp = 0
        fp = 0
        gt_matched = set()
        
        gt_by_img = {}
        for ann in coco_gt_dict["annotations"]:
            gt_by_img.setdefault(ann["image_id"], []).append(ann)
            
        preds_by_img = {}
        for p in filtered_preds:
            preds_by_img.setdefault(p["image_id"], []).append(p)
            
        for img_id, gts in gt_by_img.items():
            img_preds = preds_by_img.get(img_id, [])
            img_preds = sorted(img_preds, key=lambda x: x["score"], reverse=True)
            for p in img_preds:
                best_iou = 0.0
                best_gt_id = -1
                px, py, pw, ph = p["bbox"]
                p_x2, p_y2 = px + pw, py + ph
                
                for gt in gts:
                    if gt["id"] in gt_matched or gt["category_id"] != p["category_id"]:
                        continue
                    gx, gy, gw, gh = gt["bbox"]
                    g_x2, g_y2 = gx + gw, gy + gh
                    
                    ix1, iy1 = max(px, gx), max(py, gy)
                    ix2, iy2 = min(p_x2, g_x2), min(p_y2, g_y2)
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (pw * ph) + (gw * gh) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_id = gt["id"]
                        
                if best_iou >= 0.50 and best_gt_id != -1:
                    tp += 1
                    gt_matched.add(best_gt_id)
                else:
                    fp += 1
                    
        total_gt = len(coco_gt_dict["annotations"])
        fn = total_gt - len(gt_matched)
        p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_val = tp / total_gt if total_gt > 0 else 0.0
        f1_val = (2 * p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        return {"precision": round(p_val, 4), "recall": round(r_val, 4), "f1": round(f1_val, 4), "tp": tp, "fp": fp, "fn": fn}

    metrics_star = calc_pr(conf_star)
    metrics_25 = calc_pr(0.25)
    
    return {
        "split": split_name,
        "n_images": len(coco_gt_dict["images"]),
        "n_annotations": len(coco_gt_dict["annotations"]),
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "ap_small": round(ap_s, 4),
        "ap_medium": round(ap_m, 4),
        "ap_large": round(ap_l, 4),
        "conf_star": conf_star,
        "pr_at_conf_star": metrics_star,
        "pr_at_conf_025": metrics_25,
    }


def main():
    print("=" * 80)
    print("EVALUATION ON TEST_CCTV_V2 (92 FRAMES)")
    print("=" * 80)
    
    out_dir = WORKSPACE_ROOT / "outputs/eval/cctv_test_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    v1_weights = WORKSPACE_ROOT / "outputs/runs/baseline_m960_v1/weights/best.pt"
    v2_weights = WORKSPACE_ROOT / "outputs/runs/baseline_m960_v2/weights/best.pt"
    
    results = {}
    
    for tag, wpath, conf_star in [("v1", v1_weights, 0.28), ("v2", v2_weights, 0.30)]:
        print(f"\n>>> Evaluating {tag} (conf* = {conf_star}) on test_cctv_v2...")
        res_inno = evaluate_model_on_split(wpath, "innovatus_rob", filter_source="innovatus_rob", conf_star=conf_star)
        res_bits = evaluate_model_on_split(wpath, "bits_pilani", filter_source="bits_pilani", conf_star=conf_star)
        res_all = evaluate_model_on_split(wpath, "test_cctv_v2_all", filter_source=None, conf_star=conf_star)
        
        results[tag] = {
            "innovatus_rob": res_inno,
            "bits_pilani": res_bits,
            "test_cctv_v2_all": res_all,
        }
        
    out_file = out_dir / "cctv_test_v2_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved test_cctv_v2 evaluation to {out_file}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("TEST_CCTV_V2 BENCHMARK COMPARISON: v1 vs v2")
    print("=" * 80)
    print(f"{'Split / Subset':<25} | {'Metric':<15} | {'v1':<12} | {'v2':<12} | {'Delta (v2 - v1)':<15}")
    print("-" * 85)
    for subset in ["innovatus_rob", "bits_pilani", "test_cctv_v2_all"]:
        m1 = results["v1"][subset]
        m2 = results["v2"][subset]
        print(f"{subset:<25} | mAP50           | {m1['mAP50']:<12.4f} | {m2['mAP50']:<12.4f} | {m2['mAP50'] - m1['mAP50']:+<15.4f}")
        print(f"{'':<25} | F1 @ conf*      | {m1['pr_at_conf_star']['f1']:<12.4f} | {m2['pr_at_conf_star']['f1']:<12.4f} | {m2['pr_at_conf_star']['f1'] - m1['pr_at_conf_star']['f1']:+<15.4f}")
        print(f"{'':<25} | P/R @ conf*     | {m1['pr_at_conf_star']['precision']:.3f}/{m1['pr_at_conf_star']['recall']:.3f} | {m2['pr_at_conf_star']['precision']:.3f}/{m2['pr_at_conf_star']['recall']:.3f} |")
        print(f"{'':<25} | F1 @ 0.25       | {m1['pr_at_conf_025']['f1']:<12.4f} | {m2['pr_at_conf_025']['f1']:<12.4f} | {m2['pr_at_conf_025']['f1'] - m1['pr_at_conf_025']['f1']:+<15.4f}")
        print("-" * 85)

if __name__ == "__main__":
    main()

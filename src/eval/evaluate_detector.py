"""
src/eval/evaluate_detector.py
=============================
Standardized evaluation harness for Stage 1 Baseline Weapon Detector.
Supports like-for-like evaluation across baseline_m960_v1 and baseline_m960_v2.

Key features:
1. Valid-only operational confidence sweep (argmax F1 on valid only).
2. Standard pycocotools COCOeval across: valid, test_public, test_domain,
   test_cctv_v1 (with headline innovatus_rob and separate bits_pilani), test_hard_negatives.
3. COCO-standard mAP50, mAP50-95, AP-small, AP-med, AP-large (original-pixel definition).
4. Per-class and per-source performance breakdowns.
5. Operating point metrics at both conf* (optimal valid) and fixed conf=0.25.
6. FPPI on test_hard_negatives.
7. Group-level bootstrap 95% Confidence Intervals for valid, test_public, test_domain, and innovatus_rob.
8. Recall by GT box-size bucket (<1%, 1-5%, 5-25%, >25% of frame area).
9. Diagnostic evaluation removing predictions covering >50% of the frame.
10. Automated Error Analysis: Top 30 FNs and Top 30 FPs for test_domain and innovatus_rob
    with failure tags and saved visual galleries.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = {0: "handgun", 1: "rifle", 2: "knife"}


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads manifest.csv indexed by filename."""
    if not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path)
    return df.set_index("filename").to_dict(orient="index")


def build_coco_gt_dict(split_dir: Path, manifest_dict: Optional[Dict[str, Dict]] = None,
                       filter_source: Optional[str] = None) -> Dict[str, Any]:
    """Converts YOLO format labels and images in a split into a COCO-format ground truth dict.
    Optionally filters by manifest source (e.g. 'innovatus_rob' or 'bits_pilani').
    """
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    
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
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    img_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix in valid_exts])
    
    for img_p in img_files:
        meta = manifest_dict.get(img_p.name, {}) if manifest_dict else {}
        source = meta.get("source", "unknown")
        if filter_source and source != filter_source:
            continue
            
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        group_id = meta.get("group_id", image_id)
        
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_p.name,
            "width": w,
            "height": h,
            "source": source,
            "group_id": group_id,
            "path": str(img_p),
        })
        
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            lines = lbl_p.read_text().strip().splitlines()
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
        
    return coco_data


def run_yolo_inference(model: Any, img_paths: List[Path], image_name_to_id: Dict[str, int],
                       imgsz: int = 960, conf: float = 0.01, iou: float = 0.50,
                       device: str = "cuda:0") -> List[Dict[str, Any]]:
    """Runs YOLO batch inference over images and converts predictions to COCO format."""
    detections = []
    batch_size = 16
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i + batch_size]
        results = model.predict(
            source=[str(p) for p in batch],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            save=False
        )
        
        for p, r in zip(batch, results):
            img_id = image_name_to_id.get(p.name)
            if img_id is None:
                continue
                
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            
            for (x1, y1, x2, y2), score, cid in zip(xyxy, confs, clss):
                bw = float(x2 - x1)
                bh = float(y2 - y1)
                detections.append({
                    "image_id": img_id,
                    "category_id": int(cid),
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(bw, 2), round(bh, 2)],
                    "score": round(float(score), 4),
                })
                
    return detections


def box_iou(b1: List[float], b2: List[float]) -> float:
    """Computes IoU between two bboxes [x, y, w, h]."""
    x1_1, y1_1, x2_1, y2_1 = b1[0], b1[1], b1[0] + b1[2], b1[1] + b1[3]
    x1_2, y1_2, x2_2, y2_2 = b2[0], b2[1], b2[0] + b2[2], b2[1] + b2[3]
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_operating_point_metrics(coco_gt_dict: Dict[str, Any], detections: List[Dict[str, Any]],
                                    conf_thresh: float = 0.20, iou_thresh: float = 0.50) -> Dict[str, Any]:
    """Computes Precision, Recall, F1, TP, FP, FN per class and overall at given operating threshold."""
    gt_by_img_cls = collections.defaultdict(list)
    for ann in coco_gt_dict["annotations"]:
        gt_by_img_cls[(ann["image_id"], ann["category_id"])].append({
            "bbox": ann["bbox"],
            "matched": False
        })
        
    filtered_dt = [d for d in detections if d.get("score", 0.0) >= conf_thresh]
    filtered_dt.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    
    class_stats = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in [0, 1, 2]}
    
    for dt in filtered_dt:
        img_id = dt["image_id"]
        cid = dt["category_id"]
        if cid not in class_stats:
            continue
            
        gts = gt_by_img_cls.get((img_id, cid), [])
        best_iou = 0.0
        best_gt = None
        for gt in gts:
            if not gt["matched"]:
                iou = box_iou(dt["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
                    
        if best_iou >= iou_thresh and best_gt is not None:
            best_gt["matched"] = True
            class_stats[cid]["tp"] += 1
        else:
            class_stats[cid]["fp"] += 1
            
    for (img_id, cid), gts in gt_by_img_cls.items():
        if cid in class_stats:
            for gt in gts:
                if not gt["matched"]:
                    class_stats[cid]["fn"] += 1
                    
    results = {}
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    for cid, st in class_stats.items():
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        results[CLASS_NAMES[cid]] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        
    p_tot = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    r_tot = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    f1_tot = (2 * p_tot * r_tot) / (p_tot + r_tot) if (p_tot + r_tot) > 0 else 0.0
    results["overall"] = {"precision": p_tot, "recall": r_tot, "f1": f1_tot, "tp": tot_tp, "fp": tot_fp, "fn": tot_fn}
    return results


def sweep_valid_operating_point(coco_gt_dict: Dict[str, Any], detections: List[Dict[str, Any]],
                                iou_thresh: float = 0.50) -> Tuple[float, float, pd.DataFrame]:
    """Sweeps confidence thresholds on valid set strictly to find argmax F1."""
    confs = np.arange(0.05, 0.86, 0.01)
    sweep_data = []
    
    best_conf = 0.20
    best_f1 = -1.0
    
    for c in confs:
        c_val = round(float(c), 3)
        res = compute_operating_point_metrics(coco_gt_dict, detections, conf_thresh=c_val, iou_thresh=iou_thresh)
        overall = res["overall"]
        p, r, f1 = overall["precision"], overall["recall"], overall["f1"]
        sweep_data.append({
            "conf": c_val,
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": overall["tp"],
            "fp": overall["fp"],
            "fn": overall["fn"],
        })
        if f1 > best_f1:
            best_f1 = f1
            best_conf = c_val
            
    df_sweep = pd.DataFrame(sweep_data)
    return best_conf, best_f1, df_sweep


def evaluate_coco_metrics(coco_gt: COCO, coco_dt: COCO) -> Dict[str, Any]:
    """Runs pycocotools COCOeval and returns standard metrics."""
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = coco_eval.stats
    metrics: Dict[str, Any] = {
        "mAP50_95": float(stats[0]) if stats[0] >= 0 else 0.0,
        "mAP50": float(stats[1]) if stats[1] >= 0 else 0.0,
        "mAP75": float(stats[2]) if stats[2] >= 0 else 0.0,
        "AP_small": float(stats[3]) if stats[3] >= 0 else 0.0,
        "AP_medium": float(stats[4]) if stats[4] >= 0 else 0.0,
        "AP_large": float(stats[5]) if stats[5] >= 0 else 0.0,
    }
    
    # Extract per-class stats
    prec = coco_eval.eval["precision"]  # shape (10, 101, K, 4, 3)
    per_class = {}
    for cid, cname in CLASS_NAMES.items():
        if cid < prec.shape[2]:
            p_cat = prec[:, :, cid, 0, 2]
            p_cat_valid = p_cat[p_cat > -1]
            ap50_95 = float(np.mean(p_cat_valid)) if len(p_cat_valid) > 0 else 0.0
            
            p50_cat = prec[0, :, cid, 0, 2]
            p50_valid = p50_cat[p50_cat > -1]
            ap50 = float(np.mean(p50_valid)) if len(p50_valid) > 0 else 0.0
            
            per_class[cname] = {"mAP50": ap50, "mAP50_95": ap50_95}
        else:
            per_class[cname] = {"mAP50": 0.0, "mAP50_95": 0.0}
            
    metrics["per_class"] = per_class
    return metrics


def compute_fppi(detections: List[Dict[str, Any]], total_images: int, conf_threshold: float = 0.20) -> Tuple[float, int, Dict[str, int]]:
    """Computes False Positives Per Image (FPPI) on weapon-free negative images."""
    if total_images == 0:
        return 0.0, 0, {}
        
    filtered = [d for d in detections if d.get("score", 0.0) >= conf_threshold]
    total_fps = len(filtered)
    fppi = float(total_fps / total_images)
    
    by_cls = {cname: 0 for cname in CLASS_NAMES.values()}
    for d in filtered:
        cname = CLASS_NAMES.get(d["category_id"], str(d["category_id"]))
        by_cls[cname] = by_cls.get(cname, 0) + 1
        
    return fppi, total_fps, by_cls


def compute_recall_by_box_size_bucket(coco_gt_dict: Dict[str, Any], detections: List[Dict[str, Any]],
                                      conf_thresh: float = 0.20, iou_thresh: float = 0.50) -> Dict[str, Dict[str, Any]]:
    """Computes recall by GT box-size bucket relative to frame area:
       1: <1% of frame area
       2: 1-5% of frame area
       3: 5-25% of frame area
       4: >25% of frame area
    """
    img_map = {img["id"]: img for img in coco_gt_dict["images"]}
    
    buckets = {
        "<1%": {"min": 0.0, "max": 0.01, "gt_count": 0, "matched": 0},
        "1-5%": {"min": 0.01, "max": 0.05, "gt_count": 0, "matched": 0},
        "5-25%": {"min": 0.05, "max": 0.25, "gt_count": 0, "matched": 0},
        ">25%": {"min": 0.25, "max": 10.0, "gt_count": 0, "matched": 0},
    }
    
    gt_by_img_cls = collections.defaultdict(list)
    for ann in coco_gt_dict["annotations"]:
        img_info = img_map[ann["image_id"]]
        img_area = float(img_info["width"] * img_info["height"])
        ann_area = float(ann["bbox"][2] * ann["bbox"][3])
        ratio = ann_area / img_area if img_area > 0 else 0.0
        
        assigned_bucket = ">25%"
        for b_name, b_bounds in buckets.items():
            if b_bounds["min"] <= ratio < b_bounds["max"]:
                assigned_bucket = b_name
                break
                
        buckets[assigned_bucket]["gt_count"] += 1
        
        gt_obj = {
            "id": ann["id"],
            "bbox": ann["bbox"],
            "bucket": assigned_bucket,
            "matched": False
        }
        gt_by_img_cls[(ann["image_id"], ann["category_id"])].append(gt_obj)
        
    filtered_dt = [d for d in detections if d.get("score", 0.0) >= conf_thresh]
    filtered_dt.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    
    for dt in filtered_dt:
        img_id = dt["image_id"]
        cid = dt["category_id"]
        gts = gt_by_img_cls.get((img_id, cid), [])
        
        best_iou = 0.0
        best_gt = None
        for gt in gts:
            if not gt["matched"]:
                iou = box_iou(dt["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
                    
        if best_iou >= iou_thresh and best_gt is not None:
            best_gt["matched"] = True
            buckets[best_gt["bucket"]]["matched"] += 1
            
    bucket_results = {}
    for b_name, b_data in buckets.items():
        cnt = b_data["gt_count"]
        m = b_data["matched"]
        r = m / cnt if cnt > 0 else 0.0
        bucket_results[b_name] = {
            "gt_count": cnt,
            "matched": m,
            "recall": round(r, 4),
        }
    return bucket_results


def compute_large_box_diagnostics(coco_gt_dict: Dict[str, Any], detections: List[Dict[str, Any]],
                                  conf_star: float, fixed_conf: float = 0.25) -> Dict[str, Any]:
    """Diagnostic analysis: counts predictions covering >50% of the frame per class,
       removes them, and reports re-computed COCO and operating metrics.
    """
    img_map = {img["id"]: img for img in coco_gt_dict["images"]}
    
    large_box_counts = {cname: 0 for cname in CLASS_NAMES.values()}
    large_box_counts["total"] = 0
    
    filtered_dt = []
    for dt in detections:
        img_info = img_map.get(dt["image_id"])
        if not img_info:
            filtered_dt.append(dt)
            continue
            
        img_area = float(img_info["width"] * img_info["height"])
        dt_area = float(dt["bbox"][2] * dt["bbox"][3])
        ratio = dt_area / img_area if img_area > 0 else 0.0
        
        if ratio > 0.50:
            cname = CLASS_NAMES.get(dt["category_id"], str(dt["category_id"]))
            large_box_counts[cname] += 1
            large_box_counts["total"] += 1
        else:
            filtered_dt.append(dt)
            
    c_gt = COCO()
    c_gt.dataset = coco_gt_dict
    c_gt.createIndex()
    
    if len(filtered_dt) > 0 and len(coco_gt_dict["annotations"]) > 0:
        c_dt = c_gt.loadRes(filtered_dt)
        coco_metrics_filtered = evaluate_coco_metrics(c_gt, c_dt)
    else:
        coco_metrics_filtered = {"mAP50": 0.0, "mAP50_95": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0, "per_class": {}}
        
    op_star_filtered = compute_operating_point_metrics(coco_gt_dict, filtered_dt, conf_thresh=conf_star)
    op_fixed_filtered = compute_operating_point_metrics(coco_gt_dict, filtered_dt, conf_thresh=fixed_conf)
    
    return {
        "large_boxes_removed": large_box_counts,
        "remaining_detections": len(filtered_dt),
        "coco_metrics_filtered": coco_metrics_filtered,
        "op_metrics_conf_star_filtered": op_star_filtered,
        "op_metrics_fixed_025_filtered": op_fixed_filtered,
    }


def bootstrap_ci_grouped(coco_gt_dict: Dict[str, Any], detections: List[Dict[str, Any]],
                         n_bootstraps: int = 100, ci: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """Computes cluster/group-level bootstrap confidence intervals."""
    images = coco_gt_dict["images"]
    if len(images) == 0 or len(coco_gt_dict["annotations"]) == 0:
        return {"mAP50": (0.0, 0.0), "mAP50_95": (0.0, 0.0)}
        
    groups = collections.defaultdict(list)
    for img in images:
        gid = img.get("group_id", img["id"])
        groups[gid].append(img)
        
    unique_gids = list(groups.keys())
    n_groups = len(unique_gids)
    
    dt_by_img = collections.defaultdict(list)
    for dt in detections:
        dt_by_img[dt["image_id"]].append(dt)
        
    gt_by_img = collections.defaultdict(list)
    for ann in coco_gt_dict["annotations"]:
        gt_by_img[ann["image_id"]].append(ann)
        
    rng = np.random.default_rng(2026)
    map50_scores = []
    map50_95_scores = []
    
    for _ in range(n_bootstraps):
        sampled_gids = rng.choice(unique_gids, size=n_groups, replace=True)
        
        sub_gt = {"images": [], "annotations": [], "categories": coco_gt_dict["categories"]}
        sub_dt = []
        
        new_img_id = 1
        ann_counter = 1
        for gid in sampled_gids:
            for orig_img in groups[gid]:
                orig_id = orig_img["id"]
                sub_gt["images"].append({
                    "id": new_img_id,
                    "width": orig_img["width"],
                    "height": orig_img["height"]
                })
                for ann in gt_by_img[orig_id]:
                    ann_copy = dict(ann)
                    ann_copy["id"] = ann_counter
                    ann_copy["image_id"] = new_img_id
                    sub_gt["annotations"].append(ann_copy)
                    ann_counter += 1
                for dt in dt_by_img[orig_id]:
                    dt_copy = dict(dt)
                    dt_copy["image_id"] = new_img_id
                    sub_dt.append(dt_copy)
                new_img_id += 1
                
        try:
            c_gt = COCO()
            c_gt.dataset = sub_gt
            c_gt.createIndex()
            
            if sub_dt:
                c_dt = c_gt.loadRes(sub_dt)
                ev = COCOeval(c_gt, c_dt, "bbox")
                ev.params.imgIds = list(range(1, new_img_id))
                ev.evaluate()
                ev.accumulate()
                prec = ev.eval["precision"]
                p = prec[:, :, :, 0, 2]
                p_valid = p[p > -1]
                m50_95 = float(np.mean(p_valid)) if len(p_valid) > 0 else 0.0
                p50 = prec[0, :, :, 0, 2]
                p50_valid = p50[p50 > -1]
                m50 = float(np.mean(p50_valid)) if len(p50_valid) > 0 else 0.0
                map50_95_scores.append(m50_95)
                map50_scores.append(m50)
            else:
                map50_95_scores.append(0.0)
                map50_scores.append(0.0)
        except Exception:
            continue
            
    alpha = (1.0 - ci) / 2.0
    if map50_scores:
        m50_low = float(np.percentile(map50_scores, alpha * 100))
        m50_high = float(np.percentile(map50_scores, (1.0 - alpha) * 100))
        m_low = float(np.percentile(map50_95_scores, alpha * 100))
        m_high = float(np.percentile(map50_95_scores, (1.0 - alpha) * 100))
    else:
        m50_low, m50_high, m_low, m_high = 0.0, 0.0, 0.0, 0.0
        
    return {
        "mAP50": (round(m50_low, 4), round(m50_high, 4)),
        "mAP50_95": (round(m_low, 4), round(m_high, 4)),
    }


def analyze_image_characteristics(img: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
    """Computes blur, luminance, and size properties for an image patch / bounding box."""
    h, w = img.shape[:2]
    bx, by, bw, bh = [int(v) for v in bbox]
    bx = max(0, min(w - 1, bx))
    by = max(0, min(h - 1, by))
    bw = max(1, min(w - bx, bw))
    bh = max(1, min(h - by, bh))
    
    crop = img[by:by+bh, bx:bx+bw]
    area = bw * bh
    frame_area = w * h
    ratio = area / frame_area if frame_area > 0 else 0.0
    
    # Blur detection via Laplacian variance
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else None
    blur_score = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var()) if gray_crop is not None and gray_crop.size > 4 else 0.0
    
    # Low-light detection via HSV Value channel
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.size > 0 else None
    luminance = float(np.mean(hsv_crop[:, :, 2])) if hsv_crop is not None and hsv_crop.size > 0 else 0.0
    
    # Boundary proximity (occlusion indicator)
    is_border = (bx <= 10 or by <= 10 or (bx + bw) >= w - 10 or (by + bh) >= h - 10)
    
    return {
        "area": area,
        "ratio": ratio,
        "blur_score": blur_score,
        "luminance": luminance,
        "is_border": is_border,
        "is_small": (area < 1024 or ratio < 0.01),
        "is_blur": blur_score < 100.0,
        "is_low_light": luminance < 60.0,
    }


def generate_error_galleries(split_name: str, coco_gt_dict: Dict[str, Any], detections: List[Dict[str, Any]],
                             conf_thresh: float, output_dir: Path, max_samples: int = 30) -> Dict[str, Any]:
    """Identifies top 30 False Negatives and False Positives, assigns failure tags, and renders annotated galleries."""
    gallery_dir = output_dir / "error_galleries" / split_name
    fn_dir = gallery_dir / "false_negatives"
    fp_dir = gallery_dir / "false_positives"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)
    
    img_map = {img["id"]: img for img in coco_gt_dict["images"]}
    
    # Organize GT annotations by image
    gt_by_img = collections.defaultdict(list)
    for ann in coco_gt_dict["annotations"]:
        gt_by_img[ann["image_id"]].append({
            "id": ann["id"],
            "cid": ann["category_id"],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "matched": False,
            "max_overlap_score": 0.0,
        })
        
    filtered_dt = [d for d in detections if d.get("score", 0.0) >= conf_thresh]
    filtered_dt.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    
    dt_by_img = collections.defaultdict(list)
    for dt in detections:
        dt_by_img[dt["image_id"]].append(dt)
        
    false_positives = []
    
    # Match detections to GT
    for dt in filtered_dt:
        img_id = dt["image_id"]
        cid = dt["category_id"]
        gts = [g for g in gt_by_img[img_id] if g["cid"] == cid]
        
        best_iou = 0.0
        best_gt = None
        for g in gts:
            iou = box_iou(dt["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = g
                
        # Also update max overlap score for all gts in image regardless of class
        for g in gt_by_img[img_id]:
            ov = box_iou(dt["bbox"], g["bbox"])
            if ov > g["max_overlap_score"]:
                g["max_overlap_score"] = float(dt["score"])
                
        if best_iou >= 0.50 and best_gt is not None and not best_gt["matched"]:
            best_gt["matched"] = True
        else:
            false_positives.append(dt)
            
    false_negatives = []
    for img_id, gts in gt_by_img.items():
        for g in gts:
            if not g["matched"]:
                false_negatives.append({
                    "image_id": img_id,
                    "cid": g["cid"],
                    "bbox": g["bbox"],
                    "area": g["area"],
                    "max_overlap_score": g["max_overlap_score"],
                })
                
    # Sort FNs: lowest max_overlap_score first (hardest misses)
    false_negatives.sort(key=lambda x: (x["max_overlap_score"], x["area"]))
    # Sort FPs: highest prediction score first (most egregious false alarms)
    false_positives.sort(key=lambda x: x["score"], reverse=True)
    
    fn_samples = false_negatives[:max_samples]
    fp_samples = false_positives[:max_samples]
    
    tag_counts = collections.defaultdict(int)
    gallery_records = []
    
    # Process False Negatives
    for rank, fn in enumerate(fn_samples, 1):
        img_info = img_map[fn["image_id"]]
        img_path = img_info["path"]
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        char_info = analyze_image_characteristics(img, fn["bbox"])
        tags = []
        if char_info["is_small"]:
            tags.append("small_object [heuristic]")
        if char_info["is_blur"]:
            tags.append("motion_blur [heuristic]")
        if char_info["is_low_light"]:
            tags.append("low_light [heuristic]")
        if char_info["is_border"]:
            tags.append("near_border [heuristic]")
            
        # Check cross-class confusion (did model predict another weapon class here?)
        for dt in dt_by_img[fn["image_id"]]:
            if dt["score"] >= conf_thresh and dt["category_id"] != fn["cid"]:
                if box_iou(dt["bbox"], fn["bbox"]) >= 0.30:
                    tags.append(f"class_confusion_with_{CLASS_NAMES[dt['category_id']]} [heuristic]")
                    
        if not tags:
            tags.append("unmodeled_appearance [heuristic]")
            
        for t in tags:
            tag_counts[f"FN_{t}"] += 1
            
        # Draw on image
        annotated = img.copy()
        bx, by, bw, bh = [int(v) for v in fn["bbox"]]
        cname = CLASS_NAMES[fn["cid"]]
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(annotated, f"GT FN: {cname}", (bx, max(20, by - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        # Also draw any candidate preds in blue
        for dt in dt_by_img[fn["image_id"]]:
            if dt["score"] >= 0.05:
                px, py, pw, ph = [int(v) for v in dt["bbox"]]
                cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (255, 100, 0), 1)
                cv2.putText(annotated, f"{CLASS_NAMES[dt['category_id']]} {dt['score']:.2f}",
                            (px, py + ph + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
                            
        header = f"[FN #{rank}] {cname.upper()} | Tags: {','.join(tags)} | Area: {char_info['area']}px | Score: {fn['max_overlap_score']:.2f}"
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(annotated, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        save_p = fn_dir / f"fn_{rank:02d}_{img_info['file_name']}"
        cv2.imwrite(str(save_p), annotated)
        gallery_records.append({
            "split": split_name,
            "type": "FN",
            "rank": rank,
            "file_name": img_info["file_name"],
            "class": cname,
            "score": fn["max_overlap_score"],
            "tags": ";".join(tags),
            "area": char_info["area"],
            "saved_path": str(save_p),
        })
        
    # Process False Positives
    for rank, fp in enumerate(fp_samples, 1):
        img_info = img_map[fp["image_id"]]
        img_path = img_info["path"]
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        char_info = analyze_image_characteristics(img, fp["bbox"])
        tags = []
        if char_info["is_small"]:
            tags.append("small_object [heuristic]")
        if char_info["is_blur"]:
            tags.append("motion_blur [heuristic]")
        if char_info["is_low_light"]:
            tags.append("low_light [heuristic]")
            
        # Check cross-class confusion with GT
        has_cross_class = False
        for gt in gt_by_img[fp["image_id"]]:
            if gt["cid"] != fp["category_id"] and box_iou(fp["bbox"], gt["bbox"]) >= 0.30:
                tags.append(f"class_confusion_with_{CLASS_NAMES[gt['cid']]} [heuristic]")
                has_cross_class = True
                break
                
        if not has_cross_class:
            if char_info["ratio"] > 0.50:
                tags.append("whole_frame_hallucination [heuristic]")
            else:
                tags.append("look_alike_object [heuristic]")
                
        for t in tags:
            tag_counts[f"FP_{t}"] += 1
            
        annotated = img.copy()
        px, py, pw, ph = [int(v) for v in fp["bbox"]]
        cname = CLASS_NAMES[fp["category_id"]]
        cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
        cv2.putText(annotated, f"PRED FP: {cname} {fp['score']:.2f}", (px, max(20, py - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
        # Draw true GT boxes in green if any
        for gt in gt_by_img[fp["image_id"]]:
            gx, gy, gw, gh = [int(v) for v in gt["bbox"]]
            cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            cv2.putText(annotated, f"GT: {CLASS_NAMES[gt['cid']]}", (gx, max(20, gy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
        header = f"[FP #{rank}] {cname.upper()} | Conf: {fp['score']:.3f} | Tags: {','.join(tags)} | Area: {char_info['area']}px"
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(annotated, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        save_p = fp_dir / f"fp_{rank:02d}_{img_info['file_name']}"
        cv2.imwrite(str(save_p), annotated)
        gallery_records.append({
            "split": split_name,
            "type": "FP",
            "rank": rank,
            "file_name": img_info["file_name"],
            "class": cname,
            "score": fp["score"],
            "tags": ";".join(tags),
            "area": char_info["area"],
            "saved_path": str(save_p),
        })
        
    return {
        "total_fn": len(false_negatives),
        "total_fp": len(false_positives),
        "saved_fn": len(fn_samples),
        "saved_fp": len(fp_samples),
        "tag_counts": dict(tag_counts),
        "records": gallery_records,
        "methodology": "All failure tags are strictly heuristic rules evaluated on box geometry (area <1024px or <1% frame), Laplacian blur variance (<100), HSV luminance (<60), near-border proximity, and cross-class IoU >= 0.30, verified visually."
    }


def export_predictions_csv(detections: List[Dict[str, Any]], image_map: Dict[int, Dict[str, Any]],
                           output_path: Path, split_name: str = "eval"):
    """Outputs predictions.csv with split, image_id, filename, source, class_name, conf, x1, y1, x2, y2."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    
    with open(output_path, "a" if not write_header else "w", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["split", "image_id", "filename", "source", "class_name", "conf", "x1", "y1", "x2", "y2"])
        for dt in detections:
            img_info = image_map.get(dt["image_id"], {})
            cid = dt["category_id"]
            cname = CLASS_NAMES.get(cid, str(cid))
            bx, by, bw, bh = dt["bbox"]
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            writer.writerow([
                split_name,
                dt["image_id"],
                img_info.get("file_name", "unknown"),
                img_info.get("source", "unknown"),
                cname,
                f"{dt['score']:.4f}",
                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
            ])


def run_full_evaluation(weights_path: Path, data_root: Path, splits: List[str],
                        model_tag: str = "v1", imgsz: int = 960, fixed_conf: float = 0.25,
                        iou_op: float = 0.50, output_dir: Optional[Path] = None,
                        device: str = "cuda:0") -> Dict[str, Any]:
    """Runs end-to-end evaluation with valid-only threshold tuning and error galleries."""
    if output_dir is None:
        output_dir = Path("outputs/eval") / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"STAGE 1 BASELINE EVALUATION — MODEL [{model_tag}]: {weights_path}")
    print(f"Data Root: {data_root} | ImgSz: {imgsz} | Fixed Baseline Conf: {fixed_conf} | IoU: {iou_op}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)
    
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    
    manifest_path = data_root / "manifest.csv"
    manifest_dict = load_manifest(manifest_path)
    
    pred_csv_path = output_dir / "predictions.csv"
    if pred_csv_path.exists():
        pred_csv_path.unlink()
        
    split_data = {}
    
    # 1. Ingest and run inference on standard splits
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"[WARN] Split directory {split_dir} does not exist. Skipping.")
            continue
            
        print(f"\n[1/4] Ingesting & Running Inference on Split: {split} ...")
        coco_gt_dict = build_coco_gt_dict(split_dir, manifest_dict)
        image_name_to_id = {img["file_name"]: img["id"] for img in coco_gt_dict["images"]}
        image_map = {img["id"]: img for img in coco_gt_dict["images"]}
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
        img_files = sorted([p for p in (split_dir / "images").iterdir() if p.is_file() and p.suffix in valid_exts])
        
        detections = run_yolo_inference(
            model=model,
            img_paths=img_files,
            image_name_to_id=image_name_to_id,
            imgsz=imgsz,
            conf=0.01,
            iou=iou_op,
            device=device
        )
        print(f"      Images: {len(coco_gt_dict['images'])} | GT Annotations: {len(coco_gt_dict['annotations'])} | Detections: {len(detections)}")
        export_predictions_csv(detections, image_map, pred_csv_path, split_name=split)
        
        split_data[split] = {
            "coco_gt_dict": coco_gt_dict,
            "detections": detections,
            "image_map": image_map,
        }
        
    # Also create sub-splits for test_cctv_v1: innovatus_rob and bits_pilani
    if "test_cctv_v1" in split_data:
        cctv_full = split_data["test_cctv_v1"]
        cctv_gt = cctv_full["coco_gt_dict"]
        cctv_dt = cctv_full["detections"]
        
        for sub_src, sub_key in [("innovatus_rob", "test_cctv_v1_innovatus"), ("bits_pilani", "test_cctv_v1_bits")]:
            sub_imgs = [img for img in cctv_gt["images"] if img.get("source") == sub_src]
            sub_img_ids = set(img["id"] for img in sub_imgs)
            sub_anns = [ann for ann in cctv_gt["annotations"] if ann["image_id"] in sub_img_ids]
            sub_dts = [dt for dt in cctv_dt if dt["image_id"] in sub_img_ids]
            sub_map = {img["id"]: img for img in sub_imgs}
            
            split_data[sub_key] = {
                "coco_gt_dict": {"images": sub_imgs, "annotations": sub_anns, "categories": cctv_gt["categories"]},
                "detections": sub_dts,
                "image_map": sub_map,
            }
            print(f"      Created sub-split {sub_key}: {len(sub_imgs)} images, {len(sub_anns)} GT annotations, {len(sub_dts)} detections")
            
    # Step 2: Determine operating confidence on VALID ONLY
    print("\n[2/4] Sweeping Confidence on VALID ONLY to Determine Operational Threshold (argmax F1)...")
    valid_gt = split_data["valid"]["coco_gt_dict"]
    valid_dt = split_data["valid"]["detections"]
    conf_star, max_f1, sweep_df = sweep_valid_operating_point(valid_gt, valid_dt, iou_thresh=iou_op)
    sweep_df.to_csv(output_dir / "valid_confidence_sweep.csv", index=False)
    print(f"      ---> Optimal Valid Operational Confidence (conf*): {conf_star:.3f} (Max F1: {max_f1:.4f})")
    
    # Step 3: Compute evaluation metrics for all splits at both conf* and fixed_conf
    print(f"\n[3/4] Computing Standard COCO, Per-Class, Per-Source, Bucket Recall, and FPPI Metrics...")
    full_eval_results = {
        "model_tag": model_tag,
        "weights_path": str(weights_path),
        "conf_star": conf_star,
        "fixed_conf": fixed_conf,
        "splits": {},
    }
    
    eval_splits = ["valid", "test_public", "test_domain", "test_cctv_v1", "test_cctv_v1_innovatus", "test_cctv_v1_bits", "test_hard_negatives"]
    summary_rows = []
    
    for split in eval_splits:
        if split not in split_data:
            continue
            
        gt_dict = split_data[split]["coco_gt_dict"]
        dt_list = split_data[split]["detections"]
        
        n_imgs = len(gt_dict["images"])
        n_anns = len(gt_dict["annotations"])
        
        if split == "test_hard_negatives":
            # Weapon-free negatives: compute FPPI exclusively
            fppi_star, n_fp_star, by_cls_star = compute_fppi(dt_list, n_imgs, conf_threshold=conf_star)
            fppi_fix, n_fp_fix, by_cls_fix = compute_fppi(dt_list, n_imgs, conf_threshold=fixed_conf)
            
            full_eval_results["splits"][split] = {
                "n_images": n_imgs,
                "n_gt": 0,
                "fppi_conf_star": {"fppi": fppi_star, "total_fps": n_fp_star, "by_class": by_cls_star},
                "fppi_fixed_025": {"fppi": fppi_fix, "total_fps": n_fp_fix, "by_class": by_cls_fix},
            }
            summary_rows.append({
                "split": split,
                "n_images": n_imgs,
                "n_gt": 0,
                "mAP50": "-",
                "mAP50-95": "-",
                "P@conf*": "-", "R@conf*": "-", "F1@conf*": "-",
                "P@0.25": "-", "R@0.25": "-", "F1@0.25": "-",
                "FPPI@conf*": f"{fppi_star:.4f} ({n_fp_star} FPs)",
                "FPPI@0.25": f"{fppi_fix:.4f} ({n_fp_fix} FPs)",
            })
            continue
            
        # COCO standard evaluation
        c_gt = COCO()
        c_gt.dataset = gt_dict
        c_gt.createIndex()
        
        if len(dt_list) > 0 and n_anns > 0:
            c_dt = c_gt.loadRes(dt_list)
            coco_res = evaluate_coco_metrics(c_gt, c_dt)
        else:
            coco_res = {"mAP50": 0.0, "mAP50_95": 0.0, "mAP75": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0, "per_class": {}}
            
        # Operating point metrics
        op_star = compute_operating_point_metrics(gt_dict, dt_list, conf_thresh=conf_star, iou_thresh=iou_op)
        op_fix = compute_operating_point_metrics(gt_dict, dt_list, conf_thresh=fixed_conf, iou_thresh=iou_op)
        
        # Grouped Bootstrap Confidence Intervals
        ci_res = None
        ci_str = ""
        if split in ["valid", "test_public", "test_domain", "test_cctv_v1_innovatus"]:
            ci_res = bootstrap_ci_grouped(gt_dict, dt_list, n_bootstraps=100, ci=0.95)
            ci_str = f" [{ci_res['mAP50'][0]:.3f}-{ci_res['mAP50'][1]:.3f}]"
            print(f"      {split:22s} Bootstrap 95% CI mAP50: {ci_res['mAP50']} | mAP50-95: {ci_res['mAP50_95']}")
            
        # Recall by box-size bucket (<1%, 1-5%, 5-25%, >25%)
        bucket_recall_star = compute_recall_by_box_size_bucket(gt_dict, dt_list, conf_thresh=conf_star, iou_thresh=iou_op)
        bucket_recall_fix = compute_recall_by_box_size_bucket(gt_dict, dt_list, conf_thresh=fixed_conf, iou_thresh=iou_op)
        
        # Diagnostic: Large box removal (>50% frame area)
        large_box_diag = compute_large_box_diagnostics(gt_dict, dt_list, conf_star=conf_star, fixed_conf=fixed_conf)
        
        # Per-source metrics breakdown
        sources = sorted(list(set(img.get("source", "unknown") for img in gt_dict["images"])))
        per_source_res = {}
        if len(sources) > 1:
            for src in sources:
                src_imgs = [img for img in gt_dict["images"] if img.get("source") == src]
                src_img_ids = set(img["id"] for img in src_imgs)
                src_anns = [ann for ann in gt_dict["annotations"] if ann["image_id"] in src_img_ids]
                src_dts = [dt for dt in dt_list if dt["image_id"] in src_img_ids]
                
                if len(src_anns) > 0 and len(src_dts) > 0:
                    src_sub_gt = {"images": src_imgs, "annotations": src_anns, "categories": gt_dict["categories"]}
                    c_src_gt = COCO()
                    c_src_gt.dataset = src_sub_gt
                    c_src_gt.createIndex()
                    c_src_dt = c_src_gt.loadRes(src_dts)
                    s_eval = evaluate_coco_metrics(c_src_gt, c_src_dt)
                    per_source_res[src] = {"n_images": len(src_imgs), "n_gt": len(src_anns), "mAP50": s_eval["mAP50"], "mAP50_95": s_eval["mAP50_95"]}
                else:
                    per_source_res[src] = {"n_images": len(src_imgs), "n_gt": len(src_anns), "mAP50": 0.0, "mAP50_95": 0.0}
                    
        full_eval_results["splits"][split] = {
            "n_images": n_imgs,
            "n_gt": n_anns,
            "coco_metrics": coco_res,
            "op_metrics_conf_star": op_star,
            "op_metrics_fixed_025": op_fix,
            "bootstrap_ci": ci_res,
            "bucket_recall_conf_star": bucket_recall_star,
            "bucket_recall_fixed_025": bucket_recall_fix,
            "large_box_diagnostics": large_box_diag,
            "per_source": per_source_res,
        }
        
        summary_rows.append({
            "split": split,
            "n_images": n_imgs,
            "n_gt": n_anns,
            "mAP50": f"{coco_res['mAP50']:.4f}{ci_str}",
            "mAP50-95": f"{coco_res['mAP50_95']:.4f}",
            "P@conf*": f"{op_star['overall']['precision']:.4f}",
            "R@conf*": f"{op_star['overall']['recall']:.4f}",
            "F1@conf*": f"{op_star['overall']['f1']:.4f}",
            "P@0.25": f"{op_fix['overall']['precision']:.4f}",
            "R@0.25": f"{op_fix['overall']['recall']:.4f}",
            "F1@0.25": f"{op_fix['overall']['f1']:.4f}",
            "FPPI@conf*": "-",
            "FPPI@0.25": "-",
        })
        
    # Step 4: Error galleries for test_domain and test_cctv_v1_innovatus
    print(f"\n[4/4] Generating Error Galleries (Top 30 FNs, Top 30 FPs) for test_domain & Innovatus...")
    all_gallery_records = []
    error_summary = {}
    
    gallery_targets = [("test_domain", "test_domain"), ("test_cctv_v1_innovatus", "innovatus_rob")]
    for eval_key, gallery_name in gallery_targets:
        if eval_key in split_data:
            g_res = generate_error_galleries(
                split_name=gallery_name,
                coco_gt_dict=split_data[eval_key]["coco_gt_dict"],
                detections=split_data[eval_key]["detections"],
                conf_thresh=conf_star,
                output_dir=output_dir,
                max_samples=30
            )
            error_summary[gallery_name] = g_res
            all_gallery_records.extend(g_res["records"])
            print(f"      {gallery_name:15s}: Saved {g_res['saved_fn']} FNs and {g_res['saved_fp']} FPs to {output_dir}/error_galleries/{gallery_name}/")
            
    pd.DataFrame(all_gallery_records).to_csv(output_dir / "error_gallery_manifest.csv", index=False)
    with open(output_dir / "error_analysis_summary.json", "w") as f:
        json.dump(error_summary, f, indent=2)
        
    # Save full metrics JSON
    metrics_file = output_dir / f"baseline_m960_{model_tag}_eval_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(full_eval_results, f, indent=2)
        
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print(f"STAGE 1 BASELINE EVALUATION SUMMARY TABLE — MODEL [{model_tag}]")
    print(f"Selected Operating Confidence: conf* = {conf_star:.3f} | Fixed Baseline: conf = {fixed_conf:.2f}")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)
    
    return full_eval_results


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Baseline Weapon Detector Evaluation Harness")
    parser.add_argument("--weights", type=Path, default=Path("outputs/runs/baseline_m960_v2/weights/best.pt"), help="Path to best.pt")
    parser.add_argument("--model-tag", type=str, default="v2", help="Model version tag (e.g. v1, v2)")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/stage1_baseline"), help="Path to processed dataset")
    parser.add_argument("--splits", nargs="+", default=["valid", "test_public", "test_domain", "test_cctv_v1", "test_hard_negatives"], help="Splits to evaluate")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image resolution")
    parser.add_argument("--fixed-conf", type=float, default=0.25, help="Fixed comparison confidence threshold")
    parser.add_argument("--iou-op", type=float, default=0.50, help="IoU threshold for NMS and matching")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for eval results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    args = parser.parse_args()
    
    run_full_evaluation(
        weights_path=args.weights,
        data_root=args.data_root,
        splits=args.splits,
        model_tag=args.model_tag,
        imgsz=args.imgsz,
        fixed_conf=args.fixed_conf,
        iou_op=args.iou_op,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

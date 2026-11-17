"""
scripts/eval/run_tiled_inference.py
===================================
Tiled / Sliced (SAHI-style) inference for baseline_m960_v2:
1. Slices: 640x640 with 20% overlap (stride 512)
2. Plus full-image pass at imgsz 960
3. Coordinate restoration and merged NMS (IoU 0.50)
4. Evaluates on:
   - Innovatus CCTV (test_cctv_v1_innovatus)
   - test_domain
   - test_public
   - test_hard_negatives
5. Reports comparison with full-image inference: Recall, AP, FPPI, runtime per image.
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.eval.evaluate_detector import (
    CLASS_NAMES,
    build_coco_gt_dict,
    compute_fppi,
    compute_operating_point_metrics,
    evaluate_coco_metrics,
    load_manifest,
)


def get_slice_boxes(img_w: int, img_h: int, slice_w: int = 640, slice_h: int = 640,
                    overlap_ratio: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """Calculates grid of (x1, y1, x2, y2) slice coordinates covering the image."""
    if img_w <= slice_w and img_h <= slice_h:
        return [(0, 0, img_w, img_h)]
        
    stride_w = int(slice_w * (1.0 - overlap_ratio))
    stride_h = int(slice_h * (1.0 - overlap_ratio))
    
    xs = list(range(0, max(1, img_w - slice_w + 1), stride_w))
    if len(xs) == 0 or xs[-1] + slice_w < img_w:
        xs.append(max(0, img_w - slice_w))
    xs = sorted(list(set(xs)))
    
    ys = list(range(0, max(1, img_h - slice_h + 1), stride_h))
    if len(ys) == 0 or ys[-1] + slice_h < img_h:
        ys.append(max(0, img_h - slice_h))
    ys = sorted(list(set(ys)))
    
    slices = []
    for y in ys:
        for x in xs:
            slices.append((x, y, min(img_w, x + slice_w), min(img_h, y + slice_h)))
    return slices


def run_tiled_inference_single_image(
    model: Any,
    img: np.ndarray,
    slice_w: int = 640,
    slice_h: int = 640,
    overlap_ratio: float = 0.20,
    conf_thresh: float = 0.01,
    nms_iou: float = 0.50,
    device: str = "cuda:0",
) -> List[Dict[str, Any]]:
    """Runs tiled inference + full image pass on a single image and merges detections with batched NMS."""
    h, w = img.shape[:2]
    slice_coords = get_slice_boxes(w, h, slice_w=slice_w, slice_h=slice_h, overlap_ratio=overlap_ratio)
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    # 1. Full-image pass
    full_res = model.predict(source=img, imgsz=960, conf=conf_thresh, device=device, verbose=False)
    if full_res and len(full_res[0].boxes) > 0:
        fb = full_res[0].boxes
        xyxy = fb.xyxy.cpu().numpy()
        confs = fb.conf.cpu().numpy()
        clss = fb.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), sc, cid in zip(xyxy, confs, clss):
            all_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            all_scores.append(float(sc))
            all_classes.append(int(cid))
            
    # 2. Slices pass (batched)
    if len(slice_coords) > 1 or (w > slice_w or h > slice_h):
        crops = []
        valid_coords = []
        for (sx1, sy1, sx2, sy2) in slice_coords:
            crop = img[sy1:sy2, sx1:sx2]
            if crop.size == 0:
                continue
            # Pad crop to 640x640 if on boundary
            ch, cw = crop.shape[:2]
            if ch < slice_h or cw < slice_w:
                pad_crop = np.zeros((slice_h, slice_w, 3), dtype=np.uint8)
                pad_crop[:ch, :cw] = crop
                crops.append(pad_crop)
            else:
                crops.append(crop)
            valid_coords.append((sx1, sy1))
            
        if crops:
            slice_results = model.predict(source=crops, imgsz=slice_w, conf=conf_thresh, device=device, verbose=False, batch=len(crops))
            for (sx1, sy1), sr in zip(valid_coords, slice_results):
                if len(sr.boxes) > 0:
                    s_xyxy = sr.boxes.xyxy.cpu().numpy()
                    s_confs = sr.boxes.conf.cpu().numpy()
                    s_clss = sr.boxes.cls.cpu().numpy().astype(int)
                    for (bx1, by1, bx2, by2), sc, cid in zip(s_xyxy, s_confs, s_clss):
                        # Map back to full image coordinates
                        abs_x1 = max(0.0, min(float(w), bx1 + sx1))
                        abs_y1 = max(0.0, min(float(h), by1 + sy1))
                        abs_x2 = max(0.0, min(float(w), bx2 + sx1))
                        abs_y2 = max(0.0, min(float(h), by2 + sy1))
                        if abs_x2 > abs_x1 and abs_y2 > abs_y1:
                            all_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
                            all_scores.append(float(sc))
                            all_classes.append(int(cid))
                            
    if not all_boxes:
        return []
        
    # 3. Batched NMS
    t_boxes = torch.tensor(all_boxes, dtype=torch.float32)
    t_scores = torch.tensor(all_scores, dtype=torch.float32)
    t_classes = torch.tensor(all_classes, dtype=torch.int64)
    
    keep_indices = torchvision.ops.batched_nms(t_boxes, t_scores, t_classes, iou_threshold=nms_iou)
    keep_indices = keep_indices.cpu().numpy()
    
    merged_dets = []
    for idx in keep_indices:
        b = all_boxes[idx]
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        merged_dets.append({
            "category_id": all_classes[idx],
            "bbox": [round(b[0], 2), round(b[1], 2), round(bw, 2), round(bh, 2)],
            "score": round(all_scores[idx], 4),
        })
    return merged_dets


def evaluate_split_tiled(
    model: Any,
    split_dir: Path,
    manifest_dict: dict,
    filter_source: str | None = None,
    conf_star: float = 0.300,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Runs tiled evaluation on a given split directory."""
    gt_dict = build_coco_gt_dict(split_dir, manifest_dict, filter_source=filter_source)
    img_map = {img["file_name"]: img for img in gt_dict["images"]}
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    img_files = sorted([p for p in (split_dir / "images").iterdir() if p.is_file() and p.suffix in valid_exts and p.name in img_map])
    
    all_detections = []
    runtimes = []
    
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy, imgsz=640, device=device, verbose=False)
    
    for p in img_files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_id = img_map[p.name]["id"]
        
        t0 = time.perf_counter()
        dets = run_tiled_inference_single_image(model, img, slice_w=640, slice_h=640, overlap_ratio=0.20,
                                               conf_thresh=0.01, nms_iou=0.50, device=device)
        t1 = time.perf_counter()
        runtimes.append((t1 - t0) * 1000.0)  # ms
        
        for d in dets:
            d_copy = dict(d)
            d_copy["image_id"] = img_id
            all_detections.append(d_copy)
            
    avg_runtime_ms = float(np.mean(runtimes)) if runtimes else 0.0
    
    # Metrics
    if len(gt_dict["annotations"]) == 0:
        fppi_star, n_fp_star, by_cls_star = compute_fppi(all_detections, len(gt_dict["images"]), conf_threshold=conf_star)
        fppi_025, n_fp_025, by_cls_025 = compute_fppi(all_detections, len(gt_dict["images"]), conf_threshold=0.25)
        return {
            "n_images": len(gt_dict["images"]),
            "n_gt": 0,
            "avg_runtime_ms": round(avg_runtime_ms, 2),
            "fppi_conf_star": fppi_star,
            "fps_conf_star": n_fp_star,
            "fppi_025": fppi_025,
            "fps_025": n_fp_025,
            "by_class_conf_star": by_cls_star,
            "detections": all_detections,
        }
        
    c_gt = COCO()
    c_gt.dataset = gt_dict
    c_gt.createIndex()
    
    if all_detections:
        c_dt = c_gt.loadRes(all_detections)
        coco_res = evaluate_coco_metrics(c_gt, c_dt)
    else:
        coco_res = {"mAP50": 0.0, "mAP50_95": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0, "per_class": {}}
        
    op_star = compute_operating_point_metrics(gt_dict, all_detections, conf_thresh=conf_star)["overall"]
    op_025 = compute_operating_point_metrics(gt_dict, all_detections, conf_thresh=0.25)["overall"]
    
    return {
        "n_images": len(gt_dict["images"]),
        "n_gt": len(gt_dict["annotations"]),
        "avg_runtime_ms": round(avg_runtime_ms, 2),
        "coco_metrics": coco_res,
        "op_conf_star": op_star,
        "op_025": op_025,
        "detections": all_detections,
    }


def main():
    data_root = Path("data/processed/stage1_baseline")
    manifest_dict = load_manifest(data_root / "manifest.csv")
    
    weights_path = Path("outputs/runs/baseline_m960_v2/weights/best.pt")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    
    out_dir = Path("outputs/eval/v2_tiled")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    targets = [
        ("Innovatus CCTV", "test_cctv_v1", "innovatus_rob"),
        ("test_domain", "test_domain", None),
        ("test_public", "test_public", None),
        ("test_hard_negatives", "test_hard_negatives", None),
    ]
    
    results = {}
    print("=" * 80)
    print(f"TILED / SAHI INFERENCE EVALUATION — MODEL: {weights_path}")
    print("Parameters: 640x640 slices with 20% overlap (stride 512) + full-image 960 pass, NMS IoU=0.50")
    print("=" * 80)
    
    all_preds_records = []
    
    for display_name, split_name, filter_source in targets:
        print(f"\nProcessing {display_name} ...")
        split_dir = data_root / split_name
        res = evaluate_split_tiled(
            model=model,
            split_dir=split_dir,
            manifest_dict=manifest_dict,
            filter_source=filter_source,
            conf_star=0.300,
            device="cuda:0",
        )
        dets = res.pop("detections")
        results[display_name] = res
        
        # Format preds
        gt_dict = build_coco_gt_dict(split_dir, manifest_dict, filter_source=filter_source)
        id_to_name = {img["id"]: img["file_name"] for img in gt_dict["images"]}
        for d in dets:
            fname = id_to_name.get(d["image_id"], "unknown")
            all_preds_records.append({
                "split": split_name,
                "filename": fname,
                "class_name": CLASS_NAMES.get(d["category_id"], str(d["category_id"])),
                "conf": d["score"],
                "x1": d["bbox"][0],
                "y1": d["bbox"][1],
                "x2": d["bbox"][0] + d["bbox"][2],
                "y2": d["bbox"][1] + d["bbox"][3],
            })
            
        if split_name == "test_hard_negatives":
            print(f"  {display_name:18s} | FPPI @ 0.300: {res['fppi_conf_star']:.4f} ({res['fps_conf_star']} FPs) | Runtime: {res['avg_runtime_ms']:.1f} ms/img")
        else:
            cm = res["coco_metrics"]
            op = res["op_conf_star"]
            print(f"  {display_name:18s} | mAP50: {cm['mAP50']:.4f} | mAP50-95: {cm['mAP50_95']:.4f} | AP_s: {cm['AP_small']:.4f} | R@0.30: {op['recall']:.4f} | F1@0.30: {op['f1']:.4f} | Runtime: {res['avg_runtime_ms']:.1f} ms/img")
            
    with open(out_dir / "tiled_eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(all_preds_records).to_csv(out_dir / "predictions_tiled.csv", index=False)
    print(f"\nAll tiled inference metrics saved to {out_dir}/tiled_eval_metrics.json")


if __name__ == "__main__":
    main()

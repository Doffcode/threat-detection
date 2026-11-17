"""
scripts/eval/compute_innovatus_clean_eval.py
============================================
Computes:
1. Innovatus-only evaluation (72 frames, 75 boxes) for v1-v4 (full-image and SAHI).
2. BITS Pilani separate evaluation (20 frames, 13 boxes).
3. Shared-resample block bootstrap (B=1000) for Innovatus with paired deltas (v3-v2, v4-v2).
"""

from __future__ import annotations

import collections
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO

CLASS_NAMES = {0: "handgun", 1: "rifle", 2: "knife"}


def get_sahi_slices(img_w: int, img_h: int, slice_w: int = 640, slice_h: int = 640, overlap: float = 0.20):
    if img_w <= slice_w and img_h <= slice_h:
        return [(0, 0, img_w, img_h)]
    stride_w = int(slice_w * (1.0 - overlap))
    stride_h = int(slice_h * (1.0 - overlap))
    xs = sorted(list(set(list(range(0, max(1, img_w - slice_w + 1), stride_w)) + [max(0, img_w - slice_w)])))
    ys = sorted(list(set(list(range(0, max(1, img_h - slice_h + 1), stride_h)) + [max(0, img_h - slice_h)])))
    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, min(img_w, x + slice_w), min(img_h, y + slice_h)))
    return boxes


def build_coco_for_source(source_filter: str):
    images_dir = WORKSPACE_ROOT / "data/cctv_test_v2_candidates/images"
    labels_dir = WORKSPACE_ROOT / "data/cctv_test_v2_candidates/labels"
    csv_path = WORKSPACE_ROOT / "configs/detection/cctv_test_v2_occlusion_template.csv"
    df = pd.read_csv(csv_path)

    df_subset = df[df["source"] == source_filter].reset_index(drop=True)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "handgun"},
            {"id": 1, "name": "rifle"},
            {"id": 2, "name": "knife"},
        ],
    }
    records = []
    ann_id = 1

    for idx, row in df_subset.iterrows():
        img_id = idx + 1
        fn = row["image_filename"]
        img_p = images_dir / fn
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        record = {
            "id": img_id,
            "file_name": fn,
            "width": w,
            "height": h,
            "source": row["source"],
            "sequence_id": str(row.get("sequence_id", "none")),
            "path": str(img_p),
        }
        coco_data["images"].append(record)
        records.append(record)

        lbl_p = labels_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            lines = [l.strip() for l in lbl_p.read_text().splitlines() if l.strip()]
            for l in lines:
                parts = l.split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    abs_w = bw * w
                    abs_h = bh * h
                    abs_x1 = (cx - bw / 2.0) * w
                    abs_y1 = (cy - bh / 2.0) * h

                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cid,
                        "bbox": [round(abs_x1, 2), round(abs_y1, 2), round(abs_w, 2), round(abs_h, 2)],
                        "area": round(abs_w * abs_h, 2),
                        "norm_w": bw,
                        "norm_h": bh,
                        "norm_area": bw * bh,
                        "iscrowd": 0,
                    })
                    ann_id += 1

    return coco_data, records


def run_inference(weights_p: Path, records: List[Dict], imgsz: int, mode: str):
    torch.cuda.empty_cache()
    model = YOLO(str(weights_p))
    detections = []

    if mode == "full":
        batch_size = 8
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            results = model.predict(source=[r["path"] for r in batch], imgsz=imgsz, conf=0.01, iou=0.50, device="cuda:0", verbose=False)
            for r, res in zip(batch, results):
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for box, score, cid in zip(xyxy, confs, classes):
                    if cid in CLASS_NAMES:
                        x1, y1, x2, y2 = box
                        detections.append({
                            "image_id": r["id"],
                            "category_id": int(cid),
                            "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                            "score": float(score),
                        })
    else:
        # SAHI Tiled
        for r in records:
            img_id = r["id"]
            img_bgr = cv2.imread(r["path"])
            h, w = img_bgr.shape[:2]

            res_full = model.predict(source=img_bgr, imgsz=imgsz, conf=0.01, iou=0.50, device="cuda:0", verbose=False)[0]
            boxes_full = res_full.boxes

            cand_boxes, cand_scores, cand_classes = [], [], []
            if boxes_full is not None and len(boxes_full) > 0:
                cand_boxes.append(boxes_full.xyxy.cpu())
                cand_scores.append(boxes_full.conf.cpu())
                cand_classes.append(boxes_full.cls.cpu())

            slices = get_sahi_slices(w, h, slice_w=640, slice_h=640, overlap=0.20)
            crop_imgs = [img_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in slices]

            if crop_imgs:
                res_crops = model.predict(source=crop_imgs, imgsz=640, conf=0.01, iou=0.50, device="cuda:0", verbose=False)
                for (sx1, sy1, sx2, sy2), rc in zip(slices, res_crops):
                    b = rc.boxes
                    if b is not None and len(b) > 0:
                        c_xyxy = b.xyxy.cpu().clone()
                        c_xyxy[:, 0] += sx1
                        c_xyxy[:, 2] += sx1
                        c_xyxy[:, 1] += sy1
                        c_xyxy[:, 3] += sy1
                        cand_boxes.append(c_xyxy)
                        cand_scores.append(b.conf.cpu())
                        cand_classes.append(b.cls.cpu())

            if cand_boxes:
                all_b = torch.cat(cand_boxes, dim=0)
                all_s = torch.cat(cand_scores, dim=0)
                all_c = torch.cat(cand_classes, dim=0)
                keep = torchvision.ops.batched_nms(all_b, all_s, all_c, iou_threshold=0.50)
                keep_b = all_b[keep].numpy()
                keep_s = all_s[keep].numpy()
                keep_c = all_c[keep].numpy().astype(int)

                for box, score, cid in zip(keep_b, keep_s, keep_c):
                    if cid in CLASS_NAMES:
                        x1, y1, x2, y2 = box
                        detections.append({
                            "image_id": img_id,
                            "category_id": int(cid),
                            "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                            "score": float(score),
                        })

    return detections


def evaluate_subset(coco_gt_dict: Dict, detections: List[Dict], conf_star: float):
    gt_coco = COCO()
    gt_coco.dataset = coco_gt_dict
    gt_coco.createIndex()

    mAP50, mAP50_95 = 0.0, 0.0
    valid_img_ids = set(gt_coco.getImgIds())
    filtered_detections = [d for d in detections if d["image_id"] in valid_img_ids]

    if len(filtered_detections) > 0 and len(coco_gt_dict["annotations"]) > 0:
        pred_coco = gt_coco.loadRes(filtered_detections)
        ev = COCOeval(gt_coco, pred_coco, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        mAP50_95 = float(ev.stats[0])
        mAP50 = float(ev.stats[1])

    filtered = [d for d in filtered_detections if d["score"] >= conf_star]
    gt_by_img = {}
    for a in coco_gt_dict["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)
    preds_by_img = {}
    for p in filtered:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    tp, fp = 0, 0
    gt_matched = set()

    for img_id in valid_img_ids:
        gts = gt_by_img.get(img_id, [])
        preds = sorted(preds_by_img.get(img_id, []), key=lambda x: x["score"], reverse=True)
        for p in preds:
            px, py, pw, ph = p["bbox"]
            p_x2, p_y2 = px + pw, py + ph
            best_iou, best_gid = 0.0, -1
            for g in gts:
                if g["id"] in gt_matched or g["category_id"] != p["category_id"]:
                    continue
                gx, gy, gw, gh = g["bbox"]
                ix1, iy1 = max(px, gx), max(py, gy)
                ix2, iy2 = min(p_x2, gx + gw), min(p_y2, gy + gh)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                union = (pw * ph) + (gw * gh) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gid = g["id"]
            if best_iou >= 0.50 and best_gid != -1:
                tp += 1
                gt_matched.add(best_gid)
            else:
                fp += 1

    total_gt = len(coco_gt_dict["annotations"])
    fn = total_gt - len(gt_matched)
    p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r_val = tp / total_gt if total_gt > 0 else 0.0
    f1_val = (2 * p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0.0

    bucket_counts = {"<1%": 0, "1-5%": 0, "5-25%": 0, ">25%": 0}
    bucket_hits = {"<1%": 0, "1-5%": 0, "5-25%": 0, ">25%": 0}

    for a in coco_gt_dict["annotations"]:
        area_pct = a.get("norm_area", 0.0) * 100
        if area_pct < 1.0:
            bkey = "<1%"
        elif area_pct <= 5.0:
            bkey = "1-5%"
        elif area_pct <= 25.0:
            bkey = "5-25%"
        else:
            bkey = ">25%"
        bucket_counts[bkey] += 1
        if a["id"] in gt_matched:
            bucket_hits[bkey] += 1

    bucket_recalls = {
        k: (bucket_hits[k] / bucket_counts[k] if bucket_counts[k] > 0 else 0.0)
        for k in bucket_counts
    }

    n_imgs = len(coco_gt_dict["images"])
    fppi = fp / n_imgs if n_imgs > 0 else 0.0

    return {
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "precision": round(p_val, 4),
        "recall": round(r_val, 4),
        "f1": round(f1_val, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fppi": round(fppi, 4),
        "bucket_recalls": bucket_recalls,
        "bucket_counts": bucket_counts,
        "bucket_hits": bucket_hits,
        "total_gt": total_gt,
        "n_images": n_imgs,
    }


def main():
    print("=" * 80)
    print("INNOVATUS CCTV CLEAN EVALUATION & SHARED BLOCK BOOTSTRAP")
    print("=" * 80)

    # 1. Build Datasets
    coco_inno, records_inno = build_coco_for_source("innovatus_rob")
    coco_bits, records_bits = build_coco_for_source("bits_pilani")

    print(f"Innovatus Dataset: {len(coco_inno['images'])} images, {len(coco_inno['annotations'])} boxes")
    print(f"BITS Pilani Dataset: {len(coco_bits['images'])} images, {len(coco_bits['annotations'])} boxes")

    models = [
        ("v1", WORKSPACE_ROOT / "outputs/runs/baseline_m960_v1/weights/best.pt", 960, 0.28),
        ("v2", WORKSPACE_ROOT / "outputs/runs/baseline_m960_v2/weights/best.pt", 960, 0.30),
        ("v3", WORKSPACE_ROOT / "outputs/runs/baseline_m960_v3/weights/best.pt", 960, 0.27),
        ("v4", WORKSPACE_ROOT / "outputs/runs/baseline_m960_v4/weights/best.pt", 1280, 0.26),
    ]

    preds_inno_full = {}
    preds_inno_sahi = {}
    preds_bits_full = {}
    preds_bits_sahi = {}

    metrics_inno_full = {}
    metrics_inno_sahi = {}
    metrics_bits_full = {}
    metrics_bits_sahi = {}

    for tag, wpath, imgsz, conf_star in models:
        print(f"\nRunning inference for {tag} (conf*={conf_star}, imgsz={imgsz})...")
        # Innovatus Full & SAHI
        preds_inno_full[tag] = run_inference(wpath, records_inno, imgsz, "full")
        preds_inno_sahi[tag] = run_inference(wpath, records_inno, imgsz, "sahi")
        metrics_inno_full[tag] = evaluate_subset(coco_inno, preds_inno_full[tag], conf_star)
        metrics_inno_sahi[tag] = evaluate_subset(coco_inno, preds_inno_sahi[tag], conf_star)

        # BITS Full & SAHI
        preds_bits_full[tag] = run_inference(wpath, records_bits, imgsz, "full")
        preds_bits_sahi[tag] = run_inference(wpath, records_bits, imgsz, "sahi")
        metrics_bits_full[tag] = evaluate_subset(coco_bits, preds_bits_full[tag], conf_star)
        metrics_bits_sahi[tag] = evaluate_subset(coco_bits, preds_bits_sahi[tag], conf_star)

    # 2. Shared-Resample Block Bootstrap on Innovatus (B=1000)
    print("\n--- Running Shared-Resample Block Bootstrap on Innovatus (B=1000) ---")
    seqs = collections.defaultdict(list)
    for r in records_inno:
        seqs[r["sequence_id"]].append(r)

    blocks = []
    # seq02 (20 frames): 2 blocks of 10
    blocks.append(seqs["innovatus_cam_b_seq02"][0:10])
    blocks.append(seqs["innovatus_cam_b_seq02"][10:20])
    # seq07 (26 frames): 3 blocks of 9, 9, 8
    blocks.append(seqs["innovatus_cam_b_seq07"][0:9])
    blocks.append(seqs["innovatus_cam_b_seq07"][9:18])
    blocks.append(seqs["innovatus_cam_b_seq07"][18:26])
    # seq08 (26 frames): 3 blocks of 9, 9, 8
    blocks.append(seqs["innovatus_cam_b_seq08"][0:9])
    blocks.append(seqs["innovatus_cam_b_seq08"][9:18])
    blocks.append(seqs["innovatus_cam_b_seq08"][18:26])

    print(f"Created {len(blocks)} blocks across 3 Innovatus sequences.")

    # Cache predictions and annotations by image_id
    img_preds_by_model = {tag: collections.defaultdict(list) for tag, _, _, _ in models}
    for tag in img_preds_by_model:
        for d in preds_inno_full[tag]:
            img_preds_by_model[tag][d["image_id"]].append(d)

    img_anns = collections.defaultdict(list)
    for a in coco_inno["annotations"]:
        img_anns[a["image_id"]].append(a)

    B = 1000
    np.random.seed(2026)
    boot_map50 = {tag: [] for tag, _, _, _ in models}
    boot_f1 = {tag: [] for tag, _, _, _ in models}

    for b_idx in range(B):
        sampled_block_indices = np.random.choice(len(blocks), size=len(blocks), replace=True)
        sampled_records = []
        for bidx in sampled_block_indices:
            sampled_records.extend(blocks[bidx])

        mini_gt = {"images": [], "annotations": [], "categories": coco_inno["categories"]}
        new_img_id = 1
        new_ann_id = 1
        model_mini_preds = {tag: [] for tag, _, _, _ in models}

        for r in sampled_records:
            old_id = r["id"]
            mini_gt["images"].append({"id": new_img_id, "width": r["width"], "height": r["height"]})
            for a in img_anns.get(old_id, []):
                new_a = copy.deepcopy(a)
                new_a["id"] = new_ann_id
                new_a["image_id"] = new_img_id
                mini_gt["annotations"].append(new_a)
                new_ann_id += 1
            for tag in models:
                t_name = tag[0]
                conf_val = tag[3]
                for p in img_preds_by_model[t_name].get(old_id, []):
                    new_p = copy.deepcopy(p)
                    new_p["image_id"] = new_img_id
                    model_mini_preds[t_name].append(new_p)
            new_img_id += 1

        for tag, _, _, conf_val in models:
            res = evaluate_subset(mini_gt, model_mini_preds[tag], conf_val)
            boot_map50[tag].append(res["mAP50"])
            boot_f1[tag].append(res["f1"])

    # Compute CIs
    bootstrap_results = {}
    for tag, _, _, _ in models:
        m50_arr = np.array(boot_map50[tag])
        f1_arr = np.array(boot_f1[tag])
        bootstrap_results[tag] = {
            "mAP50_ci": (round(float(np.percentile(m50_arr, 2.5)), 4), round(float(np.percentile(m50_arr, 97.5)), 4)),
            "f1_ci": (round(float(np.percentile(f1_arr, 2.5)), 4), round(float(np.percentile(f1_arr, 97.5)), 4)),
        }

    # Paired Deltas vs v2
    v2_m50 = np.array(boot_map50["v2"])
    v2_f1 = np.array(boot_f1["v2"])

    paired_deltas = {}
    for comp in ["v3", "v4"]:
        d_m50 = np.array(boot_map50[comp]) - v2_m50
        d_f1 = np.array(boot_f1[comp]) - v2_f1
        paired_deltas[f"{comp}-v2"] = {
            "delta_mAP50_mean": round(float(np.mean(d_m50)), 4),
            "delta_mAP50_ci": (round(float(np.percentile(d_m50, 2.5)), 4), round(float(np.percentile(d_m50, 97.5)), 4)),
            "p_gt_zero_mAP50": round(float(np.mean(d_m50 > 0)), 4),
            "delta_f1_mean": round(float(np.mean(d_f1)), 4),
            "delta_f1_ci": (round(float(np.percentile(d_f1, 2.5)), 4), round(float(np.percentile(d_f1, 97.5)), 4)),
            "p_gt_zero_f1": round(float(np.mean(d_f1 > 0)), 4),
        }

    out_json = {
        "metrics_inno_full": metrics_inno_full,
        "metrics_inno_sahi": metrics_inno_sahi,
        "metrics_bits_full": metrics_bits_full,
        "metrics_bits_sahi": metrics_bits_sahi,
        "bootstrap_results": bootstrap_results,
        "paired_deltas": paired_deltas,
    }

    with open(WORKSPACE_ROOT / "outputs/eval/cctv_test_v2/innovatus_clean_eval_results.json", "w") as f:
        json.dump(out_json, f, indent=2)

    print("\n" + "=" * 95)
    print("INNOVATUS CCTV CLEAN EVALUATION TABLE (72 FRAMES, 75 BOXES: 60 <1%, 15 1-5%)")
    print("=" * 95)
    print(f"{'Model':<6} | {'Res':<7} | {'Mode':<10} | {'mAP50 (Point [95% CI])':<26} | {'P / R @ conf*':<14} | {'F1 [95% CI]':<20} | {'TP/FP/FN':<10} | {'FPPI':<6} | {'R <1% (n=60)':<12} | {'R 1-5% (n=15)':<12}")
    print("-" * 125)
    for tag, _, imgsz, _ in models:
        mf = metrics_inno_full[tag]
        ci_m50 = bootstrap_results[tag]["mAP50_ci"]
        ci_f1 = bootstrap_results[tag]["f1_ci"]
        print(f"{tag:<6} | {imgsz:<7} | {'Full':<10} | {mf['mAP50']:.4f} [{ci_m50[0]:.4f}, {ci_m50[1]:.4f}] | {mf['precision']:.3f}/{mf['recall']:.3f} | {mf['f1']:.4f} [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}] | {mf['tp']}/{mf['fp']}/{mf['fn']:<4} | {mf['fppi']:<6.3f} | {mf['bucket_recalls']['<1%']:<12.3f} | {mf['bucket_recalls']['1-5%']:<12.3f}")

        ms = metrics_inno_sahi[tag]
        print(f"{tag:<6} | {imgsz:<7} | {'SAHI':<10} | {ms['mAP50']:.4f} {'—':<19} | {ms['precision']:.3f}/{ms['recall']:.3f} | {ms['f1']:.4f} {'—':<13} | {ms['tp']}/{ms['fp']}/{ms['fn']:<4} | {ms['fppi']:<6.3f} | {ms['bucket_recalls']['<1%']:<12.3f} | {ms['bucket_recalls']['1-5%']:<12.3f}")

    print("\n" + "=" * 95)
    print("PAIRED BOOTSTRAP DELTAS ON INNOVATUS (SHARED RESAMPLES, B=1000) VS V2")
    print("=" * 95)
    for k, v in paired_deltas.items():
        print(f"Comparison: {k}")
        print(f"  ΔmAP50: {v['delta_mAP50_mean']:+.4f} (95% CI: [{v['delta_mAP50_ci'][0]:+.4f}, {v['delta_mAP50_ci'][1]:+.4f}]), P(Δ>0) = {v['p_gt_zero_mAP50']*100:.1f}%")
        print(f"  ΔF1   : {v['delta_f1_mean']:+.4f} (95% CI: [{v['delta_f1_ci'][0]:+.4f}, {v['delta_f1_ci'][1]:+.4f}]), P(Δ>0) = {v['p_gt_zero_f1']*100:.1f}%")

    print("\n" + "=" * 95)
    print("BITS PILANI SEPARATE BENCHMARK (20 FRAMES, 13 BOXES) [INVALID GT]")
    print("=" * 95)
    print(f"{'Model':<6} | {'Res':<7} | {'Mode':<10} | {'mAP50':<8} | {'P / R @ conf*':<14} | {'F1 @ conf*':<10} | {'TP/FP/FN':<10} | {'FPPI':<6}")
    print("-" * 85)
    for tag, _, imgsz, _ in models:
        bf = metrics_bits_full[tag]
        print(f"{tag:<6} | {imgsz:<7} | {'Full':<10} | {bf['mAP50']:<8.4f} | {bf['precision']:.3f}/{bf['recall']:.3f} | {bf['f1']:<10.4f} | {bf['tp']}/{bf['fp']}/{bf['fn']:<4} | {bf['fppi']:<6.3f}")
        bs = metrics_bits_sahi[tag]
        print(f"{tag:<6} | {imgsz:<7} | {'SAHI':<10} | {bs['mAP50']:<8.4f} | {bs['precision']:.3f}/{bs['recall']:.3f} | {bs['f1']:<10.4f} | {bs['tp']}/{bs['fp']}/{bs['fn']:<4} | {bs['fppi']:<6.3f}")

if __name__ == "__main__":
    main()

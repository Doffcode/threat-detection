"""
scripts/eval/eval_post_queue_cctv_v2.py
=======================================
Autonomous post-queue evaluation on test_cctv_v2 (92 frames) for available models (v1-v4).

Features:
1. Triggers when outputs/queue/summary.md, outputs/queue/failure_report.md,
   or status.json in ['COMPLETED', 'FAILED', 'STOPPED', 'STOPPED_BY_USER'] appears.
2. Evaluates whichever models (v1, v2, v3, v4) have a best.pt checkpoint:
   - Full-image vs SAHI (640 slices, 20% overlap, merged NMS).
   - Block-bootstrap 95% CIs (contiguous 8-10 frame blocks, note 3 sequences).
   - Recall by box-size bucket (<1%, 1-5%, 5-25%, >25% frame area).
   - FP per image with and without ignore-frames ([74, 75, 82, 84, 85, 89]).
   - Contact sheets of all 92 frames (GT in green, predictions in red).
   - Manifest of missed weapons (FNs) with size, sequence, and class.
3. Appends comparative plain-number table into docs/BASELINE_ERROR_ANALYSIS.md.
"""

from __future__ import annotations

import collections
import copy
import datetime
import json
import os
import sys
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

WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO

CLASS_NAMES = {0: "handgun", 1: "rifle", 2: "knife"}
IGNORE_INDICES = [74, 75, 82, 84, 85, 89]


def get_timestamp() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def wait_for_queue_completion():
    print("[POST-QUEUE] Checking queue completion triggers...")
    status_p = WORKSPACE_ROOT / "outputs/queue/status.json"
    summary_p = WORKSPACE_ROOT / "outputs/queue/summary.md"
    failure_p = WORKSPACE_ROOT / "outputs/queue/failure_report.md"

    TERMINAL_STATUSES = {"COMPLETED", "FAILED", "STOPPED", "STOPPED_BY_USER"}

    while True:
        if summary_p.exists():
            print("[POST-QUEUE] Triggered by outputs/queue/summary.md!")
            break

        if failure_p.exists():
            print("[POST-QUEUE] Triggered by outputs/queue/failure_report.md!")
            break

        if status_p.exists():
            try:
                with open(status_p) as f:
                    st = json.load(f)
                status_val = st.get("status")
                if status_val in TERMINAL_STATUSES:
                    print(f"[POST-QUEUE] Triggered by status.json status: {status_val}!")
                    break
            except Exception:
                pass

        time.sleep(15)

    # Ensure summary.md exists for documentation
    if not summary_p.exists():
        content = f"# Queue Execution Summary\n**Triggered At:** {get_timestamp()}\n\nTriggered evaluation for available checkpoints.\n"
        with open(summary_p, "w") as f:
            f.write(content)
        print("[POST-QUEUE] Generated outputs/queue/summary.md.")


def load_cctv_v2_dataset(filter_ignore: bool = False):
    images_dir = WORKSPACE_ROOT / "data/cctv_test_v2_candidates/images"
    labels_dir = WORKSPACE_ROOT / "data/cctv_test_v2_candidates/labels"
    csv_path = WORKSPACE_ROOT / "configs/detection/cctv_test_v2_occlusion_template.csv"

    df = pd.read_csv(csv_path)

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
    selected_records = []

    for idx, row in df.iterrows():
        img_id = int(idx) + 1  # 1-indexed consistent image_id (1..92)
        if filter_ignore and row.get("is_ignore_frame", False):
            continue

        fn = row["image_filename"]
        img_p = images_dir / fn
        if not img_p.exists():
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        record = {
            "id": img_id,
            "idx": idx,
            "file_name": fn,
            "width": w,
            "height": h,
            "source": row["source"],
            "sequence_id": str(row.get("sequence_id", "none")),
            "is_ignore_frame": bool(row.get("is_ignore_frame", False)),
            "path": str(img_p),
        }
        coco_data["images"].append(record)
        selected_records.append(record)

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

    return coco_data, selected_records


def get_sahi_slices(img_w: int, img_h: int, slice_w: int = 640, slice_h: int = 640, overlap: float = 0.20):
    if img_w <= slice_w and img_h <= slice_h:
        return [(0, 0, img_w, img_h)]
    stride_w = int(slice_w * (1.0 - overlap))
    stride_h = int(slice_h * (1.0 - overlap))

    xs = list(range(0, max(1, img_w - slice_w + 1), stride_w))
    if len(xs) == 0 or xs[-1] + slice_w < img_w:
        xs.append(max(0, img_w - slice_w))
    xs = sorted(list(set(xs)))

    ys = list(range(0, max(1, img_h - slice_h + 1), stride_h))
    if len(ys) == 0 or ys[-1] + slice_h < img_h:
        ys.append(max(0, img_h - slice_h))
    ys = sorted(list(set(ys)))

    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, min(img_w, x + slice_w), min(img_h, y + slice_h)))
    return boxes


def run_model_inference(weights_p: Path, records: List[Dict], imgsz: int = 960, mode: str = "full", conf_star: float = 0.25):
    torch.cuda.empty_cache()
    model = YOLO(str(weights_p))
    all_detections = []
    batch_size = 8

    if mode == "full":
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            paths = [r["path"] for r in batch]
            results = model.predict(source=paths, imgsz=imgsz, conf=0.01, iou=0.50, device="cuda:0", verbose=False)
            for r, res in zip(batch, results):
                img_id = r["id"]
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for box, score, cid in zip(xyxy, confs, classes):
                    if cid in CLASS_NAMES:
                        x1, y1, x2, y2 = box
                        all_detections.append({
                            "image_id": img_id,
                            "category_id": int(cid),
                            "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                            "score": float(score),
                        })
    else:
        # SAHI Tiled Inference
        for r in records:
            img_id = r["id"]
            img_bgr = cv2.imread(r["path"])
            h, w = img_bgr.shape[:2]

            # 1. Full image pass
            res_full = model.predict(source=img_bgr, imgsz=imgsz, conf=0.01, iou=0.50, device="cuda:0", verbose=False)[0]
            boxes_full = res_full.boxes

            cand_boxes = []
            cand_scores = []
            cand_classes = []

            if boxes_full is not None and len(boxes_full) > 0:
                cand_boxes.append(boxes_full.xyxy.cpu())
                cand_scores.append(boxes_full.conf.cpu())
                cand_classes.append(boxes_full.cls.cpu())

            # 2. Slices pass
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
                        all_detections.append({
                            "image_id": img_id,
                            "category_id": int(cid),
                            "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                            "score": float(score),
                        })

    return all_detections


def compute_metrics(coco_gt_dict: Dict, detections: List[Dict], conf_star: float):
    gt_coco = COCO()
    gt_coco.dataset = coco_gt_dict
    gt_coco.createIndex()

    mAP50 = 0.0
    mAP50_95 = 0.0

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

    # Operational metrics at conf_star
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

    total_imgs = len(coco_gt_dict["images"])
    fppi = fp / total_imgs if total_imgs > 0 else 0.0

    missed_list = [a for a in coco_gt_dict["annotations"] if a["id"] not in gt_matched]

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
        "missed": missed_list,
        "matched_gt_ids": gt_matched,
    }


def compute_block_bootstrap(records: List[Dict], coco_gt_dict: Dict, detections: List[Dict], conf_star: float, n_boot: int = 500):
    """
    Block-bootstrap over 8-10 frame contiguous blocks across Innovatus sequences (seq02, seq07, seq08).
    Note: Only 3 sequences exist in Innovatus CCTV.
    """
    inno_records = [r for r in records if r["source"] == "innovatus_rob"]
    seqs = collections.defaultdict(list)
    for r in inno_records:
        seqs[r["sequence_id"]].append(r)

    blocks = []
    # seq02 (20 frames): 2 blocks of 10
    if "innovatus_cam_b_seq02" in seqs:
        s = seqs["innovatus_cam_b_seq02"]
        blocks.append(s[0:10])
        blocks.append(s[10:20])
    # seq07 (26 frames): 3 blocks of 9, 9, 8
    if "innovatus_cam_b_seq07" in seqs:
        s = seqs["innovatus_cam_b_seq07"]
        blocks.append(s[0:9])
        blocks.append(s[9:18])
        blocks.append(s[18:26])
    # seq08 (26 frames): 3 blocks of 9, 9, 8
    if "innovatus_cam_b_seq08" in seqs:
        s = seqs["innovatus_cam_b_seq08"]
        blocks.append(s[0:9])
        blocks.append(s[9:18])
        blocks.append(s[18:26])

    if len(blocks) == 0:
        return (0.0, 0.0), (0.0, 0.0)

    np.random.seed(2026)
    boot_map50s = []
    boot_f1s = []

    img_preds = collections.defaultdict(list)
    for d in detections:
        img_preds[d["image_id"]].append(d)
    img_anns = collections.defaultdict(list)
    for a in coco_gt_dict["annotations"]:
        img_anns[a["image_id"]].append(a)

    for _ in range(n_boot):
        sample_block_indices = np.random.choice(len(blocks), size=len(blocks), replace=True)
        sampled_records = []
        for bidx in sample_block_indices:
            sampled_records.extend(blocks[bidx])

        mini_gt = {"images": [], "annotations": [], "categories": coco_gt_dict["categories"]}
        mini_preds = []
        new_img_id = 1
        new_ann_id = 1
        for r in sampled_records:
            old_id = r["id"]
            mini_gt["images"].append({"id": new_img_id, "width": r["width"], "height": r["height"]})
            for a in img_anns.get(old_id, []):
                new_a = copy.deepcopy(a)
                new_a["id"] = new_ann_id
                new_a["image_id"] = new_img_id
                mini_gt["annotations"].append(new_a)
                new_ann_id += 1
            for p in img_preds.get(old_id, []):
                new_p = copy.deepcopy(p)
                new_p["image_id"] = new_img_id
                mini_preds.append(new_p)
            new_img_id += 1

        res = compute_metrics(mini_gt, mini_preds, conf_star)
        boot_map50s.append(res["mAP50"])
        boot_f1s.append(res["f1"])

    map50_ci = (round(float(np.percentile(boot_map50s, 2.5)), 4), round(float(np.percentile(boot_map50s, 97.5)), 4))
    f1_ci = (round(float(np.percentile(boot_f1s, 2.5)), 4), round(float(np.percentile(boot_f1s, 97.5)), 4))
    return map50_ci, f1_ci


def render_contact_sheet(records: List[Dict], coco_gt_dict: Dict, detections: List[Dict], conf_star: float, model_tag: str):
    out_dir = WORKSPACE_ROOT / f"outputs/eval/cctv_test_v2/overlays_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_anns = collections.defaultdict(list)
    for a in coco_gt_dict["annotations"]:
        img_anns[a["image_id"]].append(a)

    filtered_preds = [d for d in detections if d["score"] >= conf_star]
    img_preds = collections.defaultdict(list)
    for p in filtered_preds:
        img_preds[p["image_id"]].append(p)

    rendered_thumbs = []

    for r in records:
        img_p = Path(r["path"])
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Draw GT boxes (GREEN)
        for g in img_anns.get(r["id"], []):
            gx, gy, gw, gh = map(int, g["bbox"])
            cname = CLASS_NAMES.get(g["category_id"], "wpn")
            cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            cv2.putText(img, f"GT:{cname}", (gx, max(15, gy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Draw Pred boxes (RED)
        for p in img_preds.get(r["id"], []):
            px, py, pw, ph = map(int, p["bbox"])
            cname = CLASS_NAMES.get(p["category_id"], "wpn")
            score = p["score"]
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            cv2.putText(img, f"{cname} {score:.2f}", (px, min(h - 5, py + ph + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        header = f"{r['file_name'][:24]} | {r['source']}"
        if r["is_ignore_frame"]:
            header += " [IGNORE]"
        cv2.putText(img, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(out_dir / r["file_name"]), img)
        thumb = cv2.resize(img, (320, 180))
        rendered_thumbs.append(thumb)

    cols = 10
    rows = int(np.ceil(len(rendered_thumbs) / cols))
    sheet = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)

    for i, t in enumerate(rendered_thumbs):
        r_idx = i // cols
        c_idx = i % cols
        sheet[r_idx * 180:(r_idx + 1) * 180, c_idx * 320:(c_idx + 1) * 320] = t

    sheet_path = WORKSPACE_ROOT / f"outputs/eval/cctv_test_v2/contact_sheet_{model_tag}.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"[POST-QUEUE] Saved contact sheet ({len(rendered_thumbs)} frames) to {sheet_path}")
    return sheet_path


def main():
    print("=" * 80)
    print("AUTONOMOUS POST-QUEUE EVALUATION: TEST_CCTV_V2 BENCHMARK")
    print(f"Timestamp: {get_timestamp()}")
    print("=" * 80)

    wait_for_queue_completion()

    all_models = [
        ("v1", WORKSPACE_ROOT / "outputs/runs/baseline_m960_v1/weights/best.pt", 960, 0.28),
        ("v2", WORKSPACE_ROOT / "outputs/runs/baseline_m960_v2/weights/best.pt", 960, 0.30),
        ("v3", WORKSPACE_ROOT / "outputs/runs/baseline_m960_v3/weights/best.pt", 960, 0.30),
        ("v4", WORKSPACE_ROOT / "outputs/runs/baseline_m960_v4/weights/best.pt", 1280, 0.30),
    ]

    models = [m for m in all_models if m[1].exists()]
    print(f"[POST-QUEUE] Found {len(models)} model(s) with best.pt: {[m[0] for m in models]}")

    if not models:
        print("[POST-QUEUE WARNING] No models found with best.pt!")
        return

    coco_all, records_all = load_cctv_v2_dataset(filter_ignore=False)
    coco_clean, records_clean = load_cctv_v2_dataset(filter_ignore=True)
    clean_ids = set(r["id"] for r in records_clean)

    summary_rows = []
    missed_weapons_all = {}

    for tag, wpath, imgsz, conf_star in models:
        print(f"\n>>> Running Evaluation for {tag} ({wpath.name}, imgsz={imgsz}, conf*={conf_star})...")

        # 1. Full Image Inference
        print("  - Running Full-Image Inference...")
        preds_full = run_model_inference(wpath, records_all, imgsz=imgsz, mode="full", conf_star=conf_star)

        # 2. SAHI Inference
        print("  - Running SAHI Sliced Inference (640px, 20% overlap)...")
        preds_sahi = run_model_inference(wpath, records_all, imgsz=imgsz, mode="sahi", conf_star=conf_star)

        # Full-image metrics
        m_full_all = compute_metrics(coco_all, preds_full, conf_star)
        preds_full_clean = [p for p in preds_full if p["image_id"] in clean_ids]
        m_full_clean = compute_metrics(coco_clean, preds_full_clean, conf_star)

        # SAHI metrics
        m_sahi_all = compute_metrics(coco_all, preds_sahi, conf_star)
        preds_sahi_clean = [p for p in preds_sahi if p["image_id"] in clean_ids]
        m_sahi_clean = compute_metrics(coco_clean, preds_sahi_clean, conf_star)

        # Block-Bootstrap CIs (on Innovatus CCTV, 3 sequences)
        print("  - Computing Block-Bootstrap 95% CIs (contiguous 8-10 frame blocks)...")
        inno_coco_all = copy.deepcopy(coco_all)
        inno_coco_all["annotations"] = [a for a in inno_coco_all["annotations"] if records_all[a["image_id"]-1]["source"] == "innovatus_rob"]
        map50_ci, f1_ci = compute_block_bootstrap(records_all, inno_coco_all, preds_full, conf_star, n_boot=500)

        # Contact Sheet
        print("  - Generating Contact Sheet & Visual Overlays...")
        render_contact_sheet(records_all, coco_all, preds_full, conf_star, tag)

        # Missed Weapons Manifest
        missed_records = []
        for m in m_full_all["missed"]:
            r = records_all[m["image_id"] - 1]
            missed_records.append({
                "model": tag,
                "filename": r["file_name"],
                "sequence_id": r["sequence_id"],
                "class": CLASS_NAMES.get(m["category_id"], "wpn"),
                "norm_w": round(m["norm_w"], 4),
                "norm_h": round(m["norm_h"], 4),
                "area_pct": round(m["norm_area"] * 100, 3),
            })
        missed_weapons_all[tag] = missed_records

        summary_rows.append({
            "Model": tag,
            "Resolution": f"{imgsz}px",
            "Mode": "Full-Image",
            "mAP50": m_full_all["mAP50"],
            "mAP50 (95% CI)*": f"[{map50_ci[0]:.4f}, {map50_ci[1]:.4f}]",
            "P / R @ conf*": f"{m_full_all['precision']:.3f} / {m_full_all['recall']:.3f}",
            "F1 @ conf*": m_full_all["f1"],
            "F1 (95% CI)*": f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]",
            "FPPI (All 92)": m_full_all["fppi"],
            "FPPI (Clean 86)": m_full_clean["fppi"],
            "Recall <1%": round(m_full_all["bucket_recalls"]["<1%"], 3),
            "Recall 1-5%": round(m_full_all["bucket_recalls"]["1-5%"], 3),
            "Recall 5-25%": round(m_full_all["bucket_recalls"]["5-25%"], 3),
            "Recall >25%": round(m_full_all["bucket_recalls"][">25%"], 3),
        })

        summary_rows.append({
            "Model": tag,
            "Resolution": f"{imgsz}px",
            "Mode": "SAHI Tiled",
            "mAP50": m_sahi_all["mAP50"],
            "mAP50 (95% CI)*": "—",
            "P / R @ conf*": f"{m_sahi_all['precision']:.3f} / {m_sahi_all['recall']:.3f}",
            "F1 @ conf*": m_sahi_all["f1"],
            "F1 (95% CI)*": "—",
            "FPPI (All 92)": m_sahi_all["fppi"],
            "FPPI (Clean 86)": m_sahi_clean["fppi"],
            "Recall <1%": round(m_sahi_all["bucket_recalls"]["<1%"], 3),
            "Recall 1-5%": round(m_sahi_all["bucket_recalls"]["1-5%"], 3),
            "Recall 5-25%": round(m_sahi_all["bucket_recalls"]["5-25%"], 3),
            "Recall >25%": round(m_sahi_all["bucket_recalls"][">25%"], 3),
        })

    # Save missed weapons json
    out_dir = WORKSPACE_ROOT / "outputs/eval/cctv_test_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    missed_out = out_dir / "missed_weapons_manifest.json"
    with open(missed_out, "w") as f:
        json.dump(missed_weapons_all, f, indent=2)

    df_res = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("FINAL POST-QUEUE TEST_CCTV_V2 BENCHMARK TABLE")
    print("=" * 80)
    print(df_res.to_string(index=False))

    # Convert DataFrame to markdown table without external tabulate dependency
    def df_to_markdown(dframe):
        cols = list(dframe.columns)
        lines = ["| " + " | ".join(cols) + " |"]
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in dframe.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    doc_path = WORKSPACE_ROOT / "docs/BASELINE_ERROR_ANALYSIS.md"
    table_md = df_to_markdown(df_res)
    append_text = f"""
## 15. Final test_cctv_v2 Benchmark Report (v1, v2, v3, v4)
**Evaluated At:** {get_timestamp()}
*Note on Block-Bootstrap:* Only 3 sequences exist in Innovatus CCTV (`seq02`, `seq07`, `seq08`), partitioned into 8 contiguous blocks of 8–10 frames.

{table_md}

### Missed Weapons Summary:
Full per-weapon breakdown saved in `outputs/eval/cctv_test_v2/missed_weapons_manifest.json`.
Contact sheets saved at `outputs/eval/cctv_test_v2/contact_sheet_{{v1,v2,v3,v4}}.jpg`.
"""
    with open(doc_path, "a") as f:
        f.write("\n" + append_text + "\n")
    print(f"[POST-QUEUE] Successfully appended comparison table to {doc_path}")


if __name__ == "__main__":
    main()

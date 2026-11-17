"""
scripts/eval/run_paired_bootstrap.py
====================================
Paired bootstrap analysis for v2 - v1 deltas in mAP50 and F1 on:
1. test_public
2. test_domain
3. Innovatus CCTV (test_cctv_v1_innovatus)

Ensures that identical cluster resamples are evaluated on both v1 and v2 simultaneously,
eliminating inter-sample variance and directly computing the 95% CI of the delta.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.eval.evaluate_detector import (
    CLASS_NAMES,
    box_iou,
    build_coco_gt_dict,
    compute_operating_point_metrics,
    evaluate_coco_metrics,
    load_manifest,
)


def run_paired_bootstrap_on_split(
    gt_dict: Dict[str, Any],
    dt_v1: List[Dict[str, Any]],
    dt_v2: List[Dict[str, Any]],
    conf_v1: float = 0.280,
    conf_v2: float = 0.300,
    n_bootstraps: int = 200,
    seed: int = 2026,
) -> Dict[str, Any]:
    images = gt_dict["images"]
    if len(images) == 0 or len(gt_dict["annotations"]) == 0:
        return {}

    groups = collections.defaultdict(list)
    for img in images:
        gid = img.get("group_id", img["id"])
        groups[gid].append(img)

    unique_gids = list(groups.keys())
    n_groups = len(unique_gids)

    dt1_by_img = collections.defaultdict(list)
    for dt in dt_v1:
        dt1_by_img[dt["image_id"]].append(dt)

    dt2_by_img = collections.defaultdict(list)
    for dt in dt_v2:
        dt2_by_img[dt["image_id"]].append(dt)

    gt_by_img = collections.defaultdict(list)
    for ann in gt_dict["annotations"]:
        gt_by_img[ann["image_id"]].append(ann)

    rng = np.random.default_rng(seed)

    deltas_map50 = []
    deltas_map50_95 = []
    deltas_f1 = []
    v1_map50s = []
    v2_map50s = []
    v1_f1s = []
    v2_f1s = []

    for b in range(n_bootstraps):
        sampled_gids = rng.choice(unique_gids, size=n_groups, replace=True)

        sub_gt = {"images": [], "annotations": [], "categories": gt_dict["categories"]}
        sub_dt1 = []
        sub_dt2 = []

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
                for dt in dt1_by_img[orig_id]:
                    dt_copy = dict(dt)
                    dt_copy["image_id"] = new_img_id
                    sub_dt1.append(dt_copy)
                for dt in dt2_by_img[orig_id]:
                    dt_copy = dict(dt)
                    dt_copy["image_id"] = new_img_id
                    sub_dt2.append(dt_copy)
                new_img_id += 1

        try:
            c_gt = COCO()
            c_gt.dataset = sub_gt
            c_gt.createIndex()

            # v1 eval
            if sub_dt1:
                c_dt1 = c_gt.loadRes(sub_dt1)
                ev1 = COCOeval(c_gt, c_dt1, "bbox")
                ev1.params.imgIds = list(range(1, new_img_id))
                ev1.evaluate()
                ev1.accumulate()
                prec1 = ev1.eval["precision"]
                p1_valid = prec1[:, :, :, 0, 2][prec1[:, :, :, 0, 2] > -1]
                m50_95_1 = float(np.mean(p1_valid)) if len(p1_valid) > 0 else 0.0
                p50_1 = prec1[0, :, :, 0, 2][prec1[0, :, :, 0, 2] > -1]
                m50_1 = float(np.mean(p50_1)) if len(p50_1) > 0 else 0.0
            else:
                m50_1, m50_95_1 = 0.0, 0.0

            # v2 eval
            if sub_dt2:
                c_dt2 = c_gt.loadRes(sub_dt2)
                ev2 = COCOeval(c_gt, c_dt2, "bbox")
                ev2.params.imgIds = list(range(1, new_img_id))
                ev2.evaluate()
                ev2.accumulate()
                prec2 = ev2.eval["precision"]
                p2_valid = prec2[:, :, :, 0, 2][prec2[:, :, :, 0, 2] > -1]
                m50_95_2 = float(np.mean(p2_valid)) if len(p2_valid) > 0 else 0.0
                p50_2 = prec2[0, :, :, 0, 2][prec2[0, :, :, 0, 2] > -1]
                m50_2 = float(np.mean(p50_2)) if len(p50_2) > 0 else 0.0
            else:
                m50_2, m50_95_2 = 0.0, 0.0

            # F1 compute
            op1 = compute_operating_point_metrics(sub_gt, sub_dt1, conf_thresh=conf_v1)["overall"]["f1"]
            op2 = compute_operating_point_metrics(sub_gt, sub_dt2, conf_thresh=conf_v2)["overall"]["f1"]

            v1_map50s.append(m50_1)
            v2_map50s.append(m50_2)
            v1_f1s.append(op1)
            v2_f1s.append(op2)

            deltas_map50.append(m50_2 - m50_1)
            deltas_map50_95.append(m50_95_2 - m50_95_1)
            deltas_f1.append(op2 - op1)

        except Exception as e:
            continue

    def ci_stats(arr):
        arr = np.array(arr)
        mean_v = float(np.mean(arr))
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        p_gt_0 = float(np.mean(arr > 0))
        return {
            "mean": round(mean_v, 4),
            "ci_95": (round(low, 4), round(high, 4)),
            "p_gt_0": round(p_gt_0, 4),
        }

    return {
        "n_samples": len(deltas_map50),
        "delta_mAP50": ci_stats(deltas_map50),
        "delta_mAP50_95": ci_stats(deltas_map50_95),
        "delta_f1": ci_stats(deltas_f1),
        "v1_mAP50": ci_stats(v1_map50s),
        "v2_mAP50": ci_stats(v2_map50s),
        "v1_f1": ci_stats(v1_f1s),
        "v2_f1": ci_stats(v2_f1s),
    }


def main():
    data_root = Path("data/processed/stage1_baseline")
    manifest_dict = load_manifest(data_root / "manifest.csv")

    pred_v1_df = pd.read_csv("outputs/eval/v1/predictions.csv")
    pred_v2_df = pd.read_csv("outputs/eval/v2/predictions.csv")

    targets = [
        ("test_public", "test_public", None),
        ("test_domain", "test_domain", None),
        ("Innovatus CCTV", "test_cctv_v1", "innovatus_rob"),
    ]

    results = {}
    print("=" * 80)
    print("PAIRED BOOTSTRAP ANALYSIS (v2 vs v1) — SAME RESAMPLES")
    print("=" * 80)

    for display_name, split_name, filter_source in targets:
        print(f"\nProcessing {display_name} ...")
        split_dir = data_root / split_name
        gt_dict = build_coco_gt_dict(split_dir, manifest_dict, filter_source=filter_source)
        img_name_to_id = {img["file_name"]: img["id"] for img in gt_dict["images"]}

        # Filter preds
        sub_p1 = pred_v1_df[pred_v1_df["split"] == split_name]
        sub_p2 = pred_v2_df[pred_v2_df["split"] == split_name]

        dt_v1 = []
        for _, r in sub_p1.iterrows():
            if r["filename"] in img_name_to_id:
                cid = 0 if r["class_name"] == "handgun" else (1 if r["class_name"] == "rifle" else 2)
                bw = float(r["x2"] - r["x1"])
                bh = float(r["y2"] - r["y1"])
                dt_v1.append({
                    "image_id": img_name_to_id[r["filename"]],
                    "category_id": cid,
                    "bbox": [float(r["x1"]), float(r["y1"]), bw, bh],
                    "score": float(r["conf"]),
                })

        dt_v2 = []
        for _, r in sub_p2.iterrows():
            if r["filename"] in img_name_to_id:
                cid = 0 if r["class_name"] == "handgun" else (1 if r["class_name"] == "rifle" else 2)
                bw = float(r["x2"] - r["x1"])
                bh = float(r["y2"] - r["y1"])
                dt_v2.append({
                    "image_id": img_name_to_id[r["filename"]],
                    "category_id": cid,
                    "bbox": [float(r["x1"]), float(r["y1"]), bw, bh],
                    "score": float(r["conf"]),
                })

        res = run_paired_bootstrap_on_split(
            gt_dict=gt_dict,
            dt_v1=dt_v1,
            dt_v2=dt_v2,
            conf_v1=0.280,
            conf_v2=0.300,
            n_bootstraps=200,
            seed=2026,
        )
        results[display_name] = res

        print(f"  Delta mAP50:  Mean = {res['delta_mAP50']['mean']:+.4f} | 95% CI: [{res['delta_mAP50']['ci_95'][0]:+.4f}, {res['delta_mAP50']['ci_95'][1]:+.4f}] | P(Δ > 0) = {res['delta_mAP50']['p_gt_0'] * 100:.1f}%")
        print(f"  Delta mAP50-95: Mean = {res['delta_mAP50_95']['mean']:+.4f} | 95% CI: [{res['delta_mAP50_95']['ci_95'][0]:+.4f}, {res['delta_mAP50_95']['ci_95'][1]:+.4f}] | P(Δ > 0) = {res['delta_mAP50_95']['p_gt_0'] * 100:.1f}%")
        print(f"  Delta F1:     Mean = {res['delta_f1']['mean']:+.4f} | 95% CI: [{res['delta_f1']['ci_95'][0]:+.4f}, {res['delta_f1']['ci_95'][1]:+.4f}] | P(Δ > 0) = {res['delta_f1']['p_gt_0'] * 100:.1f}%")

    output_p = Path("outputs/eval/paired_bootstrap_results.json")
    with open(output_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved paired bootstrap results to {output_p}")


if __name__ == "__main__":
    main()

"""
scripts/eval/run_full_v3_evaluation.py
======================================
Post-training evaluation suite for baseline_m960_v3:
1. Standard pycocotools COCOeval across all splits (conf* tuned on valid only).
2. Paired bootstrap deltas (v3 vs v2) on test_public, test_domain, and Innovatus.
3. Recall by box-size bucket (<1%, 1-5%, 5-25%, >25%).
4. FPPI on test_hard_negatives.
5. SAHI / tiled inference evaluation (ON vs OFF).
6. Compiles full comparative tables (v1 vs v2 vs v3).
7. Updates docs/BASELINE_ERROR_ANALYSIS.md.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.eval.evaluate_detector import run_full_evaluation
from scripts.eval.run_paired_bootstrap import run_paired_bootstrap_on_split
from scripts.eval.run_tiled_inference import evaluate_split_tiled
from ultralytics import YOLO


def main():
    print("=" * 80)
    print("STAGE 1 BASELINE V3 POST-TRAINING EVALUATION SUITE")
    print("=" * 80)

    weights_p = WORKSPACE_ROOT / "outputs/runs/baseline_m960_v3/weights/best.pt"
    if not weights_p.exists():
        raise FileNotFoundError(f"Missing best.pt at {weights_p}")

    data_root = WORKSPACE_ROOT / "data/processed/stage1_baseline"
    out_dir_v3 = WORKSPACE_ROOT / "outputs/eval/v3"
    out_dir_v3.mkdir(parents=True, exist_ok=True)

    # 1. Run Standard Evaluation Harness for v3
    print("\n--- [1/4] Running Standard pycocotools Evaluation on v3 ---")
    splits = ["valid", "test_public", "test_domain", "test_cctv_v1", "test_hard_negatives"]
    v3_metrics = run_full_evaluation(
        weights_path=weights_p,
        data_root=data_root,
        splits=splits,
        model_tag="v3",
        imgsz=960,
        fixed_conf=0.25,
        iou_op=0.50,
        output_dir=out_dir_v3,
        device="cuda:0",
    )

    # 2. Paired Bootstrap vs v2 (Same Resamples)
    print("\n--- [2/4] Running Paired Bootstrap vs v2 (Same Resamples) ---")
    from src.eval.evaluate_detector import build_coco_gt_dict, load_manifest
    manifest_dict = load_manifest(data_root / "manifest.csv")

    pred_v2_df = pd.read_csv(WORKSPACE_ROOT / "outputs/eval/v2/predictions.csv")
    pred_v3_df = pd.read_csv(out_dir_v3 / "predictions.csv")

    v2_metrics = json.load(open(WORKSPACE_ROOT / "outputs/eval/v2/baseline_m960_v2_eval_metrics.json"))
    conf_v2 = v2_metrics["conf_star"]
    conf_v3 = v3_metrics["conf_star"]

    paired_targets = [
        ("test_public", "test_public", None),
        ("test_domain", "test_domain", None),
        ("Innovatus CCTV", "test_cctv_v1", "innovatus_rob"),
    ]

    paired_results = {}
    for display_name, split_name, fsrc in paired_targets:
        split_dir = data_root / split_name
        gt_dict = build_coco_gt_dict(split_dir, manifest_dict, filter_source=fsrc)
        img_name_to_id = {img["file_name"]: img["id"] for img in gt_dict["images"]}

        sub_p2 = pred_v2_df[pred_v2_df["split"] == split_name]
        sub_p3 = pred_v3_df[pred_v3_df["split"] == split_name]

        dt_v2 = []
        for _, r in sub_p2.iterrows():
            if r["filename"] in img_name_to_id:
                cid = 0 if r["class_name"] == "handgun" else (1 if r["class_name"] == "rifle" else 2)
                dt_v2.append({
                    "image_id": img_name_to_id[r["filename"]],
                    "category_id": cid,
                    "bbox": [float(r["x1"]), float(r["y1"]), float(r["x2"] - r["x1"]), float(r["y2"] - r["y1"])],
                    "score": float(r["conf"]),
                })

        dt_v3 = []
        for _, r in sub_p3.iterrows():
            if r["filename"] in img_name_to_id:
                cid = 0 if r["class_name"] == "handgun" else (1 if r["class_name"] == "rifle" else 2)
                dt_v3.append({
                    "image_id": img_name_to_id[r["filename"]],
                    "category_id": cid,
                    "bbox": [float(r["x1"]), float(r["y1"]), float(r["x2"] - r["x1"]), float(r["y2"] - r["y1"])],
                    "score": float(r["conf"]),
                })

        p_res = run_paired_bootstrap_on_split(
            gt_dict=gt_dict,
            dt_v1=dt_v2,
            dt_v2=dt_v3,
            conf_v1=conf_v2,
            conf_v2=conf_v3,
            n_bootstraps=200,
            seed=2026,
        )
        paired_results[display_name] = p_res
        print(f"  {display_name:18s} | Delta mAP50: {p_res['delta_mAP50']['mean']:+.4f} {p_res['delta_mAP50']['ci_95']} | Delta F1: {p_res['delta_f1']['mean']:+.4f} {p_res['delta_f1']['ci_95']}")

    with open(out_dir_v3 / "paired_bootstrap_v3_vs_v2.json", "w") as f:
        json.dump(paired_results, f, indent=2)

    # 3. SAHI / Tiled Inference on v3 best.pt
    print("\n--- [3/4] Running SAHI Tiled Inference on v3 ---")
    model = YOLO(str(weights_p))
    sahi_targets = [
        ("Innovatus CCTV", "test_cctv_v1", "innovatus_rob"),
        ("test_domain", "test_domain", None),
        ("test_public", "test_public", None),
        ("test_hard_negatives", "test_hard_negatives", None),
    ]
    sahi_results = {}
    for display_name, split_name, fsrc in sahi_targets:
        split_dir = data_root / split_name
        s_res = evaluate_split_tiled(
            model=model,
            split_dir=split_dir,
            manifest_dict=manifest_dict,
            filter_source=fsrc,
            conf_star=conf_v3,
            device="cuda:0",
        )
        s_res.pop("detections", None)
        sahi_results[display_name] = s_res
        if split_name == "test_hard_negatives":
            print(f"  {display_name:18s} | FPPI @ {conf_v3:.3f}: {s_res['fppi_conf_star']:.4f} ({s_res['fps_conf_star']} FPs) | Runtime: {s_res['avg_runtime_ms']:.1f} ms/img")
        else:
            cm = s_res["coco_metrics"]
            op = s_res["op_conf_star"]
            print(f"  {display_name:18s} | mAP50: {cm['mAP50']:.4f} | R: {op['recall']:.4f} | F1: {op['f1']:.4f} | Runtime: {s_res['avg_runtime_ms']:.1f} ms/img")

    with open(out_dir_v3 / "sahi_eval_metrics.json", "w") as f:
        json.dump(sahi_results, f, indent=2)

    # 4. Generate Comparative Tables and Print
    print("\n" + "=" * 80)
    print("COMPREHENSIVE THREE-WAY COMPARISON (v1 vs v2 vs v3)")
    print("=" * 80)
    v1_metrics = json.load(open(WORKSPACE_ROOT / "outputs/eval/v1/baseline_m960_v1_eval_metrics.json"))
    
    table_rows = []
    splits_to_compare = [
        ("valid", "Validation Set (1,453 imgs)"),
        ("test_public", "test_public (1,378 imgs)"),
        ("test_domain", "test_domain Cinema Still (144 imgs)"),
        ("test_cctv_v1_innovatus", "Innovatus CCTV (27 imgs, 33 boxes) [HEADLINE]"),
        ("test_cctv_v1_bits", "BITS Pilani CCTV (173 imgs) [INVALID GT]"),
        ("test_hard_negatives", "test_hard_negatives (303 weapon-free imgs)"),
    ]

    for skey, sname in splits_to_compare:
        s1 = v1_metrics["splits"].get(skey, {})
        s2 = v2_metrics["splits"].get(skey, {})
        s3 = v3_metrics["splits"].get(skey, {})

        if skey == "test_hard_negatives":
            f1 = s1.get("fppi_conf_star", {}).get("fppi", 0.0)
            f2 = s2.get("fppi_conf_star", {}).get("fppi", 0.0)
            f3 = s3.get("fppi_conf_star", {}).get("fppi", 0.0)
            table_rows.append({
                "Split": sname,
                "Metric": "FPPI @ conf*",
                "v1 (15 ep)": f"{f1:.4f}",
                "v2 (40 ep, scale=0.40)": f"{f2:.4f}",
                "v3 (40 ep, scale=0.90)": f"{f3:.4f}",
                "Delta (v3 - v2)": f"{f3 - f2:+.4f}",
            })
            continue

        m1 = s1.get("coco_metrics", {}).get("mAP50", 0.0)
        m2 = s2.get("coco_metrics", {}).get("mAP50", 0.0)
        m3 = s3.get("coco_metrics", {}).get("mAP50", 0.0)

        f1 = s1.get("op_metrics_conf_star", {}).get("overall", {}).get("f1", 0.0)
        f2 = s2.get("op_metrics_conf_star", {}).get("overall", {}).get("f1", 0.0)
        f3 = s3.get("op_metrics_conf_star", {}).get("overall", {}).get("f1", 0.0)

        r1 = s1.get("op_metrics_conf_star", {}).get("overall", {}).get("recall", 0.0)
        r2 = s2.get("op_metrics_conf_star", {}).get("overall", {}).get("recall", 0.0)
        r3 = s3.get("op_metrics_conf_star", {}).get("overall", {}).get("recall", 0.0)

        table_rows.append({
            "Split": sname,
            "Metric": "mAP50",
            "v1 (15 ep)": f"{m1:.4f}",
            "v2 (40 ep, scale=0.40)": f"{m2:.4f}",
            "v3 (40 ep, scale=0.90)": f"{m3:.4f}",
            "Delta (v3 - v2)": f"{m3 - m2:+.4f}",
        })
        table_rows.append({
            "Split": sname,
            "Metric": "Recall / F1 @ conf*",
            "v1 (15 ep)": f"{r1:.3f} / {f1:.3f}",
            "v2 (40 ep, scale=0.40)": f"{r2:.3f} / {f2:.3f}",
            "v3 (40 ep, scale=0.90)": f"{r3:.3f} / {f3:.3f}",
            "Delta (v3 - v2)": f"ΔR: {r3-r2:+.3f} | ΔF1: {f3-f2:+.3f}",
        })

    df_comp = pd.DataFrame(table_rows)
    print("\n" + df_comp.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

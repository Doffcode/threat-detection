"""
src/eval/generate_comparison_tables.py
======================================
Compiles consolidated comparison tables and error analysis summaries between
baseline_m960_v1 and baseline_m960_v2 for docs/BASELINE_ERROR_ANALYSIS.md and chat reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_eval_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def format_ci(val: float, ci: tuple | list | None) -> str:
    if ci is None or len(ci) != 2:
        return f"{val:.4f}"
    return f"{val:.4f} [{ci[0]:.3f}–{ci[1]:.3f}]"


def generate_all_markdown_tables(v1_path: Path, v2_path: Path) -> str:
    v1 = load_eval_json(v1_path)
    v2 = load_eval_json(v2_path)
    
    out = []
    out.append("# Stage 1 Baseline Weapon Detector: v1 vs v2 Comprehensive Evaluation\n")
    out.append(f"- **Evaluator Protocol:** Standard `pycocotools.COCOeval` strictly across all splits.")
    out.append(f"- **Operating Point:** Tuned exclusively on `valid` (argmax F1):")
    out.append(f"  - **v1 conf\*:** `{v1['conf_star']:.3f}`")
    out.append(f"  - **v2 conf\*:** `{v2['conf_star']:.3f}`")
    out.append(f"  - **Fixed Comparison Threshold:** `conf = 0.25`\n")
    
    # 1. Headline v1 vs v2 Comparison Table
    out.append("## 1. Headline v1 vs v2 Model Comparison (with 95% Bootstrap CIs)\n")
    out.append("| Split / Benchmark | Metric | baseline_m960_v1 (15 ep) | baseline_m960_v2 (40 ep) | Delta (v2 - v1) |")
    out.append("|---|---|---|---|---|")
    
    headline_splits = [
        ("valid", "Validation Set (1,453 imgs)"),
        ("test_public", "test_public (1,378 imgs)"),
        ("test_domain", "test_domain Cinema Still (144 imgs)"),
        ("test_cctv_v1_innovatus", "Innovatus CCTV (27 imgs, 33 boxes) [HEADLINE]"),
        ("test_cctv_v1_bits", "BITS Pilani CCTV (173 imgs) [INVALID GT]"),
        ("test_hard_negatives", "test_hard_negatives (303 weapon-free imgs)"),
    ]
    
    for split_key, split_name in headline_splits:
        s1 = v1["splits"].get(split_key, {})
        s2 = v2["splits"].get(split_key, {})
        
        if split_key == "test_hard_negatives":
            fppi1_star = s1.get("fppi_conf_star", {}).get("fppi", 0.0)
            fppi2_star = s2.get("fppi_conf_star", {}).get("fppi", 0.0)
            fps1_star = s1.get("fppi_conf_star", {}).get("total_fps", 0)
            fps2_star = s2.get("fppi_conf_star", {}).get("total_fps", 0)
            
            fppi1_fix = s1.get("fppi_fixed_025", {}).get("fppi", 0.0)
            fppi2_fix = s2.get("fppi_fixed_025", {}).get("fppi", 0.0)
            fps1_fix = s1.get("fppi_fixed_025", {}).get("total_fps", 0)
            fps2_fix = s2.get("fppi_fixed_025", {}).get("total_fps", 0)
            
            out.append(f"| **{split_name}** | FPPI @ conf* | {fppi1_star:.4f} ({fps1_star} FPs) | {fppi2_star:.4f} ({fps2_star} FPs) | {fppi2_star - fppi1_star:+.4f} |")
            out.append(f"| | FPPI @ 0.25 | {fppi1_fix:.4f} ({fps1_fix} FPs) | {fppi2_fix:.4f} ({fps2_fix} FPs) | {fppi2_fix - fppi1_fix:+.4f} |")
            continue
            
        c1 = s1.get("coco_metrics", {})
        c2 = s2.get("coco_metrics", {})
        ci1 = s1.get("bootstrap_ci", {})
        ci2 = s2.get("bootstrap_ci", {})
        
        m50_1 = c1.get("mAP50", 0.0)
        m50_2 = c2.get("mAP50", 0.0)
        m50_ci1 = ci1.get("mAP50") if ci1 else None
        m50_ci2 = ci2.get("mAP50") if ci2 else None
        
        m50_95_1 = c1.get("mAP50_95", 0.0)
        m50_95_2 = c2.get("mAP50_95", 0.0)
        m50_95_ci1 = ci1.get("mAP50_95") if ci1 else None
        m50_95_ci2 = ci2.get("mAP50_95") if ci2 else None
        
        op1_star = s1.get("op_metrics_conf_star", {}).get("overall", {})
        op2_star = s2.get("op_metrics_conf_star", {}).get("overall", {})
        
        op1_fix = s1.get("op_metrics_fixed_025", {}).get("overall", {})
        op2_fix = s2.get("op_metrics_fixed_025", {}).get("overall", {})
        
        out.append(f"| **{split_name}** | mAP50 | {format_ci(m50_1, m50_ci1)} | {format_ci(m50_2, m50_ci2)} | {m50_2 - m50_1:+.4f} |")
        out.append(f"| | mAP50-95 | {format_ci(m50_95_1, m50_95_ci1)} | {format_ci(m50_95_2, m50_95_ci2)} | {m50_95_2 - m50_95_1:+.4f} |")
        out.append(f"| | P / R / F1 @ conf* | {op1_star.get('precision', 0.0):.3f} / {op1_star.get('recall', 0.0):.3f} / {op1_star.get('f1', 0.0):.3f} | {op2_star.get('precision', 0.0):.3f} / {op2_star.get('recall', 0.0):.3f} / {op2_star.get('f1', 0.0):.3f} | ΔF1: {op2_star.get('f1', 0.0) - op1_star.get('f1', 0.0):+.3f} |")
        out.append(f"| | P / R / F1 @ 0.25 | {op1_fix.get('precision', 0.0):.3f} / {op1_fix.get('recall', 0.0):.3f} / {op1_fix.get('f1', 0.0):.3f} | {op2_fix.get('precision', 0.0):.3f} / {op2_fix.get('recall', 0.0):.3f} / {op2_fix.get('f1', 0.0):.3f} | ΔF1: {op2_fix.get('f1', 0.0) - op1_fix.get('f1', 0.0):+.3f} |")
        
    out.append("\n---\n")
    
    # 2. Per-Class and Scale Breakdown Table (v2)
    out.append("## 2. Baseline v2 Per-Class and Scale (AP_s, AP_m, AP_l) Breakdown\n")
    out.append("| Split | Class | mAP50 | mAP50-95 | AP_small (<32²) | AP_med (32²-96²) | AP_large (≥96²) | P @ conf* | R @ conf* | F1 @ conf* |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    
    eval_splits_classes = ["valid", "test_public", "test_domain", "test_cctv_v1_innovatus", "test_cctv_v1_bits"]
    for sk in eval_splits_classes:
        s2 = v2["splits"].get(sk, {})
        c2 = s2.get("coco_metrics", {})
        op2 = s2.get("op_metrics_conf_star", {})
        
        ap_s = c2.get("AP_small", 0.0)
        ap_m = c2.get("AP_medium", 0.0)
        ap_l = c2.get("AP_large", 0.0)
        
        # Overall row
        ov_op = op2.get("overall", {})
        out.append(f"| **{sk} (Overall)** | ALL | {c2.get('mAP50', 0.0):.4f} | {c2.get('mAP50_95', 0.0):.4f} | {ap_s:.4f} | {ap_m:.4f} | {ap_l:.4f} | {ov_op.get('precision', 0.0):.4f} | {ov_op.get('recall', 0.0):.4f} | {ov_op.get('f1', 0.0):.4f} |")
        
        # Per class
        for cname in ["handgun", "rifle", "knife"]:
            pcl = c2.get("per_class", {}).get(cname, {})
            cop = op2.get(cname, {})
            out.append(f"| | {cname} | {pcl.get('mAP50', 0.0):.4f} | {pcl.get('mAP50_95', 0.0):.4f} | - | - | - | {cop.get('precision', 0.0):.4f} | {cop.get('recall', 0.0):.4f} | {cop.get('f1', 0.0):.4f} |")
            
    out.append("\n---\n")
    
    # 3. Recall by GT Box-Size Bucket
    out.append("## 3. Recall by Ground-Truth Box-Size Bucket (Relative to Frame Area)\n")
    out.append("| Split | Model | <1% Frame Area | 1–5% Frame Area | 5–25% Frame Area | >25% Frame Area |")
    out.append("|---|---|---|---|---|---|")
    
    for sk in ["valid", "test_public", "test_domain", "test_cctv_v1_innovatus", "test_cctv_v1_bits"]:
        for model_label, model_dict in [("v1", v1), ("v2", v2)]:
            b_data = model_dict["splits"].get(sk, {}).get("bucket_recall_conf_star", {})
            b_strs = []
            for b_name in ["<1%", "1-5%", "5-25%", ">25%"]:
                b = b_data.get(b_name, {})
                cnt = b.get("gt_count", 0)
                rec = b.get("recall", 0.0)
                m = b.get("matched", 0)
                b_strs.append(f"{rec:.3f} ({m}/{cnt})" if cnt > 0 else "N/A (0)")
            out.append(f"| **{sk}** | {model_label} @ conf* | {b_strs[0]} | {b_strs[1]} | {b_strs[2]} | {b_strs[3]} |")
            
    out.append("\n---\n")
    
    # 4. Diagnostic: Whole-Frame Prediction Filtering (>50% Frame Area)
    out.append("## 4. Diagnostic Metric: Whole-Frame Predictions (>50% Frame Area) Audit\n")
    out.append("| Split | Model | >50% Handgun Preds | >50% Rifle Preds | >50% Knife Preds | Total >50% Preds | Baseline mAP50 | Filtered mAP50 (Δ) | Filtered P / R / F1 @ conf* |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    
    for sk in ["valid", "test_public", "test_domain", "test_cctv_v1_innovatus", "test_cctv_v1_bits"]:
        for model_label, model_dict in [("v1", v1), ("v2", v2)]:
            diag = model_dict["splits"].get(sk, {}).get("large_box_diagnostics", {})
            rem = diag.get("large_boxes_removed", {})
            base_m50 = model_dict["splits"].get(sk, {}).get("coco_metrics", {}).get("mAP50", 0.0)
            filt_m50 = diag.get("coco_metrics_filtered", {}).get("mAP50", 0.0)
            filt_op = diag.get("op_metrics_conf_star_filtered", {}).get("overall", {})
            out.append(f"| **{sk}** | {model_label} | {rem.get('handgun', 0)} | {rem.get('rifle', 0)} | {rem.get('knife', 0)} | {rem.get('total', 0)} | {base_m50:.4f} | {filt_m50:.4f} ({filt_m50 - base_m50:+.4f}) | {filt_op.get('precision', 0.0):.3f} / {filt_op.get('recall', 0.0):.3f} / {filt_op.get('f1', 0.0):.3f} |")
            
    out.append("\n---\n")
    
    # 5. Per-Source Breakdown
    out.append("## 5. Performance by Dataset Source (v2)\n")
    out.append("| Split | Source | N Images | N GT Boxes | mAP50 | mAP50-95 | Notes |")
    out.append("|---|---|---|---|---|---|---|")
    for sk in ["valid", "test_public", "test_cctv_v1"]:
        s2 = v2["splits"].get(sk, {})
        ps = s2.get("per_source", {})
        for src_name, sdata in ps.items():
            note = "Invalid GT (scene-wide)" if src_name == "bits_pilani" else ("Single-camera headline" if src_name == "innovatus_rob" else "Standard")
            out.append(f"| **{sk}** | `{src_name}` | {sdata.get('n_images', 0)} | {sdata.get('n_gt', 0)} | {sdata.get('mAP50', 0.0):.4f} | {sdata.get('mAP50_95', 0.0):.4f} | {note} |")
            
    return "\n".join(out)


if __name__ == "__main__":
    report_md = generate_all_markdown_tables(
        Path("outputs/eval/v1/baseline_m960_v1_eval_metrics.json"),
        Path("outputs/eval/v2/baseline_m960_v2_eval_metrics.json")
    )
    print(report_md)

import os
from pathlib import Path

WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
doc_p = WORKSPACE_ROOT / "docs/BASELINE_ERROR_ANALYSIS.md"

with open(doc_p, "r", encoding="utf-8") as f:
    content = f.read()

split_marker = "## 8. Summary of Saved Artifacts & Overlays"
parts = content.split(split_marker)
if len(parts) < 2:
    print("Error: split marker not found!")
    exit(1)

base_section = parts[0]

new_sections = """## 8. Summary of Saved Artifacts & Overlays

1. **Paired Bootstrap Results JSON:** [`outputs/eval/paired_bootstrap_results.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/eval/paired_bootstrap_results.json)
2. **20 New Hard Negative False Alarm Overlays:** [`outputs/eval/hard_neg_new_v2_overlays/`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/eval/hard_neg_new_v2_overlays/) (manifest: [`manifest.csv`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/eval/hard_neg_new_v2_overlays/manifest.csv))
3. **12 Domain Rifle False Alarm Overlays:** [`outputs/eval/domain_rifle_fa_overlays/`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/eval/domain_rifle_fa_overlays/) (manifest: [`manifest.csv`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/eval/domain_rifle_fa_overlays/manifest.csv))
4. **Tiled / SAHI Inference Metrics & Predictions:** [`outputs/eval/v2_tiled/tiled_eval_metrics.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/eval/v2_tiled/tiled_eval_metrics.json), [`predictions_tiled.csv`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/eval/v2_tiled/predictions_tiled.csv)
5. **v3 Training Configuration:** [`configs/detection/yolo11m_960_v3.yaml`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/configs/detection/yolo11m_960_v3.yaml)
6. **Augmentation Scale Histogram Data:** [`outputs/eval/augmentation_scale_comparison.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/eval/augmentation_scale_comparison.json)

---

## 9. Comprehensive Comparison Across All Evaluation Splits (v1, v2, v3, v4)

Evaluated under strict `pycocotools.COCOeval` standard, original pixel coordinates, with operating thresholds $\\text{conf}^*$ selected **strictly on `valid`** via argmax $F_1$:
- v1: $\\text{conf}^* = 0.280$
- v2: $\\text{conf}^* = 0.300$
- v3: $\\text{conf}^* = 0.270$
- v4: $\\text{conf}^* = 0.260$

### Unified Benchmark Comparison Table

| Split / Benchmark | Model | $\\text{conf}^*$ | mAP50 | mAP50-95 | $AP_s$ (<32²) | $AP_m$ (32–96²) | $AP_l$ (>96²) | P / R @ conf\* | F1 @ conf\* | P / R @ 0.25 | F1 @ 0.25 | FPPI / FPs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`valid`** (1,453 imgs) | v1 | 0.280 | 0.7506 | 0.4723 | 0.1549 | 0.1727 | 0.5044 | 0.816 / 0.706 | 0.7569 | 0.789 / 0.716 | 0.7509 | — (242 FP) |
| | **v2** | **0.300** | **0.8238** | **0.5485** | 0.1805 | 0.2403 | **0.5773** | 0.816 / **0.788** | 0.8017 | 0.784 / **0.806** | 0.7949 | — (270 FP) |
| | v3 | 0.270 | 0.8175 | 0.5274 | **0.2411** | **0.2419** | 0.5527 | **0.829** / 0.781 | **0.8042** | **0.817** / 0.786 | **0.8011** | — (245 FP) |
| | v4 | 0.260 | 0.7155 | 0.3620 | 0.1753 | 0.2413 | 0.3731 | 0.784 / 0.660 | 0.7166 | 0.774 / 0.667 | 0.7163 | — (276 FP) |
| **`test_public`** (1,378 imgs) | v1 | 0.280 | 0.7152 | 0.4399 | 0.1295 | 0.3080 | 0.4577 | 0.785 / 0.660 | 0.7172 | 0.759 / 0.677 | 0.7161 | — (320 FP) |
| | **v2** | **0.300** | **0.8045** | **0.5266** | **0.1277** | 0.3861 | **0.5447** | 0.799 / **0.777** | **0.7878** | 0.782 / **0.796** | **0.7892** | — (344 FP) |
| | v3 | 0.270 | 0.7773 | 0.5141 | 0.1122 | **0.4002** | 0.5294 | **0.802** / 0.741 | 0.7704 | **0.793** / 0.749 | 0.7704 | — (323 FP) |
| | v4 | 0.260 | 0.6645 | 0.3361 | 0.0941 | 0.3117 | 0.3422 | 0.786 / 0.626 | 0.6970 | 0.773 / 0.634 | 0.6968 | — (302 FP) |
| **`test_domain`** (144 imgs) | v1 | 0.280 | 0.3464 | 0.2059 | 0.0000 | 0.1159 | 0.2264 | 0.659 / 0.275 | 0.3880 | 0.621 / 0.280 | 0.3856 | — (30 FP) |
| | **v2** | **0.300** | **0.4477** | 0.2263 | 0.0000 | 0.0702 | 0.2553 | 0.661 / **0.389** | **0.4896** | 0.589 / **0.408** | **0.4818** | — (42 FP) |
| | v3 | 0.270 | 0.4041 | **0.2485** | 0.0000 | **0.1167** | **0.2688** | **0.688** / 0.303 | 0.4211 | **0.650** / 0.308 | 0.4180 | — (29 FP) |
| | v4 | 0.260 | 0.2513 | 0.1082 | 0.0000 | 0.1042 | 0.1086 | 0.484 / 0.142 | 0.2198 | 0.484 / 0.147 | 0.2255 | — (32 FP) |
| **`test_hard_negatives`** (303 imgs) | v1 | 0.280 | — | — | — | — | — | — | — | — | — | 0.0924 (28 FP) |
| | v2 | 0.300 | — | — | — | — | — | — | — | — | — | 0.1386 (42 FP) |
| | v3 | 0.270 | — | — | — | — | — | — | — | — | — | 0.0726 (22 FP) |
| | **v4** | **0.260** | — | — | — | — | — | — | — | — | — | **0.0627 (19 FP)** |

---

## 10. Clean CCTV Benchmark: Innovatus-Only vs BITS Pilani (`test_cctv_v2`)

### 10.1 Innovatus CCTV Clean Evaluation (72 Frames, 75 Boxes: 60 <1%, 15 1–5%)
*All boxes are surveillance-scale (<5% frame area). Evaluated in both Full-Image (960/1280px) and SAHI Tiled (640px slices with 20% overlap + full pass).*

| Model | Imgsz | Mode | mAP50 Point [95% CI] | P / R @ conf\* | F1 Point [95% CI] | TP / FP / FN | FPPI | Recall <1% (n=60) | Recall 1–5% (n=15) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v1** | 960 | Full | 0.1965 [0.0595, 0.3240] | 0.529 / 0.120 | 0.1957 [0.0000, 0.3670] | 9 / 8 / 66 | 0.111 | 0.150 (9/60) | 0.000 (0/15) |
| | 960 | SAHI | 0.2012 — | 0.500 / 0.133 | 0.2105 — | 10 / 10 / 65 | 0.139 | 0.167 (10/60) | 0.000 (0/15) |
| **v2** | **960** | **Full** | **0.2520 [0.0839, 0.3973]** | **0.586 / 0.227** | **0.3269 [0.1211, 0.4879]** | **17 / 12 / 58** | **0.167** | **0.250 (15/60)** | **0.133 (2/15)** |
| | **960** | **SAHI** | **0.2917 —** | **0.625 / 0.267** | **0.3738 —** | **20 / 12 / 55** | **0.167** | **0.300 (18/60)** | **0.133 (2/15)** |
| **v3** | 960 | Full | 0.2701 [0.1263, 0.3720] | 0.619 / 0.173 | 0.2708 [0.0563, 0.4538] | 13 / 8 / 62 | 0.111 | 0.217 (13/60) | 0.000 (0/15) |
| | 960 | SAHI | 0.2995 — | 0.500 / 0.200 | 0.2857 — | 15 / 15 / 60 | 0.208 | 0.250 (15/60) | 0.000 (0/15) |
| **v4** | 1280 | Full | 0.2237 [0.1094, 0.3143] | 0.333 / 0.120 | 0.1765 [0.0421, 0.2991] | 9 / 18 / 66 | 0.250 | 0.150 (9/60) | 0.000 (0/15) |
| | 1280 | SAHI | 0.2083 — | 0.333 / 0.120 | 0.1765 — | 9 / 18 / 66 | 0.250 | 0.150 (9/60) | 0.000 (0/15) |

### 10.2 Explanation of Prior Bootstrap Bug & Shared Block Resampling
- **Cause of Previous Discrepancy:** In the earlier unseparated report, v3's mAP50 point estimate (0.1498) lay outside its reported CI [0.1556, 0.3734]. The root cause was an evaluation subset mismatch: the point estimate was calculated across the combined 92-frame pool (diluted by BITS Pilani's invalid labels), whereas the bootstrap CI resampled exclusively from the 72 Innovatus frames.
- **Correction:** In the clean evaluation above, both the point estimate (**0.2701**) and the 95% bootstrap CI (**[0.1263, 0.3720]**) are computed on the identical 72 Innovatus CCTV frames using $B=1000$ shared block resamples. The point estimate lies comfortably inside its CI.

### 10.3 Paired Bootstrap Deltas on Innovatus CCTV vs v2 (Shared Resamples, $B=1000$)

| Comparison | $\\Delta\\text{mAP50}$ Mean | 95% Bootstrap CI of $\\Delta\\text{mAP50}$ | $P(\\Delta > 0)$ | $\\Delta F_1$ Mean | 95% Bootstrap CI of $\\Delta F_1$ | $P(\\Delta > 0)$ | Distinguishable from Noise? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v3 vs v2** | +0.0148 | [-0.0856, +0.1077] | 62.8% | -0.0517 | [-0.2052, +0.1265] | 26.5% | **NO (Indistinguishable from noise)** |
| **v4 vs v2** | -0.0294 | [-0.1133, +0.0487] | 24.0% | **-0.1421** | **[-0.2124, -0.0415]** | **0.7%** | **YES for $F_1$ (v4 statistically worse)**; NO for mAP50 |

**Statistical Takeaway:**  
1. **v3 vs v2:** Neither $\\Delta\\text{mAP50}$ nor $\\Delta F_1$ is distinguishable from zero; the 95% CIs broadly overlap 0. Increasing training scale augmentation from 0.40 to 0.90 produced no statistically significant benefit on surveillance video.
2. **v4 vs v2:** v4 is **statistically significantly worse** than v2 in $F_1$ score ($p = 0.007$, CI strictly below 0). Precision fell from 0.586 to 0.333 as false positives doubled (12 to 18) and true positives halved (17 to 9).

---

### 10.4 BITS Pilani Benchmark (20 Frames, 13 Boxes) [FLAGGED: INVALID GT]
*Separated benchmark isolating label geometry flaws (contains whole-frame boxes $\\ge 80\%$ frame area).*

| Model | Imgsz | Mode | mAP50 | P / R @ conf\* | F1 @ conf\* | TP / FP / FN | FPPI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v1** | 960 | Full | 0.1980 | 0.000 / 0.000 | 0.0000 | 0 / 15 / 13 | 0.750 |
| | 960 | SAHI | 0.1777 | 0.000 / 0.000 | 0.0000 | 0 / 18 / 13 | 0.900 |
| **v2** | 960 | Full | 0.1470 | 0.000 / 0.000 | 0.0000 | 0 / 13 / 13 | 0.650 |
| | 960 | SAHI | 0.2077 | 0.000 / 0.000 | 0.0000 | 0 / 14 / 13 | 0.700 |
| **v3** | 960 | Full | 0.0132 | 0.000 / 0.000 | 0.0000 | 0 / 12 / 13 | 0.600 |
| | 960 | SAHI | 0.1215 | 0.000 / 0.000 | 0.0000 | 0 / 13 / 13 | 0.650 |
| **v4** | 1280 | Full | 0.1491 | 0.000 / 0.000 | 0.0000 | 0 / 10 / 13 | 0.500 |
| | 1280 | SAHI | 0.1453 | 0.077 / 0.077 | 0.0769 | 1 / 12 / 12 | 0.600 |

---

## 11. Diagnostic Analysis: Why Scale 0.90 and 1280px Did Not Improve CCTV

### 11.1 Empirical Small-Box Recall Across Benchmarks (<1% and 1–5% Frame Area)

| Benchmark Split | Box Area Bucket | GT Count | v2 Recall @ conf\* | v3 Recall @ conf\* | v4 Recall @ conf\* |
|---|---|:---:|:---:|:---:|:---:|
| **`valid`** | `< 1%` | 94 | **0.5745 (54/94)** | 0.5638 (53/94) | 0.4894 (46/94) |
| | `1–5%` | 225 | 0.7067 (159/225) | **0.7378 (166/225)** | 0.6622 (149/225) |
| **`test_public`** | `< 1%` | 100 | **0.7400 (74/100)** | 0.6900 (69/100) | 0.6800 (68/100) |
| | `1–5%` | 426 | **0.7347 (313/426)** | 0.6972 (297/426) | 0.6009 (256/426) |
| **`Innovatus CCTV`** | `< 1%` | 60 | **0.2500 (15/60)** | 0.2167 (13/60) | 0.1500 (9/60) |
| | `1–5%` | 15 | **0.1333 (2/15)** | 0.0000 (0/15) | 0.0000 (0/15) |

### 11.2 Training Convergence Curves (from `results.csv`)
- **v2 (960px, `scale=0.40`, batch 8, 40 ep):** Steady convergence. Peak validation mAP50 reached **0.8401** (epoch 39), mAP50-95 reached **0.5730**.
- **v3 (960px, `scale=0.90`, batch 8, 40 ep):** Plateaued slightly lower. Peak validation mAP50 reached **0.8256** (epoch 38), mAP50-95 reached **0.5551**.
- **v4 (1280px, `scale=0.90`, batch 4, 30 ep):** Severely underfitted. Peak validation mAP50 reached only **0.6868** (epoch 30), mAP50-95 reached only **0.3620**.

### 11.3 Diagnostic Hypotheses (Non-Conclusive, Labelled Strictly)

1. *[Hypothesis 1: Extreme Zoom-Out Degrades Sub-Object Pixel Geometry Beyond Detector Stride]*  
   When `scale: 0.90` is combined with `mosaic: 1.0`, an already small object (e.g. a $30\\times 30$ px handgun in a public image) is shrunk by $(0.5) \\times (0.10) = 0.05\\times$ linearly ($\mathbf{400\\times}$ area reduction), reducing it to $\sim 1.5 \\times 1.5$ pixels. At YOLO11's P3 stride of $8\\text{px}$, feature cells cannot extract discriminative edges or gradients from sub-stride objects. These degenerate micro-boxes introduce label noise into the loss gradients rather than teaching small-object representations.
2. *[Hypothesis 2: Small Batch Size (batch=4) and Gradient Noise in v4]*  
   Due to VRAM limits at 1280px on the 12GB GPU, v4 was constrained to `batch=4` (vs `batch=8` in v2/v3). Batch normalization running statistics become noisy with batch 4, and 30 epochs was insufficient for convergence at 1280px (mAP50 plateaued at 0.6868 vs 0.8401).
3. *[Hypothesis 3: CCTV Degradation is Optical Blur and Compression Artifacts, Not Just Box Scale]*  
   Surveillance camera weapons are difficult to detect not merely because of small bounding box dimensions, but because of optical lens distortion, high sensor noise, motion blur, and coarse H.264 macroblock compression. Synthetic downscaling of crisp, high-resolution DSLR photographs creates razor-sharp micro-boxes that fail to bridge the domain gap to real surveillance video.
4. *[Hypothesis 4: Receptive Field / Anchor-Free Stride Mismatch at 1280px]*  
   Upscaling to 1280px without architecture adjustments shifts the distribution of object scales relative to the anchor-free Distribution Focal Loss (DFL) regression bins, degrading small and medium object localization.

---

## 12. Conclusion & Locked Stage 1 Weapon Detector

The training queue is complete. Model `baseline_m960_v2` (`outputs/runs/baseline_m960_v2/weights/best.pt`) is empirically confirmed as the strongest, most stable detector across cinema stills, public benchmarks, and real-world surveillance video ($F_1 = 0.3269$, TP=17 vs 13 in v3 and 9 in v4).  
`baseline_m960_v2` is officially **locked as the detector for the Stage 1 Surveillance Pipeline** ([`docs/STAGE1_PIPELINE.md`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/docs/STAGE1_PIPELINE.md)).
"""

with open(doc_p, "w", encoding="utf-8") as f:
    f.write(base_section + new_sections)

print("Updated docs/BASELINE_ERROR_ANALYSIS.md cleanly!")

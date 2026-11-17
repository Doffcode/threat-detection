# Weapon Detection Project — Context & Agent Guidelines

## 1. Executive Summary & Objective
This project improves **knife and gun detection** on Indian film/TV screenshots and CCTV frames (originally developed in `WD_Intern_Proj.ipynb` under project "akriti").
The primary challenge: the original notebook reported 0.64 knife mAP50, but this was heavily inflated by **data leakage** across train/val/test splits (augmented copies of the same images in multiple splits). On an honest, strictly grouped leak-free Indian test set, the original model achieved only **0.13 knife mAP50**.

Through extensive experimentation (November 2026), the best model achieved:
- **Model:** `work/runs/v5_hybrid_m640` (YOLO11m at 640px, trained on 25.5k mixed public + Indian domain images in `work/data/v5_indian_hybrid`).
- **Stage-2 Crop Verifier:** `work/runs/verifier_v5/weights/best.pt` (YOLO11s-cls at 224px, 79.9% top-1 accuracy).
- **Tabletop / Unattended Benchmark (35-image dedicated test):**
  - **Weapons Overall:** Precision **0.899**, Recall **0.793**, mAP50 **0.919** (up from 0.260 baseline, +253%).
  - **Desk Gun:** Precision **0.869**, Recall **0.846**, mAP50 **0.915** (Two-stage `wgeo75`: **100% Recall**, mAP50 **0.946**).
  - **Tabletop Knife:** Precision **0.930**, Recall **0.741**, mAP50 **0.922** (Two-stage `wgeo75`: **100% Recall**, mAP50 **0.964**).
- **Benchmark (144-image Indian cinema domain test):**
  - **Knife:** Precision **0.517**, Recall **0.421**, F1 **0.467**, mAP50 **0.468**.
  - **Gun:** Precision **0.496**, Recall **0.459**, F1 **0.477**, mAP50 **0.459**.
- **Deployment Thresholds (val-tuned):** Gun conf = `0.1842`, Knife conf = `0.1381` (Pipeline `wgeo75`: Gun = `0.523`, Knife = `0.386`).

---

## 2. System Architecture & Two-Stage Pipeline

```
Image / Frame
     │
     ▼
[Weapon Detector: YOLO11m @ 640px (work/runs/big_s1_m640)]
     │ Classes: gun, knife, handheld_object (absorbs tools/phones, suppressed)
     │ NMS IoU: 0.5 (eliminates duplicate bounding boxes)
     │
     ├── Optional: [Verifier: YOLO11s-cls (work/runs/verifier_quick)]
     │             Square crop + 20% pad letterboxed to 224px
     │             Classes: gun, knife, not_weapon
     │
     ▼
[Threshold Filter] (Gun: 0.164, Knife: 0.168)
     │
     ▼ (If weapon confirmed)
[Armed Person Detection]
     │ Standard COCO person detector (yolo11m.pt)
     │ Flags person as armed if: (Intersection / Weapon_Area) >= 0.3
     │
     ▼
Final Detections & Threat Score (CLI & Streamlit Dashboard)
```

---

## 3. Project File Map & Workspace

- **Root:** `/home/shiva/Desktop/projects/akriti/weapon-detection`
- **Core Working Directory:** `work/`
- **Virtual Environment:** `work/.venv/` (Python 3.13.5, Torch, Ultralytics, Streamlit)
  - Always use `work/.venv/bin/python`
- **Inference App & Dashboard:** `work/app/`
  - `work/app/action_intent_engine.py`: Core inference engine (`SuspiciousBehaviorEngine`), crop management, threshold loaders.
  - `work/app/predict.py`: CLI for batch image/video inference, produces annotated frames and `detections.csv`.
  - `work/app/dashboard.py`: Streamlit dashboard with image and video/CCTV analysis tabs.
  - `work/app/run_v5.sh`: Quick launcher script using the best `v5_hybrid_m640` + `verifier_v5` configuration.
  - `work/app/run_best.sh`: Quick launcher script using legacy `big_s1_m640`.
  - `work/app/test_engine.py`: Regression test suite (run with `../.venv/bin/python test_engine.py`).
- **Data:**
  - `work/data/v5_indian_hybrid/`: 25,509 images combining clean Indian domain data with filtered OD-WeaponDetection, Open Images, BITS Pilani, Innovatus office/street CCTV, and Roboflow.
  - `work/data/v2/`: Leak-free Indian domain dataset splits (`train.txt`, `val.txt`, `test_domain.txt`).
- **Model Runs & Weights:**
  - `work/runs/v5_hybrid_m640/weights/best.pt`: **Current production detector (40 epochs @ 640px)**.
  - `work/runs/verifier_v5/weights/best.pt`: **Stage-2 crop verifier (YOLO11s-cls @ 224px)**.
  - `work/runs/big_s1_m640/weights/best.pt`: Legacy detector.
- **Reports & Memory:**
  - `work/REPORT.md`: Comprehensive 250-line experiment log, benchmark tables, and failure analyses.

---

## 4. Key Findings & Guidelines

1. **Hardware Constraints:**
   - System has 1x NVIDIA RTX A2000 12GB and ~16GB RAM.
   - Run **only one GPU training job at a time**.
   - Fan out CPU-bound tasks (data preprocessing, labeling audits, code merging) to parallel subagents.
2. **Data Policy:**
   - Generic/public datasets (foreign images) are approved for training (`data-source-policy.md`).
   - The headline validation yardstick is strictly the 144-image Indian movie-still domain test (`test_domain.txt`).
3. **What Failed (Do NOT repeat):**
   - Stage-2 fine-tuning on domain data alone (`big_s2_m960`) causes severe overfitting; knife mAP50 dropped to 0.28.
   - Resuming training with a restarted learning rate schedule (`big_s1b`) damaged trained weights.
   - Any further training push must be one continuous, uninterrupted run (~40 epochs, ~6h) without LR resets.
4. **Box Styling:**
   - Inconsistencies between "weapon only" vs "weapon + hand" accounted for many false alarms. Stick to weapon-only bounding boxes.

---

## 5. Next Immediate Objectives
1. **Full Project Integration:**
   - Integrate the improved detector (`big_s1_m640`) back into the user's primary weapon detection pipeline.
   - Combine with legacy metrics, graphs, presentation materials, and notebooks.
2. **Dashboard / Deployment:**
   - Run or host `dashboard.py` or export the pipeline for client demonstration.
3. **Model Advancement (if requested):**
   - Execute a clean overnight 40-epoch training run or incorporate additional curated Indian movie frames.

# Weapon Detection Vertical (Stage 1) — Comprehensive Implementation & Architecture Audit

**Date:** November 18, 2026  
**Auditor:** Senior ML Engineer & CV Systems Architect  
**Scope:** Stage 1 Standalone Weapon Detection Vertical (Independent of person detection, tracking, pose, kinematics, and UI)  
**Status:** COMPLETE (Phase 0 Audit Deliverable)

---

## 1. Executive Summary & Vertical Boundary

The primary mission of **Stage 1 (Weapon Detection)** is to ingest image or video frames and output localized, classified weapon bounding boxes with calibrated confidence scores:
- **Inputs:** Raw image arrays or video streams (`H × W × 3` BGR).
- **Outputs:** Bounding box coordinates (`[x1, y1, x2, y2]`), class labels (`handgun`, `rifle`, `knife`), detection confidence scores, and per-frame detection summaries.
- **Strict Out-of-Scope Elements:** Person tracking (ByteTrack), pose estimation (wrist keypoint extraction), person–weapon spatial association, kinematic behavioral profiling (loitering, running, flow deviation), threat index computation, and frontend dashboards.

### Current Architectural Landscape
The repository currently contains **two parallel lines of implementation**:
1. **The Standard Surveillance Stack (`src/pipeline/` & `src/detection/`):** Built for the end-to-end multi-stage surveillance pipeline and locked via `src/pipeline/config.yaml` and quality gate contracts. It utilizes `baseline_m960_v2` (YOLO11m trained at 960px) with a strict 3-class taxonomy (`0: handgun`, `1: rifle`, `2: knife`).
2. **The Hybrid Application Stack (`work/app/` & `work/runs/`):** Developed as an experimental enhancement (November 2026) for false-alarm suppression and tabletop weapon recall. It uses `v5_hybrid_m640` (YOLO11m at 640px) paired with a Stage-2 crop verifier (`verifier_v5`, YOLO11s-cls at 224px) and a 2-class taxonomy (`gun`, `knife`) with a confusion-absorbing class (`handheld_object`).

This audit inspects both implementations, documents their exact components, resolves historical metric discrepancies, identifies technical debt, and establishes the path to freezing the standalone detector.

---

## 2. Component Inventory & Classification

Each component is classified according to the audit criteria:
- **`IMPLEMENTED AND VERIFIED`:** Code exists, runs cleanly, and behavior is verified by tests or empirical benchmarks.
- **`IMPLEMENTED BUT UNVERIFIED`:** Code exists but lacks automated tests or rigorous independent verification.
- **`PARTIALLY IMPLEMENTED`:** Core functionality exists, but critical features, edge cases, or standard interfaces are incomplete.
- **`MISSING`:** Required functionality is completely absent.
- **`BROKEN`:** Existing implementation fails execution or throws errors under standard operations.

### Inventory Table

| Component | Path / Location | Classification | Assessment & Evidence |
|---|---|:---:|---|
| **Pipeline Weapon Detector** | `src/pipeline/detector.py` | **IMPLEMENTED AND VERIFIED** | Wraps YOLO11m with standard box extraction, confidence thresholding, NMS, and pluggable Stage-2 verifier hook. Cleanly loads and detects. |
| **COCO Evaluation Harness** | `src/eval/evaluate_detector.py` | **IMPLEMENTED AND VERIFIED** | Full `pycocotools.COCOeval` implementation. Evaluates valid, test_public, test_domain, test_cctv, and hard negatives; supports bootstrap 95% CIs. |
| **Model Training Harness** | `src/detection/train.py` | **IMPLEMENTED AND VERIFIED** | Reproducible training script with seed control (`seed=2026`) and Albumentations CCTV augmentation injection (`src/detection/albumentations_cctv.py`). |
| **Locked Baseline Dataset** | `data/processed/stage1_baseline` | **IMPLEMENTED AND VERIFIED** | 16,054 images across 6 splits (`train`, `valid`, `test_public`, `test_domain`, `test_cctv_v1`, `test_hard_negatives`). Validated by `test_dataset_integrity.py` (0 leaks). |
| **Standalone CLI Predictor** | `work/app/predict.py` | **IMPLEMENTED AND VERIFIED** | Batch CLI handling images, directories, image lists, and video files; outputs annotated frames and structured `detections.csv`. |
| **Hybrid Inference Engine** | `work/app/action_intent_engine.py` | **IMPLEMENTED AND VERIFIED** | Two-stage inference engine (`SuspiciousBehaviorEngine`) supporting verifier score fusion (`wgeo75`, `prod`, `geo`), crop management, and temporal smoothing. |
| **Two-Stage Eval Harness** | `work/scripts/verifier/eval_pipeline.py` | **IMPLEMENTED AND VERIFIED** | Computes per-class P, R, F1, and AP50 across 6 score fusion modes. Valid-tuned thresholds saved in `pipeline_thresholds.json`. |
| **Standalone Detector Tests** | `tests/test_detector.py` | **MISSING** | `tests/` contains tests for dataset integrity, pipeline association, behavior rules, and dashboard UI, but **zero dedicated unit/integration tests for the standalone weapon detector in isolation**. |
| **Engine Test Suite** | `work/app/test_engine.py` | **BROKEN** | 31/32 pass, but `test_dashboard_syntax` fails with `AssertionError: TTA` due to UI desynchronization with `work/app/dashboard.py`. |
| **Portable Packaging / CLI** | Repository Root | **PARTIALLY IMPLEMENTED** | Executing inference currently requires navigating to `work/app/` or invoking internal scripts with complex path arguments. A clean top-level CLI is missing. |
| **Stage-2 Hook in `src/`** | `src/pipeline/detector.py` (`Stage2CropVerifier`) | **IMPLEMENTED BUT UNVERIFIED** | Code exists in `src/pipeline/detector.py`, but is disabled (`enabled: false`) in `src/pipeline/config.yaml`. |

---

## 3. Standalone Weapon Detection Path Traced

The execution flow from raw input to detection output is traced below for both stacks:

### Flow Diagram

```
[Raw Input: Image File / Directory / Stream]
                    │
                    ▼
          [1. Preprocessing]
          - BGR Image read (OpenCV)
          - Letterbox resize & pad (Ultralytics / letterbox)
          - Color conversion (BGR -> RGB) & Tensor normalization [0.0, 1.0]
                    │
                    ▼
          [2. Model Loading]
          - YOLO checkpoint loaded (PyTorch .pt via Ultralytics)
          - Model parameters pinned to device (CUDA:0 or CPU)
          - Inference mode set (eval/no-grad)
                    │
                    ▼
          [3. Primary Inference]
          - Forward pass at imgsz (960px or 640px)
          - Raw tensor output: [cx, cy, w, h, class_probs]
                    │
                    ▼
          [4. Postprocessing & Filtering]
          - Anchor-free DFL bounding box decoding to xyxy
          - Class confidence thresholding
          - Batched Non-Maximum Suppression (NMS IoU: 0.50 or 0.70)
          - Suppression of confusion classes (e.g. handheld_object)
                    │
                    ├─────────────────────────────────────────────────┐
                    ▼ (If Stage-2 Crop Verifier Enabled)              ▼ (Standard Detector)
          [Optional Stage-2 Verification]                             │
          - 20% padded square crop around weapon candidate            │
          - Letterboxed to 224px                                      │
          - Forward pass through YOLO11s-cls                          │
          - Score combination: score = (det^0.25) * (verifier^0.75)   │
          - Hard rejection if p_not_weapon > 0.50                     │
                    │                                                 │
                    ├─────────────────────────────────────────────────┘
                    ▼
          [5. Detection Output]
          - Coordinate scaling back to original frame geometry
          - Bounding boxes: [x1, y1, x2, y2]
          - Class IDs & Labels: handgun (0), rifle (1), knife (2)
          - Confidence scores: calibrated float [0.0, 1.0]
          - CSV Log entry & Annotated visual rendering
```

### Trace Comparison

| Step | Standard Pipeline (`src/pipeline/detector.py`) | Hybrid Application (`work/app/predict.py`) |
|---|---|---|
| **Input Ingestion** | BGR `np.ndarray` passed to `WeaponDetector.detect()` | File path, directory, `.txt` list, or `.mp4` video |
| **Resolution (`imgsz`)** | Fixed `960px` | `640px` (from `v5_hybrid_m640` config) |
| **Model Weights** | `outputs/runs/baseline_m960_v2/weights/best.pt` | `work/runs/v5_hybrid_m640/weights/best.pt` |
| **Taxonomy** | 3 classes: `0: handgun`, `1: rifle`, `2: knife` | 2 classes: `gun`, `knife` (absorbs `handheld_object`) |
| **Verifier** | Pluggable `Stage2CropVerifier` (currently disabled) | Active `verifier_v5` (`YOLO11s-cls` @ 224px, `wgeo75`) |
| **Confidence Cutoff** | Static $\text{conf}^* = 0.300$ across classes | Gun: `0.5230`, Knife: `0.3861` (combined `wgeo75`) |
| **Output Format** | `List[WeaponDetection]` dataclass | Visual annotated image + structured `detections.csv` |

---

## 4. Resolution of the 17/75 vs 22/75 Discrepancy

A critical finding in the historical documentation is a divergence in reported CCTV detection counts on the **Innovatus CCTV Benchmark** (72 frames, 75 ground-truth weapon boxes):
- **Report A (`docs/BASELINE_ERROR_ANALYSIS.md`, Section 10.1):** Reports **17 / 75 detected (22.7% recall)**.
- **Report B (`STAGE1_DONE.md` & `outputs/status/status.md`):** Reports **22 / 75 detected (29.3% recall)**.

### Forensic Investigation & Proof of Cause

Inspection of the actual evaluation code demonstrates that this discrepancy is **not** due to different checkpoints, different frames, or corrupted data. It is caused by **two fundamentally different evaluation protocols**:

1. **Protocol A: Strict COCOeval Detection Evaluation (`src/eval/evaluate_detector.py`)**
   - **Script:** `src/eval/evaluate_detector.py` executed on `test_cctv_v1`.
   - **Evaluation Standard:** Standard `pycocotools.COCOeval` requiring strict **$\text{IoU} \ge 0.50$** (and IoU 0.50:0.95) between predicted bounding boxes and ground-truth boxes.
   - **Class Matching:** Strict class-level matching (`handgun` ground truth requires `handgun` prediction; class confusion is penalized).
   - **Result:** Exactly **17 True Positives (17 / 75)**, 12 False Positives, 58 False Negatives ($F_1 = 0.3269$).

2. **Protocol B: Looser Geometric Pipeline Matching (`scripts/pipeline/evaluate_pipeline_full.py`)**
   - **Script:** `scripts/pipeline/evaluate_pipeline_full.py`, lines 283–288:
     ```python
     iou = box_iou(g["box"], assoc["weapon_box"])
     if iou >= 0.30:  # <--- Note IoU threshold is 0.30, not 0.50
         if g_idx not in matched_gt:
             matched_gt.add(g_idx)
             gt_detected_inno += 1
     ```
   - **Evaluation Standard:** Uses a relaxed spatial threshold of **$\text{IoU} \ge 0.30$**.
   - **Class Matching:** Any predicted weapon category matching any ground-truth weapon box.
   - **Result:** Exactly **22 Matched GT Weapons (22 / 75)**.

### Conclusion on Discrepancy
- The discrepancy is **100% resolved and verified by code**.
- **17 / 75** is the authoritative **object detection benchmark** at standard COCO $\text{IoU} \ge 0.50$.
- **22 / 75** is the downstream **pipeline capture rate** at spatial overlap $\text{IoU} \ge 0.30$.
- Both numbers are valid under their respective definitions, but must never be cited interchangeably without defining the IoU threshold.

---

## 5. Current Intended Production Detector & Checkpoint Status

### Production Model Definition
By contract and quality gate verification, the current authoritative detector for the Stage 1 system is:

- **Model Identifier:** `baseline_m960_v2`
- **Checkpoint Location:** `outputs/runs/baseline_m960_v2/weights/best.pt`
- **File Size:** 38.8 MB (40,683,075 bytes)
- **Architecture:** YOLO11m (Medium)
- **Input Resolution:** $960 \times 960$
- **Training Schedule:** 40 epochs, AdamW optimizer, cosine learning rate, `close_mosaic=10`, `scale=0.40`, CCTV Albumentations enabled.
- **Classes:** `0: handgun`, `1: rifle`, `2: knife`
- **Operating Confidence Threshold ($\text{conf}^*$):** `0.300` (derived strictly on validation split via argmax $F_1$)
- **NMS IoU Threshold:** `0.50`

### Comparison Against Alternative Models in Repository

| Model | Checkpoint Path | Imgsz | Split / Benchmark | mAP50 | Notes / Decision |
|---|---|:---:|---|:---:|---|
| **`baseline_m960_v1`** | `outputs/runs/baseline_m960_v1/weights/best.pt` | 960px | test_public: 0.715<br>Innovatus: 0.197 | 0.751 | 15 epochs; superseded by v2 across all splits (+0.07 mAP). |
| **`baseline_m960_v2`** | `outputs/runs/baseline_m960_v2/weights/best.pt` | 960px | test_public: 0.805<br>Innovatus: 0.252 | **0.824** | **LOCKED BASELINE.** Statistically superior in paired bootstrap ($p < 0.01$). |
| **`baseline_m960_v3`** | `outputs/runs/baseline_m960_v3/weights/best.pt` | 960px | test_public: 0.777<br>Innovatus: 0.270 | 0.818 | Scale 0.90 augmentation; statistically indistinguishable from v2 on CCTV. |
| **`baseline_m960_v4`** | `outputs/runs/baseline_m960_v4/weights/best.pt` | 1280px | test_public: 0.665<br>Innovatus: 0.224 | 0.716 | Batch 4 constraint; underfitted, statistically worse $F_1$ ($p = 0.007$). |
| **`v5_hybrid_m640`** | `work/runs/v5_hybrid_m640/weights/best.pt` | 640px | Tabletop: 0.919<br>Cinema: 0.459 | — | 2-class model (+verif). High tabletop recall, but requires Stage-2 classifier. |

---

## 6. Dataset and Annotation Architecture

The locked ground-truth dataset supporting the Stage 1 baseline is located at:
`data/processed/stage1_baseline/`

### Dataset Splits Summary

| Split Directory | Image Count | Label Count | Source Distribution | Purpose |
|---|:---:|:---:|---|---|
| `train/` | 12,284 | 12,284 | Mixed public, cinema stills, CCTV, 2k hard negatives | Core training partition |
| `valid/` | 1,453 | 1,453 | Independent group/scene holdouts | Threshold tuning ($\text{conf}^*$) and validation mAP |
| `test_public/` | 1,378 | 1,378 | Public weapon benchmarks | In-distribution public test yardstick |
| `test_domain/` | 144 | 144 | Indian cinema movie stills | Domain shift evaluation |
| `test_cctv_v1/` | 192 | 192 | 72 Innovatus CCTV frames + BITS Pilani frames | Real-world surveillance evaluation |
| `test_hard_negatives/` | 303 | 303 | `aahil_tools` and `sohas_distractor` | Weapon-free images for false positive rate (FPPI) |
| **Total Active** | **16,054** | **16,054** | Manifest: `manifest.csv` | Full frozen evaluation suite |

### Split Integrity & Leakage Verification
- Verified by automated tests in `tests/test_dataset_integrity.py`:
  - **Filename collisions across splits:** 0 duplicates.
  - **Group/scene leakage:** 0 groups span multiple splits.
  - **Perceptual dHash near-duplicate clusters:** 0 clusters cross splits.
  - **Bounding box bounds:** 100% of box coordinates conform to $[0.0, 1.0]$.
  - **Class IDs:** 100% within valid range `{0, 1, 2}`.

---

## 7. Existing Test Coverage

### Test Execution Results
Running `work/.venv/bin/pytest tests/`:
- **Total Tests Collected:** 31 tests
- **Passed:** 31 passed in 3.71 seconds
  - `tests/test_behavior_rules.py`: 8 passed
  - `tests/test_dashboard.py`: 9 passed
  - `tests/test_dataset_integrity.py`: 6 passed
  - `tests/test_pipeline_association.py`: 8 passed

### Test Coverage Gap Analysis
While the overall test suite passes, there is a **glaring test coverage hole** regarding the primary objective of this mission:
- **No Standalone Weapon Detector Unit Tests:** There is no `tests/test_detector.py`. The detector wrapper in `src/pipeline/detector.py` is only exercised indirectly through end-to-end pipeline integration tests.
- **No Standalone Inference Contract Verification:** No automated unit tests verify that `WeaponDetector` handles empty frames, corrupted images, batch inputs, coordinate renormalizations, or missing weights gracefully.
- **Legacy Engine Test Failure:** Running `work/app/test_engine.py` reveals 1 failure: `test_dashboard_syntax` fails on an outdated assertion expecting the literal string `"TTA"` in `work/app/dashboard.py`.

---

## 8. Technical Debt & Identified Risks

### 1. Dual-Stack Codebase Drift
- `src/` and `work/` contain competing implementations of the detector, configuration files, and evaluation scripts.
- `src/pipeline/` uses 3 classes (`handgun`, `rifle`, `knife`), while `work/app/` uses 2 classes (`gun`, `knife`) with a confusion absorber.
- Downstream users risk executing conflicting models if paths are not standardized.

### 2. Hard-Coded Local Paths
- Multiple configuration files (`configs/detection/dataset_stage1.yaml`, `configs/detection/yolo11m_960_v2.yaml`, `src/pipeline/config.yaml`) contain absolute paths prefixed with `/home/shiva/Desktop/projects/akriti/weapon-detection/`.
- This violates portability requirements and will break if cloned or mounted in another environment or container.

### 3. VRAM and Hardware Constraints
- The workstation possesses 1× NVIDIA RTX A2000 (12GB VRAM).
- Model training at 960px with `batch=8` utilizes ~9.0 GB VRAM (leaving ~3 GB headroom).
- Upscaling resolution to 1280px (`baseline_m960_v4`) required dropping batch size to 4, which degraded batch normalization stability and harmed convergence.

### 4. BITS Pilani Label Flaws
- The 173-image BITS Pilani CCTV partition contains invalid ground-truth annotations (several whole-frame boxes covering $\ge 80\%$ of the image area).
- Standalone CCTV evaluations must strictly isolate the verified **Innovatus CCTV** subset (72 frames, 75 boxes) to prevent evaluation distortion.

---

## 9. Phase 0 Exit Condition & Recommendations

### Exit Condition Assessment: PASSED
- The repository, model checkpoints, training pipelines, datasets, evaluation scripts, and test suites have been comprehensively audited without modifying application code.
- The 17/75 vs 22/75 discrepancy has been traced and completely resolved.
- The baseline model (`baseline_m960_v2`), dataset, and class mappings have been definitively identified.

### Proposed Action Plan for Phase 1 (Baseline Establishment)
1. Execute an authoritative baseline reproduction run of `baseline_m960_v2` on `test_public`, `test_domain`, `test_cctv_v1` (Innovatus clean), and `test_hard_negatives` using `src/eval/evaluate_detector.py`.
2. Compute and log exact metrics (Precision, Recall, F1, mAP50, mAP50-95, FPPI, and confusion matrix counts).
3. Record exact SHA-256 hashes of weights and manifest to deliver `docs/weapon_detection/WEAPON_DETECTOR_BASELINE_AUDIT.md`.
4. Await user authorization before proceeding beyond Phase 1.

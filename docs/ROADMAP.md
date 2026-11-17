# Project Roadmap & Stage 1 Progress Checklist

## Master Stage Overview
* [ ] **Stage 1: Weapon Detection Tied to Persons** (CURRENT FOCUS)
* [ ] **Stage 2: Behavior Analysis for Harmful-Intent Signals**
* [ ] **Stage 3: Learned Anomaly Detection & Video Understanding**
* [ ] **Stage 4: Integration, TensorRT Optimization & SOC Deployment**

---

## Stage 1 Checklist (Step-by-Step)

- [x] **Step 0: Setup, Specification & Indian Domain Mandate**
  - [x] Verify environment and RTX A2000 12GB GPU access.
  - [x] Create modular repository layout (`configs/`, `data/`, `src/`, `docs/`, `outputs/`).
  - [x] Write `docs/PROJECT_CONTEXT.md` with explicit Indian context mandate (attire, desi kattas, aruvals, CCTV realities).
  - [x] Write `docs/ROADMAP.md` with detailed step-by-step review gates.
  - [ ] Await user review and approval before starting Step 1.

- [x] **Step 1: Dataset Audit (Public + Indian Context)**
  - [x] Select and audit one clean weapon dataset (OD-WeaponDetection / Sohas: 11,020 XMLs, 12,163 boxes).
  - [x] Inspect labels, class balance, and small-object pixel distributions (discovered 87.9% large, only 3.9% in 20–60 px CCTV range).
  - [x] Audit Indian-specific weapon representations (uncovered 15.1% cross-split leakage and severe label swaps in raw Roboflow/cinema collection).
  - [x] Document annotation errors, near-duplicates, and small-object biases in `docs/DATASET_AUDIT_REPORT.md`.

- [ ] **Step 2: Baseline Weapon Detector & Error Analysis**
  - [ ] Build leak-free, source-grouped 3-class dataset (`handgun`, `rifle`, `knife`) with dHash deduplication and unlabeled hard negatives.
  - [ ] Run 1-epoch plumbing check at real settings (960px, batch 8, FP16) on RTX A2000 to measure peak VRAM and step time.
  - [ ] Fine-tune YOLO11m baseline (15 epochs) with CCTV Albumentations (blur, noise, low-light, compression).
  - [ ] Evaluate per-class precision, recall, mAP@0.5, mAP@0.5:0.95, and PyCOCOtools AP-small/med/large on composite val, public test, and locked test sets (`test_domain`, `test_cctv`).
  - [ ] Measure False Positives Per Image (FPPI) on 300-image hard negative set.
  - [ ] Report per-source metrics, occlusion-level recall, bootstrap 95% confidence intervals, and structured error analysis.

- [ ] **Step 3: Person Detection & Tracking**
  - [ ] Integrate YOLO person detector with ByteTrack.
  - [ ] Benchmark tracking stability (track ID switches, track lifetime) on dense Indian crowd and loose attire footage.

- [ ] **Step 4: Pose Keypoints & Person-Weapon Association**
  - [ ] Extract hand/wrist keypoints via YOLO-Pose, testing resilience against loose sleeves, kurtas, and dupattas.
  - [ ] Implement spatial association (box overlap + hand keypoint Euclidean distance).
  - [ ] Attribute weapons to specific `track_id`s ("Person X holding handgun / knife").

- [ ] **Step 5: Hard Negatives & Temporal Filtering**
  - [ ] Integrate Indian everyday look-alikes (steel water bottles, tiffin carriers, umbrellas, tools, phones).
  - [ ] Implement $N$-frame temporal confirmation per track ID.
  - [ ] Measure false alarms per camera per hour on clean CCTV footage.

- [ ] **Step 6: Realistic CCTV Evaluation & Per-Condition Tables**
  - [ ] Evaluate on realistic CCTV-style footage (Indian store/street angles, cinema action sequences).
  - [ ] Generate per-condition breakdown tables:
    - Occlusion levels (0–25%, 25–50%, 50–75%).
    - Occluder types (hand, dhoti/lungi waistband, loose clothing, bag).
    - Lighting conditions (daylight, night IR, harsh yellow streetlamp).
  - [ ] Document failure-case analysis.

- [ ] **Step 7: Stage 1 Report & Stage 2 Handoff**
  - [ ] Document final metrics, latency profile on RTX A2000, and operational limitations.
  - [ ] Define clean API / interface contracts for Stage 2 consumption (track IDs, keypoints, weapon events).

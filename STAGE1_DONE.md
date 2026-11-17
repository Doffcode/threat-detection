# Stage 1 Person-Weapon Tracking Pipeline — Sign-Off & Review Package

**Date:** November 17, 2026  
**Pipeline Status:** COMPLETE & VERIFIED (Unit Tests: 7/7 Passed; E2E & Oracle Benchmarks Completed)  
**Locked Weapon Detector:** `baseline_m960_v2` (`outputs/runs/baseline_m960_v2/weights/best.pt`)

---

## 1. Deliverables & File Map

### Core Pipeline Code (`src/pipeline/`)
- [`src/pipeline/config.yaml`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/pipeline/config.yaml): Decoupled YAML configuration (detector, ByteTrack tracker, YOLO11-pose, associator, temporal manager, Stage 2 verifier placeholder).
- [`src/pipeline/detector.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/pipeline/detector.py): Detector wrapper with pluggable Stage 2 verification hook.
- [`src/pipeline/tracker.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/pipeline/tracker.py): Person detection and ByteTrack tracking (`persist=True`, track buffer 30).
- [`src/pipeline/pose.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/pipeline/pose.py): Pose estimator extracting wrist keypoints (indices 9 & 10) with graceful fallback.
- [`src/pipeline/associator.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/pipeline/associator.py): Spatial containment, wrist proximity relative to person height, center check, and Hungarian/max-score matching.
- [`src/pipeline/temporal.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/pipeline/temporal.py): $N=3$ frame persistence confirmation, alert records, crop export, and event logging.
- [`src/pipeline/engine.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/pipeline/engine.py): Master runner coordinating stream inference, visualization, and JSON output.

### Tests & Verification Scripts
- [`tests/test_pipeline_association.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/tests/test_pipeline_association.py): Synthetic unit tests for all 7 core behaviors (all 7 pass in 0.92s).
- [`scripts/pipeline/evaluate_pipeline_full.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/scripts/pipeline/evaluate_pipeline_full.py): Complete harness executing the Oracle test, E2E benchmarks, audit sheet generation, and gallery curation.

### Output Artifacts (`outputs/pipeline/`)
- **Annotated Demo Video:** [`outputs/pipeline/stage1_cctv_demo.mp4`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/stage1_cctv_demo.mp4) (H.264 video with bounding boxes, track IDs, yellow link lines, and top red alert banners).
- **Structured Event Audit Log:** [`outputs/pipeline/events.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/events.json) (audit trail of all alert records).
- **Comprehensive Metrics JSON:** [`outputs/pipeline/pipeline_metrics.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/pipeline_metrics.json)
- **Oracle Association Metrics JSON:** [`outputs/pipeline/oracle_metrics.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/oracle_metrics.json)
- **Association Audit Sheet:**
  - CSV: [`outputs/pipeline/audit_sheet/association_audit_sheet.csv`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/audit_sheet/association_audit_sheet.csv) (40 sampled links with blank judgment column).
  - Evidence Crops: [`outputs/pipeline/audit_sheet/crops/`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/audit_sheet/crops/) (weapon in red, person in green, link in yellow).
- **Curated Galleries:**
  - 20 Good Cases: [`outputs/pipeline/galleries/good/`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/galleries/good/) (manifest: [`manifest.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/galleries/good/manifest.json)).
  - 20 Bad Cases: [`outputs/pipeline/galleries/bad/`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/galleries/bad/) (manifest: [`manifest.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/galleries/bad/manifest.json)).

### Documentation
- [`docs/STAGE1_PIPELINE.md`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/docs/STAGE1_PIPELINE.md): Full pipeline design, thresholds, caveats, limits, and Stage 2 plug-in architecture.
- [`docs/BASELINE_ERROR_ANALYSIS.md`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/docs/BASELINE_ERROR_ANALYSIS.md): Updated with Part 0 clean Innovatus CCTV evaluation, fixed bootstrap CIs, and unified v1–v4 comparison.

---

## 2. Quantitative Summary Table

| Metric | Measured Value | Benchmark Context |
|---|:---:|---|
| **Pipeline Latency** | **59.6 ms (16.8 FPS)** | Tested on NVIDIA RTX A2000 12GB (Detector @ 960px + Tracker @ 640px + Pose @ 640px) |
| **Oracle Association Share (Innovatus)** | **98.7% (74 / 75)** | Association accuracy independent of detector |
| **Oracle Association Share (test_domain)** | **57.3% (121 / 211)** | Includes 90 close-up/tabletop weapon stills without persons |
| **End-to-End GT Weapons Detected** | 22 / 75 (29.3%) | Detector recall on surveillance-scale weapons (<5% area) |
| **End-to-End GT Weapons Linked to Person** | **21 / 75 (28.0%)** | **95.5% of all detected weapons were successfully linked** |
| **Track ID Switches on CCTV Stretches** | **0 switches** | Across all 3 continuous sequences (`seq02`, `seq07`, `seq08`) |
| **Confirmed Alert Events Generated** | 2 events | `ALERT-0001` (Track 2, knife/gun) & `ALERT-0002` (Track 1, handgun, 13 hits) |
| **False Alerts on Empty Sequences** | **0** | Isolated blips in `seq08` suppressed by $N=3$ filter |

---

## 3. 15-Minute Morning Checklist for Reviewer

1. **Watch Demo Video (3 mins):**
   - Play [`outputs/pipeline/stage1_cctv_demo.mp4`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/stage1_cctv_demo.mp4).
   - Observe tracking on Sequence 07 (actor with handgun tracked across 13 frames with red alert banner).
2. **Review Association Audit Sheet (5 mins):**
   - Open [`outputs/pipeline/audit_sheet/association_audit_sheet.csv`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/audit_sheet/association_audit_sheet.csv).
   - Inspect 5–10 sample crops in [`outputs/pipeline/audit_sheet/crops/`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/audit_sheet/crops/).
   - Fill in the blank `judgment` column with your evaluation (Correct / Incorrect / Ambiguous).
3. **Inspect Failure Mode Gallery (4 mins):**
   - Browse [`outputs/pipeline/galleries/bad/`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/galleries/bad/).
   - Check the low-link-score cases and missed weapon false negatives.
4. **Run Unit Tests (3 mins):**
   - Execute: `work/.venv/bin/pytest tests/test_pipeline_association.py -v`
   - Confirm all 7 unit tests pass cleanly.

---

---

## 5. Stage 2 Person Movement & Behavior Analysis Deliverables

### Core Behavior Engine (`src/behavior/`)
- [`src/behavior/config.yaml`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/behavior/config.yaml): YAML configuration for kinematic windows, thresholds, and zones.
- [`src/behavior/kinematics.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/behavior/kinematics.py): Normalized velocity ($v/H$), smoothed heading, direction reversal, path length, and displacement ratio.
- [`src/behavior/rules.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/behavior/rules.py): Rule engines for Loitering, Running, Abrupt Stop/Reversal, and Moving Against Flow. All events stamped `FOR_HUMAN_REVIEW_ONLY`.
- [`src/behavior/visualizer.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/src/behavior/visualizer.py): 2D trajectory plotting and live video rendering with trails, telemetry overlays, and alert banners.

### Tests & Verification Scripts
- [`tests/test_behavior_rules.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/tests/test_behavior_rules.py): Deterministic synthetic trajectory test suite (8/8 passed in 0.09s including slow walking, mid-frame track start, and short track edge cases).
- [`scripts/behavior/run_behavior_analysis.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/scripts/behavior/run_behavior_analysis.py): End-to-end CCTV runner for Innovatus sequences.

### Behavioral Outputs (`outputs/behavior/`)
- **Annotated Demo Video:** [`outputs/behavior/behavior_cctv_demo.mp4`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/behavior_cctv_demo.mp4) (H.264 video with real-time speed, heading, trajectory trails, and alert banner).
- **Trajectory Maps:**
  - Sequence 02: [`outputs/behavior/trajectories/innovatus_cam_b_seq02_trajectories.png`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/trajectories/innovatus_cam_b_seq02_trajectories.png)
  - Sequence 07: [`outputs/behavior/trajectories/innovatus_cam_b_seq07_trajectories.png`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/trajectories/innovatus_cam_b_seq07_trajectories.png)
  - Sequence 08: [`outputs/behavior/trajectories/innovatus_cam_b_seq08_trajectories.png`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/trajectories/innovatus_cam_b_seq08_trajectories.png)
- **Rule Firings CSV:** [`outputs/behavior/rule_firings.csv`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/rule_firings.csv) (32 events, each containing track ID, frame, rule, exact numbers, and review flag).
- **Summary JSON:** [`outputs/behavior/behavior_summary.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/behavior_summary.json).

### Stage 2 Documentation
- [`docs/STAGE2_MOVEMENT.md`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/docs/STAGE2_MOVEMENT.md): Comprehensive documentation covering mathematical formulas, rule definitions, synthetic testing, CCTV results, ethical bias safeguards, and operational limitations.


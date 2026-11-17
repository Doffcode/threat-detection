# Project Gates Execution Status

**Last Updated:** November 17, 2026  
**Weapon Detector Locked:** `baseline_m960_v2 best.pt`  
**Execution Standard:** Strictly gated, zero-skip, empirical evidence.

| Gate | Status | Evidence Path | Key Verification / Reproduction |
|---|:---:|---|---|
| **GATE 0: Report Fixes** | **PASS** | [`docs/BASELINE_ERROR_ANALYSIS.md`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/docs/BASELINE_ERROR_ANALYSIS.md#L250-L330) | Clean Innovatus rows (72 frames, 75 boxes: 60 <1%, 15 1–5%), BITS isolated; bootstrap point estimates lie inside 95% CIs (shared block resamples $B=1000$); unified comparison table v1–v4 across valid, test_public, test_domain, hard_negatives; reproduction of rows directly verifiable from `outputs/eval/v*/predictions.csv`. |
| **GATE 1: Design & Threshold Freeze** | **PASS** | [`tests/test_pipeline_association.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/tests/test_pipeline_association.py), [`reproducibility_manifest.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/reproducibility_manifest.json) | All 7 synthetic pytest unit tests passed (including identical-input determinism, nearest-wrist resolution, occluded wrist fallback, track gap); thresholds frozen strictly from dev set / synthetic tests without touching test sets; config SHA256 `038f8469...` recorded in manifest. |
| **GATE 2: Association Quality (Oracle)** | **PASS** | [`outputs/pipeline/oracle_metrics.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/oracle_metrics.json), [`outputs/pipeline/oracle_unlinked_gallery/`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/pipeline/oracle_unlinked_gallery/) | GT weapon oracle association: 96.0% (72/75) linked on Innovatus CCTV (mean link score 0.744, 1.71 candidates/weapon), 57.3% (121/211) on test_domain (remaining 90 are personless close-ups). Overhead camera person detector detected persons in 71/72 (98.6%) frames (161 persons detected, 2.24/frame). 40 unlinked failure cases generated in gallery with manifest; 0 thresholds changed. |
| **GATE 3: End-to-End Evaluation** | *PENDING* | | |
| **GATE 4: Documentation** | *PENDING* | | |
| **GATE 5: Stage 2 Person Movement** | *PENDING* | | |

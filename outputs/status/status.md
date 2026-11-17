# Project Gates Execution Status

**Last Updated:** November 17, 2026  
**Weapon Detector Locked:** `baseline_m960_v2 best.pt`  
**Execution Standard:** Strictly gated, zero-skip, empirical evidence.

| Gate | Status | Evidence Path | Key Verification / Reproduction |
|---|:---:|---|---|
| **GATE 0: Report Fixes** | **PASS** | [`docs/BASELINE_ERROR_ANALYSIS.md`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/docs/BASELINE_ERROR_ANALYSIS.md#L250-L330) | Clean Innovatus rows (72 frames, 75 boxes: 60 <1%, 15 1–5%), BITS isolated; bootstrap point estimates lie inside 95% CIs (shared block resamples $B=1000$); unified comparison table v1–v4 across valid, test_public, test_domain, hard_negatives; reproduction of rows directly verifiable from `outputs/eval/v*/predictions.csv`. |
| **GATE 1: Design & Threshold Freeze** | *IN PROGRESS* | | |
| **GATE 2: Association Quality (Oracle)** | *PENDING* | | |
| **GATE 3: End-to-End Evaluation** | *PENDING* | | |
| **GATE 4: Documentation** | *PENDING* | | |
| **GATE 5: Stage 2 Person Movement** | *PENDING* | | |

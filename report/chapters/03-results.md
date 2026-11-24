# Evaluation protocol and results

## Results by branch

TN, FP, FN, and TP denote true negatives, false positives, false negatives, and true positives.

The primary results table identifies the evaluation unit and operating point. Rows are drawn from different tasks or datasets. Changes in dataset, labels, detector checkpoint, and evaluation procedure limit comparisons across rows.

| Model and evaluation unit | Dataset and operating point | Verified result | Source |
|---|---|---|---|
| X3D-M, whole clip | SCFD, 300 clips, 150 per class. Threshold 0.40. | **91.0% accuracy**, **90.4% fight F1**. 146 TN, 4 FP, 23 FN, 127 TP. | F |
| YOLO11m `baseline_m960_v2`, bounding box | Innovatus CCTV, 72 images and 75 weapon annotations. Confidence 0.30 for precision, recall, and F1. Match IoU ≥ 0.50. | **0.252 mAP@0.50**, 58.6% precision, 22.7% recall, 32.7% F1. 17 TP, 12 FP, 58 FN. | W |
| YOLO11m `v5_hybrid_m640`, detector only, bounding box | Tabletop validation, 35 images and 40 annotations. 640 pixels. Average precision (AP) at IoU 0.50 across the score ranking. | **0.919 mAP@0.50**. Exported precision 89.9% and recall 79.3% at the validator's summary operating point. | T |
| Approach-and-retreat rule, cue interval | 24 SCFD clips. 15 labeled intervals, 6 cue-free clips, 3 unclear clips. Rule thresholds in methodology. | **0 alerts**, **0/15 matched intervals**, 0/6 cue-free clips flagged, 0 processing errors. Interval validity caveat below. | M |

*Table 1. Verified results and source identifiers. TN, FP, FN, and TP mean true negative, false positive, false negative, and true positive.*

## Fight classification

SCFD contains 150 fight and 150 non-fight clips in this run. All 300 received predictions. @fig-confusion uses rows for the actual label and columns for the predicted label. The frozen checkpoint was compared on a separate RWF-2000 validation sample before selection. The evaluation checked file hashes against 40 accessible RWF-2000 selection files and found no exact matches. The mirror's renamed files prevented a source identifier check. Overlap with the full training corpus and re-encoded duplicates could not be ruled out. The report therefore describes SCFD as a separate dataset evaluation with limited overlap verification.

![SCFD clip-level confusion matrix at the frozen 0.40 threshold.](figures/fight_confusion.png){#fig-confusion width=58%}

The following values were **derived from the saved clip predictions**. Positive-class metrics use fight as the positive label. F1 is the harmonic mean of precision and recall and is computed from the integer counts before rounding.

| Metric | Calculation | Result |
|---|---|---:|
| Accuracy | (127 + 146) / 300 | 91.0% |
| Fight precision | 127 / (127 + 4) | 96.9% |
| Fight recall | 127 / (127 + 23) | 84.7% |
| Fight F1 | (2 × 127) / (2 × 127 + 4 + 23) | 90.4% |
| Non-fight precision | 146 / (146 + 23) | 86.4% |
| Non-fight recall | 146 / (146 + 4) | 97.3% |
| Non-fight F1 | (2 × 146) / (2 × 146 + 23 + 4) | 91.5% |
| Balanced accuracy | (127/150 + 146/150) / 2 | 91.0% |
| False-positive rate among non-fight clips | 4 / 150 | 2.7% |
| False-negative rate among fight clips | 23 / 150 | 15.3% |

*Table 2. Derived SCFD metrics. All rows use the model, threshold, dataset, and evaluation unit in source F.*

## Weapon detection

The Innovatus operating-point export contains 17 true-positive matches, 12 false positives, and 58 misses at confidence 0.30. **Derived** precision is 17/(17 + 12), recall is 17/(17 + 58), and F1 is (2 × 17)/(2 × 17 + 12 + 58). mAP@0.50 summarizes class precision–recall curves at the stated matching IoU and cannot be recovered from these three operating-point counts alone. The complete export preserves the stricter aggregate AP metric and the alternate confidence operating point.

The tabletop detector-only export reports 0.918835319 mAP@0.50, rounded to 0.919. Its split manifest lists 35 images, and the dataset configuration points both validation and test to that same list. The result is therefore reported as tabletop validation. The export's gun and knife supports are 13 and 27, respectively. The common scalar confidence threshold for its summary precision and recall is unverified in the export. The per-class threshold estimates in that export are distinct quantities. The crop verifier was absent from the evaluation that produced 0.919.

## Movement evidence

The frozen review sample contains 24 clips, selected before movement scoring. The review presents the clip without fight labels or model outputs. The scorer counts unclear clips separately and pairs an alert with at most one labeled interval containing the alert timestamp. Saved labels record 15 cue-present clips with one interval each, 6 cue-free clips, and 3 unclear clips. The comma-separated values (CSV) alert file has no data rows, and the error file is empty. The saved result is therefore 0 matches, 15 misses under the recorded labels, and 0 unmatched alerts. The **derived recorded recall** is 0/15, or 0%.

Every positive interval is recorded as **0.00–0.02 s**. At this duration, the labels do not supply a plausible reference for a movement rule that observes a nominal 0.5-second history. The zero-alert observation is reproducible from the saved files. Valid cue recall remains unverified until the intervals are independently checked against the footage. Zero cue-free flags on six clips gives a count for this sample.

![Separate SCFD demonstration clip `fi005.mp4`. The cyan track approaches the orange track as the latter retreats. The measured image-plane gap narrows from 0.78 to 0.47 box heights near 1.25 seconds.](figures/movement_example.png){#fig-movement width=80%}

## Source artifacts

- **F:** `outputs/eval/behavior_scfd/README.md`, `manifest.csv`, `local_metrics.json`, and `local_predictions.csv`. The same directory contains the V100 predictions and metrics.
- **W:** `outputs/eval/cctv_test_v2/cctv_test_v2_metrics.json`, key `v2.innovatus_rob`. The matching procedure is documented in `docs/weapon_detection/WEAPON_DETECTOR_BASELINE_AUDIT.md`.
- **T:** `work/runs/v5_hybrid_m640/metrics_val_tabletop.json` and `work/data/v5_indian_hybrid/val_tabletop.txt`. The separate verifier export is `work/runs/v5_hybrid_m640/pipeline/pipeline_test_tabletop.json`.
- **M:** `outputs/eval/movement_review/manifest.csv`, `labels.json`, `alerts.csv`, `errors.json`, and `summary.json`. The rule and replay code are in `src/behavior/video_review.py`.

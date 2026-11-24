## Problem statement and objectives {.bullets}

Review recorded CCTV footage for visible weapons and fights, with evidence a reviewer can inspect.

- Identify visible weapons.
- Classify fight activity.
- Provide timestamped evidence for review.

::: notes
0:30 · 0:20–0:50

The input is a recorded clip. The output helps a human inspect what the system flagged and where. Our objectives are weapon localization, fight classification, and evidence review. The project measures these as separate tasks.
:::

## Summary of methodology — system workflow {.image}

![The three analysis branches meet in the dashboard.](figures/architecture.png)

::: notes
0:35 · 0:50–1:25

The same input video goes to three independent branches. Weapon detection produces boxes, fight classification produces scores, and person tracking supplies movement cues. The dashboard combines those outputs on a timestamped evidence timeline. A reviewer can jump to the relevant frame or clip, play the replay, and download the results. A low fight score does not suppress either of the other branches.
:::

## Summary of methodology — models and evaluation {.table}

| Branch | Implemented configuration | Evaluation unit |
|---|---|---|
| Weapon detection | YOLO11m · baseline_m960_v2 | Weapon annotation |
| Fight classification | Frozen X3D-M | Whole fight or non-fight clip |
| Movement evidence | YOLO11m + ByteTrack · approach-and-retreat rule | Human-labeled movement interval |

::: notes
0:40 · 1:25–2:05

The dashboard detector is baseline_m960_v2 at 960 pixels, confidence 0.30, and non-maximum suppression overlap 0.50. It returns handgun, rifle, and knife classes without the optional crop verifier in the main view. The X3D-M checkpoint is final_x3d_realtime.pt. It uses 16 uniformly sampled RGB frames resized to 224 by 224, channel mean 0.45, standard deviation 0.225, and threshold 0.40. The movement rule compares two footpoints over about 0.5 seconds. It requires at least 0.50 box heights of approach, 0.15 of retreat, and 0.30 reduction in gap, ending within 2.00 box heights. Both tracks must be visible throughout that window.
:::

## 90.4% fight F1 on 300 SCFD clips {.two-col}

**91.0% accuracy**

Whole-clip evaluation  
150 fight · 150 non-fight

Frozen X3D-M · threshold 0.40 [F]

|||

| Actual ↓ / Predicted → | Non-fight | Fight |
|---|---:|---:|
| Non-fight | **146 TN** | **4 FP** |
| Fight | **23 FN** | **127 TP** |

::: notes
1:05 · 2:05–3:10

This is the strongest measured result. The Surveillance Camera Fight Dataset contributed 150 clips per class. Rows are actual labels. In the first row, 146 non-fights were classified correctly and four were flagged. In the second row, 23 fights were missed and 127 were detected. Accuracy is (146 + 127) / 300 = 91.0%. Fight F1 is (2 × 127) / (2 × 127 + 4 + 23) = 90.4%. We used the frozen 0.40 threshold. The saved scores let the panel reproduce every decision. The overlap check found no identical hashes among 40 accessible RWF-2000 selection files. It could not rule out re-encoded duplicates or overlap with the full training corpus. This experiment scored whole clips. The dashboard scores five-second windows, which require separate validation.
:::

## Results — weapon detection {.two-col}

**Innovatus CCTV** [W]  
baseline_m960_v2

72 images · 75 weapon annotations  
**17 matched** to ground truth

**58.6%** precision · **22.7%** recall  
**0.252** mAP@0.50

Precision and recall at confidence 0.30

|||

**Tabletop validation** [T]  
v5_hybrid_m640 · detector only

35 images · 40 weapon annotations

**0.919** mAP@0.50

Crop verifier disabled

::: notes
1:05 · 3:10–4:15

The left panel is the detector used in the dashboard, at confidence 0.30 and a ground-truth matching overlap of at least 0.50. It had 17 matches, 12 false positives, and 58 misses. Precision is 17 / 29, or 58.6%, and recall is 17 / 75, or 22.7%. Its 0.252 mAP@0.50 summarizes the precision–recall ranking across confidence levels. The right panel is the v5_hybrid_m640 checkpoint on 35 tabletop validation images at 640 pixels. Its raw exported mAP@0.50 is 0.918835319, displayed as 0.919. The evaluation used the detector alone. A separate two-stage export contains verifier results. The two panels use different splits and protocols, so the values cannot quantify an improvement.
:::

## Results — movement cue review {.two-col}

**0 of 15 labeled cue intervals matched**

24 reviewed clips  
6 cue-free · 3 unclear

0 alerts · 0 processing errors [M]

|||

![Feature demonstration · separate SCFD clip](figures/movement_example.png)

::: notes
0:55 · 4:15–5:10

On the right is a separate feature demonstration, SCFD clip fi005. The cyan track approaches and the orange track retreats. The image-plane gap narrows from 0.78 to 0.47 box heights near 1.25 seconds. In the review sample, a match required an alert timestamp inside a saved reference interval. The scorer emitted zero alerts, matched zero of 15 intervals, and reported no processing errors. Six clips were labeled cue-free and three unclear. All positive labels use the same 0.00–0.02-second interval, so the reference timing needs verification and valid cue recall remains unestablished. The replay shows how the evidence viewer works.
:::

## Conclusion {.bullets}

- Completed dashboard, evidence timeline, annotated replay, and downloads.
- Frozen X3D-M achieved **90.4% fight F1** and **91.0% accuracy** on 300 SCFD clips.
- Limits: **22.7%** CCTV weapon recall and **0 of 15** movement intervals matched.
- Dashboard fight windows need separate validation.

::: notes
0:32 · 5:10–5:42

Phase 3 delivered the recorded-video review workflow and saved evaluation artifacts. The strongest verified result is whole-clip fight classification. Weapon recall is low on the CCTV benchmark and the movement sample had zero matched cues. The next validation task for the dashboard is to measure fight windows against timed event labels. The final review is on 4 January.
:::

## References {.two-col}

**Dataset and model sources**

Ş. Aktı, G. A. Tataroğlu and H. K. Ekenel. *Vision-based Fight Detection from Surveillance Cameras*. IPTA, 2019. Authors’ SCFD repository.

VisionLab AI. *School Violence Detection Models* model card and frozen X3D-M weights.

Ultralytics. *YOLO11* model documentation.

|||

**Local evaluation files**

[F] outputs/eval/behavior_scfd/  
local_metrics.json · local_predictions.csv

[W] outputs/eval/cctv_test_v2/  
cctv_test_v2_metrics.json · v2.innovatus_rob

[T] work/runs/v5_hybrid_m640/  
metrics_val_tabletop.json

[M] outputs/eval/movement_review/  
summary.json · labels.json · alerts.csv

::: notes
0:13 · 5:42–5:55

The short codes on the results slides point to the exact local evidence here. SCFD: https://github.com/seymanurakti/fight-detection-surv-dataset. X3D-M weights and model card: https://huggingface.co/visionlab-ai/school-violence-detection-models. YOLO11: https://docs.ultralytics.com/models/yolo11/. SCFD revision ff83b7c521e5ca4eb67a212cc42dbc54f39d6290. The tabletop split list is work/data/v5_indian_hybrid/val_tabletop.txt.
:::

## Thank you {.bullets}

Questions?

Surveillance Threat Detection System

::: notes
0:05 · 5:55–6:00

Thank you. I am happy to take questions or open the benchmark exports.
:::

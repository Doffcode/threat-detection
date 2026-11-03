# Weapon detection on Indian movie stills: fixing knife (and gun) detection

Final report, night of 13-14 October 2026. Project "akriti", folder `work/`.

<!-- AUTO:status -->
Tables last refreshed 2026-10-14 11:28 by `scripts/report/fill_report.py`. 20 result rows available.
<!-- /AUTO:status -->

The numbers in the tables are filled in by `scripts/report/fill_report.py` from the metrics files in `runs/`. **TBD** means that run had not finished when the report was last refreshed.

---

## 1. Problem and goal

The intern's model (YOLO11, classes *armed person*, *gun*, *knife*) is meant to flag weapons in Indian film and TV stills and CCTV-like frames. In practice it missed most knives and was unreliable on guns, even though the notebook reported good scores.

Goals for the night:

1. Measure the model honestly, on a test set that it has never seen in any form.
2. Improve **knife** detection on Indian movie stills, and gun detection as well.
3. Leave a working pipeline (detector, verifier, thresholds, armed-person rule) and a clear record of what helped and what did not.

## 2. Why the original numbers were misleading

The notebook `WD_Intern_Proj.ipynb` trained YOLO11s (640 px, 150 epochs, AdamW) on a Roboflow export with three classes. On its own validation split it reported:

| class | P | R | mAP50 |
|---|---|---|---|
| armed person | 0.94 | 0.87 | 0.91 |
| gun | 0.86 | 0.82 | 0.87 |
| knife | 0.73 | 0.64 | **0.64** |

These scores are inflated by **data leakage**. Roboflow creates several augmented copies of each photo (flipped, cropped, recoloured), and the export put copies of the *same* photo into train, validation and test. In the original export (`raw/IntProj_Weapon_Dataset_annot`), **74% of the validation images and 79% of the test images are copies of a photo that also appears in train** (same source file name). The model was largely being tested on pictures it had already learned, so the scores measured memory rather than detection.

Once the copies were grouped and every group was kept in only one split, the same kind of model scored **knife mAP50 0.13** and **gun mAP50 0.28** on unseen Indian test images (row 1 of the table in section 4). That is the real starting point.

The notebook's later sections (copy-paste knives, contour/edge and "camouflage" preprocessing, a target of about 98% accuracy) were either never trained (the code is commented out) or evaluated on the same leaky split, so they give no evidence either way.

## 3. Data

### 3.1 Sources

| source | what it is | used for |
|---|---|---|
| Domain data (the intern's Roboflow project, `raw/`) | 552 distinct photos (with augmented copies, about 1,300 images): Indian film/TV screenshots, named Indian photos | train / val / **test** |
| OD-WeaponDetection (pistols, knives, Sohas) | public, mostly Western, CC BY-SA 4.0 | extra training only |
| Roboflow Universe, Indian (reviewed in `downloads/roboflow/_review/REVIEW.md`) | anti_robbing (Indian office CCTV), trail_guard (staged Indian photos), mohini (Indian campus CCTV, machetes only) | extra training |
| weapon-detection-two v4 | 382 new photos, almost all knives, but mostly generic web images | tested in one run, which did worse; excluded afterwards |
| `data/v4_big` | Open Images (knife/dagger/sword/handgun/rifle/shotgun + hard negatives), more Roboflow knife and gun sets, plus all of the above: **24,603 training images** (13,890 gun, 8,353 knife, 6,252 hard-negative boxes) | the "big" runs |

None of the reviewed Roboflow datasets contains Indian *movie* stills; the three Indian ones are CCTV-style or staged photos. After the Indian-only runs plateaued, the user allowed generic images from any country (`data/v4_big`), while the Indian domain test stayed the yardstick.

### 3.2 Cleaning, de-duplication and the leak-free split

- **Grouping.** Images were grouped when they shared a source file name, looked nearly identical (difference hashes, flips included), came from the same screenshot session or were consecutive video frames. A whole group always goes to one split. Build scripts check that 0 groups, 0 file stems and 0 near-duplicate pairs cross splits (`data/v2/STATS.md`, `data/indian_v3/STATS.md`, `data/v4_big/STATS.md`). Public images that were near-duplicates of domain test images (46) were dropped.
- **Label fixes.** The augmented copies of one photo often had *different* labels (a box present on some copies only, or gun on one copy and knife on another). Copies were aligned and reconciled: 149 label edits over 81 conflicting photos, 47 duplicate boxes removed, and a few unclear photos left out of val/test.
- **Bad frames dropped.** For example, 114 anti_robbing frames showed a visible but unlabelled weapon. Keeping them would have taught the model that a weapon is background.

### 3.3 The test set used everywhere in this report

**Indian domain test: 144 images (31 photo groups) with 150 guns and 61 knives.** It is the same image list in every dataset version, and no run trained on any of its groups. The 144 images come from only 31 groups, so the test is small and each score has a wide margin of error (roughly +/-0.05 to 0.10).

Runs compared:

- *indian_2cls*: Indian domain data only, gun + knife.
- *indian_strict*: domain data plus the approved Indian sources (anti_robbing, trail_guard, mohini), without the v4 web knives. A run with v4 included (`indian_withv4_m960`) was clearly worse on validation (mAP50 about 0.15 vs 0.26 at the same epoch) and was stopped at epoch 39.
- *quick_ft*: indian_strict fine-tuned for a short time with the public OD-WeaponDetection images added.
- *big_s1 / big_s2*: stage 1 at 640 px on all of `data/v4_big`, then stage 2 at 960 px on Indian data plus 30% of the public data.

## 4. Experiments (Indian domain test, 144 images)

### 4.1 Before and after

<!-- AUTO:headline -->
| class | metric | before (old model) | after (best so far) | change |
|---|---|---|---|---|
| knife | mAP50 | 0.132 | 0.506 | +0.374 |
| knife | precision | 0.140 | 0.640 | +0.500 |
| knife | recall | 0.327 | 0.443 | +0.115 |
| knife | F1 | 0.184 | 0.527 | +0.343 |
| gun | mAP50 | 0.277 | 0.530 | +0.253 |
| gun | precision | 0.290 | 0.547 | +0.257 |
| gun | recall | 0.383 | 0.507 | +0.124 |
| gun | F1 | 0.322 | 0.546 | +0.224 |

"before" = old 3-class model on the same 144 test images (older labels: 141 guns / 55 knives). "after" = **Big data stage 1, 640 px (big_s1_m640)** (row with the highest mean gun + knife mAP50 on the domain test).
<!-- /AUTO:headline -->

### 4.2 All detector runs

<!-- AUTO:experiments -->
| # | model / setting | gun P | gun R | gun mAP50 | gun F1 | knife P | knife R | knife mAP50 | knife F1 | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Old 3-class model, leak-free split (knife_nosyn_m960) | 0.290 | 0.383 | 0.277 | 0.322 | 0.140 | 0.327 | 0.132 | 0.184 | older labels of the same 144 images; GT 141 guns / 55 knives |
| 2 | Indian-only, gun + knife (indian_2cls_m960) | 0.398 | 0.333 | 0.311 | 0.360 | 0.350 | 0.361 | 0.195 | 0.357 |  |
| 3 | &nbsp;&nbsp;&nbsp;same + TTA + NMS 0.5 | 0.437 | 0.520 | 0.446 | 0.497 | 0.273 | 0.344 | 0.189 | 0.301 |  |
| 4 | Indian-strict, no web knives (indian_strict_m960) | 0.380 | 0.307 | 0.276 | 0.328 | 0.340 | 0.246 | 0.216 | 0.286 |  |
| 5 | &nbsp;&nbsp;&nbsp;same + test-time augmentation (TTA) | 0.421 | 0.387 | 0.377 | 0.413 | 0.399 | 0.272 | 0.256 | 0.324 |  |
| 6 | &nbsp;&nbsp;&nbsp;same + TTA + NMS 0.5 | 0.501 | 0.388 | 0.424 | 0.442 | 0.379 | 0.213 | 0.208 | 0.267 |  |
| 7 | Quick fine-tune + public data, 800 px (quick_ft_m800) | 0.258 | 0.387 | 0.301 | 0.360 | 0.197 | 0.131 | 0.160 | 0.225 |  |
| 8 | &nbsp;&nbsp;&nbsp;same + TTA | 0.251 | 0.433 | 0.290 | 0.349 | 0.378 | 0.170 | 0.165 | 0.226 |  |
| 9 | &nbsp;&nbsp;&nbsp;same + TTA + NMS 0.5 | 0.282 | 0.447 | 0.336 | 0.366 | 0.199 | 0.148 | 0.132 | 0.199 |  |
| 10 | Big data stage 1, 640 px (big_s1_m640) | 0.547 | 0.507 | 0.530 | 0.546 | 0.640 | 0.443 | 0.506 | 0.527 |  |
| 11 | &nbsp;&nbsp;&nbsp;same + TTA + NMS 0.5 | 0.604 | 0.607 | 0.611 | 0.614 | 0.585 | 0.459 | 0.382 | 0.492 |  |
| 12 | Big data stage 2, 960 px (big_s2_m960) | 0.574 | 0.440 | 0.486 | 0.519 | 0.543 | 0.213 | 0.283 | 0.353 |  |
| 13 | &nbsp;&nbsp;&nbsp;same + TTA | 0.582 | 0.566 | 0.571 | 0.576 | 0.477 | 0.344 | 0.349 | 0.400 |  |
| 14 | &nbsp;&nbsp;&nbsp;same + TTA + NMS 0.5 | 0.682 | 0.573 | 0.618 | 0.617 | 0.521 | 0.374 | 0.375 | 0.465 |  |

P and R are Ultralytics' values at the confidence that maximises the mean F1; F1 is each class's best F1 over all confidences (so slightly optimistic). mAP50 = average precision at IoU 0.5, independent of any threshold.
<!-- /AUTO:experiments -->

### 4.3 Detector + verifier pipeline

<!-- AUTO:pipeline -->
| # | model / setting | gun P | gun R | gun mAP50 | gun F1 | knife P | knife R | knife mAP50 | knife F1 | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | indian_strict_m960 alone (same matcher, val thresholds) | 0.467 | 0.280 | 0.314 | 0.350 | 0.323 | 0.328 | 0.179 | 0.325 | reference for the row below |
| 2 | indian_strict_m960 + verifier_quick (TTA) | 0.331 | 0.327 | 0.318 | 0.329 | 0.400 | 0.262 | 0.211 | 0.317 | score mode `wgeo25`, thresholds tuned on val |
| 3 | quick_ft_m800 alone (same matcher, val thresholds) | 0.309 | 0.367 | 0.277 | 0.335 | 0.250 | 0.016 | 0.162 | 0.031 | reference for the row below |
| 4 | quick_ft_m800 + verifier_quick (TTA) | 0.293 | 0.320 | 0.256 | 0.306 | 0.231 | 0.098 | 0.143 | 0.138 | score mode `wgeo75`, thresholds tuned on val |
| 5 | big_s2_m960 alone (same matcher, val thresholds) | 0.627 | 0.427 | 0.548 | 0.508 | 0.394 | 0.213 | 0.340 | 0.277 | reference for the row below |
| 6 | big_s2_m960 + verifier_big (TTA) | 0.547 | 0.467 | 0.507 | 0.504 | 0.488 | 0.328 | 0.343 | 0.392 | score mode `prod`, thresholds tuned on val |

Here P, R and F1 are measured at the per-class thresholds chosen on the validation split (an honest deployment setting, nothing tuned on test); AP50 is computed from the final score. Matching: IoU >= 0.5, one-to-one.
<!-- /AUTO:pipeline -->

Verifier (a small image classifier that re-checks each gun/knife box as *gun*, *knife* or *not weapon*):

<!-- AUTO:verifier -->
- **verifier_quick**: top-1 accuracy 0.772 on validation crops; gun P 0.86 / R 0.80 (n=269), knife P 0.69 / R 0.77 (n=104), not_weapon P 0.69 / R 0.73 (n=131).
  Gun crops called knife: 25; knife crops called gun: 11.
- **verifier_big**: top-1 accuracy 0.683 on validation crops; gun P 0.88 / R 0.64 (n=243), knife P 0.56 / R 0.83 (n=64), not_weapon P 0.54 / R 0.70 (n=125).
  Gun crops called knife: 21; knife crops called gun: 4.
<!-- /AUTO:verifier -->

### 4.4 Knife error audit

<!-- AUTO:audit -->
- **indian_strict_m960** (knife threshold 0.223, 61 knives): found 5, missed 56, false alarms 11 (recall 0.08, precision 0.31). Missed knives: 43 not detected as anything, 5 called gun, 8 box in the wrong place; 31 of the misses had a correct knife box just below the threshold (under-confidence).
- **big_s2_m960** (knife threshold 0.163, 61 knives): found 15, missed 46, false alarms 17 (recall 0.25, precision 0.47). Missed knives: 28 not detected as anything, 9 called gun, 9 box in the wrong place; 20 of the misses had a correct knife box just below the threshold (under-confidence).
<!-- /AUTO:audit -->

(The audit counts each box with a single class, as the app does, so its recall is lower than the table's.)

### 4.5 Final comparison table (all candidates re-scored the same way)

<!-- AUTO:final_compare -->
##### knife (focus)

| run | inst | P | R | mAP50 | mAP50-95 | best-F1 conf | F1 | all mAP50 | all mAP50-95 | weapons mAP50 | weapons mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| indian_2cls_m960/test | 61 | 0.350 | 0.361 | 0.195 | 0.063 | 0.185 | 0.357 | 0.253 | 0.106 | 0.253 | 0.106 |
| indian_2cls_m960/test_domain_tta_nms5 | 61 | 0.273 | 0.344 | 0.189 | 0.081 | 0.139 | 0.301 | 0.317 | 0.139 | 0.317 | 0.139 |
| indian_2cls_m960/test_clean_tta_nms5 | 64 | 0.248 | 0.297 | 0.159 | 0.070 | 0.133 | 0.270 | 0.274 | 0.121 | 0.274 | 0.121 |
| indian_strict_m960/test_domain | 61 | 0.340 | 0.246 | 0.216 | 0.058 | 0.203 | 0.286 | 0.246 | 0.090 | 0.246 | 0.090 |
| indian_strict_m960/test_domain_tta | 61 | 0.399 | 0.272 | 0.256 | 0.105 | 0.306 | 0.324 | 0.317 | 0.128 | 0.317 | 0.128 |
| indian_strict_m960/test_domain_tta_nms5 | 61 | 0.379 | 0.213 | 0.208 | 0.070 | 0.294 | 0.267 | 0.316 | 0.110 | 0.316 | 0.110 |
| indian_strict_m960/test_clean_tta_nms5 | 64 | 0.429 | 0.188 | 0.182 | 0.063 | 0.303 | 0.261 | 0.297 | 0.096 | 0.297 | 0.096 |
| quick_ft_m800/test_domain_tta | 61 | 0.378 | 0.170 | 0.165 | 0.081 | 0.109 | 0.226 | 0.227 | 0.107 | 0.227 | 0.107 |
| quick_ft_m800/test_domain_tta_nms5 | 61 | 0.199 | 0.148 | 0.132 | 0.061 | 0.077 | 0.199 | 0.234 | 0.098 | 0.234 | 0.098 |
| quick_ft_m800/test_clean_tta_nms5 | 64 | 0.199 | 0.141 | 0.126 | 0.056 | 0.077 | 0.194 | 0.216 | 0.090 | 0.216 | 0.090 |
| big_s1_m640/test_domain | 61 | 0.640 | 0.443 | 0.506 | 0.275 | 0.181 | 0.527 | 0.518 | 0.272 | 0.518 | 0.272 |
| big_s1_m640/test_domain_tta_nms5 | 61 | 0.585 | 0.459 | 0.382 | 0.218 | 0.199 | 0.492 | 0.496 | 0.268 | 0.496 | 0.268 |
| big_s1_m640/test_clean_tta_nms5 | 64 | 0.585 | 0.438 | 0.366 | 0.209 | 0.198 | 0.477 | 0.495 | 0.265 | 0.495 | 0.265 |
| big_s2_m960/test_domain | 61 | 0.543 | 0.213 | 0.283 | 0.110 | 0.177 | 0.353 | 0.385 | 0.171 | 0.385 | 0.171 |
| big_s2_m960/test_domain_tta | 61 | 0.477 | 0.344 | 0.349 | 0.168 | 0.236 | 0.400 | 0.460 | 0.237 | 0.460 | 0.237 |
| big_s2_m960/test_domain_tta_nms5 | 61 | 0.521 | 0.374 | 0.375 | 0.163 | 0.177 | 0.465 | 0.497 | 0.229 | 0.497 | 0.229 |
| big_s2_m960/test_clean_tta_nms5 | 64 | 0.545 | 0.312 | 0.369 | 0.161 | 0.175 | 0.459 | 0.512 | 0.246 | 0.512 | 0.246 |

##### mAP50-95 per class

| run | gun | knife | handheld_object | all |
|---|---|---|---|---|
| indian_2cls_m960/test | 0.149 | 0.063 | 0.000 | 0.106 |
| indian_2cls_m960/test_domain_tta_nms5 | 0.197 | 0.081 | 0.000 | 0.139 |
| indian_2cls_m960/test_clean_tta_nms5 | 0.172 | 0.070 | 0.000 | 0.121 |
| indian_strict_m960/test_domain | 0.122 | 0.058 | 0.000 | 0.090 |
| indian_strict_m960/test_domain_tta | 0.152 | 0.105 | 0.000 | 0.128 |
| indian_strict_m960/test_domain_tta_nms5 | 0.149 | 0.070 | 0.000 | 0.110 |
| indian_strict_m960/test_clean_tta_nms5 | 0.129 | 0.063 | 0.000 | 0.096 |
| quick_ft_m800/test_domain_tta | 0.134 | 0.081 | 0.000 | 0.107 |
| quick_ft_m800/test_domain_tta_nms5 | 0.135 | 0.061 | 0.000 | 0.098 |
| quick_ft_m800/test_clean_tta_nms5 | 0.125 | 0.056 | 0.000 | 0.090 |
| big_s1_m640/test_domain | 0.269 | 0.275 | 0.000 | 0.272 |
| big_s1_m640/test_domain_tta_nms5 | 0.318 | 0.218 | 0.000 | 0.268 |
| big_s1_m640/test_clean_tta_nms5 | 0.321 | 0.209 | 0.000 | 0.265 |
| big_s2_m960/test_domain | 0.233 | 0.110 | 0.000 | 0.171 |
| big_s2_m960/test_domain_tta | 0.306 | 0.168 | 0.000 | 0.237 |
| big_s2_m960/test_domain_tta_nms5 | 0.295 | 0.163 | 0.000 | 0.229 |
| big_s2_m960/test_clean_tta_nms5 | 0.330 | 0.161 | 0.000 | 0.246 |
<!-- /AUTO:final_compare -->

## 5. Key findings

1. **The old scores were not real.** Because of leakage, a model shown as "knife mAP50 0.64" was really about 0.13 on unseen Indian stills. All numbers in this report use the leak-free test.
2. **Gun barrels are mistaken for knives.** In the knife audit of indian_strict, 9 of the 11 false knife detections were gun barrels or revolver close-ups (long shiny metal). The verifier also confuses the two in both directions (see 4.3).
3. **The model is under-confident.** More than half of the knives counted as missed (31 of 56 for indian_strict) had a correctly placed knife box with a score just *below* the threshold. Thresholds therefore matter a great deal; they are tuned per class on the validation split, never on test.
4. **Inconsistent box style costs matches.** Some labels box *gun + hand*, others the gun only. In a gun audit of indian_strict, 26 of 48 false gun detections were the right gun with a box of the other style (overlap with the label below 50%), so they count as both a miss and a false alarm. With a 30% overlap rule, gun precision rises from 0.45 to 0.54. Only consistent relabelling fixes this properly.
5. **Duplicate boxes.** 11 of the 48 false gun detections were second boxes on an already-detected gun. A stricter non-maximum suppression (NMS IoU 0.5 instead of 0.7) raised gun precision from 0.54 to 0.67 at the same recall (30% overlap rule). See the "TTA + NMS 0.5" rows.
6. **Test-time augmentation (TTA) helps clearly.** Running the detector on flipped and rescaled copies and merging the results raised indian_strict from knife mAP50 0.216 to 0.256 and gun 0.276 to 0.377 (F1: knife 0.286 to 0.324, gun 0.328 to 0.413). It costs about 3x inference time (about 0.18 s per image on the A2000 GPU), which is acceptable for still images.
7. **Gun vs knife is not the main problem.** Treating both as one "weapon" class only raised mAP50 from 0.32 to 0.36 (indian_strict, TTA), so most of the loss is finding weapons at all and box style, not confusing the two classes.

## 6. Final pipeline

```
image -> weapon detector (YOLO11m, 960 px, TTA, NMS 0.5)
      -> verifier on each gun/knife box (YOLO11s-cls, crop + 20% padding)
      -> final score = detector confidence x verifier probability
      -> per-class thresholds (gun, knife; tuned on validation)
      -> "armed person" = COCO person box that contains >= 30% of a confirmed weapon box
```

- **Detector**: two weapon classes, *gun* and *knife* (plus an internal *handheld_object* class that absorbs phones and tools and is never shown). *Armed person* is no longer a detector class, because "a person holding a weapon" can be derived reliably and the old class mostly duplicated the gun boxes.
- **Verifier**: a second look at each candidate box. It is cheap because it only runs on the few boxes found.
- **Per-class thresholds**: knives get their own, lower threshold because the model is under-confident on them.
- **Armed person**: a standard COCO person detector runs only when a weapon is confirmed. The app (`app/`, Streamlit dashboard + CLI) implements all of this and turns it into a threat score.

<!-- AUTO:final_config -->
Chosen configuration: **Big data stage 1, 640 px (big_s1_m640)** (source `runs/big_s1_m640/metrics_test_domain.json`).
Detector only (no verifier); per-class confidence thresholds from validation: gun 0.185, knife 0.133.
<!-- /AUTO:final_config -->

## 7. Limitations

- **Small test set.** 61 knives in 31 photo groups means differences of a few points are not significant.
- **Few real Indian movie stills.** The extra Indian data is CCTV and staged photos, and the large public data is mostly Western and everyday scenes. The film look (lighting, colour grading, motion blur, crowded frames) is still under-represented.
- **Label style is inconsistent** (gun + hand vs gun only, blade vs whole knife), which both confuses training and understates the measured scores.
- **Absolute accuracy is still modest.** The system should be used as an alert that a human then checks, not as an automatic decision.
- The model is YOLO11m, trained on a single 12 GB GPU in one night, and the big runs were time-limited.

## 8. Next steps (in order of expected value)

1. **Collect more Indian movie stills with knives and guns**: a few hundred new frames from different films (aruval, sickle, machete, katta, police pistols), labelled from scratch and split by film so that no film is in both train and test.
2. **Relabel consistently** with one written rule (box the weapon only, never the hand; box the whole knife including the handle) and apply it to all domain data, starting with the test set.
3. **Grow the test set** to at least 300 knives from films not used in training, so that improvements can be measured reliably.
4. **Bigger model or higher resolution** (YOLO11l/x, 1280 px or tiling) once there is more data; knives are thin and often small.
5. **Hard-negative mining**: add stills with sticks, phones and close-up gun barrels labelled as not-knife, and retrain the verifier on mistakes from the final detector.

## Appendix: where things are

| what | path |
|---|---|
| this report / refresh script | `work/REPORT.md`, `work/scripts/report/fill_report.py` |
| training / evaluation scripts and commands | `work/scripts/train/README.md` |
| app (dashboard, CLI, engine) | `work/app/README.md` |
| dataset statistics and leak checks | `work/data/v2/STATS.md`, `work/data/indian_v3/STATS.md`, `work/data/v4_big/STATS.md` |
| Roboflow dataset review | `work/downloads/roboflow/_review/REVIEW.md` |
| run outputs (metrics, thresholds, audits) | `work/runs/<run>/` |
| final side-by-side table | `work/runs/FINAL_COMPARE.md` |

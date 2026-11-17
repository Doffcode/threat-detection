# Dataset Audit Report: Weapon Detection & Indian Domain Assets

> **Step in Roadmap:** Stage 1, Step 1: Dataset Audit  
> **Date:** November 2026  
> **Script:** `src/eval/audit_dataset.py`  
> **Status:** Completed. Critical findings documented below.

---

## 1. Executive Summary

We audited the two primary candidate datasets available in the environment:
1. **OD-WeaponDetection (Sohas):** The standard open-source academic weapon benchmark (University of Granada).
2. **Raw Indian Domain Collection (`IntProj_Weapon_Dataset_annot`):** Project-inherited Indian cinema and street surveillance stills.

### Summary Scorecard
| Criterion | OD-WeaponDetection (Sohas) | Raw Indian Dataset (`IntProj`) |
| :--- | :--- | :--- |
| **Verified License** | ✅ CC BY-SA 4.0 (Pérez-Hernández et al. 2020) | ⚠️ CC BY 4.0 (Roboflow exports) |
| **Annotation Integrity** | ✅ 100% clean Pascal VOC XML (0 degenerate boxes) | ❌ 87 duplicate boxes, severe label swaps |
| **Data Leakage Risk** | ✅ Clean, grouped video partitions | ❌ **15.1% leakage** (243 stems shared in train/val/test) |
| **Class Granularity** | ⚠️ Pistols & Knives only (**No Rifles**) | ⚠️ Generic `gun` (collapses rifles & handguns) |
| **Hard Negative Ingestion** | ✅ Built-in distractors (smartphones, purses, cards)| ❌ None (only positive weapon boxes) |
| **CCTV Small-Object Ratio**| ⚠️ Only 3.9% in 20–60 px range (87.9% large boxes) | ⚠️ Only 3.2% in 20–60 px range (88.5% large boxes) |
| **Indian Weapon Coverage** | ❌ 0% (European staged/surveillance guns & knives) | ✅ High (Aruval, Desi Katta, Dacoit knives, Cinema) |

---

## 2. Quantitative Findings: OD-WeaponDetection (Sohas)

* **Source Path:** `work/downloads/odw/extracted/OD-WeaponDetection-master`
* **Total Annotation Files:** 11,020 XML files
* **Total Bounding Boxes:** 12,163 valid boxes (0 degenerate)

### 2.1 Class Distribution
| Class Name | Type | Box Count | Percentage |
| :--- | :--- | :---: | :---: |
| `pistol` | Weapon (Handgun) | 5,127 | 42.2% |
| `knife` | Weapon (Edged) | 4,504 | 37.0% |
| `smartphone` | Handheld Look-Alike (Hard Negative) | 893 | 7.3% |
| `monedero` (purse/wallet)| Handheld Look-Alike (Hard Negative) | 644 | 5.3% |
| `billete` (cash bills) | Handheld Look-Alike (Hard Negative) | 584 | 4.8% |
| `tarjeta` (card) | Handheld Look-Alike (Hard Negative) | 411 | 3.4% |
| `rifle` | Long Gun | **0** | **0.0% (Missing)** |

### 2.2 Scale & Dimension Breakdown
* **COCO Small (< 32 px):** 153 boxes (1.3%)
* **COCO Medium (32 – 96 px):** 1,318 boxes (10.8%)
* **COCO Large (> 96 px):** 10,692 boxes (87.9%)
* **CCTV Critical Range (20 – 60 px):** 469 boxes (3.9%)
* **Median Box Size:** $189.0 \times 172.0$ pixels (close-up / medium perspective).

---

## 3. Quantitative Findings: Raw Indian Collection (`IntProj`)

* **Source Path:** `work/raw/IntProj_Weapon_Dataset_annot`
* **Total YOLO Label Files:** 8,763 files
* **Total Bounding Boxes:** 20,312 boxes (0 degenerate, 87 exact duplicate duplicates)

### 3.1 Class Distribution
* `armed person` (class 0): 7,454 (36.7%) — *Should not be a detector class in Stage 1; derived via spatial association.*
* `gun` (class 1): 8,306 (40.9%) — *Merged handgun and rifle.*
* `knife` (class 2): 4,552 (22.4%)

### 3.2 Critical Quality & Integrity Problems
1. **Severe Cross-Split Data Leakage:**
   * Out of 1,606 unique base image stems, **243 stems (15.1%) appear across multiple splits simultaneously** (e.g. augmented copy A in `train`, copy B in `val`, copy C in `test`).
   * *Impact:* Models trained on this raw split report falsely high mAP (>65%) that collapses on unseen footage.
2. **Severe Label Conflicts & Noise:**
   * *Aruval* (Tamil Nadu curved machete) is labeled as `gun` in some copies and `knife` in others.
   * Actor Sanjay Dutt holding a rifle in *KGF 2* is labeled as `knife` in 1 copy and `gun` in 3 copies.
   * *Pushpa* axe/knife scene labeled as `gun` in one file and `knife` in another.
3. **Severe Scale Imbalance:**
   * 88.5% of boxes are large (>96 px). Only 0.4% are small (<32 px).

---

## 4. Key Recommendations for Step 2 (Baseline Training)

1. **Standardize Target Classes to 3-Class Hierarchy:**
   * Class 0: `handgun` (from Sohas `pistol` + audited Indian pistols)
   * Class 1: `rifle` (from audited long gun sources, e.g. Aahil / Open Images)
   * Class 2: `knife` (from Sohas `knife` + audited Indian blades/machetes)
   * Class 3 (Distractor/Negative): `handheld_object` (from Sohas phones/wallets to absorb false alarms).
2. **Enforce Leak-Free Stratified Group Splitting:**
   * Group all augmented copies and video sequences by base image stem.
   * Lock the 144 Indian cinema domain images as a zero-leakage evaluation set.
3. **Compensate for Small-Object Bias (20–60 px):**
   * Since both datasets have <4% small objects, training at standard 640px will fail on CCTV.
   * We must train at **960px or 1280px**, apply mosaic augmentation, and use copy-paste downscaling to inject small 20–60 px weapons into realistic background scenes.

# Project Context: CCTV Threat-Detection System (Operator Decision Support)

## 1. System Vision & Purpose
This system assists human security operators by flagging potential threats in CCTV footage for human review. 
Video cannot show intent directly; it captures weapons and behaviors that deviate from normal baselines. The system outputs **"needs human review"** alerts with cropped context clips and confidence scores, never automated "threat confirmed" assertions.

---

## 2. The Mandatory Indian Context (Core Non-Negotiable Pillar)
Real-world CCTV in India presents unique visual, cultural, environmental, and threat realities that standard Western datasets completely fail to address. The system **must be built from day one to operate reliably in the Indian context**:

### 2.1 Weapon Typology in India
* **Edged Weapons (Primary Threat):** Blade crimes (stabbings, robberies, chain-snatchings) far outnumber firearm incidents. Must detect diverse blades:
  * Heavy cleavers and machetes (*Aruval*, *Kathi*, *Billhook*).
  * Traditional and folding daggers (*Kirpan*, *Katar*, Rampuri knives).
  * Everyday kitchen knives and utility blades used in street crimes.
* **Firearms (Non-Standard & Country-Made):**
  * Handcrafted country pistols (*Desi Katta*, *Tamancha*, single-shot handguns) with irregular, non-standard shapes, unpolished metal, or improvised wooden grips.
  * Standard revolvers, pistols, and double-barrel hunting shotguns/rifles.

### 2.2 Traditional Indian Attire & Severe Occlusion
* **Attire Diversity:** Sarees, dhotis, lungis, kurtas, pyjamas with waist cords (*nada*), salwar-kameez, dupattas, and shawls/chadars.
* **Concealment Patterns:** Weapons are frequently tucked into lungi/dhoti waist folds or under untucked kurtas and shawls.
* **Pose & Skeleton Challenges:** Flowing fabrics obscure lower limbs and torso contours, causing standard pose models to degrade. Wrist and hand keypoint extractors must stay robust when partially covered by loose sleeves or dupattas.

### 2.3 Indian Surveillance & Environmental Realities
* **Typical CCTV Camera Angles:** High-angle, top-down mounting in narrow *kirana* stores, jewelry shops with security grilles, bank counters, and congested entryways.
* **Lighting & Image Perturbations:** Harsh mixed lighting (yellow sodium streetlamps, halogen glare, low-lux night IR with high sensor noise, ceiling fan shadows causing cyclic flicker).
* **Dense Crowd Dynamics:** High foot traffic, tight physical queues, and casual body contact in markets and transport hubs (bus stands, railway stations).

### 2.4 Cultural Confusers & Everyday Look-Alikes (Hard Negatives)
To prevent operational false alarms, the system must explicitly distinguish weapons from everyday Indian objects and activities:
* **Objects:** Stainless steel water bottles (*milton* style), cylindrical tiffin carriers, large umbrellas, bamboo lathis, coconut sickles (in vendor contexts), brass puja items/lamps.
* **Gestures & Postures:** Bending down to touch elders' feet (*Pranam*), prostrations during prayer (*Namaz*, *Sashtanga*), temple rituals (*Arti*), hailing auto-rickshaws/buses with raised arms, carrying luggage onto train berths.

---

## 3. Scope & Key Technical Decisions (Stage 1)
1. **Physical Reality of RGB:** Fully concealed weapons (100% inside pockets, completely under fabric) cannot be detected from RGB pixels alone because no light reflects from inside clothing. Do NOT claim or attempt this in Stage 1. Those belong to Stage 2 as weak behavioral cues that never alert alone.
2. **Scope of Stage 1:** Visible and partially occluded/peeking weapons (e.g. grip protruding from dhoti waistline, blade partially held in hand, weapon peeking from pocket mouth).
3. **RGB Preservation:** Grayscale discards chrominance and reflection cues. Full RGB is preserved throughout.
4. **Camouflaged Object Detection (COD):** Kept as an optional future refinement for low-contrast partially visible weapons only after baseline error analysis proves a need.

---

## 4. Multi-Stage Roadmap Overview
* **Stage 1 (Current):** Weapon detection tied to persons. Detect handguns, rifles, and knives (including partially visible and occluded instances in Indian settings) and associate each weapon with the specific person holding/carrying it.
* **Stage 2:** Behavior analysis for harmful-intent signals (loitering, sudden running/freezing, following, zone violations, and weak concealment cues: hand repeatedly touching one body spot, asymmetric gait, stiff arm, body shielding).
* **Stage 3:** Learned anomaly detection and video understanding (autoencoders, SlowFast / video transformers, ST-GCN skeleton models, per-camera baselines).
* **Stage 4:** Integration and deployment (real-time pipeline, TensorRT optimization, alert UI for operators, multi-camera logging, DPDP retention and privacy controls).

---

## 5. Stage 1 Architectural Pipeline
1. **Person Tracking:** YOLO + ByteTrack yielding persistent, stable `track_id`s in dense scenes.
2. **Small-Object Weapon Detector:** Handguns, rifles, and knives at high input resolution (960–1280px), SAHI slicing where beneficial.
3. **Person-Weapon Association:** Bounding box overlap followed by Euclidean distance to hand/wrist keypoints (YOLO-Pose).
4. **Hand Crop Classifier (Optional Refinement):** Verifies weapon vs. phone vs. tool vs. empty hand.
5. **Temporal Confirmation:** Alert triggered only when an association persists across $N$ frames on the same track ID.
6. **Human Review Payload:** Export of cropped incident video clip + confidence metadata for operator review.

---

## 6. Target Hardware
* **Primary:** Dell Precision 3660 with 1× NVIDIA RTX A2000 12GB GPU.
* **Secondary:** Dell G5 (AMD Radeon RX 5600M, CPU fallback).

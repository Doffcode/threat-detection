---
name: data-source-policy
description: Which datasets are allowed for the weapon-detection (knife) project — rule changed from Indian-only to generic-OK on 2026-10-14
metadata:
  node_type: memory
  type: project
  originSessionId: 7f6b6edd-de3a-4078-aa7f-b4e33f358931
  modified: 2026-10-13T23:55:20.995Z
---

On 2026-10-14 the user relaxed the earlier "only Indian dataset, not foreign" rule: generic images from any country are fine, and "we'll do whatever that can increase the metrics". COCO-pretrained YOLO weights were already approved.

**Why:** Indian-only training plateaued (knife mAP50 ~0.2, P ~0.35 on the 144-image Indian movie-still domain test); data volume is the bottleneck.

**How to apply:** Use public/foreign data (OD-WeaponDetection, Open Images, Roboflow Universe) for training, but keep the Indian movie-still domain test (`work/data/v2/test_domain.txt`) as the headline metric. Still dedupe against domain val/test. Related: [[knife-project-status]]

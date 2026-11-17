---
name: knife-project-status
description: "Outcome of the 2026-10-14 overnight knife/gun push — best model, honest scores, what didn't work"
metadata:
  node_type: memory
  type: project
  originSessionId: 7f6b6edd-de3a-4078-aa7f-b4e33f358931
  modified: 2026-10-14T06:00:51.122Z
---

Best model (2026-10-14): `work/runs/big_s1_m640` (YOLO11m, 640 px, stage 1 on 24.6k mixed public+Indian images in `work/data/v4_big`). Indian domain test, VAL-tuned thresholds (gun 0.164, knife 0.168), no TTA: gun P 0.53 R 0.53 F1 0.53; knife P 0.63 R 0.44 F1 0.52; mAP50 gun 0.53 / knife 0.51 (old leak-free baseline: 0.28 / 0.13). Launcher: `work/app/run_best.sh`.

Did NOT help: stage-2 Indian fine-tune at 960 (big_s2, knife mAP50 fell to 0.28 — overfits the small domain set); adding viveks v4 web photos; TTA + NMS 0.5 on big_s1 (gun up, knife down); verifier (raises knife precision to 0.74 but hurts gun); continuing big_s1 with a restarted LR schedule (big_s1b, worse: knife 0.42 / gun 0.38) and a gentle frozen Indian fine-tune after it (gentle_m640, ties big_s1: gun 0.54 / knife 0.49, gun P 0.66). Next real gain needs one long uninterrupted run (~40 epochs, no time cap), not restarts.

**Why:** data volume was the bottleneck, not tuning. Test set is tiny (61 knives / 31 groups, ±0.05–0.1 noise). Cleaned test variant `work/data/v2_testclean` (39 label fixes) gives similar numbers.

**How to apply:** start future work from big_s1 best.pt; next gains need more Indian movie stills and consistent relabelling. PC clock runs ~5.5 h fast vs real IST. Related: [[data-source-policy]], [[parallel-subagents-ok]]

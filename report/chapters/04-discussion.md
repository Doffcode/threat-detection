# Discussion and conclusion

## Interpretation and observed failures

**Fight classification.** SCFD is the strongest result because its denominator is explicit, each classification can be reconstructed from the saved scores, and the local and four-V100 runs agreed on all 300 decisions. The errors are asymmetric. The classifier missed 23 of 150 fight clips and falsely flagged 4 of 150 non-fight clips. The experiment measures short, labeled clips. Continuous video would require additional evaluation of event boundaries, false alerts per hour, and latency from incident onset.

**Weapon localization.** The tabletop result establishes performance on a small, controlled validation split with a different checkpoint. The Innovatus result describes the detector actually used in the unified dashboard. On that split, the detector missed 58 annotations at the operating threshold. The audit records 60 of 75 weapons occupying less than 1% of the image area. Small objects and occlusion are concrete conditions to investigate. The two mAP values do not measure the effect of a model upgrade because they use different splits, taxonomies, and protocols. The tabletop verifier results require a like-for-like comparison within their own export.

**Movement evidence.** The separate walkthrough clip demonstrates that the rule can generate a timestamped event and that the dashboard can produce an annotated replay. The frozen cue review yielded no alerts. Tracking gaps, identity changes, box jitter, and the requirement to satisfy all geometric conditions can prevent a cue from firing. Their contributions were unmeasured in this review. The brief, identical reference intervals limit the validity of the recall estimate. A reviewer can inspect the footage and repeat the cue evaluation with independently validated timing.

## Reproduction

From the project root, regenerate fight metrics from the fixed manifest and prediction file.

```sh
work/.venv/bin/python scripts/report_behavior_evaluation.py \
  outputs/eval/behavior_scfd/manifest.csv \
  outputs/eval/behavior_scfd/local_predictions.csv \
  --merged /tmp/phase3_scfd_predictions.csv
```

Run `work/.venv/bin/python scripts/behavior/movement_review.py report` to regenerate the movement summary from the existing artifacts. The scoring command and review instructions are in `outputs/eval/movement_review/README.md`. To reproduce the weapon results, use the locked Innovatus protocol in source W and the tabletop validation configuration in source T. The tabletop export identifies the exact local checkpoint. The original tabletop command and its run-specific confidence and NMS flags were not recorded in the JavaScript Object Notation (JSON) metrics export.

| Role | Checkpoint path |
|---|---|
| Dashboard weapon detector | `outputs/runs/baseline_m960_v2/weights/best.pt` |
| Tabletop detector | `work/runs/v5_hybrid_m640/weights/best.pt` |
| Fight classifier | `outputs/runs/behavior_runtime/final_x3d_realtime.pt` |

*Table 3. Checkpoints used in the reported branches. The SCFD protocol records the full checkpoint hash and dataset revision.*

## Conclusion

Phase 3 delivers the recorded-video analysis workflow, model and rule execution, review timeline, replay, saved evaluation artifacts, and an annotated presentation recording. The verified performance is strongest for SCFD clip-level fight classification. The tabletop validation shows strong detector performance in its specific setting, while the dashboard checkpoint has low recall on Innovatus CCTV. The movement branch is complete as a review feature. Its zero-alert review and unresolved annotation timing mean its ability to detect the target cue on the sampled footage remains unestablished. These outcomes define what the faculty panel can reproduce and which questions remain open.

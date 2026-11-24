# X3D-M on SCFD

The frozen X3D-M model scored 273 of 300 clips correctly (91.0% accuracy), with 91.5% non-fight F1 and 90.4% fight F1. It missed 23 of 150 fights and flagged 4 of 150 non-fight clips. There were zero inference errors.

| Ground-truth class | Precision | Recall | F1 | Clips |
|---|---:|---:|---:|---:|
| Non-fight | 86.4% | 97.3% | 91.5% | 150 |
| Fight | 96.9% | 84.7% | 90.4% | 150 |

The confusion matrix, in non-fight/fight order, is `[[146,4],[23,127]]` with ground truth in rows. Local single-worker latency was 50.4 ms median and 127.3 ms p95, including decoding and preprocessing, measured after warm-up on the RTX A2000 12 GB.

## Protocol

- Use the [SCFD authors’ repository](https://github.com/seymanurakti/fight-detection-surv-dataset) at commit `ff83b7c521e5ca4eb67a212cc42dbc54f39d6290` in the `scfd` directory beside the manifest. Labels follow the `fight` and `noFight` folders.
- All 300 files were fixed before inference. No SHA-256 overlap was found with the 40 accessible RWF-2000 selection files. The RWF-2000 mirror uses renamed filenames without source IDs, so SCFD's source list could not be matched against it. Re-encoded clips and overlaps with the full training corpus remain possible.
- Use the runtime and pinned upstream source documented in [the model selection report](../behavior_baseline/README.md). Use `final_x3d_realtime.pt`, SHA-256 `e833f69d110f167cad4a6c38d385564bdb2f6de63d246e45cb03ff9aa17f0349`, with 16 frames sampled uniformly over the clip and the fixed 0.40 threshold. The result measures clip-level CCTV fight detection.

```bash
python scripts/compare_behavior.py x3d "$INPUT_DIR/manifest.csv" "$WEIGHTS/final_x3d_realtime.pt" local.csv --source "$MODEL_SOURCE"
python scripts/report_behavior_evaluation.py "$INPUT_DIR/manifest.csv" local.csv --merged local_predictions.csv > local_metrics.json
```

## V100 run

Four Tesla V100-SXM2 32 GB GPUs scored 75 clips each with PyTorch 2.4.1+cu118 and torchvision 0.19.1+cu118. The workers used the tensor state extracted from the verified original checkpoint and loaded it strictly. All 300 predictions agreed with the local run.

Aggregate throughput was 16.43 clips/s over 18.26 seconds, including worker startup and warm-up, excluding environment setup and file transfer. The per-worker commands use the same inference runner with manifest rows split by row index modulo four and `CUDA_VISIBLE_DEVICES` set to a different idle V100 per worker. Merge the four worker CSVs by passing all four as positional arguments to the reporting command.

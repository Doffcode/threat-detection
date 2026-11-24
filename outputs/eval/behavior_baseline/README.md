# Pretrained behavior model selection

The comparison used 20 Fight and 20 NonFight clips sampled from RWF-2000 validation with seed 42 before inference. The mirror was `nghaanh03/RWF-2000` at revision `96a22593dc2c9ee9f89cf3c820231a86003a04ef`. Files are under `raw/val/Fight` and `raw/val/NonFight`. The publisher used this split for validation and threshold selection. Treat these results as an in-domain comparison. Hard negatives were not separately annotated.

Both runs used the local RTX A2000 with PyTorch 2.14.1+cu130 and torchvision 0.29.1+cu130. X3D-M was selected from the two completed candidates. CUE-Net was deferred because the hosted configuration requires a custom SlowFast/UniFormerV2 model and dataset implementation whose matching source was not verified. The V100 host was unavailable from this session.

## Reproduce

Use the upstream Hugging Face Space `visionlab-ai/school-violence-detection-app` at revision `67d51a3743f276b96f87fa07d83d4ac5b9d326f8`. Use the two weights under `final/` in `visionlab-ai/school-violence-detection-models` at revision `a744b6af7496f0cbfa4f0ba32acd46b65e52d4e1`. Set `MODEL_SOURCE` to the checkout and `INPUT_DIR` to the folder holding the manifest and its relative clip files. Install the upstream X3D dependencies in an isolated environment or target directory. The run used pytorchvideo 0.1.5, fvcore 0.1.5.post20221221, iopath 0.1.10, av 19.0.2, yacs 0.1.8, tabulate 0.10.0, termcolor 3.3.0, portalocker 4.4.2, and parameterized 0.9.0. The publisher's X3D wrapper resolves the PyTorchVideo backbone through Torch Hub.

```bash
python scripts/compare_behavior.py x3d "$INPUT_DIR/manifest.csv" "$WEIGHTS/final_x3d_realtime.pt" x3d.csv --source "$MODEL_SOURCE"
python scripts/compare_behavior.py r2plus1d "$INPUT_DIR/manifest.csv" "$WEIGHTS/final_best.pt" r2plus1d.csv --source "$MODEL_SOURCE"
python scripts/summarize_behavior.py x3d.csv r2plus1d.csv
```

For separate V100 workers, set `CUDA_VISIBLE_DEVICES` separately for each inference process. Rerun sequentially on the demo machine for latency comparisons.

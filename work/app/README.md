# Weapon detection app (inference side)

Two-stage pipeline, matching `scripts/verifier/eval_pipeline.py` exactly:

1. **Weapon detector** (YOLO detect). Classes `gun`, `knife`, `handheld_object`. `handheld_object` is never
   reported: the detector is called with `classes=[gun, knife]`, like the eval. It only gives the detector a place to
   put phones, wallets, tools and so on, instead of calling them guns or knives. Old 3-class models
   (`armed person`, `gun`, `knife`) still work.
2. **Verifier** (optional, YOLO classify; currently `runs/verifier_quick/weights/best.pt`, yolo11s-cls, classes
   `gun`, `knife`, `not_weapon`). It runs on a square crop of every gun/knife box, built by `square_crop`: pad 20%
   per side, letterbox to 224 px, taken from the original frame. `square_crop` is identical to
   `scripts/verifier/common.py`, and a test enforces that. The final score combines detector confidence `d` and
   verifier `p` = p(detected class):

   | `combine` | final score | alias |
   |---|---|---|
   | `det` | `d` | `detector` |
   | `cls` | `p` | `verifier` |
   | `prod` (default without a pipeline file) | `d × p` | `product` |
   | `geo` | `sqrt(d × p)` | `geomean` |
   | `wgeo25` | `d^0.75 × p^0.25` | |
   | `wgeo75` | `d^0.25 × p^0.75` | |

   A detection is kept if `score >= thr[class]`. Without a verifier, the score is `d`.
3. **Armed person** (derived). A COCO person detector (default `raw/IntProj_Weapon_Dataset_annot/yolo11m.pt`,
   class 0, conf 0.35) runs only when the frame has at least one confirmed weapon box. A person box becomes
   `armed person` if some confirmed weapon box has `intersection / weapon_area >= 0.3` with it.

| File | Purpose |
|---|---|
| `action_intent_engine.py` | `SuspiciousBehaviorEngine`, threshold/model-dir loading, crops, score modes |
| `predict.py` | CLI: images / folders / `.txt` lists / videos, writing annotated outputs and `detections.csv` |
| `dashboard.py` | Streamlit dashboard with an image tab and a CCTV video tab |
| `test_engine.py` | CPU tests (fakes, the real COCO path, and the real detector+verifier compared against the eval code) |

All commands assume `cd work/app` and `PY=../.venv/bin/python`. Add `--device cpu` while the GPU is busy
training.

## Final inference recipe (defaults)

| Setting | Default | Override |
|---|---|---|
| NMS IoU | **0.5**. Measured: it removes duplicate boxes, and gun precision rises from 0.54 to 0.67 at the same recall. | `--nms-iou`, `nms_iou=` |
| TTA | **on for still images, off for video**. ultralytics `augment=True`, plus verifier h-flip averaging like `eval_pipeline.py --tta`. | `--tta` / `--no-tta` (all sources); `tta_image=`, `tta_video=`, `augment=` |
| Thresholds | per class (gun/knife), from the run dir (see below) | `--thresholds FILE`, `--gun-conf`, `--knife-conf` |
| Detector imgsz | the pipeline file's `det_imgsz`, else the run's `args.yaml`, else the checkpoint, else 960 | `--imgsz` |
| Verifier crop | the pipeline file's `pad` / `cls_imgsz`, else 0.2 / 224 | `verifier_pad=`, `verifier_imgsz=` |
| Batching | **one image per predict call**. A list source would be loaded as one batch and runs out of RAM. Tiles are separate calls too, and the verifier runs in batches of 32 crops. | |

## Point it at a run: `--model-dir`

```bash
$PY predict.py --model-dir ../runs/indian_strict_m960 --verifier ../runs/verifier_quick/weights/best.pt \
    --source ../data/v2/test_domain.txt --out ../out/pred_strict --device cpu
```

`--model-dir ../runs/<run>` picks up:

- **weights** from `<run>/weights/best.pt`
- **thresholds**, in this lookup order (first file found wins):
  - with a verifier:
    1. `<run>/pipeline_tta/pipeline_thresholds.json` for TTA calls (images), or `<run>/pipeline/pipeline_thresholds.json` for non-TTA calls (video).
    2. The other one of those two, with a warning that it was tuned with the other TTA setting.
    3. `pipeline_thresholds.json` next to the weights or in `<run>`.
    4. `<run>/thresholds.json`, with a warning that it is detector-calibrated, so it is applied to the combined score.
    5. The built-in defaults.
  - without a verifier: `thresholds.json` first, then the detector-only thresholds in a pipeline file.
- **imgsz**, see the table above
- **verifier**: if `--verifier` is not given, the verifier path recorded in `pipeline_thresholds.json` is used. Pass `--verifier none` to disable it.

When the best model is ready, swap the run dir:

```bash
$PY predict.py --model-dir ../runs/quick_ft_m800 --verifier ../runs/verifier_quick/weights/best.pt --source IMG_DIR --out OUT
$PY predict.py --model-dir ../runs/big_s2_m960  --verifier ../runs/verifier_quick/weights/best.pt --source IMG_DIR --out OUT
```

For exact pipeline thresholds, first run the eval on the new detector. It writes
`<run>/pipeline/pipeline_thresholds.json`, or `<run>/pipeline_tta/...` with `--tta`. Run it twice to get both:

```bash
cd .. && .venv/bin/python scripts/verifier/eval_pipeline.py --det-weights runs/<run>/weights/best.pt \
    --verifier runs/verifier_quick/weights/best.pt --val-data data/v2/data_all.yaml --nms-iou 0.5 [--tta]
```

After that, `--model-dir runs/<run>` alone (no `--verifier`) loads the same verifier, the selected score mode, and
the thresholds.

### `pipeline_thresholds.json` (written by eval_pipeline.py)

The engine reads these fields:

- `selected_mode`: the score mode.
- `per_mode[mode][cls].thr`: the thresholds. `--combine X` switches to mode X's thresholds.
- `classes[cls].score_thr`: a fallback for the selected mode's thresholds.
- `detector_only[cls].det_conf_thr`: used when there is no verifier.
- `det_conf_floor`: the candidate floor. It is raised to the lowest detector score that can still pass, which never changes the output.
- `det_imgsz`, `cls_imgsz`, `pad`, `tta`, `verifier`.
- `nms_iou`: if it differs from the running NMS IoU, a note is printed.

Pipeline thresholds are used as-is, without the 0.05 clamp. Older files are also accepted:

```jsonc
{"combine": "product", "classes": {"gun": {"thr": 0.41}, "knife": {"thr": 0.33}}}   // old pipeline format
{"gun": 0.4, "knife": 0.3}                                                         // flat
{"source_split": "val", "classes": {"knife": {"class_id": 1, "conf": 0.2}}}        // scripts/train/eval.py thresholds.json
```

These simple formats are clamped to a 0.05 minimum. Explicit `--gun-conf`, `--knife-conf` and `--combine` override the file.

### Config file

Instead of flags, you can use `--config app_config.json`, a JSON object of engine arguments. Relative paths are
resolved against the config file, and CLI flags override it.

```json
{"model_dir": "../runs/indian_strict_m960", "verifier_weights": "../runs/verifier_quick/weights/best.pt",
 "nms_iou": 0.5, "tta_image": true, "tta_video": false, "device": "cpu"}
```

```bash
$PY predict.py --config app_config.json --source IMG_DIR --out OUT
```

In Python: `SuspiciousBehaviorEngine(model_dir=..., verifier_weights=...)` or `SuspiciousBehaviorEngine.from_config(path)`.

## Run

```bash
# smoke run: 3 images from the domain test list, CPU (outputs in work/app/_smoke/)
awk 'NR%48==1' ../data/v2/test_domain.txt | head -3 > /tmp/smoke3.txt
$PY predict.py --model-dir ../runs/indian_strict_m960 --verifier ../runs/verifier_quick/weights/best.pt \
    --source /tmp/smoke3.txt --out _smoke --device cpu

# a whole list / folder (one image per predict call; --limit N for the first N)
$PY predict.py --model-dir ../runs/indian_strict_m960 --verifier ../runs/verifier_quick/weights/best.pt \
    --source ../data/v2/test_domain.txt --out ../out/domain --device cpu --limit 20

# detector only
$PY predict.py --model-dir ../runs/indian_strict_m960 --verifier none --source IMG_DIR --out OUT

# video (video mode: 2-of-5 smoothing, TTA off), no person model
$PY predict.py --model-dir ../runs/indian_strict_m960 --source clip.mp4 --out ../out/clip --tile \
    --max-frames 300 --person-weights none

# dashboard
WEAPON_MODEL_DIR=../runs/indian_strict_m960 VERIFIER_WEIGHTS=../runs/verifier_quick/weights/best.pt \
    DEMO_VIDEO_DIR=./videos ../.venv/bin/streamlit run dashboard.py

# tests (CPU; about 1 minute)
CUDA_VISIBLE_DEVICES= $PY test_engine.py
```

`--source` accepts an image, a directory, a `.txt` list of image paths (one per line; relative paths are relative
to the list), or a video.

Other `predict.py` options:

- `--mode image|video` forces one mode for every source.
- `--combine det|cls|prod|geo|wgeo25|wgeo75`
- `--thresholds FILE`
- `--person-weights PATH|none`, `--person-det-conf 0.35`
- `--pose-weights ../raw/IntProj_Weapon_Dataset_annot/yolo11n-pose.pt` adds the `RAISED_ARM` flag.
- `--tile` runs the full frame plus 2x2 tiles, about 5x slower.
- `--contour-edge` / `--camouflage`: only if the weights were trained on images preprocessed the same way.

`predict.py` prints the resolved settings first: weights, imgsz and where it came from, NMS IoU, TTA per mode,
verifier, and the thresholds and their source per TTA profile.

## Weights resolution (without `--model-dir`)

In order: `--weights`, then `$WEAPON_WEIGHTS`, then the newest `work/runs/*/weights/best.pt`. Run dirs starting
with `_` are skipped. If none is found, it raises `WeightsError`. It never falls back to COCO, and any detector
without `gun` and `knife` classes is refused.

## Output (`meta`)

`threat_score`, `threat_level`, `weapons`, `armed_persons`, `flags`, `raw_counts`, `confirmed`, `mode`, `tta`,
`combine`.

- `weapons`: a list of dicts with `name`, `conf` (the final score), `det_conf`, `verifier_p` (`None` without a verifier), `box`, `cls` and `confirmed`.
- `armed_persons`: a list of `[x1, y1, x2, y2]`.

`detections.csv` columns: `source, frame, class, det_conf, verifier_p, score, x1, y1, x2, y2, threat_score, flags`.
Armed-person rows have empty score columns.

## Flags and score

Confirmed classes produce `KNIFE_DETECTED`, `GUN_DETECTED` and `ARMED_PERSON`. In video mode, a detection that is
not yet confirmed only produces `<CLASS>_UNCONFIRMED`, which adds nothing to the score.

| Event | Points |
|---|---|
| knife | +45 |
| gun | +50 |
| armed person | +20 |
| raised arm | +10 |
| raised arm with a confirmed weapon | +15 more |

The score is capped at 100. Levels: 75 or more is `CRITICAL THREAT / HARM INTENDED`, 45 or more is
`SUSPICIOUS MOVEMENT`, anything lower is `NORMAL`.

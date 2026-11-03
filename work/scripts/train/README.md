# Weapon detector training / evaluation (YOLO11)

Class sets: v1 `data/clean` = 0 armed person, 1 gun, 2 knife; **v2** `data/v2` = 0 gun, 1 knife,
2 handheld_object (a confusion-absorbing class, ignored at inference). All scripts resolve classes
by name, so both work. Metrics files contain an extra `weapons` row (mean over gun + knife); use it
instead of `all` for v2, because `all` includes handheld_object.

The v2 two-stage pipeline (detector -> crop verifier -> armed-person rule) is in section 0; the
v1 knife experiments follow in sections 1-5.

All commands assume:

```bash
cd /home/shiva/Desktop/projects/akriti/weapon-detection/work
PY=/home/shiva/Desktop/projects/akriti/weapon-detection/work/.venv/bin/python
S=scripts/train
```

Runs go to `work/runs/<name>/`. `train.py` automatically evaluates `best.pt` on **val** (writes
`thresholds.json` = per-class best-F1 conf) and then on **test** (`metrics_test.json/.md`,
thresholds not re-tuned on test).

## Before launching (GPU memory)

An `ollama llama-server` was holding ~7.3 GB of the 12 GB A2000 during setup. yolo11m @ 960 with
batch 8 needs roughly 9-10 GB, so stop ollama first (`nvidia-smi` to check), or use `--batch -1`
(AutoBatch) / `--batch 4`. RAM is tight: keep `--workers <= 4`; `cache=False` is the default.

## 0. v2 two-stage pipeline: full-run command sequence

```bash
cd /home/shiva/Desktop/projects/akriti/weapon-detection/work
PY=/home/shiva/Desktop/projects/akriti/weapon-detection/work/.venv/bin/python
S=scripts/train; V=scripts/verifier; D=data/v2

# 1) stage 1: detector on all v2 data (val -> thresholds.json; test + domain + public test metrics)
$PY $S/train.py --data $D/data_all.yaml --name v2_stage1 \
    --test-data $D/data_domain_test.yaml --test-data $D/data_public_test.yaml

# 2) stage 2: fine-tune stage-1 best.pt (recommended stage-2 hyp; optimizer must be explicit, see below)
$PY $S/train.py --data $D/data_stage2.yaml --name v2_stage2 \
    --init-weights runs/v2_stage1/weights/best.pt --epochs 40 \
    --hyp optimizer=SGD --hyp lr0=0.002 --hyp warmup_epochs=1 --hyp close_mosaic=10 \
    --test-data $D/data_domain_test.yaml --test-data $D/data_public_test.yaml

# 3) verifier crops: GT gun/knife + not_weapon (handheld GT, mined detector FPs, background)
$PY $V/make_crops.py --data $D/data_all.yaml --weights runs/v2_stage2/weights/best.pt \
    --out data/verifier_crops

# 4) verifier: YOLO11s-cls, 224 px, 50 epochs (writes runs/verifier_s224/verifier_val.{json,md})
$PY $V/train_verifier.py --data data/verifier_crops --name verifier_s224

# 5) pipeline eval: detector alone vs detector+verifier. Mode + per-class thresholds tuned on VAL,
#    reported on each TEST yaml -> runs/v2_stage2/pipeline/ (with --tta: pipeline_tta/)
$PY $V/eval_pipeline.py --det-weights runs/v2_stage2/weights/best.pt \
    --verifier runs/verifier_s224/weights/best.pt --val-data $D/data_all.yaml \
    --test-data $D/data_domain_test.yaml --test-data $D/data_public_test.yaml
$PY $V/eval_pipeline.py --det-weights runs/v2_stage2/weights/best.pt \
    --verifier runs/verifier_s224/weights/best.pt --val-data $D/data_all.yaml \
    --test-data $D/data_domain_test.yaml --test-data $D/data_public_test.yaml --tta

# 6) armed person = COCO person (yolo11m) x confirmed weapon, on data/clean test ('armed person' GT)
#    -> runs/v2_stage2/armed_person/armed_test.{json,md}: rows 'gt' (upper bound) and 'pred'
$PY scripts/armed_person/eval_armed.py --weights runs/v2_stage2/weights/best.pt \
    --verifier runs/verifier_s224/weights/best.pt \
    --thresholds runs/v2_stage2/pipeline/pipeline_thresholds.json

# 7) compare detector runs per test set (focus knife; 'weapons' columns = gun+knife mean)
$PY $S/eval.py --compare runs/v2_stage1/metrics_test_domain.json runs/v2_stage2/metrics_test_domain.json \
    --out runs/compare_v2_domain.md
$PY $S/eval.py --compare runs/v2_stage1/metrics_test_public.json runs/v2_stage2/metrics_test_public.json \
    --out runs/compare_v2_public.md
$PY $S/eval.py --compare runs/v2_stage2/metrics_test_domain.json --focus gun
```

Pipeline result files: `pipeline_val.md` (every score mode on VAL), `pipeline_thresholds.json`
(selected mode + per-class score thresholds + detector-only thresholds; this is the deployment
config), `pipeline_test_<tag>.md/json` (detector alone vs pipeline, the delta table, all modes).

Notes:
- **Stage 2 / `--init-weights`**: starts a *new* run from a previous best.pt (fresh optimizer and
  schedule, new data, your hyp). This is different from `--resume`. Recommended stage-2 hyp:
  `epochs=40, lr0=0.002, warmup_epochs=1, close_mosaic=10`. With ultralytics' default
  `optimizer=auto`, lr0 is **ignored** (it picks SGD 0.01 or AdamW with its own lr), so pass
  `--hyp optimizer=SGD`; train.py warns about this. For long runs (>10k iterations) `auto` picks
  SGD at lr0=0.01, so 0.002 is a 5x lower fine-tune lr.
- **`--test-data [TAG=]YAML`** (repeatable) evaluates the yaml's `test` split (or `val` if it has no
  test split) after training, writing `metrics_test_<tag>.json/.md`. The default tag is the yaml stem
  without `data_` / `_test`, so `data_domain_test.yaml` gives `domain` and `data_public_test.yaml`
  gives `public`. Thresholds are still tuned only on `--data`'s val split. To re-run one by hand:
  `$PY $S/eval.py --weights W --data $D/data_domain_test.yaml --split test --tag domain` (add `--tta`).
- **Score modes** (eval_pipeline.py): `det` (= detector alone), `cls` = p_class, `prod` =
  det_conf x p_class, `geo` = sqrt(prod), and weighted geometric means `wgeo25/wgeo75`
  (det^0.75 p^0.25, det^0.25 p^0.75). geo and prod rank boxes identically, so they give the same AP/F1
  and only their thresholds differ. That is why the weighted variants are included. The verifier
  mode with the best mean gun/knife F1 on VAL is selected.
- Tune the pipeline on the val split closest to deployment. Use `--val-data $D/data_stage2.yaml` if
  that yaml has a domain val split. To evaluate another test set with already-tuned thresholds, run
  `--thresholds runs/v2_stage2/pipeline/pipeline_thresholds.json --test-data X.yaml`.
- **FP mining**: a detector mined on its own training images finds few FPs (it has memorised them).
  The val crops are fine. If a detector exists that did not train on those images (e.g. stage 1 when
  stage 2 used other data), use it for `--weights`. not_weapon crops whose padded crop region shows
  >= 50% of a GT weapon are dropped (`--nw-max-cover`), so a big FP box around a gun is not labelled
  not_weapon.
- The verifier weights `yolo11s-cls.pt` are downloaded by ultralytics into the current directory.
- `audit.py` works with v2 too (FN causes become `misclassified-as-gun` / `-as-handheld_object`).

## Knife-friendly defaults (train.py)

degrees=0, shear=0, perspective=0, mixup=0, copy_paste=0 (no-op without segment labels), cutmix=0,
mosaic=1.0, close_mosaic=15, scale=0.5, translate=0.1, fliplr=0.5, hsv 0.015/0.7/0.4, cos_lr,
seed=0, epochs=100, patience=30, imgsz=960, batch=8. Override anything with `--hyp key=value`
(repeatable, YAML-typed), e.g. `--hyp lr0=0.005 --hyp scale=0.7`.

## 1. Baseline (clean dataset)

```bash
$PY $S/train.py --data data/clean/data.yaml --name knife_baseline_m960
```

## 2. No-synthetic ablation

```bash
$PY $S/train.py --data data/clean/data_nosyn.yaml --name knife_nosyn_m960
```

## 3. Knife-oversampled run

```bash
$PY $S/train.py --data data/clean/data_oversample.yaml --name knife_oversample_m960
```

Run them one at a time (single GPU). Long runs: launch with `nohup ... > runs/<name>.log 2>&1 &`.
Resume a crashed run: `$PY $S/train.py --data X --name Y --resume runs/Y/weights/last.pt`.

## 4. Re-evaluate / compare

```bash
# re-run eval (val writes thresholds.json; test only writes metrics_test.*)
$PY $S/eval.py --weights runs/knife_baseline_m960/weights/best.pt --data data/clean/data.yaml --split val
$PY $S/eval.py --weights runs/knife_baseline_m960/weights/best.pt --data data/clean/data.yaml --split test

# side-by-side, knife-focused
$PY $S/eval.py --compare runs/knife_baseline_m960/metrics_test.json \
    runs/knife_nosyn_m960/metrics_test.json runs/knife_oversample_m960/metrics_test.json \
    --out runs/compare_knife_test.md
```

Note: all three runs must be evaluated on the same test split for the comparison to be meaningful
(evaluate the ablations with `--data data/clean/data.yaml` if their yaml points to a different test set).

P/R in the table are ultralytics' values at the conf maximising mean F1 across classes; the
`best-F1 conf`/`F1` columns are per class.

## 5. Knife error audit

```bash
$PY $S/audit.py --weights runs/knife_baseline_m960/weights/best.pt --data data/clean/data.yaml --split test
# fixed knife threshold instead of thresholds.json:
$PY $S/audit.py --weights runs/knife_baseline_m960/weights/best.pt --data data/clean/data.yaml --conf 0.25
```

Writes `runs/<name>/audit/` (`audit_<split>/` for non-test splits): `fn_sheet.jpg`, `fp_sheet.jpg`
(max 48 crops; green=GT knife, red=knife pred, orange=other-class box), `summary.json/.md`
(FN by size bucket <1%/1-5%/5-15%/>15% x cause: misclassified-as-gun, misclassified-as-person,
missed, localisation; FP by overlapped GT), and `errors.json` (every FN/FP).
The audit uses `model.predict` (single label per box, as in deployment), whereas `val` uses
multi-label NMS, so recall can differ slightly from the val table.

## Smoke test (what was used to validate the scripts)

v1 class set:

```bash
$PY $S/train.py --data $S/smoke_raw.yaml --model raw/IntProj_Weapon_Dataset_annot/yolo11n.pt \
    --name _smoke --epochs 1 --fraction 0.1 --imgsz 320
$PY $S/audit.py --weights runs/_smoke/weights/best.pt --data $S/smoke_raw.yaml
rm -rf runs/_smoke
```

v2 pipeline: build a tiny v2-mapped subset of data/clean in `$S/_smoke/data/` (images symlinked,
labels remapped gun 1->0, knife 2->1, armed person 0->2 just to exercise class 2). Write
`data_all.yaml` (train/val/test), `data_stage2.yaml`, `data_domain_test.yaml` and
`data_public_test.yaml`, then run with `P=$S/_smoke`:

```bash
$PY $S/train.py --data $P/data/data_all.yaml --model raw/IntProj_Weapon_Dataset_annot/yolo11n.pt --name s1 \
    --project $P/runs --epochs 1 --imgsz 320 --batch 4 --workers 2 \
    --test-data $P/data/data_domain_test.yaml --test-data $P/data/data_public_test.yaml
$PY $S/train.py --data $P/data/data_stage2.yaml --init-weights $P/runs/s1/weights/best.pt --name s2 \
    --project $P/runs --epochs 1 --imgsz 320 --batch 4 --workers 2 --hyp optimizer=SGD --hyp lr0=0.002 \
    --hyp warmup_epochs=1 --hyp close_mosaic=10 --test-data $P/data/data_domain_test.yaml
$PY $S/audit.py --weights $P/runs/s2/weights/best.pt --data $P/data/data_all.yaml
$PY $V/make_crops.py --data $P/data/data_all.yaml --weights $P/runs/s1/weights/best.pt --out $P/crops
$PY $V/train_verifier.py --data $P/crops --model yolo11n-cls.pt --name ver --project $P/runs --epochs 1 \
    --batch 4 --workers 2
$PY $V/eval_pipeline.py --det-weights $P/runs/s2/weights/best.pt --verifier $P/runs/ver/weights/best.pt \
    --val-data $P/data/data_all.yaml --test-data $P/data/data_domain_test.yaml [--tta --max-images 20]
$PY scripts/armed_person/eval_armed.py --max-images 25 --weights $P/runs/s2/weights/best.pt \
    --verifier $P/runs/ver/weights/best.pt --thresholds $P/runs/s2/pipeline/pipeline_thresholds.json --out $P/armed
rm -rf $P yolo11n-cls.pt
```

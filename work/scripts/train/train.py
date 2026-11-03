#!/usr/bin/env python
"""Train a YOLO11 weapon detector with knife-friendly augmentation, then evaluate best.pt on val
(-> thresholds.json) and test (-> metrics_test.json/.md), plus every --test-data yaml
(-> metrics_test_<tag>.json/.md). Class set comes from the data yaml (old: armed person/gun/knife,
v2: 0 gun, 1 knife, 2 handheld_object).

Knife-friendly defaults: no rotation / shear / perspective (they distort thin elongated boxes and
inflate axis-aligned labels), no mixup, no copy_paste (needs segment labels; no-op for boxes),
mosaic on with the last 15 epochs mosaic-free, scale=0.5, horizontal flip only.

Example:
  python train.py --data ../../data/clean/data.yaml --name baseline_m960
  python train.py --data D.yaml --name exp --hyp lr0=0.005 --hyp scale=0.7 --hyp cls=1.0
  # stage 2: fine-tune a stage-1 best.pt on another dataset, evaluate on two extra test yamls
  python train.py --data data/v2/data_stage2.yaml --name v2_stage2 --init-weights runs/v2_stage1/weights/best.pt \
      --epochs 40 --hyp optimizer=SGD --hyp lr0=0.002 --hyp warmup_epochs=1 --hyp close_mosaic=10 \
      --test-data data/v2/data_domain_test.yaml --test-data data/v2/data_public_test.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent  # .../weapon-detection/work
RAW = WORK / "raw" / "IntProj_Weapon_Dataset_annot"
sys.path.insert(0, str(HERE))

KNIFE_HYP = dict(
    degrees=0.0, shear=0.0, perspective=0.0, mixup=0.0, copy_paste=0.0, cutmix=0.0,
    mosaic=1.0, close_mosaic=15, scale=0.5, translate=0.1, fliplr=0.5, flipud=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    cos_lr=True, plots=True, seed=0, deterministic=False,
    cache=False,  # RAM is tight
)


def parse_hyp(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--hyp expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)  # typed: 0.5 -> float, true -> bool, [1,2] -> list
    return out


def parse_test_data(items: list[str]) -> list[tuple[str, Path]]:
    """--test-data [TAG=]PATH -> [(tag, path)]. Default tag: yaml stem minus 'data_' / '_test'
    (data_domain_test.yaml -> domain, data_public_test.yaml -> public)."""
    out = []
    for it in items or []:
        tag, path = it.split("=", 1) if "=" in it and not Path(it).exists() else (None, it)
        path = Path(path).resolve()
        if not path.exists():
            raise SystemExit(f"--test-data: {path} not found")
        if not tag:
            tag = path.stem
            tag = tag[5:] if tag.startswith("data_") else tag
            tag = tag[:-5] if tag.endswith("_test") else tag
        if tag in [t for t, _ in out]:
            raise SystemExit(f"--test-data: duplicate tag {tag!r}; use TAG=PATH")
        out.append((tag, path))
    return out


def eval_split_of(yaml_path) -> str | None:
    """'test' if the yaml has a test split, else 'val' (None if neither)."""
    with open(yaml_path) as f:
        d = yaml.safe_load(f) or {}
    return "test" if d.get("test") else ("val" if d.get("val") else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=str(RAW / "yolo11m.pt"))
    ap.add_argument("--init-weights", default=None, metavar="BEST_PT",
                    help="start from a previous run's best.pt (stage-2 fine-tuning); overrides --model. Unlike "
                         "--resume this starts a fresh run (new optimizer/schedule, new data, your hyp).")
    ap.add_argument("--test-data", action="append", default=[], metavar="[TAG=]YAML",
                    help="extra test yaml(s) evaluated after training (its 'test' split, else 'val') -> "
                         "metrics_test_<tag>.json/.md; repeatable. Thresholds stay tuned on --data val.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=float, default=8,
                    help="batch size; -1 = AutoBatch (~60%% of free GPU mem), 0.x = that fraction of GPU mem")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers (keep <=4, RAM is tight)")
    ap.add_argument("--project", default=str(WORK / "runs"))
    ap.add_argument("--exist-ok", action="store_true")
    ap.add_argument("--hyp", action="append", default=[], metavar="KEY=VALUE",
                    help="override any ultralytics train arg, repeatable")
    ap.add_argument("--resume", default=None, metavar="LAST_PT", help="resume from run/weights/last.pt")
    ap.add_argument("--no-eval", action="store_true", help="skip post-training val/test evaluation")
    a = ap.parse_args()

    test_sets = parse_test_data(a.test_data)  # validate paths before a long run
    if a.workers > 4:
        print(f"[warn] workers={a.workers} > 4 may exhaust RAM")
    from ultralytics import YOLO

    if a.resume:
        model = YOLO(a.resume)
        model.train(resume=True)
    else:
        batch = int(a.batch) if a.batch >= 1 or a.batch == -1 else a.batch
        cfg = dict(KNIFE_HYP)
        cfg.update(data=str(Path(a.data).resolve()), epochs=a.epochs, imgsz=a.imgsz, batch=batch,
                   patience=a.patience, fraction=a.fraction, device=a.device, workers=a.workers,
                   project=str(Path(a.project).resolve()), name=a.name, exist_ok=a.exist_ok)
        cfg.update(parse_hyp(a.hyp))
        if "lr0" in cfg and cfg.get("optimizer", "auto") == "auto":
            print("[warn] optimizer=auto ignores lr0 (ultralytics picks SGD/AdamW + lr itself); "
                  "add --hyp optimizer=SGD (or AdamW) for lr0 to take effect")
        if cfg.get("cache"):
            print(f"[warn] cache={cfg['cache']} requested; RAM is tight")
        print("[train] config:\n" + json.dumps(cfg, indent=2, default=str))
        src = a.init_weights or a.model
        if a.init_weights:
            if not Path(a.init_weights).exists():
                raise SystemExit(f"--init-weights {a.init_weights} not found")
            print(f"[train] stage-2: initialising from {Path(a.init_weights).resolve()}")
        model = YOLO(src)
        model.train(**cfg)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    print(f"[train] done: {save_dir}  weights: {best}")
    if a.no_eval:
        return

    from eval import run_eval

    data = model.trainer.args.data
    with open(data) as f:
        has_test = bool(yaml.safe_load(f).get("test"))
    imgsz = model.trainer.args.imgsz
    # 1) val: chooses per-class best-F1 thresholds (thresholds.json)
    run_eval(best, data, "val", imgsz=imgsz, device=a.device, run_dir=save_dir, write_thresholds=True,
             workers=a.workers)
    # 2) test: unbiased final numbers
    if has_test:
        run_eval(best, data, "test", imgsz=imgsz, device=a.device, run_dir=save_dir, write_thresholds=False,
                 workers=a.workers)
    else:
        print("[eval] data yaml has no 'test' split; only val evaluated")
    # 3) extra test sets (e.g. v2 domain / public): metrics_test_<tag>.json
    for tag, path in test_sets:
        split = eval_split_of(path)
        if split is None:
            print(f"[eval] {path} has neither test nor val split; skipped")
            continue
        try:
            run_eval(best, path, split, imgsz=imgsz, device=a.device, run_dir=save_dir, write_thresholds=False,
                     workers=a.workers, tag=tag)
        except Exception as e:  # one broken yaml must not lose the other evaluations
            print(f"[eval] {tag} ({path}) failed: {e!r}")
    if test_sets:
        print("[eval] test metrics: " + ", ".join(str(save_dir / f"metrics_{eval_split_of(p) or 'test'}_{t}.json")
                                                   for t, p in test_sets))


if __name__ == "__main__":
    main()

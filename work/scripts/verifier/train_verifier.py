#!/usr/bin/env python
"""Train the stage-2 crop verifier (YOLO11s-cls, classes gun / knife / not_weapon) on make_crops.py output,
then write <run>/verifier_val.json/.md (top-1 + per-class precision/recall + confusion matrix on val).

Defaults: yolo11s-cls.pt (ultralytics downloads it), imgsz 224, 50 epochs, batch 64, patience 15.
Ultralytics' classification augmentation (RandAugment, random erasing, flips, hsv) is kept; flipud is
off and scale is mild so thin knife crops stay recognisable. Override with --hyp key=value.

Example:
  python train_verifier.py --data data/verifier_crops --name verifier_s224
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "train"))

from train import parse_hyp  # noqa: E402

CLS_HYP = dict(fliplr=0.5, flipud=0.0, scale=0.3, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, erasing=0.3,
               cos_lr=True, seed=0, deterministic=False, plots=True, cache=False)


def eval_verifier(weights, data, device="0", imgsz=224, batch=64, workers=4, out_dir=None) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out_dir = Path(out_dir or Path(weights).resolve().parent.parent)
    m = model.val(data=str(data), split="val", imgsz=imgsz, batch=batch, device=device, workers=workers,
                  project=str(out_dir), name="val_verifier", exist_ok=True, plots=True, verbose=False)
    names = model.names
    cm = np.asarray(m.confusion_matrix.matrix)  # rows = predicted, cols = true (ultralytics convention)
    n = len(names)
    cm = cm[:n, :n]
    rows = []
    for i in range(n):
        tp = cm[i, i]
        prec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        rec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rows.append({"class": names[i], "n_val": int(cm[:, i].sum()), "precision": round(float(prec), 4),
                     "recall": round(float(rec), 4)})
    res = {"weights": str(Path(weights).resolve()), "data": str(data), "top1": round(float(m.top1), 4),
           "top5": round(float(m.top5), 4), "per_class": rows,
           "confusion_matrix_pred_x_true": cm.astype(int).tolist(), "names": [names[i] for i in range(n)]}
    (out_dir / "verifier_val.json").write_text(json.dumps(res, indent=2))
    md = [f"### verifier val — {out_dir.name}", "", f"top-1 = {res['top1']:.4f}", "",
          "| class | n | precision | recall |", "|---|---|---|---|"]
    md += [f"| {r['class']} | {r['n_val']} | {r['precision']:.3f} | {r['recall']:.3f} |" for r in rows]
    md += ["", "confusion (rows = predicted, cols = true): " + ", ".join(res["names"]), ""]
    md += ["| pred \\ true | " + " | ".join(res["names"]) + " |", "|---|" + "---|" * n]
    md += [f"| {res['names'][i]} | " + " | ".join(str(int(v)) for v in cm[i]) + " |" for i in range(n)]
    (out_dir / "verifier_val.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="ImageFolder root with train/ and val/ (make_crops.py --out)")
    ap.add_argument("--model", default="yolo11s-cls.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default=str(WORK / "runs"))
    ap.add_argument("--exist-ok", action="store_true")
    ap.add_argument("--hyp", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()

    data = Path(a.data).resolve()
    for sub in ("train", "val"):
        if not (data / sub).is_dir():
            raise SystemExit(f"{data / sub} missing; run make_crops.py first")
    from ultralytics import YOLO

    cfg = dict(CLS_HYP)
    cfg.update(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
               fraction=a.fraction, device=a.device, workers=a.workers, project=str(Path(a.project).resolve()),
               name=a.name, exist_ok=a.exist_ok)
    cfg.update(parse_hyp(a.hyp))
    print("[verifier] config:\n" + json.dumps(cfg, indent=2, default=str))
    model = YOLO(a.model)
    model.train(**cfg)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    print(f"[verifier] done: {save_dir}  weights: {best}")
    if not a.no_eval:
        eval_verifier(best, data, a.device, a.imgsz, a.batch, a.workers, save_dir)


if __name__ == "__main__":
    main()

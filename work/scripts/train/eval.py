#!/usr/bin/env python
"""Evaluate a trained YOLO11 weapon detector and write per-class metrics + per-class thresholds.

Usage:
  python eval.py --weights RUN/weights/best.pt --data DATA.yaml [--split test] [--tag domain]
  python eval.py --compare RUN_A/metrics_test.json RUN_B/metrics_test.json ...

Outputs (in <run_dir>, which defaults to the run folder that contains weights/best.pt):
  metrics_<split>.json / metrics_<split>.md   per-class P, R, mAP50, mAP50-95, instances, best-F1 conf
                                              (metrics_<split>_<tag>.* with --tag, e.g. one per test yaml)
  thresholds.json                             recommended per-class conf thresholds (best F1 on <split>)
  val_<split>/                                ultralytics plots (PR / F1 curves, confusion matrix)

thresholds.json is only (over)written when evaluating the val split (or with --write-thresholds), so
thresholds are tuned on 'val' and test numbers stay unbiased. train.py evaluates val first, then test.
Best-F1 confs for the evaluated split are always included inside metrics_<split>.json.

Works with any class set (old: armed person/gun/knife; v2: gun/knife/handheld_object). Besides 'all',
the json/md contain a 'weapons' row = mean over the gun + knife classes that exist, which is the
number that matters when handheld_object is only a confusion-absorbing class ignored at inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KNIFE_NAME = "knife"
WEAPON_NAMES = ("gun", "knife")


def weapons_row(rows: list[dict]) -> dict | None:
    """Mean of per-class metrics over gun/knife rows (None if neither class exists)."""
    w = [r for r in rows if r["class"].lower() in WEAPON_NAMES]
    if not w:
        return None
    out = {"class_id": -2, "class": "weapons", "instances": int(sum(r["instances"] for r in w))}
    for k in ("precision", "recall", "map50", "map50_95"):
        out[k] = float(np.mean([r[k] for r in w]))
    out["best_f1_conf"], out["best_f1"] = float("nan"), float(np.nanmean([r["best_f1"] for r in w]))
    return out


def run_dir_from_weights(weights: str | Path) -> Path:
    w = Path(weights).resolve()
    if w.parent.name == "weights":
        return w.parent.parent
    return w.parent


def _smooth(y: np.ndarray, f: float = 0.1) -> np.ndarray:
    """Same box-filter smoothing ultralytics uses for its F1 curve."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def per_class_thresholds(box, names: dict) -> dict:
    """Best-F1 confidence per class from metrics.box.f1_curve / px (rows follow ap_class_index)."""
    out = {}
    px = np.asarray(getattr(box, "px", []))
    f1c = np.asarray(getattr(box, "f1_curve", []))
    pc = np.asarray(getattr(box, "p_curve", []))
    rc = np.asarray(getattr(box, "r_curve", []))
    if px.size == 0 or f1c.size == 0:
        return out
    for row, ci in enumerate(box.ap_class_index):
        ci = int(ci)
        f1s = _smooth(f1c[row])
        j = int(f1s.argmax())
        out[names[ci]] = {
            "class_id": ci,
            # floor at 0.01: an (under-trained) curve can peak at conf~0, which is useless operationally
            "conf": round(max(float(px[j]), 0.01), 4) if f1s[j] > 0 else 0.25,
            "conf_raw": round(float(px[j]), 4),
            "f1": round(float(f1c[row][j]), 4),
            "precision": round(float(pc[row][j]), 4) if pc.size else None,
            "recall": round(float(rc[row][j]), 4) if rc.size else None,
        }
    return out


def table_md(rows: list[dict], title: str = "") -> str:
    hdr = "| class | instances | P | R | mAP50 | mAP50-95 | best-F1 conf | F1 |\n|---|---|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        body += (f"| {r['class']} | {r['instances']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                 f"{r['map50']:.3f} | {r['map50_95']:.3f} | {r.get('best_f1_conf', float('nan')):.3f} | "
                 f"{r.get('best_f1', float('nan')):.3f} |\n")
    return (f"### {title}\n\n" if title else "") + hdr + body


def run_eval(weights, data, split="test", imgsz=None, batch=8, device="0", conf=0.001, iou=0.7,
             run_dir=None, write_thresholds=True, workers=4, plots=True, tag=None, augment=False) -> dict:
    """tag: suffix for output names (metrics_<split>_<tag>.json, val_<split>_<tag>/), e.g. 'domain'."""
    from ultralytics import YOLO

    run_dir = Path(run_dir) if run_dir else run_dir_from_weights(weights)
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{tag}" if tag else split
    model = YOLO(str(weights))
    if imgsz is None:  # use training imgsz if recorded in checkpoint
        imgsz = (getattr(model, "overrides", {}) or {}).get("imgsz") or 640
    m = model.val(data=str(data), split=split, imgsz=imgsz, batch=batch, device=device, conf=conf, iou=iou,
                  workers=workers, plots=plots, project=str(run_dir), name=f"val_{stem}", exist_ok=True,
                  verbose=False, augment=augment)
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    thr = per_class_thresholds(m.box, names)
    nt = m.nt_per_class if m.nt_per_class is not None else np.zeros(len(names), int)
    rows = []
    for i, ci in enumerate(m.box.ap_class_index):
        ci = int(ci)
        p, r, ap50, ap = m.box.class_result(i)
        t = thr.get(names[ci], {})
        rows.append({"class_id": ci, "class": names[ci], "instances": int(nt[ci]), "precision": float(p),
                     "recall": float(r), "map50": float(ap50), "map50_95": float(ap),
                     "best_f1_conf": t.get("conf", float("nan")), "best_f1": t.get("f1", float("nan"))})
    present = {r["class_id"] for r in rows}
    for ci, n in names.items():  # classes with no GT in this split
        if ci not in present:
            rows.append({"class_id": ci, "class": n, "instances": int(nt[ci]), "precision": 0.0, "recall": 0.0,
                         "map50": 0.0, "map50_95": 0.0, "best_f1_conf": float("nan"), "best_f1": float("nan")})
    rows.sort(key=lambda r: r["class_id"])
    mp, mr, map50, map_ = m.box.mean_results()
    allrow = {"class_id": -1, "class": "all", "instances": int(nt.sum()), "precision": float(mp),
              "recall": float(mr), "map50": float(map50), "map50_95": float(map_),
              "best_f1_conf": float("nan"), "best_f1": float("nan")}
    wrow = weapons_row(rows)
    res = {"weights": str(Path(weights).resolve()), "data": str(Path(data).resolve()), "split": split,
           "tag": tag, "imgsz": imgsz, "augment": augment, "run_dir": str(run_dir), "all": allrow,
           "weapons": wrow, "per_class": rows, "thresholds": thr, "speed_ms": m.speed}
    (run_dir / f"metrics_{stem}.json").write_text(json.dumps(res, indent=2, default=float))
    md = table_md([allrow] + ([wrow] if wrow else []) + rows,
                  f"{run_dir.name} — {stem} (imgsz={imgsz}{', TTA' if augment else ''})")
    (run_dir / f"metrics_{stem}.md").write_text(md)
    if write_thresholds and thr:
        (run_dir / "thresholds.json").write_text(json.dumps({"source_split": split, "classes": thr}, indent=2))
    print("\n" + md)
    print(f"saved: {run_dir / f'metrics_{stem}.json'}  {run_dir / f'metrics_{stem}.md'}"
          + (f"  {run_dir / 'thresholds.json'}" if write_thresholds and thr else ""))
    return res


def compare(files: list[str], focus: str = KNIFE_NAME) -> str:
    runs = [json.loads(Path(f).read_text()) for f in files]

    def get(run, cls):
        if cls == "all":
            return run["all"]
        return next((r for r in run["per_class"] if r["class"] == cls), None)

    def label(run, f):
        t = f"_{run['tag']}" if run.get("tag") else ""
        return f"{Path(run.get('run_dir', Path(f).parent)).name}/{run.get('split', '?')}{t}"

    def wcols(run):  # gun+knife mean ('weapons' row; older json files lack it -> recompute)
        w = run.get("weapons") or weapons_row(run["per_class"])
        return f"{w['map50']:.3f} | {w['map50_95']:.3f}" if w else "- | -"

    lines = [f"### {focus} (focus)\n",
             "| run | inst | P | R | mAP50 | mAP50-95 | best-F1 conf | F1 | all mAP50 | all mAP50-95 "
             "| weapons mAP50 | weapons mAP50-95 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, f in zip(runs, files):
        k, a = get(run, focus), run["all"]
        if k is None:
            lines.append(f"| {label(run, f)} | - | - | - | - | - | - | - | {a['map50']:.3f} | {a['map50_95']:.3f} "
                         f"| {wcols(run)} |")
            continue
        lines.append(f"| {label(run, f)} | {k['instances']} | {k['precision']:.3f} | {k['recall']:.3f} | "
                     f"{k['map50']:.3f} | {k['map50_95']:.3f} | {k['best_f1_conf']:.3f} | {k['best_f1']:.3f} | "
                     f"{a['map50']:.3f} | {a['map50_95']:.3f} | {wcols(run)} |")
    classes = []
    for run in runs:
        for r in run["per_class"]:
            if r["class"] not in classes:
                classes.append(r["class"])
    lines += ["", "### mAP50-95 per class\n", "| run | " + " | ".join(classes + ["all"]) + " |",
              "|---|" + "---|" * (len(classes) + 1)]
    for run, f in zip(runs, files):
        vals = []
        for c in classes + ["all"]:
            r = get(run, c)
            vals.append(f"{r['map50_95']:.3f}" if r else "-")
        lines.append(f"| {label(run, f)} | " + " | ".join(vals) + " |")
    out = "\n".join(lines)
    print(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=None, help="output suffix: metrics_<split>_<tag>.json (e.g. domain, public)")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (ultralytics augment=True)")
    ap.add_argument("--imgsz", type=int, default=None, help="default: imgsz stored in checkpoint")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--run-dir", default=None, help="where to write outputs (default: run dir of weights)")
    ap.add_argument("--write-thresholds", action="store_true",
                    help="write thresholds.json even when split is not val (default: only for split=val)")
    ap.add_argument("--compare", nargs="+", metavar="METRICS_JSON", help="side-by-side table of metrics json files")
    ap.add_argument("--focus", default=KNIFE_NAME)
    ap.add_argument("--out", default=None, help="(compare mode) write markdown here")
    a = ap.parse_args()
    if a.compare:
        md = compare(a.compare, a.focus)
        if a.out:
            Path(a.out).write_text(md + "\n")
        return
    if not a.weights or not a.data:
        ap.error("--weights and --data are required unless --compare is used")
    run_eval(a.weights, a.data, a.split, a.imgsz, a.batch, a.device, iou=a.iou, run_dir=a.run_dir,
             write_thresholds=a.write_thresholds or (a.split in ("val", "valid") and not a.tag),
             workers=a.workers, tag=a.tag, augment=a.tta)


if __name__ == "__main__":
    sys.exit(main())

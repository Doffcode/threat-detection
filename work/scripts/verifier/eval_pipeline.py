#!/usr/bin/env python
"""Evaluate the two-stage pipeline (detector -> crop verifier) against the detector alone.

1. Detector at a low conf (--det-conf 0.01) on VAL and on each TEST yaml. Only gun / knife predictions
   are kept (handheld_object is a confusion-absorbing class, ignored at inference).
2. Each kept box is cropped exactly like make_crops.py (pad 20%, square-ish, 224) and the verifier gives
   p_class = P(verifier class == detected class). Score modes:
       det      det_conf                       (detector alone, the baseline)
       cls      p_class
       prod     det_conf * p_class
       geo      sqrt(det_conf * p_class)        (same ranking as prod -> same AP/F1, other thresholds)
       wgeo25   det_conf^0.75 * p_class^0.25    (weighted geometric means; these DO rank differently)
       wgeo75   det_conf^0.25 * p_class^0.75
3. VAL: per class and mode, the best-F1 score threshold; the verifier mode with the highest mean
   (gun, knife) F1 on VAL is selected (tie -> mean AP50). -> pipeline_thresholds.json
4. TEST: per class P / R / F1 at the VAL thresholds (TP = IoU >= 0.5, one-to-one, score-ordered) and AP50
   from the score itself, for the detector alone and the selected pipeline (+ all modes in the json).

Outputs (default <det run dir>/pipeline[_tta]/): pipeline_thresholds.json, pipeline_val.{json,md},
pipeline_test_<tag>.{json,md}.

Examples:
  python eval_pipeline.py --det-weights runs/v2_stage2/weights/best.pt --verifier runs/verifier_s224/weights/best.pt \\
      --val-data data/v2/data_all.yaml --test-data data/v2/data_domain_test.yaml --test-data data/v2/data_public_test.yaml
  # re-use thresholds from an earlier run (skips VAL):
  python eval_pipeline.py ... --thresholds runs/v2_stage2/pipeline/pipeline_thresholds.json --test-data X.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import (WEAPONS, Verifier, ap50, best_f1, load_gt, load_split, match, name_ids, prf_at,
                    square_crop)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from eval import run_dir_from_weights  # noqa: E402
from train import eval_split_of, parse_test_data  # noqa: E402

MODES = {
    "det": lambda d, p: d,
    "cls": lambda d, p: p,
    "prod": lambda d, p: d * p,
    "geo": lambda d, p: np.sqrt(d * p),
    "wgeo25": lambda d, p: d ** 0.75 * p ** 0.25,
    "wgeo75": lambda d, p: d ** 0.25 * p ** 0.75,
}
VERIFIER_MODES = [m for m in MODES if m != "det"]


def collect(det, ver, data_yaml, split, a) -> list[dict]:
    """Per image: GT gun/knife boxes and gun/knife predictions with det_conf and verifier p_class."""
    data, imgs = load_split(data_yaml, split, a.max_images, a.seed)
    gid = name_ids(data["names"])
    missing = [n for n in WEAPONS if n not in gid]
    if missing:
        raise SystemExit(f"{data_yaml}: classes {missing} not in {data['names']}")
    did = name_ids(det.names)
    det_w = {did[n]: n for n in WEAPONS if n in did}
    imgsz = a.imgsz or (det.overrides or {}).get("imgsz") or 640
    t0 = time.time()
    recs = []
    # one image per call: a list source is loaded as a single batch and runs out of RAM
    stream = (det.predict(str(q), imgsz=imgsz, conf=a.det_conf, iou=a.nms_iou, device=a.device, verbose=False,
                          max_det=a.max_det, augment=a.tta, classes=list(det_w))[0] for q in imgs)
    for p, res in zip(imgs, stream):
        H, W = res.orig_shape
        gc, gb = load_gt(p, W, H)
        rec = {"image": str(p), "gt": {n: gb[gc == gid[n]] for n in WEAPONS}, "pred": {}}
        if res.boxes is not None and len(res.boxes):
            pb = res.boxes.xyxy.cpu().numpy()
            pc = res.boxes.cls.cpu().numpy().astype(int)
            ps = res.boxes.conf.cpu().numpy()
        else:
            pb, pc, ps = np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
        if ver is not None and len(pb):
            probs = ver.probs([square_crop(res.orig_img, b, a.pad, a.cls_imgsz) for b in pb])
        for ci, n in det_w.items():
            m = pc == ci
            pcls = probs[m, ver.ids[n]] if ver is not None and len(pb) else np.ones(int(m.sum()))
            rec["pred"][n] = (ps[m].astype(float), pcls.astype(float), pb[m])
        recs.append(rec)
    print(f"[pipeline] {Path(str(data_yaml)).name}:{split} {len(recs)} images in {time.time() - t0:.0f}s "
          f"(imgsz={imgsz}, det_conf={a.det_conf}, tta={a.tta})")
    return recs


def score_eval(recs, mode, cls, iou_thr):
    f = MODES[mode]
    preds = [(f(r["pred"][cls][0], r["pred"][cls][1]), r["pred"][cls][2]) for r in recs]
    return match(preds, [r["gt"][cls] for r in recs], iou_thr)


def tune(recs, modes, iou_thr) -> dict:
    out = {}
    for mode in modes:
        out[mode] = {}
        for c in WEAPONS:
            s, tp, n = score_eval(recs, mode, c, iou_thr)
            out[mode][c] = {**best_f1(s, tp, n), "ap50": round(ap50(tp, n), 4), "n_gt": n}
        out[mode]["mean_f1"] = round(float(np.mean([out[mode][c]["f1"] for c in WEAPONS])), 4)
        out[mode]["mean_ap50"] = round(float(np.mean([out[mode][c]["ap50"] for c in WEAPONS])), 4)
    return out


def test_eval(recs, modes, thr: dict, iou_thr) -> dict:
    out = {}
    for mode in modes:
        out[mode] = {}
        for c in WEAPONS:
            s, tp, n = score_eval(recs, mode, c, iou_thr)
            t = thr[mode][c]
            out[mode][c] = {"thr": t, **prf_at(s, tp, n, t), "ap50": round(ap50(tp, n), 4), "n_gt": n}
        for k in ("f1", "precision", "recall", "ap50"):
            out[mode][f"mean_{k}"] = round(float(np.mean([out[mode][c][k] for c in WEAPONS])), 4)
    return out


def table(res: dict, modes, title: str, mark: str | None = None) -> list[str]:
    lines = [f"### {title}", "", "| mode | class | GT | thr | P | R | F1 | TP | FP | FN | AP50 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode in modes:
        lab = f"**{mode}**" if mode == mark else mode
        for c in WEAPONS:
            r = res[mode][c]
            lines.append(f"| {lab} | {c} | {r['n_gt']} | {r['thr']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                         f"{r['f1']:.3f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['ap50']:.3f} |")
        mf = res[mode].get("mean_f1", float("nan"))
        lines.append(f"| {lab} | mean | | | {res[mode].get('mean_precision', float('nan')):.3f} | "
                     f"{res[mode].get('mean_recall', float('nan')):.3f} | {mf:.3f} | | | | "
                     f"{res[mode]['mean_ap50']:.3f} |")
    return lines + [""]


def delta_lines(res: dict, sel: str) -> list[str]:
    if sel == "det" or sel not in res:
        return []
    lines = [f"### verifier effect ({sel} - det)", "", "| class | dP | dR | dF1 | dAP50 | dFP | dTP |",
             "|---|---|---|---|---|---|---|"]
    for c in WEAPONS:
        d, s = res["det"][c], res[sel][c]
        lines.append(f"| {c} | {s['precision'] - d['precision']:+.3f} | {s['recall'] - d['recall']:+.3f} | "
                     f"{s['f1'] - d['f1']:+.3f} | {s['ap50'] - d['ap50']:+.3f} | {s['fp'] - d['fp']:+d} | "
                     f"{s['tp'] - d['tp']:+d} |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-weights", required=True)
    ap.add_argument("--verifier", default=None, help="verifier best.pt (omit: detector-only evaluation)")
    ap.add_argument("--val-data", default=None, help="yaml whose VAL split tunes mode + thresholds")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--test-data", action="append", default=[], metavar="[TAG=]YAML",
                    help="test yaml(s) (its test split, else val); repeatable. Default: --val-data's test split")
    ap.add_argument("--thresholds", default=None, help="existing pipeline_thresholds.json -> skip VAL tuning")
    ap.add_argument("--det-conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=None, help="detector imgsz (default: from checkpoint)")
    ap.add_argument("--cls-imgsz", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.2, help="crop padding (must match make_crops.py)")
    ap.add_argument("--tta", action="store_true", help="detector augment=True + verifier h-flip averaging")
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-images", type=int, default=None, help="random subset per split (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO

    out = Path(a.out) if a.out else run_dir_from_weights(a.det_weights) / ("pipeline_tta" if a.tta else "pipeline")
    out.mkdir(parents=True, exist_ok=True)
    det = YOLO(a.det_weights)
    ver = Verifier(a.verifier, a.device, a.cls_imgsz, a.tta) if a.verifier else None
    modes = list(MODES) if ver else ["det"]

    tests = parse_test_data(a.test_data)
    if not tests:
        if not a.val_data:
            ap.error("give --test-data and/or --val-data")
        tests = [(Path(a.val_data).stem.removeprefix("data_") or "test", Path(a.val_data).resolve())]
        tests = [(t, p) for t, p in tests if eval_split_of(p) == "test"]
        if not tests:
            ap.error(f"{a.val_data} has no test split; pass --test-data")

    if a.thresholds:
        th = json.loads(Path(a.thresholds).read_text())
        thr = {m: {c: v["thr"] for c, v in th["per_mode"][m].items()} for m in th["per_mode"] if m in modes}
        selected = th["selected_mode"] if th["selected_mode"] in modes else "det"
        modes = [m for m in modes if m in thr]
        val_res = None
        print(f"[pipeline] thresholds from {a.thresholds}: selected={selected}")
        for k, v in (("tta", a.tta), ("det_conf_floor", a.det_conf), ("pad", a.pad)):
            if k in th and th[k] != v:
                print(f"[warn] thresholds were tuned with {k}={th[k]} but this run uses {v}")
    else:
        if not a.val_data:
            ap.error("--val-data is required unless --thresholds is given")
        val_res = tune(collect(det, ver, a.val_data, a.val_split, a), modes, a.iou_match)
        cand = VERIFIER_MODES if ver else ["det"]
        selected = max(cand, key=lambda m: (val_res[m]["mean_f1"], val_res[m]["mean_ap50"]))
        thr = {m: {c: val_res[m][c]["thr"] for c in WEAPONS} for m in modes}
        th = {"selected_mode": selected, "formula": {
                  "det": "det_conf", "cls": "p_class", "prod": "det_conf*p_class", "geo": "sqrt(det_conf*p_class)",
                  "wgeo25": "det_conf**0.75 * p_class**0.25", "wgeo75": "det_conf**0.25 * p_class**0.75"},
              "classes": {c: {"score_thr": thr[selected][c], "val_f1": val_res[selected][c]["f1"]} for c in WEAPONS},
              "detector_only": {c: {"det_conf_thr": thr["det"][c], "val_f1": val_res["det"][c]["f1"]} for c in WEAPONS},
              "per_mode": {m: {c: {"thr": thr[m][c], "val_f1": val_res[m][c]["f1"], "val_ap50": val_res[m][c]["ap50"]}
                               for c in WEAPONS} for m in modes},
              "det_weights": str(Path(a.det_weights).resolve()),
              "verifier": str(Path(a.verifier).resolve()) if a.verifier else None,
              "val_data": str(Path(a.val_data).resolve()), "val_split": a.val_split, "det_conf_floor": a.det_conf,
              "det_imgsz": a.imgsz or (det.overrides or {}).get("imgsz"), "cls_imgsz": a.cls_imgsz, "pad": a.pad,
              "tta": a.tta, "nms_iou": a.nms_iou, "ignored_classes": ["handheld_object"]}
        (out / "pipeline_thresholds.json").write_text(json.dumps(th, indent=2))
        md = table(val_res, modes, f"VAL ({Path(a.val_data).name}:{a.val_split}) — best-F1 thresholds", selected)
        md = [f"# Pipeline VAL — selected mode: {selected}", ""] + md
        (out / "pipeline_val.md").write_text("\n".join(md) + "\n")
        (out / "pipeline_val.json").write_text(json.dumps({"selected_mode": selected, "results": val_res}, indent=2))
        print("\n".join(md))

    for tag, path in tests:
        split = eval_split_of(path)
        recs = collect(det, ver, path, split, a)
        res = test_eval(recs, modes, thr, a.iou_match)
        shown = ["det"] + ([selected] if selected != "det" else [])
        md = [f"# Pipeline TEST {tag} ({path.name}:{split}) — det vs det+verifier ({selected})", "",
              f"detector {Path(a.det_weights).resolve()}  verifier {a.verifier}  det_conf floor {a.det_conf}  "
              f"tta {a.tta}. Thresholds tuned on VAL; AP50 from the mode's score.", ""]
        md += table(res, shown, "detector alone vs pipeline", selected) + delta_lines(res, selected)
        if len(modes) > 2:
            md += table(res, modes, "all score modes (VAL thresholds)", selected)
        (out / f"pipeline_test_{tag}.md").write_text("\n".join(md) + "\n")
        (out / f"pipeline_test_{tag}.json").write_text(json.dumps(
            {"tag": tag, "data": str(path), "split": split, "selected_mode": selected, "results": res,
             "thresholds": thr}, indent=2))
        print("\n".join(md))
    print(f"[pipeline] saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Derived 'armed person' evaluation: COCO person detector + confirmed weapon boxes.

armed person := a person box (pretrained COCO yolo11m.pt, class 0) whose intersection with a confirmed
weapon box covers >= --link (0.3) of the WEAPON box area (intersection / weapon area).
Score of an armed-person box = person_conf x max(score of linked confirmed weapons).

Weapon sources (--weapons):
  gt    GT gun/knife boxes of the same images, score 1 (upper bound: perfect weapon detector)
  pred  --weights detector (gun/knife, by name) [+ --verifier], confirmed with per-class thresholds:
          pipeline_thresholds.json (selected mode with --verifier, else its detector_only thresholds), or
          a train/eval.py thresholds.json (detector only). Default: <run>/thresholds.json
  both  (default when --weights is given)

GT: the 'armed person' boxes in --data (work/data/clean: class 0 'armed person'; resolved by name).
Metrics: AP50 (from the score) and P/R/F1 at an armed-person score threshold tuned for best F1 on
--tune-split (val); with --tune-split none the best F1 on the eval split itself is reported (oracle).

Example:
  python eval_armed.py --weights runs/v2_stage2/weights/best.pt --verifier runs/verifier_s224/weights/best.pt \\
      --thresholds runs/v2_stage2/pipeline/pipeline_thresholds.json
Outputs: <out>/armed_<split>.json/.md (default out: <det run>/armed_person, or runs/armed_person_gt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "verifier"))

from common import WEAPONS, WORK, Verifier, ap50, best_f1, load_gt, load_split, match, name_ids, prf_at  # noqa: E402

RAW = WORK / "raw" / "IntProj_Weapon_Dataset_annot"


def link_scores(pboxes: np.ndarray, wboxes: np.ndarray, wscores: np.ndarray, thr: float) -> np.ndarray:
    """Per person: max score of weapons with inter/weapon_area >= thr (0 if none)."""
    if len(pboxes) == 0 or len(wboxes) == 0:
        return np.zeros(len(pboxes))
    lt = np.maximum(pboxes[:, None, :2], wboxes[None, :, :2])
    rb = np.minimum(pboxes[:, None, 2:], wboxes[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    cover = inter / np.maximum((wboxes[:, 2:] - wboxes[:, :2]).prod(1), 1e-9)[None]
    return np.where(cover >= thr, wscores[None], 0).max(1)


def load_weapon_thresholds(path: Path, use_verifier: bool) -> tuple[str, dict, float | None]:
    """-> (mode, {class: thr}, det_conf floor or None)."""
    d = json.loads(path.read_text())
    if "selected_mode" in d:  # pipeline_thresholds.json
        if use_verifier:
            return d["selected_mode"], {c: v["score_thr"] for c, v in d["classes"].items()}, d.get("det_conf_floor")
        return "det", {c: v["det_conf_thr"] for c, v in d["detector_only"].items()}, d.get("det_conf_floor")
    if use_verifier:
        raise SystemExit(f"{path} is a detector thresholds.json; --verifier needs pipeline_thresholds.json")
    return "det", {n.lower(): float(v["conf"]) for n, v in d["classes"].items() if n.lower() in WEAPONS}, None


def person_pass(pmodel, imgs, a) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    stream = pmodel.predict([str(p) for p in imgs], imgsz=a.person_imgsz, conf=a.person_conf, iou=0.7,
                            device=a.device, classes=[a.person_class], stream=True, verbose=False, max_det=100)
    for res in stream:
        if res.boxes is None or len(res.boxes) == 0:
            out.append((np.zeros(0), np.zeros((0, 4))))
        else:
            out.append((res.boxes.conf.cpu().numpy().astype(float), res.boxes.xyxy.cpu().numpy()))
    return out


def run_split(split, a, pmodel, det, ver, wthr, mode) -> dict:
    data, imgs = load_split(a.data, split, a.max_images, a.seed)
    ids = name_ids(data["names"])
    armed = a.armed_class if a.armed_class is not None else ids.get("armed person")
    if armed is None:
        raise SystemExit(f"no 'armed person' class in {data['names']}; pass --armed-class")
    persons = person_pass(pmodel, imgs, a)
    gt_armed, gt_weap, sizes = [], [], []
    from PIL import Image
    from ultralytics.data.utils import exif_size

    for p in imgs:
        with Image.open(p) as im:
            W, H = exif_size(im)
        c, b = load_gt(p, W, H)
        gt_armed.append(b[c == armed])
        gt_weap.append(b[np.isin(c, [ids[n] for n in WEAPONS if n in ids])])
    sources = {}
    if a.weapons in ("gt", "both"):
        sources["gt"] = [(gb, np.ones(len(gb))) for gb in gt_weap]
    if a.weapons in ("pred", "both"):
        from eval_pipeline import MODES, collect

        recs = collect(det, ver, a.data, split, a)  # same image list (same seed / max_images)
        f = MODES[mode]
        lst = []
        for r in recs:
            bs, ss = [], []
            for c in WEAPONS:
                d, pc, b = r["pred"][c]
                s = f(d, pc)
                k = s >= wthr[c]
                bs.append(b[k])
                ss.append(s[k])
            lst.append((np.concatenate(bs), np.concatenate(ss)))
        sources["pred"] = lst
    out = {"split": split, "images": len(imgs), "gt_armed": int(sum(len(g) for g in gt_armed))}
    for src, weap in sources.items():
        preds = []
        for (pc, pb), (wb, ws) in zip(persons, weap):
            ls = link_scores(pb, wb, ws, a.link)
            k = ls > 0
            preds.append(((pc * ls)[k], pb[k]))
        s, tp, n = match(preds, gt_armed, a.iou_match)
        out[src] = {"scores": s, "tp": tp, "n_gt": n, "ap50": round(ap50(tp, n), 4),
                    "n_pred": int(len(s)), "best_f1": best_f1(s, tp, n)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(WORK / "data" / "clean" / "data.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--tune-split", default="val", help="split for the armed-person threshold; 'none' = oracle")
    ap.add_argument("--armed-class", type=int, default=None, help="default: class named 'armed person'")
    ap.add_argument("--person-weights", default=str(RAW / "yolo11m.pt"))
    ap.add_argument("--person-class", type=int, default=0)
    ap.add_argument("--person-conf", type=float, default=0.01)
    ap.add_argument("--person-imgsz", type=int, default=640)
    ap.add_argument("--weapons", choices=["gt", "pred", "both"], default=None)
    ap.add_argument("--weights", default=None, help="weapon detector (gun/knife classes)")
    ap.add_argument("--verifier", default=None)
    ap.add_argument("--thresholds", default=None, help="pipeline_thresholds.json or thresholds.json")
    ap.add_argument("--det-conf", type=float, default=None, help="detector floor (default: from thresholds or 0.01)")
    ap.add_argument("--imgsz", type=int, default=None, help="detector imgsz (default: checkpoint)")
    ap.add_argument("--cls-imgsz", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.2)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--link", type=float, default=0.3, help="min intersection / weapon area")
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.weapons = a.weapons or ("both" if a.weights else "gt")
    if a.weapons != "gt" and not a.weights:
        ap.error("--weapons pred/both needs --weights")

    from ultralytics import YOLO

    sys.path.insert(0, str(HERE.parent / "train"))
    from eval import run_dir_from_weights

    det = ver = None
    mode, wthr = "det", {}
    if a.weapons != "gt":
        det = YOLO(a.weights)
        thf = Path(a.thresholds) if a.thresholds else run_dir_from_weights(a.weights) / "thresholds.json"
        if not thf.exists():
            raise SystemExit(f"thresholds file {thf} not found; pass --thresholds")
        mode, wthr, floor = load_weapon_thresholds(thf, bool(a.verifier))
        a.det_conf = a.det_conf if a.det_conf is not None else (floor or 0.01)
        if a.verifier:
            ver = Verifier(a.verifier, a.device, a.cls_imgsz, a.tta)
        print(f"[armed] weapon thresholds ({thf.name}, mode={mode}): {wthr}, det floor {a.det_conf}")
    out = Path(a.out) if a.out else (run_dir_from_weights(a.weights) / "armed_person" if a.weights
                                     else WORK / "runs" / "armed_person_gt")
    out.mkdir(parents=True, exist_ok=True)
    pmodel = YOLO(a.person_weights)
    print(f"[armed] person model {a.person_weights} class {a.person_class} = {pmodel.names[a.person_class]!r}")

    ev = run_split(a.split, a, pmodel, det, ver, wthr, mode)
    tu = run_split(a.tune_split, a, pmodel, det, ver, wthr, mode) if a.tune_split.lower() != "none" else None
    res = {"data": str(Path(a.data).resolve()), "split": a.split, "tune_split": a.tune_split,
           "images": ev["images"], "gt_armed": ev["gt_armed"], "link": a.link, "weapon_mode": mode,
           "weapon_thresholds": wthr, "weights": a.weights, "verifier": a.verifier, "results": {}}
    md = [f"# Armed person = COCO person ∩ confirmed weapon (cover >= {a.link}) — {a.split}", "",
          f"images {ev['images']}, GT armed persons {ev['gt_armed']}; threshold tuned on "
          f"{a.tune_split if tu else a.split + ' (ORACLE)'}; weapons mode {mode}", "",
          "| weapons | AP50 | thr | P | R | F1 | TP | FP | FN | #pred |", "|---|---|---|---|---|---|---|---|---|---|"]
    for src in [s for s in ("gt", "pred") if s in ev]:
        e = ev[src]
        thr = tu[src]["best_f1"]["thr"] if tu else e["best_f1"]["thr"]
        r = {"ap50": e["ap50"], "thr": thr, **prf_at(e["scores"], e["tp"], e["n_gt"], thr), "n_pred": e["n_pred"],
             "oracle_best_f1": e["best_f1"]}
        res["results"][src] = r
        md.append(f"| {src} | {r['ap50']:.3f} | {thr:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                  f"{r['tp']} | {r['fp']} | {r['fn']} | {r['n_pred']} |")
    md += ["", "'gt' = GT weapon boxes (upper bound for the person+weapon rule); 'pred' = detector"
           + (" + verifier" if a.verifier else "") + " weapons at their thresholds."]
    (out / f"armed_{a.split}.json").write_text(json.dumps(res, indent=2))
    (out / f"armed_{a.split}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[armed] saved {out / f'armed_{a.split}.json'}")


if __name__ == "__main__":
    main()

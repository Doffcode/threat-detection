#!/usr/bin/env python
"""Knife error audit.

Runs the model on a split at a very low conf (0.001), then applies per-class operating thresholds
(from <run_dir>/thresholds.json, else 0.25; --conf overrides the knife threshold) and matches
predictions to GT knife boxes one-to-one at IoU >= 0.5.

Works with any class set that has a 'knife' class (old: armed person/gun/knife; v2: gun/knife/
handheld_object). Knife false negatives (unmatched GT knives) get a cause, checked in this order:
  misclassified-as-<cls>   a prediction of another class (>= its thr) with IoU >= 0.5 to the GT knife,
                           'gun' first, then the other classes in id order
                           (v2: as-gun, as-handheld_object; old: as-gun, as-armed person)
  localisation             a knife prediction (>= knife thr) with 0.1 <= IoU < 0.5
  missed                   none of the above (field below_thr_conf tells whether a sub-threshold
                           knife detection with IoU >= 0.5 existed, i.e. a threshold problem)
Knife false positives (unmatched knife preds >= thr) are tagged with what GT they overlap best:
  <other class name> (IoU >= 0.5), knife-loc (GT knife 0.1 <= IoU < 0.5), duplicate
  (IoU >= 0.5 with an already-matched GT knife), background.

Outputs in <run_dir>/audit[_<split>]/: fn_sheet.jpg, fp_sheet.jpg, summary.json, summary.md, errors.json
Crops: green = GT knife, red = knife pred, orange = other-class pred that stole it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

SIZE_BUCKETS = [(0.0, 0.01, "<1%"), (0.01, 0.05, "1-5%"), (0.05, 0.15, "5-15%"), (0.15, 1.01, ">15%")]


def size_bucket(area_frac: float) -> str:
    for lo, hi, n in SIZE_BUCKETS:
        if lo <= area_frac < hi:
            return n
    return ">15%"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def list_images(data: dict, split: str) -> list[Path]:
    from ultralytics.data.utils import IMG_FORMATS

    key = "val" if split in ("val", "valid") else split
    src = data.get(key)
    if not src:
        raise SystemExit(f"split {split!r} not in data yaml")
    srcs = src if isinstance(src, list) else [src]
    files = []
    for s in srcs:
        p = Path(s)
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix[1:].lower() in IMG_FORMATS]
        elif p.suffix == ".txt":
            base = p.parent
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    f = Path(ln)
                    files.append(f if f.is_absolute() else (base / f).resolve())
        else:
            files.append(p)
    return files


def load_gt(img: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    from ultralytics.data.utils import img2label_paths

    lp = Path(img2label_paths([str(img)])[0])
    cls, boxes = [], []
    if lp.exists():
        for ln in lp.read_text().splitlines():
            v = ln.split()
            if len(v) < 5:
                continue
            c = int(float(v[0]))
            if len(v) == 5:
                x, y, bw, bh = map(float, v[1:5])
            else:  # polygon -> bbox
                pts = np.array(list(map(float, v[1:]))).reshape(-1, 2)
                (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
                x, y, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            cls.append(c)
            boxes.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return np.array(cls, int), np.array(boxes, float).reshape(-1, 4)


def load_thresholds(run_dir: Path, names: dict, default=0.25) -> dict[int, float]:
    thr = {i: default for i in names}
    f = run_dir / "thresholds.json"
    if f.exists():
        d = json.loads(f.read_text()).get("classes", {})
        for n, v in d.items():
            thr[int(v["class_id"])] = float(v["conf"])
        print(f"[audit] thresholds from {f}: {thr}")
    else:
        print(f"[audit] no thresholds.json; using {default} for all classes")
    return thr


def make_sheet(items: list[dict], out: Path, title: str, tile=200, cols=8, max_n=48):
    items = items[:max_n]
    if not items:
        img = np.full((80, 600, 3), 255, np.uint8)
        cv2.putText(img, f"{title}: none", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(str(out), img)
        return
    rows = (len(items) + cols - 1) // cols
    lab_h = 34
    sheet = np.full((40 + rows * (tile + lab_h), cols * tile, 3), 255, np.uint8)
    cv2.putText(sheet, f"{title} (showing {len(items)})", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cache = {}
    for k, it in enumerate(items):
        p = it["image"]
        if p not in cache:
            cache = {p: cv2.imread(p)}  # keep only one image in memory
        im = cache[p]
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = [b for b in (it.get("gt_box"), it.get("pred_box"), it.get("other_box")) if b is not None]
        allb = np.array(boxes)
        x0, y0 = allb[:, :2].min(0)
        x1, y1 = allb[:, 2:].max(0)
        pad = max(0.25 * max(x1 - x0, y1 - y0), 24)
        cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        cx1, cy1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
        crop = im[cy0:cy1, cx0:cx1].copy()
        s = tile / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        lw = 2
        for key, col in (("other_box", (0, 140, 255)), ("pred_box", (0, 0, 255)), ("gt_box", (0, 200, 0))):
            b = it.get(key)
            if b is None:
                continue
            q = [int((b[0] - cx0) * s), int((b[1] - cy0) * s), int((b[2] - cx0) * s), int((b[3] - cy0) * s)]
            cv2.rectangle(crop, (q[0], q[1]), (q[2], q[3]), col, lw)
        r, c = divmod(k, cols)
        oy, ox = 40 + r * (tile + lab_h), c * tile
        sheet[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        for li, txt in enumerate(it["label"].split("\n")[:2]):
            cv2.putText(sheet, txt[:30], (ox + 3, oy + tile + 13 + li * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=None, help="default: imgsz stored in checkpoint")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=None, help="knife operating threshold (default thresholds.json/0.25)")
    ap.add_argument("--default-conf", type=float, default=0.25, help="threshold for classes not in thresholds.json")
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--knife-class", type=int, default=None, help="default: class named 'knife'")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--max-crops", type=int, default=48)
    ap.add_argument("--max-det", type=int, default=1000, help="per-image cap at conf=0.001 (300 can crowd out knives)")
    ap.add_argument("--low-conf", type=float, default=0.001)
    a = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    from eval import run_dir_from_weights

    run_dir = Path(a.run_dir) if a.run_dir else run_dir_from_weights(a.weights)
    out = run_dir / ("audit" if a.split == "test" else f"audit_{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    names = model.names
    K = a.knife_class if a.knife_class is not None else next(i for i, n in names.items() if n.lower() == "knife")
    # other classes that can "steal" a knife; gun first (most common confusion), then id order
    others = sorted((i for i in names if i != K), key=lambda i: (names[i].lower() != "gun", i))
    mis_causes = [(i, f"misclassified-as-{names[i]}") for i in others]
    imgsz = a.imgsz or (model.overrides or {}).get("imgsz") or 640
    thr = load_thresholds(run_dir, names, a.default_conf)
    if a.conf is not None:
        thr[K] = a.conf
    data = check_det_dataset(a.data)
    imgs = list_images(data, a.split)
    print(f"[audit] {len(imgs)} images, split={a.split}, imgsz={imgsz}, knife thr={thr[K]:.3f}")

    fns, fps, tps = [], [], 0
    n_gt = 0
    # one image per call: a list source is predicted as a single batch and OOMs a 12 GB GPU
    preds = (r for p in imgs for r in model.predict(str(p), imgsz=imgsz, conf=a.low_conf, iou=a.nms_iou,
                                                     device=a.device, verbose=False, max_det=a.max_det))
    for res in preds:
        H, W = res.orig_shape
        img = res.path
        gcls, gbox = load_gt(Path(img), W, H)
        pb = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        pc = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros(0, int)
        ps = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
        keep = ps >= np.array([thr.get(int(c), a.default_conf) for c in pc])
        kb_all, ks_all = pb[pc == K], ps[pc == K]  # all knife preds incl. sub-threshold
        b, c, s = pb[keep], pc[keep], ps[keep]
        kb, ks = b[c == K], s[c == K]
        order = np.argsort(-ks)
        kb, ks = kb[order], ks[order]
        gk = gbox[gcls == K]
        n_gt += len(gk)
        iou_kk = iou_matrix(gk, kb)
        matched_gt, matched_pred = set(), set()
        for j in range(len(kb)):  # greedy by confidence
            if len(gk) == 0:
                break
            cand = [(iou_kk[i, j], i) for i in range(len(gk)) if i not in matched_gt and iou_kk[i, j] >= a.iou_match]
            if cand:
                _, i = max(cand)
                matched_gt.add(i)
                matched_pred.add(j)
        tps += len(matched_gt)
        # ---- false negatives
        for i in range(len(gk)):
            if i in matched_gt:
                continue
            g = gk[i]
            area = float((g[2] - g[0]) * (g[3] - g[1]) / (W * H))
            rec = {"image": img, "gt_box": g.tolist(), "area_frac": round(area, 5), "size": size_bucket(area),
                   "pred_box": None, "other_box": None, "conf": None}
            gi = iou_matrix(g[None], b)[0] if len(b) else np.zeros(0)
            cause = None
            for cls_id, cname in mis_causes:
                m = (c == cls_id) & (gi >= a.iou_match)
                if m.any():
                    j = int(np.argmax(np.where(m, s, -1)))
                    cause, rec["other_box"], rec["conf"] = cname, b[j].tolist(), float(s[j])
                    rec["iou"] = float(gi[j])
                    break
            if cause is None and len(kb):
                ii = iou_kk[i]
                m = (ii >= 0.1) & (ii < a.iou_match)
                if m.any():
                    j = int(np.argmax(np.where(m, ii, -1)))
                    cause, rec["pred_box"], rec["conf"], rec["iou"] = "localisation", kb[j].tolist(), float(ks[j]), float(ii[j])
            if cause is None:
                cause = "missed"
                ia = iou_matrix(g[None], kb_all)[0] if len(kb_all) else np.zeros(0)
                m = ia >= a.iou_match
                rec["below_thr_conf"] = round(float(ks_all[m].max()), 4) if m.any() else None
            rec["cause"] = cause
            extra = f" {rec['conf']:.3f}" if rec["conf"] is not None else (
                f" sub{rec['below_thr_conf']:.3f}" if rec.get("below_thr_conf") else "")
            rec["label"] = f"{cause.replace('misclassified-as-', 'as-')}{extra}\n{area * 100:.2f}% {Path(img).stem[:18]}"
            fns.append(rec)
        # ---- false positives
        for j in range(len(kb)):
            if j in matched_pred:
                continue
            p = kb[j]
            gi = iou_matrix(p[None], gbox)[0] if len(gbox) else np.zeros(0)
            area = float((p[2] - p[0]) * (p[3] - p[1]) / (W * H))
            rec = {"image": img, "pred_box": p.tolist(), "conf": float(ks[j]), "area_frac": round(area, 5),
                   "size": size_bucket(area), "gt_box": None}
            ov = "background"
            if len(gbox):
                kn = gcls == K
                gi_k = np.where(kn, gi, 0)
                gi_o = np.where(~kn, gi, 0)
                if gi_k.max() >= a.iou_match:
                    ov = "duplicate"
                    rec["gt_box"] = gbox[int(gi_k.argmax())].tolist()
                elif gi_o.max() >= a.iou_match:
                    jj = int(gi_o.argmax())
                    ov = names[int(gcls[jj])]
                    rec["other_box"] = gbox[jj].tolist()
                elif gi_k.max() >= 0.1:
                    ov = "knife-loc"
                    rec["gt_box"] = gbox[int(gi_k.argmax())].tolist()
                rec["max_iou_gt"] = round(float(gi.max()), 3)
                rec["max_iou_gt_class"] = names[int(gcls[int(gi.argmax())])]
            rec["overlap"] = ov
            rec["label"] = f"{ov} {rec['conf']:.3f}\n{area * 100:.2f}% {Path(img).stem[:18]}"
            fps.append(rec)

    fns.sort(key=lambda r: (r["cause"], r["area_frac"]))
    fps.sort(key=lambda r: -r["conf"])
    cause_order = [c for _, c in mis_causes] + ["missed", "localisation"]
    size_order = [b[2] for b in SIZE_BUCKETS]
    fn_cause = Counter(r["cause"] for r in fns)
    fn_size = Counter(r["size"] for r in fns)
    cross = {sz: {cs: sum(1 for r in fns if r["size"] == sz and r["cause"] == cs) for cs in cause_order}
             for sz in size_order}
    missed_sub = sum(1 for r in fns if r["cause"] == "missed" and r.get("below_thr_conf"))
    summary = {
        "weights": str(Path(a.weights).resolve()), "split": a.split, "imgsz": imgsz, "images": len(imgs),
        "knife_threshold": thr[K], "thresholds": {names[k]: v for k, v in thr.items()},
        "gt_knives": n_gt, "tp": tps, "fn": len(fns), "fp": len(fps),
        "recall": round(tps / n_gt, 4) if n_gt else None,
        "precision": round(tps / (tps + len(fps)), 4) if tps + len(fps) else None,
        "fn_by_cause": {k: fn_cause.get(k, 0) for k in cause_order},
        "fn_missed_with_subthreshold_detection": missed_sub,
        "fn_by_size": {k: fn_size.get(k, 0) for k in size_order},
        "fn_size_x_cause": cross,
        "fp_by_overlap": dict(Counter(r["overlap"] for r in fps).most_common()),
        "fp_by_size": {k: sum(1 for r in fps if r["size"] == k) for k in size_order},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    strip = lambda r: {k: v for k, v in r.items() if k != "label"}
    (out / "errors.json").write_text(json.dumps({"fn": [strip(r) for r in fns], "fp": [strip(r) for r in fps]},
                                                indent=1))
    md = [f"# Knife audit — {run_dir.name} / {a.split}", "",
          f"knife thr={thr[K]:.3f}, imgsz={imgsz}, images={len(imgs)}, GT knives={n_gt}, "
          f"TP={tps}, FN={len(fns)}, FP={len(fps)}, recall={summary['recall']}, precision={summary['precision']}",
          "", "## FN by size x cause", "",
          "| size | " + " | ".join(cause_order) + " | total |", "|---|" + "---|" * (len(cause_order) + 1)]
    for sz in size_order:
        md.append(f"| {sz} | " + " | ".join(str(cross[sz][cs]) for cs in cause_order) + f" | {fn_size.get(sz, 0)} |")
    md.append("| total | " + " | ".join(str(fn_cause.get(cs, 0)) for cs in cause_order) + f" | {len(fns)} |")
    md += ["", f"'missed' with a sub-threshold knife detection (IoU>=0.5): {missed_sub}", "",
           "## FP by overlapped GT", "", "| overlap | count |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary["fp_by_overlap"].items()]
    md += ["", "## FP by size", "", "| size | count |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary["fp_by_size"].items()]
    md += ["", "Sheets: fn_sheet.jpg (round-robin over causes, small boxes first), fp_sheet.jpg (sorted by conf desc). "
           "Green=GT knife, red=knife pred, orange=other-class box."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    # sheet: round-robin over causes so every cause is represented within max_crops
    by_cause = [[r for r in fns if r["cause"] == cs] for cs in cause_order]
    fn_rr = [g[i] for i in range(max(map(len, by_cause), default=0)) for g in by_cause if i < len(g)]
    make_sheet(fn_rr, out / "fn_sheet.jpg", f"Knife FN @thr {thr[K]:.3f}", max_n=a.max_crops)
    make_sheet(fps, out / "fp_sheet.jpg", f"Knife FP @thr {thr[K]:.3f}", max_n=a.max_crops)
    print("\n".join(md))
    print(f"[audit] saved to {out}")


if __name__ == "__main__":
    main()

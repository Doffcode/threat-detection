"""Shared helpers for the two-stage pipeline (detector -> crop verifier) and the armed-person eval.

- crops: square_crop() is THE crop definition; make_crops.py (training data) and eval_pipeline.py
  (inference) both use it, so the verifier sees identical crops at train and test time.
- metrics: greedy score-ordered matching at IoU>=0.5, AP50 (ultralytics' 101-point compute_ap),
  best-F1 threshold and P/R/F1 at a fixed threshold.
- dataset helpers are reused from scripts/train/audit.py (list_images, load_gt, iou_matrix).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
WORK = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / "train"))

from audit import iou_matrix, list_images, load_gt  # noqa: E402,F401

WEAPONS = ("gun", "knife")
NOT_WEAPON = "not_weapon"
HANDHELD = "handheld_object"


def name_ids(names) -> dict[str, int]:
    """{lower-case class name: id} from a names dict/list."""
    names = names if isinstance(names, dict) else dict(enumerate(names))
    return {str(n).lower(): int(i) for i, n in names.items()}


def load_split(data_yaml: str, split: str, max_images: int | None = None, seed: int = 0):
    """-> (data dict from check_det_dataset, unique image paths). max_images: random subset (smoke tests)."""
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    seen, imgs = set(), []
    for p in list_images(data, split):  # oversample txt lists repeat images -> dedupe
        k = str(p)
        if k not in seen:
            seen.add(k)
            imgs.append(p)
    if max_images and len(imgs) > max_images:
        rng = np.random.default_rng(seed)
        imgs = [imgs[i] for i in sorted(rng.choice(len(imgs), max_images, replace=False))]
    return data, imgs


def short_id(path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:6]


# ----------------------------------------------------------------------------------------------- crops
def crop_region(box, W: int, H: int, pad: float = 0.2) -> tuple[int, int, int, int]:
    """Pad the box by `pad` of its size on every side, grow the short side to the long side (square),
    shift it back inside the image where possible, clip. Returns int xyxy."""
    x0, y0, x1, y1 = map(float, box)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    side = max(w, h) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sw, sh = min(side, W), min(side, H)
    nx0 = min(max(cx - sw / 2, 0), W - sw)
    ny0 = min(max(cy - sh / 2, 0), H - sh)
    return int(round(nx0)), int(round(ny0)), int(round(nx0 + sw)), int(round(ny0 + sh))


def square_crop(img: np.ndarray, box, pad: float = 0.2, size: int = 224, fill: int = 114) -> np.ndarray:
    """BGR crop around box (see crop_region), letterboxed to size x size (only non-square near borders)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = crop_region(box, W, H, pad)
    c = img[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    s = size / max(c.shape[:2])
    nw, nh = max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))
    c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if nw == size and nh == size:
        return c
    out = np.full((size, size, 3), fill, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = c
    return out


# -------------------------------------------------------------------------------------------- verifier
class Verifier:
    """YOLO11-cls crop classifier. probs(crops) -> (N, C) softmax; .ids maps lower-case name -> column."""

    def __init__(self, weights, device="0", imgsz: int = 224, tta: bool = False, batch: int = 32):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.ids = name_ids(self.model.names)
        missing = [n for n in WEAPONS if n not in self.ids]
        if missing:
            raise SystemExit(f"verifier {weights} lacks classes {missing}; has {list(self.ids)}")
        self.device, self.imgsz, self.tta, self.batch = device, imgsz, tta, batch

    def _run(self, crops):
        out = []
        for i in range(0, len(crops), self.batch):
            res = self.model.predict(crops[i:i + self.batch], imgsz=self.imgsz, device=self.device, verbose=False)
            out += [r.probs.data.float().cpu().numpy() for r in res]
        return np.stack(out)

    def probs(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, len(self.ids)), np.float32)
        p = self._run(crops)
        if self.tta:  # + horizontal flip, averaged
            p = (p + self._run([np.ascontiguousarray(c[:, ::-1]) for c in crops])) / 2
        return p


# --------------------------------------------------------------------------------------------- metrics
def match(preds: list[tuple[np.ndarray, np.ndarray]], gts: list[np.ndarray], iou_thr: float = 0.5):
    """Greedy, score-ordered one-to-one matching for ONE class over a set of images.

    preds[i] = (scores (n,), boxes (n,4)) for image i; gts[i] = GT boxes (m,4) of that class.
    Returns (scores sorted desc, tp flags aligned with them, number of GT boxes)."""
    allv, alli, allj = [], [], []
    for i, (s, _) in enumerate(preds):
        allv.append(np.asarray(s, float))
        alli.append(np.full(len(s), i))
        allj.append(np.arange(len(s)))
    n_gt = int(sum(len(g) for g in gts))
    if not allv or sum(len(v) for v in allv) == 0:
        return np.zeros(0), np.zeros(0, bool), n_gt
    v, ii, jj = np.concatenate(allv), np.concatenate(alli), np.concatenate(allj)
    order = np.argsort(-v, kind="stable")
    ious = {i: iou_matrix(preds[i][1], gts[i]) for i in range(len(preds)) if len(gts[i]) and len(preds[i][0])}
    used = {i: np.zeros(len(gts[i]), bool) for i in ious}
    tp = np.zeros(len(order), bool)
    for k, o in enumerate(order):
        i, j = ii[o], jj[o]
        if i not in ious:
            continue
        row = np.where(used[i], -1.0, ious[i][j])
        g = int(row.argmax())
        if row[g] >= iou_thr:
            used[i][g] = True
            tp[k] = True
    return v[order], tp, n_gt


def ap50(tp: np.ndarray, n_gt: int) -> float:
    from ultralytics.utils.metrics import compute_ap

    if n_gt == 0 or len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    rec = ctp / n_gt
    prec = ctp / np.arange(1, len(tp) + 1)
    return float(compute_ap(rec, prec)[0])


def best_f1(scores: np.ndarray, tp: np.ndarray, n_gt: int, floor: float = 0.0) -> dict:
    """Threshold (keep score >= thr) maximising F1; only cuts at the end of tie groups are considered."""
    if len(scores) == 0 or n_gt == 0:
        return {"thr": 1.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": int(len(scores)), "fn": n_gt}
    ctp = np.cumsum(tp)
    k = np.arange(1, len(tp) + 1)
    f1 = 2 * ctp / (k + n_gt)
    valid = np.r_[scores[1:] != scores[:-1], True] & (scores >= floor)
    if not valid.any():
        valid[-1] = True
    j = int(np.argmax(np.where(valid, f1, -1)))
    return {"thr": float(scores[j]), **prf_at(scores, tp, n_gt, float(scores[j]))}


def prf_at(scores: np.ndarray, tp: np.ndarray, n_gt: int, thr: float) -> dict:
    keep = scores >= thr
    t = int(tp[keep].sum())
    fp = int(keep.sum()) - t
    p = t / (t + fp) if t + fp else 0.0
    r = t / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"f1": round(f1, 4), "precision": round(p, 4), "recall": round(r, 4), "tp": t, "fp": fp, "fn": n_gt - t}

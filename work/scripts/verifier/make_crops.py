#!/usr/bin/env python
"""Build the stage-2 verifier dataset (ImageFolder) from a detection dataset yaml.

  <out>/{train,val}/{gun,knife,not_weapon}/*.jpg   + manifest.csv + summary.json

Positive crops: every GT gun / knife box (class ids resolved by name from the yaml).
not_weapon crops, three sources:
  (a) handheld   GT handheld_object boxes (if the class exists)
  (b) mined_fp   detector false positives: --weights run on the split at --mine-conf (0.1); gun/knife
                 predictions whose IoU with every GT gun/knife box is < --mine-iou (0.1)
  (c) background random squares not overlapping any GT gun/knife box, at most --bg-frac of not_weapon
Label-noise guard: a not_weapon crop is dropped if its actual (padded, squared) crop region covers
>= --nw-max-cover (0.5) of any GT gun/knife box (background: >= 0.1), e.g. a big FP box around a
poster that contains a small gun has IoU < 0.1 with it but would show the gun.
Every crop: box padded by --pad (20%) per side, short side grown to the long side (square-ish, clipped
at image borders, letterboxed), resized to --size (224). Same crop function as eval_pipeline.py.

Balancing (per split):
  - train only: the minority weapon class gets jittered extra crops (box shift +-10%, scale 0.85-1.15)
    up to min(majority count, --max-dup x its own count). Disable with --no-balance-weapons.
  - not_weapon is capped at --nw-cap (1.5) x the larger weapon class; within the cap background takes
    at most --bg-frac, the rest is split evenly between mined FPs and handheld (unused share flows over).

Tip: mine train-split FPs with a detector that did NOT train on that split if you have one (e.g. a
cross-fold model); a detector mined on its own train split finds few FPs. The val split is fine.

Example:
  python make_crops.py --data data/v2/data_all.yaml --weights runs/v2_stage2/weights/best.pt \\
      --out data/verifier_crops
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (HANDHELD, NOT_WEAPON, WEAPONS, WORK, crop_region, iou_matrix, load_gt, load_split,
                    name_ids, short_id, square_crop)


def image_size(p: Path) -> tuple[int, int]:
    from PIL import Image
    from ultralytics.data.utils import exif_size

    with Image.open(p) as im:
        return exif_size(im)  # (w, h) after EXIF rotation, like ultralytics / cv2.imread


def crop_covers_weapon(box, W, H, wboxes: np.ndarray, pad: float, thr: float) -> bool:
    """Does the padded/squared crop region of `box` cover >= thr of any weapon box's area?"""
    if len(wboxes) == 0:
        return False
    r = np.array(crop_region(box, W, H, pad), float)
    lt = np.maximum(r[:2], wboxes[:, :2])
    rb = np.minimum(r[2:], wboxes[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    return bool((inter / np.maximum((wboxes[:, 2:] - wboxes[:, :2]).prod(1), 1e-9) >= thr).any())


def jitter(box, rng, shift=0.1, scale=(0.85, 1.15)):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2 + rng.uniform(-shift, shift) * w, (y0 + y1) / 2 + rng.uniform(-shift, shift) * h
    s = rng.uniform(*scale)
    return [cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2]


def collect(a, split: str, rng, det=None) -> list[dict]:
    data, imgs = load_split(a.data, split, a.max_images, a.seed)
    ids = name_ids(data["names"])
    missing = [n for n in WEAPONS if n not in ids]
    if missing:
        raise SystemExit(f"{a.data}: classes {missing} not found in names {data['names']}")
    wid = {ids[n]: n for n in WEAPONS}
    hh = ids.get(HANDHELD)
    print(f"[crops] {split}: {len(imgs)} images; weapon ids {wid}; handheld id {hh}")
    specs, gt_w, sizes = [], {}, {}
    dropped = Counter()
    for p in imgs:
        W, H = image_size(p)
        sizes[str(p)] = (W, H)
        cls, boxes = load_gt(p, W, H)
        wmask = np.isin(cls, list(wid))
        gt_w[str(p)] = boxes[wmask]
        for c, b in zip(cls, boxes):
            if min(b[2] - b[0], b[3] - b[1]) < a.min_px:
                continue
            if int(c) in wid:
                specs.append({"image": str(p), "box": b.tolist(), "cls": wid[int(c)], "src": "gt"})
            elif hh is not None and int(c) == hh:
                if crop_covers_weapon(b, W, H, gt_w[str(p)], a.pad, a.nw_max_cover):
                    dropped["handheld"] += 1
                    continue
                specs.append({"image": str(p), "box": b.tolist(), "cls": NOT_WEAPON, "src": "handheld"})
        # (c) background candidates: a couple of random squares per image, 8-40% of the short side
        for _ in range(a.bg_per_image):
            for _try in range(10):
                side = rng.uniform(0.08, 0.4) * min(W, H)
                x0, y0 = rng.uniform(0, W - side), rng.uniform(0, H - side)
                b = np.array([x0, y0, x0 + side, y0 + side])
                if not crop_covers_weapon(b, W, H, gt_w[str(p)], a.pad, 0.1):
                    specs.append({"image": str(p), "box": b.tolist(), "cls": NOT_WEAPON, "src": "background"})
                    break
    # (b) mined detector false positives
    if det is not None:
        dids = name_ids(det.names)
        dw = {dids[n]: n for n in WEAPONS if n in dids}
        n_fp = 0
        # one image per call: a list source is loaded as a single batch and runs out of RAM
        dimgsz = a.imgsz or (det.overrides or {}).get("imgsz") or 640
        stream = (det.predict(str(q), imgsz=dimgsz, conf=a.mine_conf, iou=0.7, device=a.device, verbose=False,
                              classes=list(dw), max_det=100, agnostic_nms=True)[0] for q in imgs)
        for p, res in zip(imgs, stream):  # same order as the input list
            if res.boxes is None or len(res.boxes) == 0:
                continue
            pb = res.boxes.xyxy.cpu().numpy()
            pc = res.boxes.cls.cpu().numpy().astype(int)
            ps = res.boxes.conf.cpu().numpy()
            g = gt_w[str(p)]
            iou = iou_matrix(pb, g) if len(g) else np.zeros((len(pb), 0))
            for j in range(len(pb)):
                if (iou.shape[1] and iou[j].max() >= a.mine_iou) or min(pb[j, 2] - pb[j, 0], pb[j, 3] - pb[j, 1]) < a.min_px:
                    continue
                if crop_covers_weapon(pb[j], *sizes[str(p)], g, a.pad, a.nw_max_cover):
                    dropped["mined_fp"] += 1
                    continue
                specs.append({"image": str(p), "box": pb[j].tolist(), "cls": NOT_WEAPON, "src": "mined_fp",
                              "det_cls": dw[int(pc[j])], "det_conf": round(float(ps[j]), 4)})
                n_fp += 1
        print(f"[crops] {split}: mined {n_fp} detector FPs (conf>={a.mine_conf}, IoU<{a.mine_iou})")
    if dropped:
        print(f"[crops] {split}: dropped not_weapon crops whose crop region shows a GT weapon: {dict(dropped)}")
    return specs


def balance(a, specs: list[dict], split: str, rng) -> list[dict]:
    pos = {n: [s for s in specs if s["cls"] == n] for n in WEAPONS}
    if split == "train" and a.balance_weapons and all(pos.values()):
        mi, ma = sorted(WEAPONS, key=lambda n: len(pos[n]))
        target = min(len(pos[ma]), int(a.max_dup * len(pos[mi])))
        extra = []
        for k in range(target - len(pos[mi])):
            s = pos[mi][rng.integers(len(pos[mi]))]
            extra.append({**s, "box": jitter(s["box"], rng), "src": "gt_jitter"})
        pos[mi] += extra
    budget = int(round(a.nw_cap * max(len(v) for v in pos.values())))
    by = {k: [s for s in specs if s["cls"] == NOT_WEAPON and s["src"] == k] for k in ("mined_fp", "handheld", "background")}
    for v in by.values():
        rng.shuffle(v)
    n_bg = min(len(by["background"]), int(a.bg_frac * budget))
    rest = budget - n_bg
    n_fp = min(len(by["mined_fp"]), rest // 2)
    n_hh = min(len(by["handheld"]), rest - n_fp)
    n_fp = min(len(by["mined_fp"]), rest - n_hh)  # give handheld's unused share back to FPs
    # mined FPs: prefer the most confident (hardest) ones
    fps = sorted(by["mined_fp"], key=lambda s: -s.get("det_conf", 0))[:n_fp]
    nw = fps + by["handheld"][:n_hh] + by["background"][:n_bg]
    return pos["gun"] + pos["knife"] + nw


def write(a, specs: list[dict], split: str, out: Path, writer) -> Counter:
    for c in (*WEAPONS, NOT_WEAPON):
        (out / split / c).mkdir(parents=True, exist_ok=True)
    by_img = defaultdict(list)
    for s in specs:
        by_img[s["image"]].append(s)
    cnt = Counter()
    for img, ss in by_img.items():
        im = cv2.imread(img)
        if im is None:
            print(f"[warn] cannot read {img}")
            continue
        sid = short_id(img)
        for k, s in enumerate(ss):
            crop = square_crop(im, s["box"], a.pad, a.size)
            fn = out / split / s["cls"] / f"{Path(img).stem[:40]}_{sid}_{k}_{s['src']}.jpg"
            cv2.imwrite(str(fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cnt[(s["cls"], s["src"])] += 1
            writer.writerow([split, s["cls"], s["src"], fn.relative_to(out), img,
                             *[round(v, 1) for v in s["box"]], s.get("det_cls", ""), s.get("det_conf", "")])
    return cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="detection yaml with gun/knife[/handheld_object] classes")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="yaml splits; train->train/, val->val/")
    ap.add_argument("--out", default=str(WORK / "data" / "verifier_crops"))
    ap.add_argument("--overwrite", action="store_true", help="delete --out first if it exists")
    ap.add_argument("--weights", default=None, help="detector for FP mining (omit to skip source b)")
    ap.add_argument("--mine-conf", type=float, default=0.1)
    ap.add_argument("--mine-iou", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=None, help="detector imgsz (default: from checkpoint)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--pad", type=float, default=0.2)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--min-px", type=float, default=4, help="skip boxes with a side shorter than this")
    ap.add_argument("--nw-max-cover", type=float, default=0.5,
                    help="drop handheld/mined-FP crops whose crop region covers >= this of a GT weapon box")
    ap.add_argument("--nw-cap", type=float, default=1.5, help="not_weapon <= this x larger weapon class")
    ap.add_argument("--bg-frac", type=float, default=0.2, help="max share of background within not_weapon")
    ap.add_argument("--bg-per-image", type=int, default=1, help="background candidates per image")
    ap.add_argument("--no-balance-weapons", dest="balance_weapons", action="store_false")
    ap.add_argument("--max-dup", type=float, default=3.0, help="minority weapon class up-sampling cap (train)")
    ap.add_argument("--max-images", type=int, default=None, help="random image subset per split (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        if not a.overwrite:
            raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    det = None
    if a.weights:
        from ultralytics import YOLO

        det = YOLO(a.weights)
    rng = np.random.default_rng(a.seed)
    summary = {"data": str(Path(a.data).resolve()), "weights": a.weights, "args": vars(a), "splits": {}}
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "src", "file", "image", "x0", "y0", "x1", "y1", "det_cls", "det_conf"])
        for split in a.splits:
            dst = "train" if split == "train" else "val" if split in ("val", "valid") else split
            specs = balance(a, collect(a, split, rng, det), dst, rng)
            cnt = write(a, specs, split=dst, out=out, writer=w)
            per_cls = Counter()
            for (c, _), n in cnt.items():
                per_cls[c] += n
            summary["splits"][dst] = {"per_class": dict(per_cls),
                                      "per_source": {f"{c}/{s}": n for (c, s), n in sorted(cnt.items())}}
            print(f"[crops] {dst}: {dict(per_cls)}  sources: {summary['splits'][dst]['per_source']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[crops] saved {out}  (manifest.csv, summary.json)")


if __name__ == "__main__":
    main()

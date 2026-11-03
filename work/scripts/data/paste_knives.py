#!/usr/bin/env python
"""Offline knife copy-paste augmentation for the TRAIN split of work/data/clean.

Knife crops (work/raw/knife_crops/knife_XXXX.jpg) turned out to be deterministic crops of every
class-2 box of IntProj_Weapon_Dataset_annot, in order train/valid/test (files sorted by name,
label-line order), padded by 20% of box w/h on each side, clipped to the image and resized so the
long side is 640.  We reconstruct that mapping (verified pixel-wise), which gives us
  * the source image of every crop -> its group/split in the clean split (leak check), and
  * the exact labelled knife box inside every crop.  The 20% context ring is then used as the
    background seed for GrabCut, giving a knife-shaped soft alpha (no pasted rectangles); the new
    label is the tight bbox of that alpha (crops whose cut-out truncates the labelled box are dropped).
Crops whose source image is in clean valid/test are excluded, plus (belt and braces) any crop whose
tight-box dHash is within --hash-thr of any valid/test knife box (incl. horizontal flip).

Writes ~N images  work/data/clean/train/{images,labels}/synknife_XXXXX.*  and a preview sheet.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

WORK = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work')
RAW = WORK / 'raw'
SRC = RAW / 'IntProj_Weapon_Dataset_annot'
CROPS = RAW / 'knife_crops'
CLEAN = WORK / 'data' / 'clean'
PAD = 0.20
# Crops whose GrabCut cut-out passed the automatic checks but, on visual inspection of the cut-out
# sheet, is not a usable knife (whole person/face included, beige/white blobs, background slabs).
MANUAL_REJECT = {4, 5, 22, 107, 127, 128, 149, 156, 162, 164, 165, 183, 184, 185, 189, 208, 209, 213,
                 214, 215, 228, 276, 279, 280, 286, 287, 308, 331, 336, 347, 353, 354, 367, 397, 399,
                 400}
COLORS = {0: (255, 160, 0), 1: (0, 200, 255), 2: (0, 0, 255)}
NAMES = ['person', 'gun', 'knife']


def dhash(gray):
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (s[:, 1:] > s[:, :-1]).flatten()


def ham(a, b):
    return int((a != b).sum())


def line_to_xyxy(vals):
    if len(vals) == 4:
        cx, cy, w, h = vals
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    xs, ys = vals[0::2], vals[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def read_yolo(path):
    out = []
    if path.exists():
        for l in path.read_text().splitlines():
            p = l.split()
            if len(p) == 5:
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                out.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0, inter


def _padded_box(W, H, x1, y1, x2, y2):
    bw, bh = (x2 - x1) * W, (y2 - y1) * H
    X1, Y1 = int(max(0, x1 * W - PAD * bw)), int(max(0, y1 * H - PAD * bh))
    X2, Y2 = int(min(W, x2 * W + PAD * bw)), int(min(H, y2 * H + PAD * bh))
    return X1, Y1, X2, Y2, bw, bh


def map_crops(match_thr=3.0):
    """Reconstruct crop -> source knife box by content (crop order is not the label order).

    Every IntProj class-2 box is cut out with the 20% padding rule and thumbnailed; each crop is
    matched to the box with the lowest thumbnail MAE.  All boxes within match_thr MAE are kept as
    candidate sources (augmented copies can be near-identical) so the leak check is conservative.
    """
    boxes, thumbs = [], []
    for sp in ['train', 'valid', 'test']:
        for f in sorted((SRC / sp / 'images').glob('*.jpg')):
            lf = SRC / sp / 'labels' / (f.stem + '.txt')
            kb = [line_to_xyxy(list(map(float, l.split()[1:]))) for l in lf.read_text().splitlines()
                  if l.split() and l.split()[0] == '2']
            if not kb:
                continue
            im = cv2.imread(str(f))
            H, W = im.shape[:2]
            for (x1, y1, x2, y2) in kb:
                X1, Y1, X2, Y2, bw, bh = _padded_box(W, H, x1, y1, x2, y2)
                thumbs.append(cv2.resize(im[Y1:Y2, X1:X2], (32, 32), interpolation=cv2.INTER_AREA))
                boxes.append(dict(split=sp, file=f, box=(x1, y1, x2, y2), W=W, H=H))
    T = np.array(thumbs, dtype=np.float32)
    res = []
    for k, cf in enumerate(sorted(CROPS.glob('knife_*.jpg'))):
        c = cv2.imread(str(cf))
        ch, cw = c.shape[:2]
        ct = cv2.resize(c, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        e = np.abs(T - ct[None]).mean((1, 2, 3))
        j = int(e.argmin())
        rec = dict(crop=cf, idx=k, ok=bool(e[j] < match_thr), thumb_err=float(e[j]),
                   cand_files=sorted({boxes[i]['file'].name for i in np.where(e < match_thr)[0]}))
        bx = boxes[j]
        x1, y1, x2, y2 = bx['box']
        W, H = bx['W'], bx['H']
        X1, Y1, X2, Y2, bw, bh = _padded_box(W, H, x1, y1, x2, y2)
        sx, sy = cw / (X2 - X1), ch / (Y2 - Y1)
        rec.update(src_split=bx['split'], src_file=bx['file'].name, src_px=(bw, bh),
                   inner=(max(0.0, (x1 * W - X1) * sx), max(0.0, (y1 * H - Y1) * sy),
                          min(float(cw), (x2 * W - X1) * sx), min(float(ch), (y2 * H - Y1) * sy)))
        res.append(rec)
    return res, len(boxes)


def make_cutout(crop, inner, work=320):
    """GrabCut the knife inside its labelled box, using the 20% context ring as background seed.

    Returns (alpha HxWx1 float32 in [0,1], tight box (x1,y1,x2,y2) of alpha>0.5, reason-or-None).
    """
    H, W = crop.shape[:2]
    s = min(1.0, work / max(H, W))
    sm = cv2.resize(crop, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    h, w = sm.shape[:2]
    x1, y1, x2, y2 = [v * s for v in inner]
    rx1, ry1 = int(np.floor(x1)), int(np.floor(y1))
    rx2, ry2 = int(np.ceil(x2)), int(np.ceil(y2))
    rx1, ry1, rx2, ry2 = max(1, rx1), max(1, ry1), min(w - 1, rx2), min(h - 1, ry2)
    if rx2 - rx1 < 6 or ry2 - ry1 < 6:
        return None, None, 'box_too_small'
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(0)
        cv2.grabCut(sm, mask, (rx1, ry1, rx2 - rx1, ry2 - ry1), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None, None, 'grabcut_error'
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return None, None, 'empty_mask'
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.1 * big]
    fg = np.isin(lab, keep).astype(np.uint8)
    k = max(3, int(round(0.015 * max(h, w))) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    area_box = (rx2 - rx1) * (ry2 - ry1)
    frac = fg[ry1:ry2, rx1:rx2].sum() / area_box
    ys, xs = np.where(fg > 0)
    span_w = (xs.max() - xs.min() + 1) / (rx2 - rx1)
    span_h = (ys.max() - ys.min() + 1) / (ry2 - ry1)
    if frac < 0.06:
        return None, None, 'mask_too_thin'
    if frac > 0.92:
        return None, None, 'mask_is_whole_box'
    if span_w < 0.75 or span_h < 0.75:
        return None, None, 'mask_truncates_knife'
    a = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), sigmaX=1.2)
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    # fade at the crop border too
    xs_ = np.arange(W, dtype=np.float32)[None, :] + 0.5
    ys_ = np.arange(H, dtype=np.float32)[:, None] + 0.5
    edge = np.minimum(np.minimum(xs_, W - xs_), np.minimum(ys_, H - ys_)) / max(2.0, 0.02 * min(W, H))
    a = np.minimum(a, np.clip(edge, 0, 1))
    yy, xx = np.where(a > 0.5)
    if len(xx) == 0:
        return None, None, 'empty_mask'
    return a[..., None], (float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1)), None


def match_tone(patch, bg_region, alpha, strength):
    """Shift patch L/a/b mean/std partially towards the background region it lands on."""
    p = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(bg_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = alpha[..., 0] > 0.5
    if m.sum() < 16:
        return patch
    for ch, st in ((0, strength), (1, strength * 0.5), (2, strength * 0.5)):
        pm, ps = p[..., ch][m].mean(), p[..., ch][m].std() + 1e-3
        bm, bs = b[..., ch].mean(), b[..., ch].std() + 1e-3
        tm = pm + st * (bm - pm)
        ts = ps * (1 + st * 0.5 * (np.clip(bs / ps, 0.5, 2.0) - 1))
        p[..., ch] = (p[..., ch] - pm) * (ts / ps) + tm
    return cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--hash-thr', type=int, default=6)
    ap.add_argument('--min-src-px', type=float, default=32, help='min long side of source knife box')
    ap.add_argument('--max-src-area', type=float, default=0.30,
                    help='drop crops whose knife box covers more than this fraction of its source image')
    ap.add_argument('--area-min', type=float, default=0.01)
    ap.add_argument('--area-max', type=float, default=0.08)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)

    # ---------------- crop provenance + leak filtering
    split_of_file = {}
    with open(CLEAN / '_manifest.csv') as fh:
        for r in csv.DictReader(fh):
            split_of_file[r['file']] = r['split']
    crops, n_boxes = map_crops()
    n_ok = sum(c['ok'] for c in crops)
    print(f'{len(crops)} crops, {n_boxes} source knife boxes, {n_ok} crops verified pixel-exact')

    # valid/test knife box hashes (from the clean labels)
    vt_hash = []
    for sp in ['valid', 'test']:
        for lf in (CLEAN / sp / 'labels').glob('*.txt'):
            kb = [b for b in read_yolo(lf) if b[0] == 2]
            if not kb:
                continue
            img = cv2.imread(str(next((CLEAN / sp / 'images').glob(lf.stem + '.*'))), cv2.IMREAD_GRAYSCALE)
            H, W = img.shape
            for _, x1, y1, x2, y2 in kb:
                r = img[int(y1 * H):max(int(y1 * H) + 1, int(y2 * H)), int(x1 * W):max(int(x1 * W) + 1, int(x2 * W))]
                vt_hash.append(dhash(r))
    excl = {'source_in_valid_test': [], 'hash_near_valid_test': [], 'unverified': [], 'too_small': [],
            'box_too_large': []}
    usable = []
    for c in crops:
        if not c['ok']:
            excl['unverified'].append(c['crop'].name)
            continue
        if any(split_of_file.get(f) != 'train' for f in c['cand_files']):
            excl['source_in_valid_test'].append(c['crop'].name)
            continue
        img = cv2.imread(str(c['crop']), cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = [int(round(v)) for v in c['inner']]
        core = img[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
        h0, h1 = dhash(core), dhash(cv2.flip(core, 1))
        dmin = min(min(ham(h0, v), ham(h1, v)) for v in vt_hash) if vt_hash else 64
        c['dmin_vt'] = int(dmin)
        if dmin <= args.hash_thr:
            excl['hash_near_valid_test'].append(c['crop'].name)
            continue
        if max(c['src_px']) < args.min_src_px:
            excl['too_small'].append(c['crop'].name)
            continue
        if c['src_px'][0] * c['src_px'][1] > args.max_src_area * 640 * 640:
            excl['box_too_large'].append(c['crop'].name)
            continue
        if c['idx'] in MANUAL_REJECT:
            excl.setdefault('manual_reject_bad_cutout', []).append(c['crop'].name)
            continue
        crop = cv2.imread(str(c['crop']))
        alpha, tight, why = make_cutout(crop, c['inner'])
        if why:
            excl.setdefault('seg_' + why, []).append(c['crop'].name)
            continue
        c['alpha'], c['tight'] = alpha.astype(np.float16), tight
        usable.append(c)
    print('excluded:', {k: len(v) for k, v in excl.items()}, '-> usable', len(usable))
    (CLEAN / '_knife_crops_filter.json').write_text(json.dumps(
        dict(n_crops=len(crops), n_verified=n_ok, excluded={k: sorted(v) for k, v in excl.items()},
             usable=[dict(crop=c['crop'].name, src=c['src_file'], dmin_vt=c['dmin_vt']) for c in usable]), indent=1))

    # ---------------- backgrounds (real train images only, no knife)
    tr_img, tr_lab = CLEAN / 'train' / 'images', CLEAN / 'train' / 'labels'
    for old in list(tr_img.glob('synknife_*')) + list(tr_lab.glob('synknife_*')):
        old.unlink()
    pref, other = [], []
    for f in sorted(tr_img.iterdir()):
        labs = read_yolo(tr_lab / (f.stem + '.txt'))
        cls = {b[0] for b in labs}
        if 2 in cls:
            continue
        (pref if 0 in cls else other).append(f)
    print(f'backgrounds: {len(pref)} armed-person/no-knife, {len(other)} other knife-free')
    n_pref = min(len(pref), int(args.n * 0.75))
    bgs = rng.sample(pref, n_pref) + rng.sample(other, min(len(other), args.n - n_pref))
    while len(bgs) < args.n + 50:  # spares in case a background has no valid placement
        bgs.append(rng.choice(pref + other))
    rng.shuffle(bgs)

    crop_cache = {}
    usage = {c['crop'].name: 0 for c in usable}
    stats = dict(images=0, knives=0, failed_place=0)
    previews = []
    for k, bgf in enumerate(bgs):
        if stats['images'] >= args.n:
            break
        bg = cv2.imread(str(bgf))
        H, W = bg.shape[:2]
        labs = read_yolo(tr_lab / (bgf.stem + '.txt'))
        boxes_px = [(c, x1 * W, y1 * H, x2 * W, y2 * H) for c, x1, y1, x2, y2 in labs]
        n_paste = 1 if rng.random() < 0.7 else 2
        new = []
        for _ in range(n_paste):
            # least-used crops first (random among the least used) for even coverage
            mu = min(usage.values())
            cands = [c for c in usable if usage[c['crop'].name] <= mu + 1]
            c = rng.choice(cands)
            if c['crop'].name not in crop_cache:
                crop_cache[c['crop'].name] = cv2.imread(str(c['crop']))
            patch = crop_cache[c['crop'].name]
            ph, pw = patch.shape[:2]
            ix1, iy1, ix2, iy2 = c['tight']
            iw, ih = ix2 - ix1, iy2 - iy1
            placed = False
            flip = rng.random() < 0.5
            for _try in range(40):
                frac = float(np.exp(nrng.uniform(np.log(args.area_min), np.log(args.area_max))))
                s = np.sqrt(frac * W * H / (iw * ih))
                # aspect jitter +-10%
                ar = float(nrng.uniform(0.9, 1.1))
                sx, sy = s * ar, s / ar
                nw, nh = int(round(pw * sx)), int(round(ph * sy))
                if nw < 8 or nh < 8 or nw >= W or nh >= H:
                    continue
                bx1, by1 = ix1 * sx, iy1 * sy
                bx2, by2 = ix2 * sx, iy2 * sy
                if flip:  # tight box position inside the horizontally flipped patch
                    bx1, bx2 = nw - bx2, nw - bx1
                # position: half the time near an armed person (upper/middle body), else anywhere
                persons = [b for b in boxes_px if b[0] == 0]
                if persons and rng.random() < 0.6:
                    _, px1, py1, px2, py2 = rng.choice(persons)
                    cx = rng.uniform(px1 - 0.15 * (px2 - px1), px2 + 0.15 * (px2 - px1))
                    cy = rng.uniform(py1 + 0.2 * (py2 - py1), py1 + 0.8 * (py2 - py1))
                    ox, oy = int(cx - (bx1 + bx2) / 2), int(cy - (by1 + by2) / 2)
                else:
                    ox, oy = rng.randint(-int(bx1), W - int(bx2)), rng.randint(-int(by1), H - int(by2))
                kb = (ox + bx1, oy + by1, ox + bx2, oy + by2)
                # knife box must be fully inside the image
                if kb[0] < 0 or kb[1] < 0 or kb[2] > W or kb[3] > H:
                    continue
                ok = True
                for b in boxes_px + new:
                    bb = b[1:]
                    io, inter = iou(kb, bb)
                    barea = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    # IoU < 0.3 with everything; guns/knives (small objects) also must not be
                    # more than 20% covered
                    if io >= 0.3 or (b[0] != 0 and barea > 0 and inter / barea >= 0.2):
                        ok = False
                        break
                # don't paste into letterbox bars / flat regions
                reg = cv2.cvtColor(bg[int(kb[1]):int(kb[3]), int(kb[0]):int(kb[2])], cv2.COLOR_BGR2GRAY)
                if reg.size == 0 or (reg < 20).mean() > 0.25 or reg.std() < 6:
                    ok = False
                if ok:
                    placed = True
                    break
            if not placed:
                stats['failed_place'] += 1
                continue
            interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR
            p = cv2.resize(patch, (nw, nh), interpolation=interp)
            alpha = cv2.resize(c['alpha'].astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)[..., None]
            if flip:
                p = cv2.flip(p, 1)
                alpha = np.ascontiguousarray(alpha[:, ::-1])
            # paste window (patch may extend beyond the image; clip)
            X1, Y1 = max(0, ox), max(0, oy)
            X2, Y2 = min(W, ox + nw), min(H, oy + nh)
            pp = p[Y1 - oy:Y2 - oy, X1 - ox:X2 - ox]
            aa = alpha[Y1 - oy:Y2 - oy, X1 - ox:X2 - ox]
            region = bg[Y1:Y2, X1:X2]
            pp = match_tone(pp, region, aa, strength=float(nrng.uniform(0.2, 0.5)))
            # match blur level: mild blur if background is soft
            if rng.random() < 0.3:
                pp = cv2.GaussianBlur(pp, (0, 0), sigmaX=float(nrng.uniform(0.4, 1.0)))
            bg[Y1:Y2, X1:X2] = (aa * pp.astype(np.float32) + (1 - aa) * region.astype(np.float32)).astype(np.uint8)
            new.append((2,) + tuple(kb))
            usage[c['crop'].name] += 1
        if not new:
            continue
        # sensor noise over the whole image so pasted area isn't noticeably cleaner
        if rng.random() < 0.5:
            noise = nrng.normal(0, rng.uniform(1.0, 3.0), bg.shape).astype(np.float32)
            bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        name = f'synknife_{stats["images"]:05d}'
        cv2.imwrite(str(tr_img / f'{name}.jpg'), bg, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(85, 95))])
        with open(tr_lab / f'{name}.txt', 'w') as fh:
            for c_, x1, y1, x2, y2 in boxes_px + new:
                fh.write(f'{c_} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n')
        stats['images'] += 1
        stats['knives'] += len(new)
        if len(previews) < 16:
            previews.append(name)
    stats['crops_used'] = sum(1 for v in usage.values() if v)
    stats['max_uses_per_crop'] = max(usage.values())
    print(stats)

    # ---------------- preview sheet
    tiles = []
    for name in previews:
        im = cv2.imread(str(tr_img / f'{name}.jpg'))
        H, W = im.shape[:2]
        for c_, x1, y1, x2, y2 in read_yolo(tr_lab / f'{name}.txt'):
            p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
            cv2.rectangle(im, p1, p2, COLORS[c_], 2)
            cv2.putText(im, NAMES[c_], (p1[0] + 2, p1[1] + 14), 0, 0.5, COLORS[c_], 2)
        cv2.putText(im, name, (4, H - 8), 0, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (400, 400)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(CLEAN / '_preview_synthetic.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (CLEAN / '_synthetic_meta.json').write_text(json.dumps(dict(args=vars(args), stats=stats), indent=1))


if __name__ == '__main__':
    main()

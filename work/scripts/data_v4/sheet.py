#!/usr/bin/env python
"""Contact-sheet helpers.

draw_sheet(items, out, cols=6, cell=260): items = [(img_path, [(name, cx, cy, w, h), ...], caption)]
CLI: sheet.py rfclass <dataset_dir> <class_id> <out.jpg> [n]   -> one image per .rf. stem containing class
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

COL = {'gun': (0, 0, 255), 'knife': (0, 200, 0), 'handheld_object': (255, 128, 0)}


def draw_sheet(items, out, cols=6, cell=260):
    rows = (len(items) + cols - 1) // cols or 1
    sheet = np.full((rows * cell, cols * cell, 3), 40, np.uint8)
    for k, (p, boxes, cap) in enumerate(items):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = cell / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        hh, ww = im.shape[:2]
        for (n, cx, cy, bw, bh) in boxes:
            c = COL.get(n, (0, 255, 255))
            x1, y1 = int((cx - bw / 2) * ww), int((cy - bh / 2) * hh)
            x2, y2 = int((cx + bw / 2) * ww), int((cy + bh / 2) * hh)
            cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
            cv2.putText(im, str(n)[:12], (x1 + 2, max(10, y1 + 11)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
        r, cc = divmod(k, cols)
        sheet[r * cell:r * cell + hh, cc * cell:cc * cell + ww] = im
        cv2.putText(sheet, str(cap)[:34], (cc * cell + 2, r * cell + cell - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def read_yolo(p):
    out = []
    if not Path(p).exists():
        return out
    for line in Path(p).read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c = int(float(v[0]))
        f = [float(x) for x in v[1:]]
        if len(f) == 4:
            out.append((c, *f))
        else:
            xs, ys = f[0::2], f[1::2]
            out.append((c, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)))
    return out


def rf_class(d, cid, out, n=24, seed=0):
    import yaml
    d = Path(d)
    names = yaml.safe_load(open(d / 'data.yaml'))['names']
    seen = {}
    for s in ('train', 'valid', 'test'):
        for lp in sorted((d / s / 'labels').glob('*.txt')) if (d / s / 'labels').exists() else []:
            stem = lp.stem.split('.rf.')[0]
            if stem in seen:
                continue
            b = read_yolo(lp)
            if any(x[0] == cid for x in b):
                ip = next((d / s / 'images').glob(lp.stem + '.*'), None)
                if ip:
                    seen[stem] = (ip, [(names[x[0]] if x[0] == cid else '.' + names[x[0]], *x[1:]) for x in b], stem)
    items = list(seen.values())
    random.Random(seed).shuffle(items)
    draw_sheet(items[:n], out)
    print(f'{names[cid]}: {len(items)} unique images')


if __name__ == '__main__':
    if sys.argv[1] == 'rfclass':
        rf_class(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 24)

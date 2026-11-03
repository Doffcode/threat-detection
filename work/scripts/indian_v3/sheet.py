"""Contact sheet helper: sheet.py OUT.jpg TILE COLS img1 [img2 ...]  (labels looked up as ../labels/<stem>.txt;
draws boxes, class colour 0 red, 1 blue, others green; caption = filename prefix). Use with python -I."""
import sys
from pathlib import Path
import cv2
import numpy as np

COL = {0: (0, 0, 255), 1: (255, 0, 0)}


def boxes_for(img_path, lab_path=None):
    p = Path(img_path)
    lab = Path(lab_path) if lab_path else p.parent.parent / 'labels' / (p.stem + '.txt')
    out = []
    if lab.exists():
        for line in lab.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            c = int(float(v[0])); f = [float(x) for x in v[1:]]
            if len(f) == 4:
                cx, cy, w, h = f
            else:
                xs, ys = f[0::2], f[1::2]
                cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
            out.append((c, cx, cy, w, h))
    return out


def tile(img_path, size, caption=None, boxes=None, names=None):
    im = cv2.imread(str(img_path))
    if im is None:
        im = np.zeros((size, size, 3), np.uint8)
    h, w = im.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 40, np.uint8)
    canvas[:im.shape[0], :im.shape[1]] = im
    if boxes is None:
        boxes = boxes_for(img_path)
    for c, cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * im.shape[1]), int((cy - bh / 2) * im.shape[0])
        x2, y2 = int((cx + bw / 2) * im.shape[1]), int((cy + bh / 2) * im.shape[0])
        col = COL.get(c, (0, 200, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        if names:
            cv2.putText(canvas, names[c] if c < len(names) else str(c), (x1 + 2, max(10, y1 + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cap = caption if caption is not None else Path(img_path).name.split('.rf.')[0][:28]
    cv2.rectangle(canvas, (0, size - 16), (size, size), (0, 0, 0), -1)
    cv2.putText(canvas, cap, (3, size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return canvas


def sheet(out, tiles, cols):
    size = tiles[0].shape[0]
    rows = (len(tiles) + cols - 1) // cols
    S = np.full((rows * size, cols * size, 3), 0, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        S[r * size:(r + 1) * size, c * size:(c + 1) * size] = t
    cv2.imwrite(str(out), S, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    out, size, cols = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sheet(out, [tile(p, size) for p in sys.argv[4:]], cols)

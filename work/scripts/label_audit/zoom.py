"""Zoom view of one or more test images (by index from sheets/index.csv or path) with GT boxes and a 5% grid.
Usage: python -I zoom.py IDX [IDX ...] [--crop x1,y1,x2,y2]  -> data/test_audit/zoom/zoom_<idxs>.jpg"""
import sys, os, csv, cv2, numpy as np
ROOT = '/home/shiva/Desktop/projects/akriti/weapon-detection/work'
A = f'{ROOT}/data/test_audit'
args = sys.argv[1:]; crop = (0, 0, 1, 1)
if '--crop' in args:
    i = args.index('--crop'); crop = tuple(map(float, args[i+1].split(','))); args = args[:i] + args[i+2:]
idx = {r['idx']: r['image'] for r in csv.DictReader(open(f'{A}/sheets/index.csv'))}
COL = [(0, 0, 255), (0, 200, 0), (255, 128, 0)]; NAMES = ['gun', 'knife', 'handheld']
S = 900 if len(args) == 1 else 640
tiles = []
for a in args:
    p = idx.get(a, a); im = cv2.imread(p); h, w = im.shape[:2]
    x1, y1, x2, y2 = crop
    sub = im[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
    cw, ch = x2-x1, y2-y1; sc = S / max(cw, ch)
    W, H = int(cw*sc), int(ch*sc); sub = cv2.resize(sub, (W, H), interpolation=cv2.INTER_CUBIC)
    tx = lambda v: int((v-x1)*sc); ty = lambda v: int((v-y1)*sc)
    ov = sub.copy()
    for k in range(0, 101, 5):
        v = k/100; th = 1
        if x1 < v < x2:
            cv2.line(ov, (tx(v), 0), (tx(v), H), (255, 255, 255), th); cv2.putText(sub, f'{k}', (tx(v)+1, 10), 0, 0.35, (0, 255, 255), 1)
        if y1 < v < y2:
            cv2.line(ov, (0, ty(v)), (W, ty(v)), (255, 255, 255), th); cv2.putText(sub, f'{k}', (1, ty(v)-1), 0, 0.35, (0, 255, 255), 1)
    sub = cv2.addWeighted(ov, 0.2, sub, 0.8, 0)
    lp = os.path.splitext(p.replace('/images/', '/labels/'))[0] + '.txt'
    for j, r in enumerate([l.split() for l in open(lp)] if os.path.exists(lp) else []):
        c = int(r[0]); x, y, bw, bh = map(float, r[1:5])
        cv2.rectangle(sub, (tx(x-bw/2), ty(y-bh/2)), (tx(x+bw/2), ty(y+bh/2)), COL[c], 1)
        cv2.putText(sub, f'{j}:{NAMES[c]}', (tx(x-bw/2)+2, ty(y-bh/2)+12), 0, 0.4, COL[c], 1)
    cv2.putText(sub, f'#{a}', (W-60, H-8), 0, 0.6, (255, 0, 255), 2)
    tiles.append(sub)
Hm = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.zeros((Hm-t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
os.makedirs(f'{A}/zoom', exist_ok=True)
if len(tiles) % 2 and len(tiles) > 1: tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
o = f'{A}/zoom/zoom_{"_".join(args)}.jpg'; cv2.imwrite(o, grid, [cv2.IMWRITE_JPEG_QUALITY, 90]); print(o)

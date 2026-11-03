"""Render domain-test images with GT boxes + 10% grid into contact sheets.
Usage: python -I render_sheets.py [list.txt] [out_dir] [--labels-root-replace]
Images are sorted by base photo name (part before '.rf.') so Roboflow copies sit together.
Also writes index.csv (sheet, slot, image) to out_dir."""
import sys, os, csv, cv2, numpy as np
ROOT = '/home/shiva/Desktop/projects/akriti/weapon-detection/work'
lst = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/data/v2/test_domain.txt'
out = sys.argv[2] if len(sys.argv) > 2 else f'{ROOT}/data/test_audit/sheets'
NAMES = ['gun', 'knife', 'handheld']
COL = [(0, 0, 255), (0, 200, 0), (255, 128, 0)]
T, PER, NC = 620, 6, 3
os.makedirs(out, exist_ok=True)
paths = [l.strip() for l in open(lst) if l.strip()]
paths.sort(key=lambda p: os.path.basename(p))

def lab_path(p):
    return os.path.splitext(p.replace('/images/', '/labels/'))[0] + '.txt'

def tile(p, idx):
    im = cv2.imread(p); h, w = im.shape[:2]
    im = cv2.resize(im, (T, T)); ov = im.copy()
    for k in range(1, 10):
        c = int(T * k / 10)
        cv2.line(ov, (c, 0), (c, T), (255, 255, 255), 1); cv2.line(ov, (0, c), (T, c), (255, 255, 255), 1)
    im = cv2.addWeighted(ov, 0.25, im, 0.75, 0)
    for k in range(1, 10):
        cv2.putText(im, str(k), (int(T*k/10)+2, 12), 0, 0.4, (255, 255, 0), 1)
        cv2.putText(im, str(k), (2, int(T*k/10)-2), 0, 0.4, (255, 255, 0), 1)
    lp = lab_path(p)
    rows = [l.split() for l in open(lp)] if os.path.exists(lp) else []
    for j, r in enumerate(rows):
        if len(r) < 5: continue
        c = int(r[0]); x, y, bw, bh = map(float, r[1:5])
        x1, y1, x2, y2 = int((x-bw/2)*T), int((y-bh/2)*T), int((x+bw/2)*T), int((y+bh/2)*T)
        cv2.rectangle(im, (x1, y1), (x2, y2), COL[c], 2)
        cv2.putText(im, f'{j}:{NAMES[c]}', (x1+2, max(y1+14, 14)), 0, 0.5, COL[c], 2)
    bar = np.zeros((40, T, 3), np.uint8)
    nm = os.path.basename(p)
    cv2.putText(bar, f'#{idx} {nm[:40]}', (4, 16), 0, 0.45, (255, 255, 255), 1)
    cv2.putText(bar, f'{nm[40:80]} n={len(rows)}', (4, 34), 0, 0.45, (200, 200, 200), 1)
    return np.vstack([bar, im])

with open(f'{out}/index.csv', 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['sheet', 'idx', 'image'])
    for s in range(0, len(paths), PER):
        tiles = [tile(p, s+i) for i, p in enumerate(paths[s:s+PER])]
        for i, p in enumerate(paths[s:s+PER]): wr.writerow([s//PER, s+i, p])
        while len(tiles) % NC: tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[r:r+NC]) for r in range(0, len(tiles), NC)]
        cv2.imwrite(f'{out}/sheet_{s//PER:02d}.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(paths), 'images ->', out)

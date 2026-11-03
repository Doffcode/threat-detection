"""Flag gold/yellow blobs (sword hilts) lying outside labelled boxes. gold_scan.py LIST.txt -> prints frame, blobs.
Run with python -I. Heuristic aid for the manual anti_robbing review only."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet import boxes_for  # noqa: E402

for p in Path(sys.argv[1]).read_text().split():
    im = cv2.imread(p)
    H, W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 32) & (hsv[..., 1] >= 110) & (hsv[..., 2] >= 120)).astype(np.uint8)
    for c, cx, cy, bw, bh in boxes_for(p):
        x1, y1 = int((cx - bw / 2 - 0.03) * W), int((cy - bh / 2 - 0.03) * H)
        x2, y2 = int((cx + bw / 2 + 0.03) * W), int((cy + bh / 2 + 0.03) * H)
        m[max(0, y1):y2, max(0, x1):x2] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    bl = [(int(cen[i][0]) / W, int(cen[i][1]) / H, int(st[i][4])) for i in range(1, n) if st[i][4] >= 25]
    print(Path(p).name.split('_jpg')[0], ' '.join(f'({x:.2f},{y:.2f},{a})' for x, y, a in bl))

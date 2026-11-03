"""Crop-region contact sheet: crop_sheet.py OUT.jpg x0 y0 x1 y1 TILE_W COLS img... (python -I).
Draws the label boxes (red gun / blue knife) clipped to the crop."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet import boxes_for, COL  # noqa: E402

out = sys.argv[1]
x0, y0, x1, y1 = map(float, sys.argv[2:6])
tw, cols = int(sys.argv[6]), int(sys.argv[7])
paths = sys.argv[8:]
tiles = []
for p in paths:
    im = cv2.imread(p)
    H, W = im.shape[:2]
    for c, cx, cy, bw, bh in boxes_for(p):
        cv2.rectangle(im, (int((cx - bw / 2) * W), int((cy - bh / 2) * H)), (int((cx + bw / 2) * W), int((cy + bh / 2) * H)), COL.get(c, (0, 200, 0)), 2)
    cr = im[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
    th = int(tw * cr.shape[0] / cr.shape[1])
    cr = cv2.resize(cr, (tw, th), interpolation=cv2.INTER_AREA)
    cap = Path(p).name.split('.rf.')[0]
    cv2.rectangle(cr, (0, 0), (90, 16), (0, 0, 0), -1)
    cv2.putText(cr, cap, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    tiles.append(cr)
th = tiles[0].shape[0]
rows = (len(tiles) + cols - 1) // cols
S = np.zeros((rows * (th + 2), cols * (tw + 2), 3), np.uint8)
for k, t in enumerate(tiles):
    r, c = divmod(k, cols)
    S[r * (th + 2):r * (th + 2) + th, c * (tw + 2):c * (tw + 2) + tw] = t
cv2.imwrite(out, S, [cv2.IMWRITE_JPEG_QUALITY, 88])

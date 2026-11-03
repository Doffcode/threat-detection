"""Shared constants/helpers for the v2 (gun / knife / handheld_object) dataset build."""
import sys
from pathlib import Path

WORK = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work')
sys.path.insert(0, str(WORK / 'scripts' / 'data'))
import split as v1  # noqa: E402  (reuse dHash / grouping helpers from the v1 split)

CLEAN = WORK / 'data' / 'clean'
OUT = WORK / 'data' / 'v2'
STAGE = OUT / '_staging'
DOWNLOADS = WORK / 'downloads'
SPLITS = ['train', 'valid', 'test']
NAMES = ['gun', 'knife', 'handheld_object']
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def read_yolo(path):
    """-> list of (cls, cx, cy, w, h); polygons converted to bbox."""
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        v = [float(x) for x in p[1:]]
        if len(v) == 4:
            out.append((c, *v))
        elif len(v) >= 6 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            out.append((c, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)))
    return out


def write_yolo(path, boxes):
    with open(path, 'w') as fh:
        for c, cx, cy, w, h in boxes:
            fh.write(f'{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')


def xyxy(b):
    _, cx, cy, w, h = b
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

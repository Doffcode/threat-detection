"""Build a review sheet + stats for a downloaded Roboflow YOLO dataset (untrusted data: only read).
Usage: python -I review.py <dataset_dir> [<dataset_dir> ...]
Writes work/downloads/roboflow/_review/<name>.jpg and <name>.json."""
import collections, json, random, sys
from pathlib import Path
import cv2
import numpy as np

WORK = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work')
sys.path.insert(0, str(WORK / 'scripts' / 'data'))
import split as v1  # dhash16 / trim_borders / hamming_matrix / norm_stem

REVIEW = WORK / 'downloads' / 'roboflow' / '_review'
REF_CACHE = REVIEW / '_dom_hashes.npz'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DUP_THR = 24  # of 256 dHash16 bits (flip-aware)


def hash_img(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = v1.trim_borders(g)
    return v1.dhash16(g), v1.dhash16(g[:, ::-1])


def ref_hashes():
    if REF_CACHE.exists():
        z = np.load(REF_CACHE)
        return z['H'], list(z['names'])
    files = sorted((WORK / 'data' / 'v2' / 'train' / 'images').glob('dom_*'))
    H, names = [], []
    for f in files:
        h = hash_img(f)
        if h is not None:
            H.append(h[0]); names.append(f.name)
    H = np.array(H)
    REVIEW.mkdir(parents=True, exist_ok=True)
    np.savez(REF_CACHE, H=H, names=np.array(names))
    return H, names


def read_names(ds):
    for y in ds.glob('data.yaml'):
        txt = y.read_text(errors='ignore')
        # minimal parse of 'names:' (list or inline) without yaml.load on untrusted file
        import yaml
        d = yaml.safe_load(txt)
        n = d.get('names')
        if isinstance(n, dict):
            n = [n[k] for k in sorted(n)]
        return list(n)
    return []


def boxes(lbl):
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text(errors='ignore').splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0])); v = list(map(float, p[1:]))
        if len(v) == 4:
            cx, cy, w, h = v
        else:  # polygon
            xs, ys = v[0::2], v[1::2]
            cx, cy, w, h = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, max(xs)-min(xs), max(ys)-min(ys)
        out.append((c, cx, cy, w, h))
    return out


def tile(img_path, lbl, names, W=360, H=270):
    im = cv2.imread(str(img_path))
    canvas = np.full((H + 22, W, 3), 255, np.uint8)
    if im is None:
        return canvas
    ih, iw = im.shape[:2]
    s = min(W / iw, H / ih)
    im = cv2.resize(im, (max(1, int(iw * s)), max(1, int(ih * s))))
    rh, rw = im.shape[:2]
    for c, cx, cy, w, h in boxes(lbl):
        x1, y1 = int((cx - w/2) * rw), int((cy - h/2) * rh)
        x2, y2 = int((cx + w/2) * rw), int((cy + h/2) * rh)
        col = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)][c % 6]
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
        nm = names[c] if c < len(names) else str(c)
        cv2.putText(im, nm[:16], (x1 + 2, max(12, y1 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(im, nm[:16], (x1 + 2, max(12, y1 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    canvas[:rh, :rw] = im
    cap = img_path.name.split('.rf.')[0][:52]
    cv2.putText(canvas, cap, (2, H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    return canvas


def review(ds, grep=None, tag=None):
    ds = Path(ds)
    name = ds.name + (f'__{tag}' if tag else '')
    names = read_names(ds)
    imgs = sorted(p for p in ds.rglob('*') if p.suffix.lower() in IMG_EXT and p.parent.name == 'images')
    if grep:
        import re
        imgs = [p for p in imgs if re.search(grep, p.name, re.I)]
    split_counts = collections.Counter(p.parent.parent.name for p in imgs)
    cls_inst = collections.Counter(); cls_img = collections.Counter(); empty = 0
    lbl_of = {}
    for p in imgs:
        lbl = p.parent.parent / 'labels' / (p.stem + '.txt')
        lbl_of[p] = lbl
        b = boxes(lbl)
        if not b:
            empty += 1
        for c in {x[0] for x in b}:
            cls_img[names[c] if c < len(names) else str(c)] += 1
        for x in b:
            cls_inst[names[x[0]] if x[0] < len(names) else str(x[0])] += 1
    groups = collections.defaultdict(list)
    for p in imgs:
        groups[v1.norm_stem(p.name)].append(p)
    n_rf = sum(1 for p in imgs if '.rf.' in p.name)
    # one representative per source stem
    reps = [g[0] for g in groups.values()]
    rng = random.Random(0)
    sample = rng.sample(reps, min(24, len(reps)))
    tiles = [tile(p, lbl_of[p], names) for p in sample]
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sheet = np.vstack(rows)
    REVIEW.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(REVIEW / f'{name}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # duplicates vs our dom_* train images
    RH, rnames = ref_hashes()
    Hs, HFs = [], []
    for p in reps:
        h = hash_img(p)
        if h is None:
            h = (np.zeros(256, bool), np.zeros(256, bool))
        Hs.append(h[0]); HFs.append(h[1])
    Hs, HFs = np.array(Hs), np.array(HFs)
    d = np.minimum(v1.hamming_matrix(Hs, RH), v1.hamming_matrix(HFs, RH)).min(1) if len(Hs) else np.array([])
    ndup = int((d <= DUP_THR).sum())
    stats = dict(dataset=name, files=len(imgs), splits=dict(split_counts), unique_sources=len(groups),
                 rf_files=n_rf, aug_copies=len(imgs) - len(groups), empty_label_files=empty,
                 class_instances=dict(cls_inst.most_common()), class_images=dict(cls_img.most_common()),
                 names=names, dup_vs_dom=ndup, dup_frac=round(ndup / max(1, len(reps)), 3),
                 sample=[p.name for p in sample])
    json.dump(stats, open(REVIEW / f'{name}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != 'sample'}))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--grep', help='only images whose filename matches this regex')
    ap.add_argument('--tag', help='suffix for the sheet name when --grep is used')
    a = ap.parse_args()
    for d in a.dirs:
        review(d, a.grep, a.tag)

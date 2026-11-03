#!/usr/bin/env python
"""Leak-free group-level split of the pooled weapon datasets.

Pools work/raw/IntProj_Weapon_Dataset_annot (0 armed person, 1 gun, 2 knife) and
work/raw/roboflow_v1 (0 gun, 1 knife -> remapped to 1 gun, 2 knife), groups images by
normalised source stem + perceptual-hash near-duplicates, then does a seeded,
knife-stratified group split into work/data/clean/{train,valid,test}.

roboflow_v1 is (almost entirely) the same source photos as IntProj but annotated without the
'armed person' class.  By default roboflow_v1 images whose group also contains an IntProj
image are used for grouping only and NOT written out (they are redundant and would teach the
model that armed persons are background).  Use --keep-rv1-dups to write them anyway.

Labels: Roboflow exported a mix of bbox lines and polygon lines; everything is converted to
YOLO bbox (cls cx cy w h), clipped to [0,1].  Anomalies are logged to _label_issues.txt.
"""
import argparse
import collections
import json
import os
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

WORK = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work')
RAW = WORK / 'raw'
OUT = WORK / 'data' / 'clean'
SPLITS = ['train', 'valid', 'test']
NAMES = ['armed person', 'gun', 'knife']
DATASETS = {
    'IntProj_Weapon_Dataset_annot': {0: 0, 1: 1, 2: 2},
    'roboflow_v1': {0: 1, 1: 2},
}
EXT_RE = re.compile(r'_(jpe?g|png|webp|avif|jfif|gif|bmp|tiff?)$', re.I)
COPY_RE = re.compile(r'-copy(-\d+-)?$', re.I)


def norm_stem(name):
    s = name.split('.rf.')[0]
    if s == name:  # no roboflow hash
        s = os.path.splitext(name)[0]
    while True:  # strip stacked extension tokens: foo_jpeg_jpg -> foo
        s2 = EXT_RE.sub('', s)
        if s2 == s:
            break
        s = s2
    s = COPY_RE.sub('', s)  # "x - Copy", "x - Copy (2)" -> x
    s = EXT_RE.sub('', s)
    return s.lower().strip('-_ ')


def dhash_bits(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def trim_borders(gray, thr=16):
    """Crop near-black letterbox/pillarbox borders (screenshots of videos)."""
    m = gray > thr
    rows = np.where(m.mean(1) > 0.02)[0]
    cols = np.where(m.mean(0) > 0.02)[0]
    if len(rows) < 8 or len(cols) < 8:
        return gray
    return gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def dhash16(gray):
    small = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


SHOT_RE1 = re.compile(r'screenshot-(\d{4})-(\d\d)-(\d\d)-at-(\d+)-(\d+)-(\d+)-(am|pm)')
SHOT_RE2 = re.compile(r'screenshot-(\d{4})-(\d\d)-(\d\d)-(\d\d)(\d\d)(\d\d)$')


def shot_time(stem):
    """Seconds since epoch for 'Screenshot-YYYY-MM-DD-at-H-M-S-PM' / 'Screenshot-YYYY-MM-DD-HHMMSS' stems."""
    import datetime
    m = SHOT_RE1.match(stem)
    if m:
        y, mo, d, h, mi, se, ap = m.groups()
        h = int(h) % 12 + (12 if ap == 'pm' else 0)
    else:
        m = SHOT_RE2.match(stem)
        if not m:
            return None
        y, mo, d, h, mi, se = m.groups()
        h = int(h)
    return datetime.datetime(int(y), int(mo), int(d), int(h), int(mi), int(se)).timestamp()


def pack(bits):
    return int(''.join('1' if b else '0' for b in bits), 2)


_POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def hamming_matrix(A, B=None, chunk=256):
    """A,B: (n,bits) bool arrays -> (n,m) int16 Hamming distances, chunked to bound memory."""
    Ap = np.packbits(A.astype(bool), axis=1)
    Bp = Ap if B is None else np.packbits(B.astype(bool), axis=1)
    out = np.empty((len(Ap), len(Bp)), dtype=np.int16)
    for s in range(0, len(Ap), chunk):
        x = Ap[s:s + chunk, None, :] ^ Bp[None, :, :]
        out[s:s + chunk] = _POP[x].sum(-1, dtype=np.int16)
    return out


def min_dist(H, HF, rows=None):
    """Distance allowing a horizontal flip; rows restricts to a subset."""
    if rows is not None:
        H, HF = H[rows], HF[rows]
    return np.minimum(hamming_matrix(H), hamming_matrix(H, HF))


def popcount(x):
    return bin(x).count('1')


def parse_label(path, cmap, issues, tag):
    boxes = []
    if not path.exists():
        issues.append(f'{tag}: missing label file')
        return boxes
    seen = set()
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        p = line.split()
        if not p:
            continue
        try:
            c = int(float(p[0]))
            v = [float(x) for x in p[1:]]
        except ValueError:
            issues.append(f'{tag}:{ln}: unparsable line {line!r}')
            continue
        if c not in cmap:
            issues.append(f'{tag}:{ln}: unknown class {c}')
            continue
        c = cmap[c]
        if len(v) == 4:
            cx, cy, w, h = v
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif len(v) >= 6 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            issues.append(f'{tag}:{ln}: bad number of values ({len(v)})')
            continue
        if min(x1, y1) < -0.01 or max(x2, y2) > 1.01:
            issues.append(f'{tag}:{ln}: box out of range ({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}) -> clipped')
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 - x1 < 2e-3 or y2 - y1 < 2e-3:
            issues.append(f'{tag}:{ln}: degenerate box dropped')
            continue
        key = (c, round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))
        if key in seen:
            issues.append(f'{tag}:{ln}: duplicate line dropped')
            continue
        seen.add(key)
        boxes.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)
            return True
        return False


def assign(groups, weight_key, targets, rng, split_of):
    """Greedy: largest groups first (random tiebreak), each to split with largest relative deficit."""
    gl = list(groups)
    rng.shuffle(gl)
    gl.sort(key=lambda g: -g[weight_key])
    cur = {s: 0.0 for s in SPLITS}
    total = sum(g[weight_key] for g in gl) or 1
    for g in gl:
        best = max(SPLITS, key=lambda s: (targets[s] * total - cur[s]) / (targets[s] * total))
        split_of[g['gid']] = best
        cur[best] += g[weight_key]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--hash-thr', type=int, default=4)
    ap.add_argument('--hash16-thr', type=int, default=32, help='256-bit trimmed dHash threshold')
    ap.add_argument('--shot-gap', type=float, default=30, help='seconds; consecutive screenshots merged')
    ap.add_argument('--min-test-knife', type=int, default=55)
    ap.add_argument('--keep-rv1-dups', action='store_true')
    ap.add_argument('--copy', action='store_true', help='copy images instead of hardlinking')
    args = ap.parse_args()
    rng = random.Random(args.seed)

    issues = []
    recs = []
    for ds, cmap in DATASETS.items():
        for sp in SPLITS:
            idir = RAW / ds / sp / 'images'
            for f in sorted(idir.iterdir()):
                if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    continue
                lab = RAW / ds / sp / 'labels' / (f.stem + '.txt')
                tag = f'{ds}/{sp}/{f.name}'
                boxes = parse_label(lab, cmap, issues, tag)
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    issues.append(f'{tag}: unreadable image, skipped')
                    continue
                h = dhash_bits(img)
                hf = dhash_bits(cv2.flip(img, 1))
                tr = trim_borders(img)
                stem = norm_stem(f.name)
                recs.append(dict(ds=ds, src_split=sp, path=f, name=f.name, stem=stem,
                                 boxes=boxes, h=pack(h), hf=pack(hf), shape=img.shape[:2],
                                 h16=dhash16(tr), h16f=dhash16(cv2.flip(tr, 1)), t=shot_time(stem)))
    n = len(recs)
    print(f'pooled {n} images')

    # --- group by stem
    uf = UF(n)
    by_stem = collections.defaultdict(list)
    for i, r in enumerate(recs):
        by_stem[r['stem']].append(i)
    for idx in by_stem.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    n_stem_groups = len(by_stem)

    # --- perceptual near-duplicate merge (incl. horizontal flip)
    H = np.array([[(r['h'] >> k) & 1 for k in range(64)] for r in recs], dtype=np.uint8)
    HF = np.array([[(r['hf'] >> k) & 1 for k in range(64)] for r in recs], dtype=np.uint8)
    dmin = min_dist(H, HF)
    # guard against near-constant images whose dHash is meaningless
    low_info = np.array([popcount(r['h']) in range(0, 4) or popcount(r['h']) > 60 for r in recs])
    merges = []
    ii, jj = np.where(np.triu(dmin <= args.hash_thr, 1))
    for i, j in zip(ii, jj):
        if low_info[i] or low_info[j]:
            continue
        if uf.find(i) != uf.find(j):
            merges.append((recs[i]['stem'], recs[j]['stem'], int(dmin[i, j])))
            uf.union(i, j)
    del dmin

    # --- finer near-duplicate check: 256-bit dHash on border-trimmed image (catches letterboxed
    # consecutive video frames that the 64-bit hash misses)
    H16 = np.array([r['h16'] for r in recs], dtype=np.uint8)
    H16F = np.array([r['h16f'] for r in recs], dtype=np.uint8)
    d16 = min_dist(H16, H16F)
    merges16 = []
    ii, jj = np.where(np.triu(d16 <= args.hash16_thr, 1))
    for i, j in zip(ii, jj):
        if low_info[i] or low_info[j]:
            continue
        if uf.find(i) != uf.find(j):
            merges16.append((recs[i]['stem'], recs[j]['stem'], int(d16[i, j])))
            uf.union(i, j)
    del d16

    # --- screenshot sessions: 'Screenshot-...-at-H-M-S' files taken <= --shot-gap seconds apart are
    # consecutive frames of the same video scene -> same group
    shots = sorted({(r['t'], r['stem']) for r in recs if r['t'] is not None})
    first_idx = {}
    for i, r in enumerate(recs):
        first_idx.setdefault(r['stem'], i)
    shot_merges = 0
    for (t0, s0), (t1, s1) in zip(shots, shots[1:]):
        if t1 - t0 <= args.shot_gap:
            if uf.union(first_idx[s0], first_idx[s1]):
                shot_merges += 1
    print(f'{len(merges16)} merges from 256-bit trimmed dHash (<= {args.hash16_thr}), '
          f'{shot_merges} screenshot-session merges (gap <= {args.shot_gap}s) over {len(shots)} screenshot stems')

    print(f'{len(merges)} merges from 64-bit dHash (<= {args.hash_thr})')

    gmap = collections.defaultdict(list)
    for i in range(n):
        gmap[uf.find(i)].append(i)
    print(f'stem groups {n_stem_groups} -> {len(gmap)} groups after hash + screenshot-session merging')

    # --- choose which records to write
    keep = [True] * n
    dropped_rv1 = 0
    for root, idx in gmap.items():
        has_ip = any(recs[i]['ds'] != 'roboflow_v1' for i in idx)
        if has_ip and not args.keep_rv1_dups:
            for i in idx:
                if recs[i]['ds'] == 'roboflow_v1':
                    keep[i] = False
                    dropped_rv1 += 1

    # label-consistency check within groups (knife vs gun confusion)
    conflicts = []
    for root, idx in gmap.items():
        sig = {}
        for i in idx:
            cnt = collections.Counter(b[0] for b in recs[i]['boxes'])
            sig[f"{recs[i]['ds'][:4]}/{recs[i]['src_split']}/{recs[i]['name'][:60]}"] = (cnt[1], cnt[2])
        vals = set(sig.values())
        if len(vals) > 1:
            knife_any = any(v[1] for v in vals)
            gun_any = any(v[0] for v in vals)
            if knife_any and any(v[1] == 0 for v in vals):
                conflicts.append((recs[idx[0]]['stem'], sig))
            elif gun_any and knife_any:
                conflicts.append((recs[idx[0]]['stem'], sig))

    groups = []
    for gid, (root, idx) in enumerate(sorted(gmap.items())):
        kidx = [i for i in idx if keep[i]]
        if not kidx:
            continue
        cls = collections.Counter(b[0] for i in kidx for b in recs[i]['boxes'])
        groups.append(dict(gid=gid, idx=kidx, all_idx=idx, knife=cls[2], gun=cls[1], person=cls[0],
                           nimg=len(kidx), stems=sorted({recs[i]['stem'] for i in idx})))

    # --- stratified split
    split_of = {}
    knife_groups = [g for g in groups if g['knife'] > 0]
    total_knife = sum(g['knife'] for g in knife_groups)
    t_test = max(0.10, min(0.2, args.min_test_knife / max(total_knife, 1)))
    t_valid = 0.15
    kt = {'train': 1 - t_valid - t_test, 'valid': t_valid, 'test': t_test}
    assign(knife_groups, 'knife', kt, rng, split_of)
    rest = [g for g in groups if g['knife'] == 0]
    tt = {'train': 0.75, 'valid': 0.15, 'test': 0.10}
    assign([g for g in rest if g['gun'] > 0], 'nimg', tt, rng, split_of)
    assign([g for g in rest if g['gun'] == 0], 'nimg', tt, rng, split_of)

    # --- verify no overlap (by group, by normalised stem, by hash)
    stems_by_split = {s: set() for s in SPLITS}
    for g in groups:
        stems_by_split[split_of[g['gid']]].update(g['stems'])
    overlap = {}
    for a in range(3):
        for b in range(a + 1, 3):
            sa, sb = SPLITS[a], SPLITS[b]
            overlap[f'{sa}&{sb} stems'] = len(stems_by_split[sa] & stems_by_split[sb])
    rec_split = {}
    for g in groups:
        for i in g['idx']:
            rec_split[i] = split_of[g['gid']]
    kidx = sorted(rec_split)
    sp_arr = np.array([rec_split[i] for i in kidx])
    dd = min_dist(H, HF, kidx)
    cross = (dd <= args.hash_thr) & (sp_arr[:, None] != sp_arr[None, :])
    li = low_info[kidx]
    cross &= ~li[:, None] & ~li[None, :]
    overlap['cross-split pHash pairs (<=thr)'] = int(np.triu(cross, 1).sum())
    dd16 = min_dist(H16, H16F, kidx)
    cross16 = (dd16 <= args.hash16_thr) & (sp_arr[:, None] != sp_arr[None, :]) & ~li[:, None] & ~li[None, :]
    overlap['cross-split 256-bit dHash pairs (<=thr16)'] = int(np.triu(cross16, 1).sum())
    del dd16
    for k, v in overlap.items():
        print(f'  overlap {k}: {v}')
    assert all(v == 0 for v in overlap.values()), 'LEAK DETECTED'

    # --- write
    if OUT.exists():
        for s in SPLITS:
            shutil.rmtree(OUT / s, ignore_errors=True)
    for s in SPLITS:
        (OUT / s / 'images').mkdir(parents=True, exist_ok=True)
        (OUT / s / 'labels').mkdir(parents=True, exist_ok=True)
    manifest = []
    used_names = set()
    for g in groups:
        s = split_of[g['gid']]
        for i in g['idx']:
            r = recs[i]
            prefix = 'rv1_' if r['ds'] == 'roboflow_v1' else ''
            name = prefix + r['name']
            assert name not in used_names, name
            used_names.add(name)
            dst = OUT / s / 'images' / name
            if args.copy:
                shutil.copy2(r['path'], dst)
            else:
                os.link(r['path'], dst)
            with open(OUT / s / 'labels' / (Path(name).stem + '.txt'), 'w') as fh:
                for c, cx, cy, w, h in r['boxes']:
                    fh.write(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
            manifest.append(dict(split=s, group=g['gid'], file=name, ds=r['ds'], src_split=r['src_split'],
                                 stem=r['stem'], n_knife=sum(b[0] == 2 for b in r['boxes'])))

    meta = dict(seed=args.seed, hash_thr=args.hash_thr, n_pooled=n, n_stem_groups=n_stem_groups,
                n_groups=len(groups), n_phash_merges=len(merges), n_hash16_merges=len(merges16), n_shot_session_merges=shot_merges,
                dropped_rv1_dups=dropped_rv1,
                knife_targets=kt, overlap=overlap,
                groups_per_split={s: sum(1 for g in groups if split_of[g['gid']] == s) for s in SPLITS},
                knife_groups_per_split={s: sum(1 for g in knife_groups if split_of[g['gid']] == s) for s in SPLITS})
    np.savez_compressed(OUT / '_hashes.npz', names=np.array([('rv1_' if recs[i]['ds'] == 'roboflow_v1' else '') + recs[i]['name'] for i in kidx]),
                        split=sp_arr, h64=H[kidx], h64f=HF[kidx], h256=H16[kidx], h256f=H16F[kidx])
    (OUT / '_split_meta.json').write_text(json.dumps(meta, indent=1))
    import csv
    with open(OUT / '_manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    with open(OUT / '_phash_merges.txt', 'w') as fh:
        for a, b, dist in merges:
            fh.write(f'dhash64\t{dist}\t{a}\t{b}\n')
        for a, b, dist in merges16:
            fh.write(f'dhash256\t{dist}\t{a}\t{b}\n')
    with open(OUT / '_label_issues.txt', 'w') as fh:
        fh.write('\n'.join(issues) + '\n')
        fh.write(f'\n# {len(conflicts)} groups whose members disagree on gun/knife labels\n')
        for stem, sig in conflicts:
            fh.write(f'{stem}: ' + '; '.join(f'{k}=(gun {v[0]},knife {v[1]})' for k, v in sig.items()) + '\n')
    print(json.dumps(meta, indent=1))
    print(f'label issues: {len(issues)} lines, {len(conflicts)} gun/knife-conflict groups (see _label_issues.txt)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Domain data for v2: remap work/data/clean labels and resolve copy-level label conflicts.

* source: work/data/clean/{train,valid,test} (classes 0 armed person, 1 gun, 2 knife), excluding synknife_*.
* new classes: 1 gun -> 0, 2 knife -> 1, 0 armed person dropped.
* conflict resolution: the unit is a *photo* = all Roboflow-augmented copies sharing a normalised
  stem (split.norm_stem). Copies are aligned up to h-flip + small shift/zoom; the per-copy transform
  is estimated from matched boxes (incl. the armed-person boxes, used here only as anchors).
  Weapon boxes are tracked across copies; for each track every copy votes gun / knife / none
  ('none' only when the track would be >=50% inside that copy). If >=2 copies agree and that option
  strictly beats the others, it is applied to all copies (relabel / add box / remove box);
  otherwise the track is unresolved and listed in _conflicts.csv / _conflicts_sheet.jpg.
* unresolved photos in valid/test are excluded (cannot be moved to train without leaking their
  group); unresolved photos in train keep their (partially resolved) current labels.

Writes work/data/v2/_staging/domain/{labels/*.txt, manifest.csv} and work/data/v2/_conflicts.{csv,jpg},
_autoresolved.csv, _autoresolved_sheet.jpg (for spot checks).
"""
import collections
import csv
import shutil

import cv2
import numpy as np

from common import CLEAN, OUT, STAGE, SPLITS, read_yolo, write_yolo, xyxy, iou, v1

REMAP = {1: 0, 2: 1}
# Visually verified exceptions to the majority rule (see STATS.md):
# aruval_tamil_nadu-1 is a single machete; 2 of 3 copies call it gun -> force knife.
OVERRIDE_CLASS = {'aruval_tamil_nadu-1': 1}
# tiny, ambiguous object; majority says gun but session neighbours are knives -> leave to the user
FORCE_REVIEW = {'screenshot-2026-07-02-at-2-25-01-pm'}
ALIGN_THR = 0.35  # max (1 - NCC) after warping a copy onto the reference copy
CN = {0: 'gun', 1: 'knife', None: 'none'}


def small(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def kind(c):
    return 'p' if c == 0 else 'w'


def greedy_match(A, B, thr, same_kind=True):
    """A,B lists of (cls, x1,y1,x2,y2) -> list of (i,j)."""
    cand = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if same_kind and kind(a[0]) != kind(b[0]):
                continue
            v = iou(a[1:], b[1:])
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    ui, uj, out = set(), set(), []
    for v, i, j in cand:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        out.append((i, j))
    return out


class Xform:
    """copy coords -> reference coords: optional h-flip then per-axis x' = a x + b."""

    def __init__(self, flip, ax=1.0, bx=0.0, ay=1.0, by=0.0):
        self.flip, self.ax, self.bx, self.ay, self.by = flip, ax, bx, ay, by

    def fwd(self, r):
        x1, y1, x2, y2 = r
        if self.flip:
            x1, x2 = 1 - x2, 1 - x1
        return (self.ax * x1 + self.bx, self.ay * y1 + self.by, self.ax * x2 + self.bx, self.ay * y2 + self.by)

    def inv(self, r):
        x1, y1, x2, y2 = r
        x1, x2 = (x1 - self.bx) / self.ax, (x2 - self.bx) / self.ax
        y1, y2 = (y1 - self.by) / self.ay, (y2 - self.by) / self.ay
        if self.flip:
            x1, x2 = 1 - x2, 1 - x1
        return (x1, y1, x2, y2)


def fit_axis(src, dst):
    src, dst = np.asarray(src), np.asarray(dst)
    if len(src) < 2 or np.ptp(src) < 0.05:
        return 1.0, float(np.mean(dst - src)) if len(src) else 0.0
    a, b = np.polyfit(src, dst, 1)
    if not 0.6 < a < 1.6:
        return 1.0, float(np.mean(dst - src))
    return float(a), float(b)


def warp_residual(X, ref_img, img):
    """1 - normalised cross-correlation between ref and copy warped into the ref frame (robust to the
    brightness/contrast augmentations), on the valid area."""
    N = img.shape[0]
    sx = -X.ax if X.flip else X.ax
    tx = (X.ax + X.bx) if X.flip else X.bx
    M = np.float32([[sx, 0, tx * N], [0, X.ay, X.by * N]])
    w = cv2.warpAffine(img, M, (N, N), flags=cv2.INTER_LINEAR, borderValue=-1)
    m = w >= 0
    if m.mean() < 0.5:
        return 2.0
    a, b = ref_img[m], w[m]
    a, b = a - a.mean(), b - b.mean()
    return float(1 - (a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))  # 1 - NCC


def estimate_xform(ref_boxes, ref_img, boxes, img):
    """Try both flip hypotheses; per-axis scale/shift fitted on matched boxes; pick lowest warped residual."""
    best = None
    for flip in (False, True):
        X = Xform(flip)
        R = [(b[0], *xyxy(b)) for b in ref_boxes]
        B = [(b[0], *X.fwd(xyxy(b))) for b in boxes]
        m = greedy_match(R, B, 0.2)
        if m:
            sx, dx, sy, dy = [], [], [], []
            for i, j in m:
                sx += [B[j][1], B[j][3]]
                dx += [R[i][1], R[i][3]]
                sy += [B[j][2], B[j][4]]
                dy += [R[i][2], R[i][4]]
            ax, bx = fit_axis(sx, dx)
            ay, by = fit_axis(sy, dy)
            X = Xform(flip, ax, bx, ay, by)
        res = warp_residual(X, ref_img, img)
        if best is None or res < best[1]:
            best = (X, res)
    return best


def visible_frac(r):
    x1, y1, x2, y2 = r
    a = max(1e-9, (x2 - x1) * (y2 - y1))
    ix = max(0.0, min(1, x2) - max(0, x1))
    iy = max(0.0, min(1, y2) - max(0, y1))
    return ix * iy / a


def clip_box(c, r):
    x1, y1, x2, y2 = [min(1.0, max(0.0, v)) for v in r]
    return (c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def main():
    group_of = {}
    with open(CLEAN / '_manifest.csv') as fh:
        for r in csv.DictReader(fh):
            group_of[r['file']] = int(r['group'])

    recs = []
    for sp in SPLITS:
        for f in sorted((CLEAN / sp / 'images').iterdir()):
            if f.name.startswith('synknife_') or f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                continue
            boxes = read_yolo(CLEAN / sp / 'labels' / (f.stem + '.txt'))
            recs.append(dict(split=sp, path=f, name=f.name, group=group_of.get(f.name, -1),
                             stem=v1.norm_stem(f.name.removeprefix('rv1_')), orig=boxes))
    print(f'domain images (no synknife): {len(recs)}  per split:',
          dict(collections.Counter(r['split'] for r in recs)))
    assert all(r['group'] >= 0 for r in recs), 'file missing from clean manifest'

    by_stem = collections.defaultdict(list)
    for i, r in enumerate(recs):
        by_stem[r['stem']].append(i)
        r['new'] = [list(b) for b in r['orig']]  # mutable working copy, old class ids for now
        r['status'] = 'ok'

    auto_rows, conf_rows = [], []
    n_photo_conf = n_photo_resolved = n_photo_unres = 0
    n_track_resolved = n_track_unres = n_unaligned = 0
    n_dup_dropped = [0]
    for stem, idx in by_stem.items():
        if len(idx) < 2:
            continue
        cp = [recs[i] for i in idx]
        imgs = [small(r['path']) for r in cp]
        ref = cp[0]
        xf, aligned = [Xform(False)], [True]
        for r, im in zip(cp[1:], imgs[1:]):
            X, res = estimate_xform(ref['orig'], imgs[0], r['orig'], im)
            xf.append(X)
            aligned.append(res <= ALIGN_THR)
        n_unaligned += aligned.count(False)
        # tracks of weapon boxes in reference frame: list of dict(rect, members{copy_k: box_index})
        tracks = []
        for k, r in enumerate(cp):
            if not aligned[k]:
                continue  # cannot be put in correspondence reliably -> abstains, labels untouched
            W = [(bi, xf[k].fwd(xyxy(b))) for bi, b in enumerate(r['orig']) if b[0] != 0]
            cand = []
            for ti, t in enumerate(tracks):
                for bi, rect in W:
                    v = iou(t['rect'], rect)
                    if v >= 0.3:
                        cand.append((v, ti, bi))
            cand.sort(reverse=True)
            used_t, used_b = set(), set()
            for v, ti, bi in cand:
                if ti in used_t or bi in used_b:
                    continue
                used_t.add(ti)
                used_b.add(bi)
                tracks[ti]['members'][k] = bi
            for bi, rect in W:
                if bi not in used_b:
                    tracks.append(dict(rect=rect, members={k: bi}))
        photo_conf = photo_unres = False
        for ti, t in enumerate(tracks):
            votes = {}
            for k, r in enumerate(cp):
                if not aligned[k]:
                    continue
                if k in t['members']:
                    votes[k] = REMAP[r['orig'][t['members'][k]][0]]
                elif visible_frac(xf[k].inv(t['rect'])) >= 0.5:
                    votes[k] = None
            cnt = collections.Counter(votes.values())
            if len(cnt) <= 1:
                continue
            photo_conf = True
            ccnt = collections.Counter(v for v in votes.values() if v is not None)
            (top, nt), *rest = ccnt.most_common()
            second = rest[0][1] if rest else 0
            # class: majority (>=2 agreeing copies, strictly more than the other class) or unanimous among
            # the copies that drew the box. presence: a box drawn on any copy is added to aligned copies
            # missing it (the missing copies are annotation omissions, see STATS.md)
            if stem not in FORCE_REVIEW and ((len(ccnt) == 1) or (nt >= 2 and nt > second)):
                n_track_resolved += 1
                kind_ = 'presence' if len(ccnt) == 1 else 'class'
                src_k = next(k for k in t['members'] if votes.get(k) == top)
                for k, vv in votes.items():
                    if vv == top:
                        continue
                    r = cp[k]
                    if vv is None:  # add box transferred from a majority copy
                        sb = cp[src_k]['orig'][t['members'][src_k]]
                        rect = xf[k].inv(xf[src_k].fwd(xyxy(sb)))
                        r['new'].append(list(clip_box(sb[0], rect)))
                        act = f'add {CN[top]} box'
                    else:
                        r['new'][t['members'][k]][0] = {0: 1, 1: 2}[top]
                        act = f'{CN[vv]} -> {CN[top]}'
                        rect = xyxy(r['orig'][t['members'][k]])
                    auto_rows.append(dict(group=r['group'], stem=stem, split=r['split'], file=r['name'], track=ti,
                                          kind=kind_, votes=' '.join(CN[votes[q]] for q in sorted(votes)),
                                          action=act, rect=rect))
            else:
                n_track_unres += 1
                photo_unres = True
                for k, vv in votes.items():
                    r = cp[k]
                    rect = xyxy(r['orig'][t['members'][k]]) if k in t['members'] else xf[k].inv(t['rect'])
                    conf_rows.append(dict(group=r['group'], stem=stem, split=r['split'], file=r['name'], track=ti,
                                          current=CN[vv], votes=' '.join(CN[votes[q]] for q in sorted(votes)),
                                          x1=round(rect[0], 4), y1=round(rect[1], 4), x2=round(rect[2], 4), y2=round(rect[3], 4),
                                          path=str(r['path'])))
        # unaligned copies: class-count level check against the (fixed) aligned copies
        al = [r for k, r in enumerate(cp) if aligned[k]]
        if len(al) < len(cp):
            def sig(r):
                c = collections.Counter(b[0] for b in r['new'] if b is not None and b[0] != 0)
                return (frozenset(c), c[2])
            ref_sigs = collections.Counter(sig(r) for r in al)
            ref_sig = ref_sigs.most_common(1)[0][0]
            def full(r):
                return collections.Counter(b[0] for b in r['new'] if b is not None and b[0] != 0)
            fulls = collections.Counter(tuple(sorted(full(r).items())) for r in cp)
            (ref_full, n_ref), = fulls.most_common(1)
            for k, r in enumerate(cp):
                if aligned[k] or sig(r) == ref_sig:
                    continue
                photo_conf = True
                mine = full(r)
                if (n_ref >= 2 and len(ref_full) == 1 and sum(mine.values()) == ref_full[0][1]
                        and fulls[tuple(sorted(mine.items()))] < n_ref):
                    # same number of weapon boxes, all aligned copies agree on one class -> majority class
                    tgt = ref_full[0][0]
                    for b in r['new']:
                        if b is not None and b[0] != 0 and b[0] != tgt:
                            auto_rows.append(dict(group=r['group'], stem=stem, split=r['split'], file=r['name'],
                                                  track='U', kind='class', votes=f'unaligned vs {n_ref} agreeing copies',
                                                  action=f'{CN[REMAP[b[0]]]} -> {CN[REMAP[tgt]]}', rect=xyxy(b)))
                            b[0] = tgt
                    n_track_resolved += 1
                    continue
                photo_unres = True
                n_track_unres += 1
                want = '+'.join(sorted(CN[REMAP[c]] for c in ref_sig[0])) or 'none'
                bx = [b for b in r['orig'] if b[0] != 0] or [(None, 0.5, 0.5, 1.0, 1.0)]
                for b in bx:
                    x1, y1, x2, y2 = xyxy(b)
                    conf_rows.append(dict(group=r['group'], stem=stem, split=r['split'], file=r['name'], track='U',
                                          current=CN[REMAP.get(b[0])], votes=f'unaligned; other copies: {want} ({ref_sig[1]} knife)',
                                          x1=round(x1, 4), y1=round(y1, 4), x2=round(x2, 4), y2=round(y2, 4),
                                          path=str(r['path'])))
        if photo_conf:
            n_photo_conf += 1
            if photo_unres:
                n_photo_unres += 1
                for r in cp:
                    r['status'] = 'unresolved'
            else:
                n_photo_resolved += 1
                for r in cp:
                    r['status'] = 'resolved'

    for r in recs:
        if r['stem'] in OVERRIDE_CLASS:
            tgt = {0: 1, 1: 2}[OVERRIDE_CLASS[r['stem']]]
            for b in r['new']:
                if b is not None and b[0] != 0 and b[0] != tgt:
                    auto_rows.append(dict(group=r['group'], stem=r['stem'], split=r['split'], file=r['name'], track='M',
                                          kind='manual', votes='manual override (visually verified)',
                                          action=f'{CN[REMAP[b[0]]]} -> {CN[REMAP[tgt]]}', rect=xyxy(b)))
                    b[0] = tgt
            r['status'] = 'resolved'
        # drop same-class near-duplicate boxes (IoU > 0.8)
        kept = []
        for b in r['new']:
            if b is None:
                continue
            if any(k[0] == b[0] and iou(xyxy(k), xyxy(b)) > 0.8 for k in kept):
                n_dup_dropped[0] += 1
                continue
            kept.append(b)
        r['new'] = kept

    # finalise labels: drop person, remap
    lab_dir = STAGE / 'domain' / 'labels'
    if lab_dir.exists():
        shutil.rmtree(lab_dir)
    lab_dir.mkdir(parents=True)
    rows = []
    excluded = 0
    for r in recs:
        boxes = [(REMAP[b[0]], *b[1:]) for b in r['new'] if b is not None and b[0] != 0]
        keep = not (r['status'] == 'unresolved' and r['split'] != 'train')
        if not keep:
            excluded += 1
        stem = r['path'].stem
        write_yolo(lab_dir / f'{stem}.txt', boxes)
        rows.append(dict(name=r['name'], split=r['split'], group=r['group'], stem=r['stem'], status=r['status'],
                         keep=int(keep), n_gun=sum(b[0] == 0 for b in boxes), n_knife=sum(b[0] == 1 for b in boxes),
                         n_person_dropped=sum(b[0] == 0 for b in r['orig']), path=str(r['path'])))
    with open(STAGE / 'domain' / 'manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # old (session-level) conflict list from v1, for the record
    with open(OUT / '_conflicts.csv', 'w', newline='') as fh:
        flds = ['group', 'stem', 'split', 'file', 'track', 'current', 'votes', 'x1', 'y1', 'x2', 'y2', 'path']
        w = csv.DictWriter(fh, fieldnames=flds)
        w.writeheader()
        w.writerows(conf_rows)
    with open(OUT / '_autoresolved.csv', 'w', newline='') as fh:
        flds = ['group', 'stem', 'split', 'file', 'track', 'kind', 'votes', 'action']
        w = csv.DictWriter(fh, fieldnames=flds, extrasaction='ignore')
        w.writeheader()
        w.writerows(auto_rows)

    contact_sheet(conf_rows, OUT / '_conflicts_sheet.jpg', key='current',
                  title='UNRESOLVED gun/knife conflicts (copies of the same photo)')
    contact_sheet([dict(r, current=r['action'], path=str(next(q['path'] for q in recs if q['name'] == r['file'])))
                   for r in auto_rows], OUT / '_autoresolved_sheet.jpg', key='current',
                  title='AUTO-RESOLVED by majority (action applied to this copy)', use_rect=True)

    summary = dict(photos_multi_copy=sum(len(v) > 1 for v in by_stem.values()), photos_total=len(by_stem),
                   photos_conflicting=n_photo_conf, photos_auto_resolved=n_photo_resolved,
                   photos_unresolved=n_photo_unres, tracks_auto_resolved=n_track_resolved,
                   tracks_unresolved=n_track_unres, copies_unaligned_abstained=n_unaligned,
                   class_fixes=sum(r['kind'] == 'class' for r in auto_rows),
                   presence_fixes=sum(r['kind'] == 'presence' for r in auto_rows),
                   manual_overrides=sum(r['kind'] == 'manual' for r in auto_rows), duplicate_boxes_dropped=n_dup_dropped[0], label_edits=len(auto_rows),
                   unresolved_images=sum(r['status'] == 'unresolved' for r in recs),
                   unresolved_images_excluded_from_val_test=excluded)
    import json
    (STAGE / 'domain' / 'conflict_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def contact_sheet(rows, out, key, title, use_rect=False, tile=200, cols=8):
    if not rows:
        im = np.full((80, 900, 3), 255, np.uint8)
        cv2.putText(im, title + ': none', (10, 50), 0, 0.7, (0, 0, 0), 2)
        cv2.imwrite(str(out), im)
        return
    tiles = []
    cache = {}
    for r in rows:
        p = r['path']
        if p not in cache:
            cache.clear()
            cache[p] = cv2.imread(p)
        im = cache[p]
        H, W = im.shape[:2]
        if use_rect:
            x1, y1, x2, y2 = r['rect']
        else:
            x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']
        bw, bh = x2 - x1, y2 - y1
        px1, py1 = int(max(0, x1 - 0.15 * bw) * W), int(max(0, y1 - 0.15 * bh) * H)
        px2, py2 = int(min(1, x2 + 0.15 * bw) * W), int(min(1, y2 + 0.15 * bh) * H)
        crop = im[py1:max(py2, py1 + 2), px1:max(px2, px1 + 2)].copy()
        s = (tile - 4) / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        t = np.full((tile + 46, tile, 3), 255, np.uint8)
        t[2:2 + crop.shape[0], 2:2 + crop.shape[1]] = crop
        cur = str(r[key])
        col = (0, 0, 220) if 'gun' in cur.split(' -> ')[-1] else (0, 150, 0) if 'knife' in cur else (120, 120, 120)
        cv2.rectangle(t, (0, 0), (tile - 1, tile + 45), col, 2)
        cv2.putText(t, f"g{r['group']} {r['split']} {cur}"[:30], (3, tile + 12), 0, 0.38, col, 1)
        cv2.putText(t, r['file'][:34], (3, tile + 26), 0, 0.3, (0, 0, 0), 1)
        cv2.putText(t, f"{r['stem'][:22]} t{r['track']} [{r['votes']}]"[:40], (3, tile + 40), 0, 0.3, (60, 60, 60), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    head = np.full((40, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title + f' ({len(rows)} crops)', (10, 28), 0, 0.8, (0, 0, 0), 2)
    cv2.imwrite(str(out), np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == '__main__':
    main()

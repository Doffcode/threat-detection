#!/usr/bin/env python
"""Build work/data/v2: merge domain (work/data/clean) + public sources, dedupe, group, leak-free split.

Run with `python -I` (reads untrusted downloads). Prerequisites:
  python domain.py                 -> _staging/domain   (remap + conflict resolution)
  python -I convert_odw.py         -> _staging/odw      (OD-WeaponDetection VOC -> YOLO)
  optional: --kaggle / --roboflow  -> hooks in sources.py (skipped when credentials are missing)

Grouping (reuses work/scripts/data/split.py dHash helpers and thresholds):
  * domain images keep their work/data/clean group and split (domain test stays comparable);
  * public images named like video-frame / photo-session series are grouped by name prefix;
  * union over ALL sources of 64-bit dHash (<=4, h-flip aware) and 256-bit border-trimmed dHash (<=32);
  * components that touch domain images take the domain split; components touching domain images of
    >1 split drop their public images; public-only components get a seeded ~80/10/10 group split
    (series groups forced to train). Near-identical public copies inside a group are deduplicated.
"""
import argparse
import collections
import csv
import json
import os
import random
import re
import shutil
from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN, NAMES, OUT, SPLITS, STAGE, read_yolo, v1
import sources  # noqa: E402

GENERIC_PREFIXES = {'armas', 'knife', 'pistol', 'smartphone', 'monedero', 'billete', 'tarjeta', 'online'}
PUB_TARGETS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
SRC_PRIORITY = {'odw_soh': 0, 'odw_knf': 1, 'odw_pis': 2}


def series_key(name):
    """Public series key (video frames / photo sessions) or None for generic web images."""
    s = re.sub(r'^odw_[a-z]{3}_', '', Path(name).stem)
    s = re.sub(r'_\d+$', '', s)  # rename suffix
    p = re.sub(r'\d+', '', s).strip('_').rstrip('_')
    p = re.sub(r'_+', '_', p)
    if p.lower() in GENERIC_PREFIXES or p.lower().startswith('armas'):
        return None
    return p


def hash_image(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    tr = v1.trim_borders(g)
    h = v1.dhash_bits(g)
    return dict(h64=np.packbits(h), h64f=np.packbits(v1.dhash_bits(cv2.flip(g, 1))),
                h256=np.packbits(v1.dhash16(tr)), h256f=np.packbits(v1.dhash16(cv2.flip(tr, 1))),
                low=int(h.sum()) < 4 or int(h.sum()) > 60, shape=g.shape[:2])


def near_pairs(P, PF, thr, rows=None, chunk=128):
    """All (i<j) with min(hamming(P_i,P_j), hamming(P_i,PF_j)) <= thr; chunked (bounded memory)."""
    n = len(P)
    rows = range(n) if rows is None else rows
    out = []
    rows = np.asarray(list(rows))
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        d = v1._POP[P[r][:, None, :] ^ P[None, :, :]].sum(-1, dtype=np.int16)
        d = np.minimum(d, v1._POP[P[r][:, None, :] ^ PF[None, :, :]].sum(-1, dtype=np.int16))
        ii, jj = np.where(d <= thr)
        for a, j in zip(ii, jj):
            i = r[a]
            if j > i:
                out.append((int(i), int(j), int(d[a, j])))
    return out


def load_records(extra_tags):
    recs = []
    with open(STAGE / 'domain' / 'manifest.csv') as fh:
        for r in csv.DictReader(fh):
            recs.append(dict(source='dom', name='dom_' + r['name'], src=r['path'],
                             label=str(STAGE / 'domain' / 'labels' / (Path(r['name']).stem + '.txt')),
                             fixed_split=r['split'], dom_group=int(r['group']), keep=r['keep'] == '1',
                             status=r['status']))
    for tag in ['odw'] + extra_tags:
        mf = STAGE / tag / 'manifest.csv'
        if not mf.exists():
            continue
        with open(mf) as fh:
            for r in csv.DictReader(fh):
                recs.append(dict(source=r['source'], name=r['name'], src=r['src'], label=r['label'],
                                 fixed_split=None, dom_group=None, keep=True, status='ok'))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--hash-thr', type=int, default=4)
    ap.add_argument('--hash16-thr', type=int, default=32)
    ap.add_argument('--dup-thr', type=int, default=6, help='256-bit dist (no flip) for identical-copy removal')
    ap.add_argument('--bg-frac', type=float, default=0.10, help='max background fraction of final train')
    ap.add_argument('--stage2-public-frac', type=float, default=0.30)
    ap.add_argument('--kaggle', action='store_true', help='add Kaggle iqmansingh/guns-knives-object-detection')
    ap.add_argument('--roboflow', action='store_true', help='add 1-2 Roboflow Universe knife/pistol sets')
    args = ap.parse_args()
    rng = random.Random(args.seed)

    extra, skipped = [], []
    if args.kaggle:
        (extra.append('kag') if sources.fetch_kaggle() else skipped.append('kaggle (no ~/.kaggle/kaggle.json)'))
    else:
        skipped.append('kaggle (flag not given; ~/.kaggle/kaggle.json %s)' % ('present' if sources.kaggle_available() else 'absent'))
    if args.roboflow:
        ok = sources.fetch_roboflow()
        if ok:
            extra += [p.name for p in STAGE.iterdir() if p.name.startswith('rf')]
        else:
            skipped.append('roboflow (ROBOFLOW_API_KEY not set)')
    else:
        skipped.append('roboflow (flag not given; ROBOFLOW_API_KEY %s)' % ('set' if sources.roboflow_available() else 'unset'))

    recs = load_records(extra)
    print(f'records: {len(recs)}', dict(collections.Counter(r['source'] for r in recs)))

    # ---- hash (streamed)
    good = []
    unreadable = 0
    for i, r in enumerate(recs):
        h = hash_image(r['src'])
        if h is None:
            unreadable += 1
            continue
        r.update(h)
        r['boxes'] = read_yolo(r['label'])
        good.append(r)
        if i % 2000 == 0:
            print(f'  hashed {i}/{len(recs)}')
    recs = good
    n = len(recs)
    print(f'hashed {n}, unreadable {unreadable}')
    H64 = np.stack([r['h64'] for r in recs])
    H64F = np.stack([r['h64f'] for r in recs])
    H256 = np.stack([r['h256'] for r in recs])
    H256F = np.stack([r['h256f'] for r in recs])
    low = np.array([r['low'] for r in recs])

    # ---- union-find
    uf = v1.UF(n)
    first = {}
    for i, r in enumerate(recs):
        if r['source'] == 'dom':
            k = ('dom', r['dom_group'])
        else:
            sk = series_key(r['name'])
            k = ('series', sk) if sk else None
            r['series'] = sk
        if k is not None:
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
    pairs64 = near_pairs(H64, H64F, args.hash_thr)
    pairs256 = near_pairs(H256, H256F, args.hash16_thr)
    merge_stats = collections.Counter()
    for tag, pairs in (('dhash64', pairs64), ('dhash256', pairs256)):
        for i, j, d in pairs:
            if low[i] or low[j]:
                continue
            kind = '-'.join(sorted({'dom' if recs[i]['source'] == 'dom' else 'pub',
                                    'dom' if recs[j]['source'] == 'dom' else 'pub'}))
            if kind == 'dom-pub':
                # domain<->public: white-background product shots / low-texture frames give many single-hash
                # false positives that chain unrelated images; require BOTH hashes, otherwise drop the
                # public image (so no single-hash near-duplicate pair can cross splits either)
                d64 = min(int(v1._POP[H64[i] ^ H64[j]].sum()), int(v1._POP[H64[i] ^ H64F[j]].sum()))
                d256 = min(int(v1._POP[H256[i] ^ H256[j]].sum()), int(v1._POP[H256[i] ^ H256F[j]].sum()))
                # any domain<->public hash match: drop the public image instead of merging, so public chains are
                # never pulled into a domain split (and domain groups stay exactly as in work/data/clean)
                both = d64 <= args.hash_thr and d256 <= args.hash16_thr
                p_ = j if recs[i]['source'] == 'dom' else i
                if recs[p_]['keep']:
                    recs[p_]['keep'] = False
                    recs[p_]['drop_reason'] = (f"{'near-duplicate' if both else 'single-hash match'} of a domain image "
                                               f'(d64={d64}, d256={d256})')
                    merge_stats[f"dom-pub {'both-hash' if both else 'single-hash'} -> public dropped"] += 1
                continue
            if uf.union(i, j):
                merge_stats[f'{tag} {kind}'] += 1
    comps = collections.defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)
    print(f'{len(comps)} components; merges: {dict(merge_stats)}')

    # ---- split assignment
    split_of = {}
    dropped_bridge = 0
    pub_groups = []
    for root, idx in comps.items():
        dsplits = {recs[i]['fixed_split'] for i in idx if recs[i]['source'] == 'dom'}
        if len(dsplits) == 1:
            split_of[root] = dsplits.pop()
        elif len(dsplits) > 1:
            # public image(s) bridge domain groups of different splits -> drop the public images
            for i in idx:
                if recs[i]['source'] != 'dom':
                    recs[i]['keep'] = False
                    recs[i]['drop_reason'] = 'bridges domain splits'
                    dropped_bridge += 1
            for i in idx:
                if recs[i]['source'] == 'dom':
                    recs[i]['comp_split'] = recs[i]['fixed_split']
            split_of[root] = 'MIXED'
        else:
            cls = collections.Counter(b[0] for i in idx for b in recs[i]['boxes'])
            pub_groups.append(dict(gid=root, nimg=len(idx), knife=cls[1], gun=cls[0], hh=cls[2],
                                   series=any(recs[i].get('series') for i in idx)))
    # public: series groups -> train; others stratified greedy group split
    for g in pub_groups:
        if g['series']:
            split_of[g['gid']] = 'train'
    rest = [g for g in pub_groups if not g['series']]
    strata = [[g for g in rest if g['knife']], [g for g in rest if not g['knife'] and g['gun']],
              [g for g in rest if not g['knife'] and not g['gun'] and g['hh']],
              [g for g in rest if not g['knife'] and not g['gun'] and not g['hh']]]
    # series images already sit in train: correct the train target so the overall public ratio is ~80/10/10
    S = sum(g['nimg'] for g in pub_groups if g['series'])
    R = sum(g['nimg'] for g in rest) or 1
    T = S + R
    tg = {'valid': PUB_TARGETS['valid'] * T / R, 'test': PUB_TARGETS['test'] * T / R}
    tg['train'] = max(0.05, 1 - tg['valid'] - tg['test'])
    print(f'public: {S} series images forced to train; non-series targets {tg}')
    for st in strata:
        v1.assign(st, 'nimg', tg, rng, split_of)
    for i, r in enumerate(recs):
        s = split_of[uf.find(i)]
        r['split'] = r.get('comp_split', s) if s == 'MIXED' else s
        r['gid'] = uf.find(i) if s != 'MIXED' else ('dom', r['dom_group'])

    # ---- near-identical public copies within a group -> keep one
    dup_removed = collections.Counter()
    for root, idx in comps.items():
        pub = [i for i in idx if recs[i]['source'] != 'dom' and recs[i]['keep']]
        dom = [i for i in idx if recs[i]['source'] == 'dom']
        if not pub:
            continue
        pub.sort(key=lambda i: (SRC_PRIORITY.get(recs[i]['source'], 9), -len(recs[i]['boxes']), recs[i]['name']))
        kept = list(dom)
        for i in pub:
            dup = False
            for k in kept:
                d = int(v1._POP[H256[i] ^ H256[k]].sum())
                if d <= args.dup_thr and int(v1._POP[H64[i] ^ H64[k]].sum()) <= 2:
                    dup = True
                    break
            if dup:
                recs[i]['keep'] = False
                recs[i]['drop_reason'] = 'near-identical copy of ' + ('domain' if recs[k]['source'] == 'dom' else recs[k]['source'])
                dup_removed[f"{recs[i]['source']} dup of {'dom' if recs[k]['source'] == 'dom' else recs[k]['source']}"] += 1
            else:
                kept.append(i)

    # ---- background cap (train)
    train_pos = [r for r in recs if r['keep'] and r['split'] == 'train' and r['boxes']]
    train_bg = [r for r in recs if r['keep'] and r['split'] == 'train' and not r['boxes']]
    max_bg = int(args.bg_frac / (1 - args.bg_frac) * len(train_pos))
    rng.shuffle(train_bg)
    for r in train_bg[max_bg:]:
        r['keep'] = False
        r['drop_reason'] = 'background cap'
    bg_info = dict(train_positive=len(train_pos), train_background_available=len(train_bg),
                   train_background_kept=min(max_bg, len(train_bg)))

    # ---- write
    for s in SPLITS:
        shutil.rmtree(OUT / s, ignore_errors=True)
        (OUT / s / 'images').mkdir(parents=True)
        (OUT / s / 'labels').mkdir(parents=True)
    used = set()
    W = [r for r in recs if r['keep']]
    for r in W:
        assert r['name'] not in used, r['name']
        used.add(r['name'])
        dst = OUT / r['split'] / 'images' / r['name']
        try:
            os.link(r['src'], dst)
        except OSError:
            shutil.copy2(r['src'], dst)
        with open(OUT / r['split'] / 'labels' / (Path(r['name']).stem + '.txt'), 'w') as fh:
            for c, cx, cy, w, h in r['boxes']:
                fh.write(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        r['out'] = str(dst)

    # ---- leak checks on what was written
    widx = [i for i, r in enumerate(recs) if r['keep']]
    sp = np.array([recs[i]['split'] for i in widx])
    src_dom = np.array([recs[i]['source'] == 'dom' for i in widx])
    wl = low[widx]
    overlap = {}
    gsplits = collections.defaultdict(set)
    for i in widx:
        gsplits[recs[i]['gid']].add(recs[i]['split'])
    overlap['groups in >1 split'] = sum(len(v) > 1 for v in gsplits.values())
    for tag, P, PF, thr in (('dhash64', H64, H64F, args.hash_thr), ('dhash256', H256, H256F, args.hash16_thr)):
        Pw, PFw = P[widx], PF[widx]
        cross = [(a, b) for a, b, d in near_pairs(Pw, PFw, thr) if sp[a] != sp[b] and not wl[a] and not wl[b]]
        overlap[f'cross-split {tag} pairs (<= {thr}) all'] = len(cross)
        overlap[f'cross-split {tag} pairs (<= {thr}) domain-public'] = sum(src_dom[a] != src_dom[b] for a, b in cross)
    # domain stems (copies of the same photo) must not cross splits
    stems = collections.defaultdict(set)
    for i in widx:
        if recs[i]['source'] == 'dom':
            stems[v1.norm_stem(recs[i]['name'][4:].removeprefix('rv1_'))].add(recs[i]['split'])
    overlap['domain photo stems in >1 split'] = sum(len(v) > 1 for v in stems.values())
    # domain split unchanged vs work/data/clean
    overlap['domain images whose split changed vs clean'] = sum(
        recs[i]['split'] != recs[i]['fixed_split'] for i in widx if recs[i]['source'] == 'dom')
    for k, v in overlap.items():
        print(f'  overlap check {k}: {v}')
    assert all(v == 0 for v in overlap.values()), 'LEAK DETECTED'

    # ---- lists + yamls
    def lst(fn, items):
        (OUT / fn).write_text(''.join(r['out'] + '\n' for r in sorted(items, key=lambda r: r['name'])))
    dom = [r for r in W if r['source'] == 'dom']
    pub = [r for r in W if r['source'] != 'dom']
    lst('test_domain.txt', [r for r in dom if r['split'] == 'test'])
    lst('test_public.txt', [r for r in pub if r['split'] == 'test'])
    lst('valid_domain.txt', [r for r in dom if r['split'] == 'valid'])
    lst('valid_public.txt', [r for r in pub if r['split'] == 'valid'])
    pub_train = sorted([r for r in pub if r['split'] == 'train'], key=lambda r: r['name'])
    samp = random.Random(args.seed + 1).sample(pub_train, int(round(args.stage2_public_frac * len(pub_train))))
    stage2 = [r for r in dom if r['split'] == 'train'] + samp
    lst('train_stage2.txt', stage2)
    names = "['gun', 'knife', 'handheld_object']"
    assert NAMES == ['gun', 'knife', 'handheld_object']
    Y = {
        'data_all.yaml': ('# all sources (dom_ + odw_), leak-free group split',
                          'train/images', 'valid/images', 'test/images'),
        'data_domain_test.yaml': ('# all-source train/val; test = domain-only test (comparable to work/data/clean test)',
                                  'train/images', 'valid/images', 'test_domain.txt'),
        'data_public_test.yaml': ('# all-source train/val; test = public-only (OD-WeaponDetection) test',
                                  'train/images', 'valid/images', 'test_public.txt'),
        'data_stage2.yaml': (f'# stage-2 fine-tune: all domain train + {args.stage2_public_frac:.0%} random public train; '
                             'val = domain valid; test = domain test', 'train_stage2.txt', 'valid_domain.txt', 'test_domain.txt'),
    }
    for fn, (cmt, tr, va, te) in Y.items():
        (OUT / fn).write_text(f'{cmt}\npath: {OUT}\ntrain: {tr}\nval: {va}\ntest: {te}\nnc: 3\nnames: {names}\n')

    # ---- manifest
    with open(OUT / '_manifest.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['name', 'source', 'split', 'group', 'keep', 'drop_reason', 'n_gun', 'n_knife', 'n_handheld', 'src'])
        for r in recs:
            c = collections.Counter(b[0] for b in r['boxes'])
            w.writerow([r['name'], r['source'], r.get('split'), str(r['gid']), int(r['keep']), r.get('drop_reason', ''),
                        c[0], c[1], c[2], r['src']])

    preview(pub_train, OUT / '_preview_public.jpg', rng=random.Random(args.seed + 2))

    meta = dict(seed=args.seed, thresholds=dict(dhash64=args.hash_thr, dhash256=args.hash16_thr, dup=args.dup_thr),
                n_records=n, unreadable=unreadable, n_components=len(comps), merges=dict(merge_stats),
                public_dropped_bridging_domain_splits=dropped_bridge, near_identical_removed=dict(dup_removed),
                background=bg_info, overlap=overlap, skipped_sources=skipped,
                stage2=dict(domain_train=len(stage2) - len(samp), public_sample=len(samp), public_train=len(pub_train)))
    (OUT / '_build_meta.json').write_text(json.dumps(meta, indent=1, default=str))
    write_stats(recs, W, meta)
    print(json.dumps(meta, indent=1, default=str))


def preview(items, out, rng, k=16, tile=320):
    cols = [(0, 0, 230), (0, 180, 0), (230, 120, 0)]
    tiles = []
    for r in rng.sample(items, min(k, len(items))):
        im = cv2.imread(r['out'])
        h, w = im.shape[:2]
        s = tile / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        h, w = im.shape[:2]
        for c, cx, cy, bw, bh in r['boxes']:
            p1 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
            p2 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
            cv2.rectangle(im, p1, p2, cols[c], 2)
            cv2.putText(im, NAMES[c], (p1[0] + 2, max(12, p1[1] + 14)), 0, 0.5, cols[c], 2)
        t = np.full((tile + 18, tile, 3), 255, np.uint8)
        t[:h, :w] = im
        cv2.putText(t, r['name'][:44], (2, tile + 13), 0, 0.38, (0, 0, 0), 1)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def write_stats(recs, W, meta):
    dsum = json.loads((STAGE / 'domain' / 'conflict_summary.json').read_text())
    srcs = sorted({r['source'] for r in W}, key=lambda s: (s != 'dom', s))
    L = ['# work/data/v2 dataset statistics', '',
         'Classes: 0 gun, 1 knife, 2 handheld_object. Prefixes: `dom_` = domain (work/data/clean, no synknife), '
         '`odw_knf_` / `odw_pis_` / `odw_soh_` = OD-WeaponDetection Knife_detection / Pistol detection / Sohas_weapon-Detection.',
         '', '## Instances and images per split x source', '',
         '| split | source | images | bg images | gun | knife | handheld_object | groups |', '|---|---|---|---|---|---|---|---|']
    tot = collections.defaultdict(collections.Counter)
    for s in SPLITS:
        for src in srcs + ['ALL']:
            rr = [r for r in W if r['split'] == s and (src == 'ALL' or r['source'] == src)]
            c = collections.Counter(b[0] for r in rr for b in r['boxes'])
            g = len({str(r['gid']) for r in rr})
            L.append(f"| {s} | {'**all**' if src == 'ALL' else src} | {len(rr)} | {sum(not r['boxes'] for r in rr)} | "
                     f"{c[0]} | {c[1]} | {c[2]} | {g} |")
    c = collections.Counter(b[0] for r in W for b in r['boxes'])
    L.append(f"| **total** | | {len(W)} | {sum(not r['boxes'] for r in W)} | {c[0]} | {c[1]} | {c[2]} | "
             f"{len({str(r['gid']) for r in W})} |")
    L += ['', '## Grouping / dedupe', '',
          f"- records hashed: {meta['n_records']} (unreadable: {meta['unreadable']}); components after merging: {meta['n_components']}",
          f"- dHash merges (new unions): {meta['merges']}",
          f"- public images dropped because they bridged domain groups of different splits: {meta['public_dropped_bridging_domain_splits']}",
          '- domain<->public hash matches are not merged: the public image is dropped instead (most are single-hash false positives on '
          'white-background product shots, checked visually; merging them chained ~350 public images into the domain test split). '
          'Domain groups therefore stay exactly as in work/data/clean.',
          '- public<->public merging is deliberately aggressive (either hash): many generic-named public images are consecutive video frames.',
          f"- near-identical public copies removed (kept one per group): {meta['near_identical_removed']}",
          f"- train background cap: {meta['background']}",
          '- domain groups keep their work/data/clean split; public video/photo-session series (name prefix) are one group each and go to train; '
          'other public groups: seeded greedy ~80/10/10 split stratified by knife / gun / handheld.',
          f"- skipped optional sources: {meta['skipped_sources']}",
          '', '## Overlap checks (all must be 0)', '']
    for k, v in meta['overlap'].items():
        L.append(f'- {k}: **{v}**')
    L += ['', '## Domain label-conflict resolution', '',
          'Unit = one photo = all Roboflow-augmented copies sharing a normalised stem (the 27 groups in '
          '`work/data/clean/_label_issues.txt` are screenshot *sessions*; different frames legitimately differ, so they were '
          're-examined per photo). Copies are aligned (h-flip + per-axis scale/shift fitted on matched boxes, checked with NCC); '
          'weapon boxes are tracked across copies.', '',
          '- **class** disagreement: majority (>= 2 agreeing copies, strictly more than the other class) applied to the others.',
          '- **presence** disagreement (a box drawn on some copies only): the box is added to the aligned copies missing it. '
          'A literal majority vote would have *deleted* clearly visible guns/knives from ~60 photos (checked visually), so union was used instead.',
          '- copies that could not be aligned (NCC residual > 0.35) abstain; if their gun/knife counts differ from the other copies '
          'they are left for the user (or relabelled when box counts match and >= 2 copies agree on one class).',
          '- manual exceptions (visually verified): `aruval_tamil_nadu-1` is a machete, majority said gun -> forced knife; '
          '`screenshot-2026-07-02-at-2-25-01-pm` (tiny ambiguous object, majority gun) left for the user.',
          '- exact duplicate boxes (same class, IoU > 0.8) removed.', '']
    for k, v in dsum.items():
        L.append(f'- {k}: {v}')
    L += ['', 'Files: `_conflicts.csv` + `_conflicts_sheet.jpg` (unresolved, for the user), `_autoresolved.csv` + '
          '`_autoresolved_sheet.jpg` (every automatic edit). Unresolved photos in valid/test are excluded; in train they keep current labels.',
          '', '## Licences', '',
          '- OD-WeaponDetection (ari-dasci, https://github.com/ari-dasci/OD-WeaponDetection): CC BY-SA 4.0 '
          '(attribution + share-alike apply to derived datasets). Cite Pérez-Hernández et al. 2020 (KBS 194:105590) for Sohas, '
          'Olmos et al. 2018 (Neurocomputing 275) for pistols.',
          '- Domain data: project-internal (work/raw).', '']
    (OUT / 'STATS.md').write_text('\n'.join(L))


if __name__ == '__main__':
    main()

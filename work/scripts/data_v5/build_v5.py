#!/usr/bin/env python
"""Build work/data/v5_indian_hybrid: High-accuracy Indian Hybrid Dataset for Weapon Detection.

Combines:
  1. Indian Domain Data (dom): 999 clean Indian cinema stills (fixed splits: train_domain, valid_domain, test_domain).
  2. Verified Indian Datasets:
     - rob (anti_robbing): Indian office CCTV (ROB_DROP unannotated desk frames excluded).
     - trg (trail_guard): Staged Indian poses (toy Pistol/Rifle purged; Machete/Knife -> 1, Rod -> 2).
     - mob (mohini): Indian campus CCTV (toy Gun and Fuel Tank purged; Machete -> 1, Axe -> 2).
     - bits_thr (threatdetect2): Indian blade/knife/cutter scenes (lazy whole-frame boxes dropped).
     - knife_ijigt (vivek knife): Indian cinema knife stills.
     - gun_twl (gun-knife): Indian cinema / real gun and knife frames.
     - snatch (chain_snatching): Indian street CCTV background negatives.
     - Tabletop evaluation set (val_tabletop): Dedicated 35-image benchmark (20 rob desk weapons + 15 bits tabletop knives).
  3. Public Foundation Data:
     - Open Images V7 (oi): filtered firearms, knives, and hard negatives.
     - OD-WeaponDetection (odw): pistols, knives, Sohas.
     - Roboflow curated exports: adelios (ade), jaeyuncho (jae), harami (har), aahil (aah), karan (kar), gowtham (gow), rohan (roh), csvtu (csv).

Guardrails:
  - Strict dHash64 (<=4) and dHash256 (<=32) deduplication ensuring 0 LEAKAGE against test_domain.
  - Background negatives capped at 10% of training positives.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import random
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

WORK = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work')
sys.path.insert(0, str(WORK / 'scripts' / 'data'))
import split as v1  # noqa: E402

V2 = WORK / 'data' / 'v2'
IV3 = WORK / 'data' / 'indian_v3'
OUT = WORK / 'data' / 'v5_indian_hybrid'
RF = WORK / 'downloads' / 'roboflow'
RF4 = WORK / 'downloads' / 'roboflow_v4'
OI = WORK / 'downloads' / 'openimages'
CACHE = WORK / 'scripts' / 'data_v5' / '_cache'

NAMES = ['gun', 'knife', 'handheld_object']
SEED = 2026
T64, T256 = 4, 32
OI_HARDNEG_CAP = 2500
BG_FRAC = 0.10
BIG_GROUP = 30
TARGET = {'train': 0.80, 'valid': 0.09, 'test_public': 0.11}
D = 'DROP'  # drop whole image
I = None    # ignore box

# anti_robbing frames with visible unlabelled weapons to drop
ROB_DROP = set(
    list(range(44, 73)) +
    list(range(235, 264)) + [290] +
    [n for n in range(565, 634) if n not in (604, 605)]
)

# 20 well-spaced anti_robbing desk stems held out for val_tabletop
ROB_TABLETOP_STEMS = {0, 17, 36, 82, 98, 116, 138, 177, 225, 297, 364, 400, 452, 467, 482, 502, 517, 560, 650, 666}

RFSRC = {
    'aah': (RF / 'aahil-m8yrw__weapon-detection-revmd__v3',
            {'0': 0, '1': D, '2': D, 'Celurit': 1, 'Linggis': 2, 'axe': 2, 'bomb': D, 'bow': D, 'cleaver': 1,
             'cutlass': 1, 'handgun': 0, 'heavyweapon': 0, 'katana': 1, 'knife': 1, 'mace': 2, 'machine gun': 0,
             'morningstar': 2, 'pistol': 0, 'rifle': 0, 'rocket launcher': D, 'scabbard': I, 'scope': I,
             'shield': I, 'shotgun': 0, 'sickle': 1, 'smg': 0, 'sniper rifle': 0, 'spear': D, 'sword': 1,
             'war hammer': 2}),
    'har': (RF / 'harami__other-h2hcu__v2',
            {'YOLOv5 Fall Detection - v1 2023-03-03 10-23pm': I, 'axe': 2, 'bomb': D, 'bow': D, 'cleaver': 1,
             'cutlass': 1, 'fall': I, 'katana': 1, 'knife': 1, 'mace': 2, 'machine gun': 0, 'morningstar': 2,
             'pistol': 0, 'rifle': 0, 'rocket launcher': D, 'scabbard': I, 'scope': I, 'shield': I, 'shotgun': 0,
             'sickle': 1, 'smg': 0, 'sniper rifle': 0, 'spear': D, 'sword': 1, 'war hammer': 2}),
    'kar': (RF / 'karan-bedi-nikgk__weapon-detector-ntwwv__v3', {'Gun': 0, 'Sword': 1}),
    'csv': (RF / 'csvtu__gun-ubawg__v1', {'gun': 0}),
    'gow': (RF / 'gowtham-6tn02__movie-yxxm0__v5', {'gun': 0, 'person': I}),
    'roh': (RF / 'rohan-gupta-tk6ru__weapon-eqpbe__v1', {'gun': 0, 'melee': 'MELEE'}),
    'jae': (RF4 / 'jaeyuncho__knife-gabdq__v1', {'0': 'VIOL', '1': 1}),
    'ade': (RF4 / 'adelios-workspace__holding-handgun-and-knife-detection__v11', {'pisau': 1, 'pistol': 0}),
}


def read_boxes(p, filter_lazy=False):
    out = []
    if not Path(p).exists():
        return out
    for line in Path(p).read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        try:
            c = int(float(v[0]))
            f = [float(x) for x in v[1:]]
        except ValueError:
            continue
        if len(f) == 4:
            x1, y1, x2, y2 = f[0] - f[2] / 2, f[1] - f[3] / 2, f[0] + f[2] / 2, f[1] + f[3] / 2
        elif len(f) >= 6 and len(f) % 2 == 0:
            x1, y1, x2, y2 = min(f[0::2]), min(f[1::2]), max(f[0::2]), max(f[1::2])
        else:
            continue
        x1, y1, x2, y2 = max(0., x1), max(0., y1), min(1., x2), min(1., y2)
        w, h = x2 - x1, y2 - y1
        if w < 2e-3 or h < 2e-3:
            continue
        if filter_lazy:
            if (w * h > 0.65) or (w > 0.85 and h > 0.85):
                continue
        out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return out


def rec(name, src, path, boxes, split=None, stem=None, extra=None):
    r = dict(name=name, src=src, path=str(path), boxes=boxes, split=split,
             stem=stem or v1.norm_stem(Path(path).name))
    if extra:
        r.update(extra)
    return r


def collect_domain(stats):
    recs = []
    for lst, sp in (('train_domain.txt', 'train'), ('valid_domain.txt', 'valid'), ('test_domain.txt', 'test_domain')):
        for p in (V2 / lst).read_text().split():
            p = Path(p)
            lp = p.parent.parent / 'labels' / (p.stem + '.txt')
            recs.append(rec(p.name, 'dom', p, read_boxes(lp), sp, extra=dict(label_src=str(lp), is_fixed=True)))
    stats['domain images collected'] = len(recs)
    return recs


def collect_indian_sources(stats):
    recs = []
    import yaml

    # 1. anti_robbing (rob)
    rob_dir = RF / 'innovatus-gv1rg__anti_robbing__v1'
    if rob_dir.exists():
        seen = set()
        for sp in ('train', 'valid', 'test'):
            idir = rob_dir / sp / 'images'
            if not idir.exists():
                continue
            for ip in sorted(idir.iterdir()):
                stem_raw = ip.name.split('_jpg')[0]
                if not stem_raw.isdigit():
                    continue
                frame_no = int(stem_raw)
                if frame_no in ROB_DROP:
                    stats['rob: unlabelled frame dropped'] += 1
                    continue
                key = ip.name.split('.rf.')[0]
                if key in seen:
                    continue
                seen.add(key)
                raw = read_boxes(rob_dir / sp / 'labels' / (ip.stem + '.txt'), filter_lazy=False)
                # Map 0 -> 0 (Gun), 1 -> 1 (Knife)
                out = [(b[0], *b[1:]) for b in raw if b[0] in (0, 1)]
                if not out:
                    continue
                is_tt = frame_no in ROB_TABLETOP_STEMS
                split = 'val_tabletop' if is_tt else 'train'
                recs.append(rec(f'rob_{ip.name}', 'rob', ip, out, split=split,
                                extra=dict(is_fixed=True, is_tabletop=is_tt)))
                if is_tt:
                    stats['tabletop: rob frames'] += 1

    # 2. trail_guard (trg)
    trg_dir = RF / 'sathishcrisp2__trail_guard__v1'
    if trg_dir.exists():
        # Classes: ['Knife', 'Machete', 'Person', 'Pistol', 'Rifle', 'Rod']
        # IDs: 0: Knife, 1: Machete, 2: Person, 3: Pistol, 4: Rifle, 5: Rod
        seen = set()
        for sp in ('train', 'valid', 'test'):
            idir = trg_dir / sp / 'images'
            if not idir.exists():
                continue
            for ip in sorted(idir.iterdir()):
                key = ip.name.split('.rf.')[0]
                if key in seen:
                    continue
                seen.add(key)
                raw = read_boxes(trg_dir / sp / 'labels' / (ip.stem + '.txt'))
                # Guardrail: Drop images containing toy Pistol (3) or Rifle (4)
                if any(b[0] in (3, 4) for b in raw):
                    stats['trg: toy firearm frame dropped'] += 1
                    continue
                out = []
                for b in raw:
                    if b[0] in (0, 1):   # Knife, Machete -> knife (1)
                        out.append((1, *b[1:]))
                    elif b[0] == 5:      # Rod -> handheld_object (2)
                        out.append((2, *b[1:]))
                    # Person (2) ignored
                if not out and raw:
                    continue
                recs.append(rec(f'trg_{ip.name}', 'trg', ip, out, split='train', extra=dict(is_fixed=True)))

    # 3. mohini (mob)
    mob_dir = RF / 'mohini-bamankar-esrz6__my-first-project-uffsi__v1'
    if mob_dir.exists():
        # Classes: ['Axe', 'Chainsaw', 'Fuel Tank', 'Gun', 'Machete']
        # IDs: 0: Axe, 1: Chainsaw, 2: Fuel Tank, 3: Gun, 4: Machete
        seen = set()
        for sp in ('train', 'valid', 'test'):
            idir = mob_dir / sp / 'images'
            if not idir.exists():
                continue
            for ip in sorted(idir.iterdir()):
                key = ip.name.split('.rf.')[0]
                if key in seen:
                    continue
                seen.add(key)
                raw = read_boxes(mob_dir / sp / 'labels' / (ip.stem + '.txt'))
                # Guardrail: Drop toy Gun (3), Fuel Tank (2), Chainsaw (1)
                if any(b[0] in (1, 2, 3) for b in raw):
                    stats['mob: toy gun/fuel tank frame dropped'] += 1
                    continue
                out = []
                for b in raw:
                    if b[0] == 4:      # Machete -> knife (1)
                        out.append((1, *b[1:]))
                    elif b[0] == 0:    # Axe -> handheld_object (2)
                        out.append((2, *b[1:]))
                if not out:
                    continue
                recs.append(rec(f'mob_{ip.name}', 'mob', ip, out, split='train', extra=dict(is_fixed=True)))

    # 4. threatdetect2 (bits_thr)
    thr_dir = RF / 'bits-pilani__threatdetect2__v7'
    if thr_dir.exists():
        seen = set()
        tabletop_bits_count = 0
        for sp in ('valid', 'test', 'train'):
            idir = thr_dir / sp / 'images'
            if not idir.exists():
                continue
            for ip in sorted(idir.iterdir()):
                key = ip.name.split('.rf.')[0]
                if key in seen:
                    continue
                seen.add(key)
                # Filter lazy whole-frame boxes
                raw = read_boxes(thr_dir / sp / 'labels' / (ip.stem + '.txt'), filter_lazy=True)
                if not raw:
                    stats['bits_thr: lazy boxes only dropped'] += 1
                    continue
                # Classes: blade, knife, scissor -> all mapped to knife (1)
                out = [(1, *b[1:]) for b in raw]
                is_tt = False
                if tabletop_bits_count < 15 and sp in ('valid', 'test'):
                    is_tt = True
                    tabletop_bits_count += 1
                    stats['tabletop: bits frames'] += 1
                split = 'val_tabletop' if is_tt else 'train'
                recs.append(rec(f'bitsthr_{ip.name}', 'bitsthr', ip, out, split=split,
                                extra=dict(is_fixed=True, is_tabletop=is_tt)))

    # 5. knife_ijigt (vivek cinema knife stills)
    knf_dir = RF / 'viveks-workspace-85aoc__knife-images-ijigt__v1'
    if knf_dir.exists():
        seen = set()
        for sp in ('train', 'valid', 'test'):
            idir = knf_dir / sp / 'images'
            if not idir.exists():
                continue
            for ip in sorted(idir.iterdir()):
                key = ip.name.split('.rf.')[0]
                if key in seen:
                    continue
                seen.add(key)
                # Classes: 0: armed_person, 1: knife
                raw = read_boxes(knf_dir / sp / 'labels' / (ip.stem + '.txt'), filter_lazy=True)
                out = [(1, *b[1:]) for b in raw if b[0] == 1]
                if not out:
                    continue
                recs.append(rec(f'knfiji_{ip.name}', 'knfiji', ip, out, split='train', extra=dict(is_fixed=True)))

    # 6. gun_twl (gun-knife cinema / real stills)
    twl_dir = RF / 'gun-knife__gun-twllw__v2'
    if twl_dir.exists():
        seen = set()
        for sp in ('train', 'valid', 'test'):
            idir = twl_dir / sp / 'images'
            if not idir.exists():
                continue
            for ip in sorted(idir.iterdir()):
                key = ip.name.split('.rf.')[0]
                if key in seen:
                    continue
                seen.add(key)
                # Classes: 0: gun, 1: knife
                raw = read_boxes(twl_dir / sp / 'labels' / (ip.stem + '.txt'), filter_lazy=True)
                out = [(b[0], *b[1:]) for b in raw if b[0] in (0, 1)]
                if not out:
                    continue
                recs.append(rec(f'guntwl_{ip.name}', 'guntwl', ip, out, split='train', extra=dict(is_fixed=True)))

    # 7. snatch (chain_snatching CCTV backgrounds)
    snatch_dir = RF / 'innovatus-gv1rg__chain_snatching-jk1lw__v1'
    if snatch_dir.exists():
        seen = set()
        for sp in ('train', 'valid', 'test'):
            idir = snatch_dir / sp / 'images'
            if not idir.exists():
                continue
            for ip in sorted(idir.iterdir()):
                key = ip.name.split('.rf.')[0]
                if key in seen:
                    continue
                seen.add(key)
                # Pure negative frame: empty boxes
                recs.append(rec(f'snatch_{ip.name}', 'snatch', ip, [], split='train', extra=dict(is_fixed=True)))
                stats['snatch: cctv negative frame added'] += 1

    stats['total indian records collected'] = len(recs)
    return recs


def collect_public(stats):
    recs = []
    import yaml

    # 1. ODW (from v2 manifest)
    with open(V2 / '_manifest.csv') as fh:
        for r in csv.DictReader(fh):
            if r['source'].startswith('odw') and r['keep'] == '1':
                p = V2 / r['split'] / 'images' / r['name']
                lp = p.parent.parent / 'labels' / (p.stem + '.txt')
                recs.append(rec(r['name'], r['source'], p, read_boxes(lp), extra=dict(v2group=r['group'])))

    # 2. Roboflow exports
    for pre, (d, cmap) in RFSRC.items():
        if not (d / 'data.yaml').exists():
            continue
        names = yaml.safe_load(open(d / 'data.yaml'))['names']
        seen = set()
        for sp in ('train', 'valid', 'test'):
            idir = d / sp / 'images'
            if not idir.exists():
                continue
            for ip in sorted(idir.iterdir()):
                key = ip.name.split('.rf.')[0]
                if key in seen:
                    continue
                seen.add(key)
                if pre == 'aah' and key.lower().startswith('knife_'):
                    stats[f'{pre}: synthetic knife_* render dropped'] += 1
                    continue
                raw = read_boxes(d / sp / 'labels' / (ip.stem + '.txt'), filter_lazy=True)
                out, drop = [], None
                for b in raw:
                    cname = names[b[0]] if b[0] < len(names) else str(b[0])
                    m = cmap.get(cname, None)
                    if m == D:
                        drop = f'class {cname}'
                    elif m == 'MELEE':
                        if key.lower().startswith('knife-'):
                            out.append((1, *b[1:]))
                        else:
                            drop = 'ambiguous melee'
                    elif m == 'VIOL':
                        continue
                    elif m is not None:
                        out.append((m, *b[1:]))
                if pre == 'jae' and not any(b[0] == 1 for b in out):
                    drop = 'no knife box'
                if drop:
                    stats[f'{pre}: dropped ({drop})'] += 1
                    continue
                recs.append(rec(f'{pre}_{ip.name}', pre, ip, out))

    # 3. Open Images V7
    rng = random.Random(SEED)
    hard = []
    if (OI / 'selected.csv').exists():
        for iid, sub, kind, bj in csv.reader(open(OI / 'selected.csv')):
            p = OI / 'images' / f'{iid}.jpg'
            if not p.exists():
                stats[f'oi {kind}: not downloaded'] += 1
                continue
            r = rec(f'oi_{iid}.jpg', 'oi', p, [tuple(b) for b in json.loads(bj)], stem=iid, extra=dict(oi_kind=kind))
            (hard if kind == 'hardneg' else recs).append(r)
        rng.shuffle(hard)
        stats['oi hardneg available'] = len(hard)
        recs += hard[:OI_HARDNEG_CAP]

    stats['total public records collected'] = len(recs)
    return recs


def _hash(path):
    g = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if g is None:
        return path, None
    tr = v1.trim_borders(g)
    return path, (np.packbits(v1.dhash_bits(g)), np.packbits(v1.dhash_bits(cv2.flip(g, 1))),
                  np.packbits(v1.dhash16(tr)), np.packbits(v1.dhash16(cv2.flip(tr, 1))))


def add_hashes(recs):
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / 'hashes.npz'
    cache = {}
    if cf.exists():
        z = np.load(cf)
        for k, a, b, c, d in zip(z['keys'], z['a'], z['b'], z['c'], z['d']):
            cache[str(k)] = (a, b, c, d)
    todo = sorted({r['path'] for r in recs if r['path'] not in cache})
    print(f'[hash] Hashing {len(todo)} new images (out of {len(recs)} total)...', flush=True)
    cv2.setNumThreads(1)
    if todo:
        with Pool(4) as pool:
            for i, (p, h) in enumerate(pool.imap_unordered(_hash, todo, chunksize=32)):
                if h is not None:
                    cache[p] = h
                if (i + 1) % 500 == 0 or (i + 1) == len(todo):
                    print(f'  hashed {i + 1}/{len(todo)}', flush=True)
        keys = sorted(cache)
        np.savez(cf, keys=np.array(keys), a=np.array([cache[k][0] for k in keys]),
                 b=np.array([cache[k][1] for k in keys]), c=np.array([cache[k][2] for k in keys]),
                 d=np.array([cache[k][3] for k in keys]))
    bad = [r['name'] for r in recs if r['path'] not in cache]
    for r in recs:
        if r['path'] in cache:
            r['h'] = cache[r['path']]
    return [r for r in recs if 'h' in r], bad


_POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def close_pairs(A, AF, thr, chunk=128):
    out = []
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk, None, :]
        d = np.minimum(_POP[a ^ A[None]].sum(-1, dtype=np.int16), _POP[a ^ AF[None]].sum(-1, dtype=np.int16))
        ii, jj = np.where(d <= thr)
        ii = ii + s
        k = ii < jj
        out.extend(zip(ii[k].tolist(), jj[k].tolist()))
    return out


def main():
    rng = random.Random(SEED)
    stats = collections.Counter()

    print('[v5 build] Collecting domain, Indian, and public records...', flush=True)
    dom_recs = collect_domain(stats)
    ind_recs = collect_indian_sources(stats)
    pub_recs = collect_public(stats)
    recs = dom_recs + ind_recs + pub_recs
    print(f'[v5 build] Total collected raw records: {len(recs)}', flush=True)

    recs, bad = add_hashes(recs)
    stats['unreadable images dropped'] = len(bad)
    n = len(recs)
    print(f'[v5 build] Total readable records: {n}', flush=True)

    H = np.array([r['h'][0] for r in recs])
    HF = np.array([r['h'][1] for r in recs])
    H2 = np.array([r['h'][2] for r in recs])
    H2F = np.array([r['h'][3] for r in recs])
    pc = _POP[H].sum(1)
    low = (pc < 4) | (pc > 60)

    pairs = []
    for tag, A, AF, thr in (('dhash64', H, HF, T64), ('dhash256', H2, H2F, T256)):
        pp = [(i, j, tag) for i, j in close_pairs(A, AF, thr) if not (low[i] or low[j])]
        print(f'[v5 build] {tag}: {len(pp)} near-duplicate pairs found', flush=True)
        pairs += pp

    # Check leakage against test_domain
    test_dom_indices = {i for i, r in enumerate(recs) if r['split'] == 'test_domain'}
    dropped = set()
    for i, j, tag in pairs:
        if (i in test_dom_indices) != (j in test_dom_indices):
            t_idx, o_idx = (i, j) if i in test_dom_indices else (j, i)
            if o_idx not in dropped:
                dropped.add(o_idx)
                stats[f'LEAK GUARD: dropped {recs[o_idx]["name"]} (near-dup of test_domain {recs[t_idx]["name"]})'] += 1

    # Group public records using Union-Find
    fixed = [r.get('is_fixed', False) for r in recs]
    uf = v1.UF(n)
    first = {}
    ser = collections.defaultdict(list)
    for i, r in enumerate(recs):
        if fixed[i] or i in dropped:
            continue
        for k in [('stem', r['src'][:3], r['stem'])] + ([('v2g', r['v2group'])] if 'v2group' in r else []):
            if k in first:
                stats['union ' + k[0]] += uf.union(first[k], i)
            else:
                first[k] = i
        if r['src'] != 'oi':
            ser[(r['src'], re.sub(r'\d+$', '#', r['stem']))].append(i)
    for k, idx in ser.items():
        if len(idx) >= 5:
            for j in idx[1:]:
                stats['union name-series'] += uf.union(idx[0], j)
    for i, j, tag in pairs:
        if not fixed[i] and not fixed[j] and i not in dropped and j not in dropped:
            stats['union ' + tag] += uf.union(i, j)

    groups = collections.defaultdict(list)
    for i in range(n):
        if not fixed[i] and i not in dropped:
            groups[uf.find(i)].append(i)

    split_of = {}
    small = []
    for g, idx in groups.items():
        if len(idx) > BIG_GROUP:
            split_of[g] = 'train'
        else:
            small.append(dict(root=g,
                              kn=sum(any(b[0] == 1 for b in recs[i]['boxes']) for i in idx),
                              gu=sum(any(b[0] == 0 for b in recs[i]['boxes']) for i in idx),
                              im=len(idx)))

    for key in ('kn', 'gu', 'im'):
        part = [g for g in small if g['root'] not in split_of and (key == 'im' or g[key] > 0)]
        rng.shuffle(part)
        part.sort(key=lambda g: -g[key])
        tot = sum(g[key] for g in part) or 1
        cur = {s: 0.0 for s in TARGET}
        for g in part:
            best = max(TARGET, key=lambda s: (TARGET[s] * tot - cur[s]) / (TARGET[s] * tot))
            split_of[g['root']] = best
            cur[best] += g[key]

    for i, r in enumerate(recs):
        if not fixed[i] and i not in dropped:
            r['split'] = split_of[uf.find(i)]
            r['group'] = f'p{uf.find(i)}'
        elif fixed[i]:
            r['group'] = f'f{r["src"]}'

    keep = [i for i in range(n) if i not in dropped]

    # Background capping in train
    bg = [i for i in keep if recs[i]['split'] == 'train' and not recs[i]['boxes']]
    pos = sum(1 for i in keep if recs[i]['split'] == 'train' and recs[i]['boxes'])
    cap = int(BG_FRAC * pos / (1 - BG_FRAC))
    rng.shuffle(bg)
    keep_bg = {i for i in bg if recs[i]['src'] in ('dom', 'trg', 'snatch')}
    per = collections.Counter()
    for i in bg:
        if len(keep_bg) >= cap:
            break
        if per[recs[i]['src']] >= cap // 3:
            continue
        per[recs[i]['src']] += 1
        keep_bg.add(i)

    drop_bg = set(bg) - keep_bg
    stats['train background available'] = len(bg)
    stats['train background kept'] = len(keep_bg)
    stats['train background by source'] = json.dumps(collections.Counter(recs[i]['src'] for i in keep_bg))

    keep = [i for i in keep if i not in drop_bg]
    keepset = set(keep)

    # Verification checks
    checks = collections.Counter()
    for i, j, tag in pairs:
        if i in keepset and j in keepset and recs[i]['split'] != recs[j]['split']:
            ff = 'fixed-fixed' if fixed[i] and fixed[j] else ('public-public' if not fixed[i] and not fixed[j] else 'fixed-public')
            checks[f'cross-split {tag} pairs ({ff})'] += 1

    v2test = sorted(Path(p).name for p in (V2 / 'test_domain.txt').read_text().split())
    mine_test = sorted(recs[i]['name'] for i in keep if recs[i]['split'] == 'test_domain')
    checks['test_domain mismatch with v2'] = int(v2test != mine_test)

    v2val = sorted(Path(p).name for p in (V2 / 'valid_domain.txt').read_text().split())
    mine_val = sorted(recs[i]['name'] for i in keep if recs[i]['split'] == 'valid' and recs[i]['src'] == 'dom')
    checks['valid_domain mismatch with v2'] = int(v2val != mine_val)

    tabletop_count = sum(1 for i in keep if recs[i]['split'] == 'val_tabletop')
    print(f'[v5 build] Dedicated val_tabletop images: {tabletop_count}', flush=True)

    print('[v5 build] Integrity checks:\n', json.dumps(checks, indent=2))
    if checks['test_domain mismatch with v2'] != 0:
        print('[error] test_domain was modified or corrupted!')
        sys.exit(1)

    # Output generation
    for s in ('train', 'valid', 'test_domain', 'test_public', 'val_tabletop'):
        for sub in ('images', 'labels'):
            d = OUT / s / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    lists = collections.defaultdict(list)
    man = []

    for i in keep:
        r = recs[i]
        s = r['split']
        name = r['name']
        if os.path.splitext(name)[1].lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            name += '.jpg'
        ip = OUT / s / 'images' / name
        os.symlink(r['path'], ip)
        lp = OUT / s / 'labels' / (os.path.splitext(name)[0] + '.txt')
        if r['src'] == 'dom' and 'label_src' in r:
            shutil.copyfile(r['label_src'], lp)
        else:
            lp.write_text(''.join(f'{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n' for b in r['boxes']))
        lists[s].append(str(ip))
        if s == 'valid' and r['src'] == 'dom':
            lists['valid_domain'].append(str(ip))
        man.append([name, r['src'], s, r['group'], r['stem'],
                    *[sum(b[0] == c for b in r['boxes']) for c in range(3)], r['path']])

    for k, v in lists.items():
        (OUT / f'{k}.txt').write_text('\n'.join(sorted(v)) + '\n')

    with open(OUT / '_manifest.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['name', 'source', 'split', 'group', 'stem', 'n_gun', 'n_knife', 'n_handheld', 'src_path'])
        w.writerows(man)

    ya = lambda tr, va, te, c: (f'# {c}\npath: {OUT}\ntrain: {tr}\nval: {va}\ntest: {te}\nnc: 3\n'
                                f"names: ['gun', 'knife', 'handheld_object']\n")
    (OUT / 'data_hybrid.yaml').write_text(ya('train.txt', 'valid_domain.txt', 'test_domain.txt',
                                            'v5_indian_hybrid: hybrid train, locked domain val, locked domain test'))
    (OUT / 'data_tabletop.yaml').write_text(ya('train.txt', 'val_tabletop.txt', 'val_tabletop.txt',
                                              'v5_indian_hybrid: tabletop benchmark'))
    (OUT / 'data_public.yaml').write_text(ya('train.txt', 'valid_domain.txt', 'test_public.txt',
                                            'v5_indian_hybrid: held-out public test'))

    (OUT / '_BUILD_DONE').write_text('DONE\n')
    json.dump(dict(stats=stats, checks=checks, n_lists={k: len(v) for k, v in lists.items()}),
              open(OUT / '_build_meta.json', 'w'), indent=2)

    print('\n[v5 build] Completed successfully!')
    print('Split counts:', {k: len(v) for k, v in lists.items()})
    print('Key stats:', json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()

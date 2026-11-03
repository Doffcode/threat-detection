#!/usr/bin/env python
"""Build work/data/v4_big (run: .venv/bin/python -I scripts/data_v4/build.py).

Fixed-split records: dom (data/v2 domain lists) + ind (indian_v3 nov4 lists: rob/trg/mob).
Public records (re-split by group): odw (v2 keep=1), Roboflow exports, Open Images.
Any public image near-duplicate (dHash64<=4 or trimmed dHash256<=32, h-flip) of ANY fixed image is dropped.
Public grouping: stem, v2 group, hash pairs, numeric name series (>=5 members). Groups >30 imgs -> train;
others greedy ~90/5/5 (train/valid/test_public) stratified knife > gun > images.
"""
import collections
import csv
import json
import os
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

WORK = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work')
sys.path.insert(0, str(WORK / 'scripts' / 'data'))
import split as v1  # noqa: E402

V2, IV3 = WORK / 'data' / 'v2', WORK / 'data' / 'indian_v3'
OUT = WORK / 'data' / 'v4_big'
RF, RF4, OI = WORK / 'downloads' / 'roboflow', WORK / 'downloads' / 'roboflow_v4', WORK / 'downloads' / 'openimages'
CACHE = WORK / 'scripts' / 'data_v4' / '_cache'
NAMES = ['gun', 'knife', 'handheld_object']
SEED = 2026
T64, T256 = 4, 32
OI_HARDNEG_CAP = 2500
BG_FRAC = 0.10
BIG_GROUP = 30
TARGET = {'train': 0.80, 'valid': 0.09, 'test_public': 0.11}
D = 'DROP'  # drop whole image if class present
I = None    # ignore box

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


def read_boxes(p):
    out = []
    if not Path(p).exists():
        return out
    for line in Path(p).read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c = int(float(v[0]))
        f = [float(x) for x in v[1:]]
        if len(f) == 4:
            x1, y1, x2, y2 = f[0] - f[2] / 2, f[1] - f[3] / 2, f[0] + f[2] / 2, f[1] + f[3] / 2
        elif len(f) >= 6 and len(f) % 2 == 0:
            x1, y1, x2, y2 = min(f[0::2]), min(f[1::2]), max(f[0::2]), max(f[1::2])
        else:
            continue
        x1, y1, x2, y2 = max(0., x1), max(0., y1), min(1., x2), min(1., y2)
        if x2 - x1 < 2e-3 or y2 - y1 < 2e-3:
            continue
        out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out


def rec(name, src, path, boxes, split=None, stem=None, extra=None):
    r = dict(name=name, src=src, path=str(path), boxes=boxes, split=split, stem=stem or v1.norm_stem(Path(path).name))
    if extra:
        r.update(extra)
    return r


def collect(stats):
    recs = []
    # ---- domain (fixed)
    for lst, sp in (('train_domain.txt', 'train'), ('valid_domain.txt', 'valid'), ('test_domain.txt', 'test_domain')):
        for p in (V2 / lst).read_text().split():
            p = Path(p)
            lp = p.parent.parent / 'labels' / (p.stem + '.txt')
            recs.append(rec(p.name, 'dom', p, read_boxes(lp), sp, extra=dict(label_src=str(lp))))
    # ---- indian_v3 nov4 non-domain (fixed)
    for lst, sp in (('train_nov4.txt', 'train'), ('valid_nov4.txt', 'valid'), ('test_new_nov4.txt', 'test_public')):
        for p in (IV3 / lst).read_text().split():
            p = Path(p)
            if p.name.startswith('dom_'):
                continue
            lp = p.parent.parent / 'labels' / (p.stem + '.txt')
            recs.append(rec(p.name, p.name.split('_')[0], p, read_boxes(lp), sp))
    # ---- ODW (public)
    with open(V2 / '_manifest.csv') as fh:
        for r in csv.DictReader(fh):
            if r['source'].startswith('odw') and r['keep'] == '1':
                p = V2 / r['split'] / 'images' / r['name']
                lp = p.parent.parent / 'labels' / (p.stem + '.txt')
                recs.append(rec(r['name'], r['source'], p, read_boxes(lp), extra=dict(v2group=r['group'])))
    # ---- Roboflow (public), one image per .rf. stem
    import yaml
    for pre, (d, cmap) in RFSRC.items():
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
                raw = read_boxes(d / sp / 'labels' / (ip.stem + '.txt'))
                out, drop = [], None
                for b in raw:
                    m = cmap[names[b[0]]]
                    if m == D:
                        drop = f'class {names[b[0]]}'
                    elif m == 'MELEE':
                        if key.lower().startswith('knife-'):
                            out.append((1, *b[1:]))
                        else:
                            drop = 'ambiguous melee (stick/pen/knife)'
                    elif m == 'VIOL':
                        continue
                    elif m is not None:
                        out.append((m, *b[1:]))
                if pre == 'jae' and not any(b[0] == 1 for b in out):
                    drop = 'no knife box (violence-only frame)'
                if pre == 'har' and not out and raw:
                    pass  # fall-detection frame: background candidate
                if drop:
                    stats[f'{pre}: dropped ({drop})'] += 1
                    continue
                recs.append(rec(f'{pre}_{ip.name}', pre, ip, out))
    # ---- Open Images (public)
    rng = random.Random(SEED)
    hard = []
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
    print('hashing', len(todo), 'of', len(recs), flush=True)
    cv2.setNumThreads(1)
    with Pool(4) as pool:
        for i, (p, h) in enumerate(pool.imap_unordered(_hash, todo, chunksize=32)):
            if h is not None:
                cache[p] = h
            if i % 2000 == 0:
                print('  hashed', i, flush=True)
    keys = sorted(cache)
    np.savez(cf, keys=np.array(keys), a=np.array([cache[k][0] for k in keys]), b=np.array([cache[k][1] for k in keys]),
             c=np.array([cache[k][2] for k in keys]), d=np.array([cache[k][3] for k in keys]))
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
    recs = collect(stats)
    print('records', len(recs), flush=True)
    recs, bad = add_hashes(recs)
    stats['unreadable'] = len(bad)
    n = len(recs)
    H = np.array([r['h'][0] for r in recs]); HF = np.array([r['h'][1] for r in recs])
    H2 = np.array([r['h'][2] for r in recs]); H2F = np.array([r['h'][3] for r in recs])
    pc = _POP[H].sum(1)
    low = (pc < 4) | (pc > 60)
    pairs = []
    for tag, A, AF, thr in (('dhash64', H, HF, T64), ('dhash256', H2, H2F, T256)):
        pp = [(i, j, tag) for i, j in close_pairs(A, AF, thr) if not (low[i] or low[j])]
        print(tag, len(pp), 'pairs', flush=True)
        pairs += pp
    fixed = [r['split'] is not None for r in recs]
    # ---- drop public near-dups of fixed (domain / Indian) images
    dropped = set()
    for i, j, tag in pairs:
        if fixed[i] != fixed[j]:
            f, p = (i, j) if fixed[i] else (j, i)
            if p not in dropped:
                dropped.add(p)
                stats[f'public dropped: near-dup of {recs[f]["src"]} {recs[f]["split"]} ({recs[p]["src"]})'] += 1
    # ---- group public
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
            small.append(dict(root=g, kn=sum(any(b[0] == 1 for b in recs[i]['boxes']) for i in idx),
                              gu=sum(any(b[0] == 0 for b in recs[i]['boxes']) for i in idx), im=len(idx)))
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
    # ---- background cap in train
    bg = [i for i in keep if recs[i]['split'] == 'train' and not recs[i]['boxes']]
    pos = sum(1 for i in keep if recs[i]['split'] == 'train' and recs[i]['boxes'])
    cap = int(BG_FRAC * pos / (1 - BG_FRAC))
    rng.shuffle(bg)
    keep_bg = {i for i in bg if recs[i]['src'] in ('dom', 'trg')}
    per = collections.Counter()
    for i in bg:
        if len(keep_bg) >= cap:
            break
        if per[recs[i]['src']] >= cap // 3:
            continue
        per[recs[i]['src']] += 1
        keep_bg.add(i)
    stats['train background by source'] = json.dumps(collections.Counter(recs[i]['src'] for i in keep_bg))
    drop_bg = set(bg) - keep_bg
    stats['train background available'] = len(bg)
    stats['train background kept'] = len(keep_bg)
    keep = [i for i in keep if i not in drop_bg]
    keepset = set(keep)
    # ---- leak checks
    checks = collections.Counter()
    for i, j, tag in pairs:
        if i in keepset and j in keepset and recs[i]['split'] != recs[j]['split']:
            ff = 'fixed-fixed' if fixed[i] and fixed[j] else ('public-public' if not fixed[i] and not fixed[j] else 'fixed-public')
            checks[f'cross-split {tag} pairs ({ff})'] += 1
    for t in ('dhash64', 'dhash256'):
        for ff in ('fixed-public', 'public-public', 'fixed-fixed'):
            checks.setdefault(f'cross-split {t} pairs ({ff})', 0)
    gs = collections.defaultdict(set)
    for i in keep:
        if not fixed[i]:
            gs[recs[i]['group']].add(recs[i]['split'])
    checks['public groups in >1 split'] = sum(len(v) > 1 for v in gs.values())
    st = collections.defaultdict(set)
    for i in keep:
        st[(recs[i]['src'][:3], recs[i]['stem'])].add(recs[i]['split'])
    checks['(source, stem) in >1 split'] = sum(len(v) > 1 for v in st.values())
    v2test = sorted(Path(p).name for p in (V2 / 'test_domain.txt').read_text().split())
    mine = sorted(recs[i]['name'] for i in keep if recs[i]['split'] == 'test_domain')
    checks['test_domain != v2 test_domain'] = int(v2test != mine)
    v2val = sorted(Path(p).name for p in (V2 / 'valid_domain.txt').read_text().split())
    checks['valid_domain != v2 valid_domain'] = int(v2val != sorted(recs[i]['name'] for i in keep if recs[i]['split'] == 'valid' and recs[i]['src'] == 'dom'))
    print(json.dumps(checks, indent=1))
    # ---- write
    import shutil
    for s in ('train', 'valid', 'test_domain', 'test_public'):
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
        if r['src'] == 'dom':
            shutil.copyfile(r['label_src'], lp)  # byte-identical domain labels
        else:
            lp.write_text(''.join(f'{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n' for b in r['boxes']))
        lists[s].append(str(ip))
        if s == 'valid' and r['src'] == 'dom':
            lists['valid_domain'].append(str(ip))
        if s == 'train' and fixed[i]:
            lists['train_fixed'].append(str(ip))
        if s == 'train' and not fixed[i]:
            lists['train_public'].append(str(ip))
        man.append([name, r['src'], s, r['group'], r['stem'], *[sum(b[0] == c for b in r['boxes']) for c in range(3)], r['path']])
    pubtr = list(lists['train_public'])
    rng.shuffle(pubtr)
    lists['train_stage2'] = lists['train_fixed'] + sorted(pubtr[:int(0.3 * len(pubtr))])
    for k, v in lists.items():
        (OUT / f'{k}.txt').write_text('\n'.join(sorted(v)) + '\n')
    with open(OUT / '_manifest.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['name', 'source', 'split', 'group', 'stem', 'n_gun', 'n_knife', 'n_handheld', 'src_path'])
        w.writerows(man)
    ya = lambda tr, va, te, c: (f'# {c}\npath: {OUT}\ntrain: {tr}\nval: {va}\ntest: {te}\nnc: 3\n'
                                f"names: ['gun', 'knife', 'handheld_object']\n")
    (OUT / 'data_big.yaml').write_text(ya('train.txt', 'valid_domain.txt', 'test_domain.txt', 'v4_big: all train, domain val, domain test'))
    (OUT / 'data_big_pubtest.yaml').write_text(ya('train.txt', 'valid_domain.txt', 'test_public.txt', 'v4_big: all train, domain val, held-out public test'))
    (OUT / 'data_stage2.yaml').write_text(ya('train_stage2.txt', 'valid_domain.txt', 'test_domain.txt', 'v4_big stage2: domain+Indian train + 30% public train'))
    json.dump(dict(stats=stats, checks=checks, n_lists={k: len(v) for k, v in lists.items()}),
              open(OUT / '_build_meta.json', 'w'), indent=1)
    print(json.dumps(stats, indent=1))
    print({k: len(v) for k, v in lists.items()})
    bad_checks = {k: v for k, v in checks.items() if v}
    if bad_checks:
        print('LEAK CHECK FAILURES', bad_checks)
        sys.exit(2)


if __name__ == '__main__':
    main()

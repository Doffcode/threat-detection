#!/usr/bin/env python
"""Build work/data/indian_v3: Indian-only, leak-free gun/knife dataset.

Run with:  .venv/bin/python -I scripts/indian_v3/build.py   (downloads are untrusted -> isolated mode)

Sources
  dom  work/data/v2 domain images (dom_*), labels/splits FIXED (incl. v2 label fixes); hardlinked.
  v4   viveks-workspace-85aoc/weapon-detection-two v4: armed person drop, gun->0, knife->1.
       Only genuinely NEW source photos are kept: any v4 image whose group (normalised stem, dHash64/256,
       screenshot session) touches a domain image or a raw v1 image (IntProj / roboflow_v1, reference only)
       is dropped.
  rob  innovatus-gv1rg/anti_robbing v1: Gun->0, Knife->1; frames with a visible UNLABELLED weapon dropped
       (manual review with contact sheets, see ROB_DROP below).
  trg  sathishcrisp2/trail_guard v1: Knife, Machete->1; Pistol, Rifle->0; Person, Rod dropped (images kept).
  mob  mohini-bamankar-esrz6/my-first-project-uffsi v1: Machete->1 only; image kept only if it has >=1 machete
       and NO (toy) Gun box.

Grouping (union-find): normalised stem (per stem namespace), v2 domain group, dHash64 <= 4 OR trimmed
dHash256 <= 32 (incl. h-flip; low-information hashes ignored), screenshot sessions (<= 30 s), video-frame series
(rob/mob: frame numbers within 2; trg: timestamps within 30 s). Groups touching the domain inherit the domain
split; new groups are split ~80/10/10 per source, stratified knife > gun > background.
"""
import collections
import csv
import datetime
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

WORK = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work')
sys.path.insert(0, str(WORK / 'scripts' / 'data'))
import split as v1  # noqa: E402  norm_stem, dhash, trim_borders, shot_time, UF

V2 = WORK / 'data' / 'v2'
OUT = WORK / 'data' / 'indian_v3'
CACHE = WORK / 'scripts' / 'indian_v3' / '_cache'
RF = WORK / 'downloads' / 'roboflow'
SPLITS = ['train', 'valid', 'test']
NAMES = ['gun', 'knife', 'handheld_object']
T64, T256 = 4, 32
SHOT_GAP = 30
SEED = 2026
TARGET = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
BIG_FRAC = 0.3  # a new group holding > 30% of its source goes to train

SRC = {
    'v4': dict(dir=RF / 'viveks-workspace-85aoc__weapon-detection-two__v4', cmap={1: 0, 2: 1},
               url='https://universe.roboflow.com/viveks-workspace-85aoc/weapon-detection-two/dataset/4'),
    'rob': dict(dir=RF / 'innovatus-gv1rg__anti_robbing__v1', cmap={0: 0, 1: 1},
                url='https://universe.roboflow.com/innovatus-gv1rg/anti_robbing/dataset/1'),
    'trg': dict(dir=RF / 'sathishcrisp2__trail_guard__v1', cmap={0: 1, 1: 1, 3: 0, 4: 0},
                url='https://universe.roboflow.com/sathishcrisp2/trail_guard/dataset/1'),
    'mob': dict(dir=RF / 'mohini-bamankar-esrz6__my-first-project-uffsi__v1', cmap={4: 1},
                url='https://universe.roboflow.com/mohini-bamankar-esrz6/my-first-project-uffsi/dataset/1'),
}
REF = [WORK / 'raw' / 'IntProj_Weapon_Dataset_annot', WORK / 'raw' / 'roboflow_v1']

# anti_robbing frames with a visible unlabelled weapon (manual contact-sheet review; frame number = stem)
ROB_DROP = {
    'pistol lying on desk near lower-left keyboard area': list(range(44, 73)),
    'revolver lying on desk (brown grip)': list(range(235, 264)) + [290],
    'sword lying on back desk (gold hilt, faint blade)': [n for n in range(565, 634) if n not in (604, 605)],
}

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
NUM_RE = re.compile(r'^(\d+)$')
TS_RE = re.compile(r'^(\d{8})_(\d{6})')


def read_labels(path):
    """-> list of (cls, cx, cy, w, h) with polygons converted to clipped bbox; raw class ids."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        v = [float(x) for x in p[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif len(v) >= 6 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        x1, y1, x2, y2 = max(0., x1), max(0., y1), min(1., x2), min(1., y2)
        if x2 - x1 < 2e-3 or y2 - y1 < 2e-3:
            continue
        out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out


def remap(boxes, cmap):
    return [(cmap[b[0]], *b[1:]) for b in boxes if b[0] in cmap]


def hashes(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    tr = v1.trim_borders(g)
    return (v1.dhash_bits(g), v1.dhash_bits(cv2.flip(g, 1)), v1.dhash16(tr), v1.dhash16(cv2.flip(tr, 1)))


# ----------------------------------------------------------------------------------------------- records
def collect():
    recs = []
    # domain (fixed)
    man = {}
    with open(V2 / '_manifest.csv') as fh:
        for r in csv.DictReader(fh):
            if r['source'] == 'dom':
                man[r['name']] = r['group']
    for sp in SPLITS:
        for line in (V2 / f'{sp}_domain.txt').read_text().split():
            p = Path(line)
            name = p.name
            base = name[len('dom_'):]
            if base.startswith('rv1_'):
                base = base[4:]
            recs.append(dict(src='dom', split=sp, path=p, name=name, ns='v1', stem=v1.norm_stem(base),
                             boxes=read_labels(p.parent.parent / 'labels' / (p.stem + '.txt')),
                             v2group=man[name]))
    # raw v1 reference (never written)
    for root in REF:
        for sp in SPLITS:
            for f in sorted((root / sp / 'images').iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    recs.append(dict(src='ref', split=None, path=f, name=f'{root.name}/{sp}/{f.name}', ns='v1',
                                     stem=v1.norm_stem(f.name), boxes=[]))
    # new sources
    dropped = collections.defaultdict(list)
    rob_drop = {n: why for why, ns in ROB_DROP.items() for n in ns}
    for key, cfg in SRC.items():
        for sp in SPLITS:
            idir = cfg['dir'] / sp / 'images'
            if not idir.exists():
                continue
            for f in sorted(idir.iterdir()):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                raw = read_labels(cfg['dir'] / sp / 'labels' / (f.stem + '.txt'))
                stem = v1.norm_stem(f.name)
                rec = dict(src=key, split=None, src_split=sp, path=f, name=f'{key}_{f.name}',
                           ns='v1' if key == 'v4' else key, stem=stem, boxes=remap(raw, cfg['cmap']), raw=raw)
                if key == 'rob':
                    n = int(stem.split('_')[0]) if stem.split('_')[0].isdigit() else None
                    rec['frame'] = n
                    if n in rob_drop:
                        dropped['rob'].append((f.name, rob_drop[n]))
                        continue
                if key == 'mob':
                    cls = {b[0] for b in raw}
                    rec['frame'] = int(stem) if stem.isdigit() else None
                    if 4 not in cls:
                        dropped['mob'].append((f.name, 'no machete' + (' (toy gun only/with other)' if 3 in cls else '')))
                        continue
                    if 3 in cls:
                        dropped['mob'].append((f.name, 'machete + unlabelled-after-remap toy Gun'))
                        continue
                if key == 'trg':
                    m = TS_RE.match(stem)
                    rec['t'] = (datetime.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').timestamp()
                                if m else None)
                recs.append(rec)
    return recs, dropped


def add_hashes(recs):
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / 'hashes.npz'
    cache = {}
    if cf.exists():
        z = np.load(cf, allow_pickle=False)
        for k, a, b, c, d in zip(z['keys'], z['h64'], z['h64f'], z['h256'], z['h256f']):
            cache[str(k)] = (a, b, c, d)
    bad = []
    for i, r in enumerate(recs):
        k = f"{r['path']}|{os.path.getsize(r['path'])}"
        if k not in cache:
            h = hashes(r['path'])
            if h is None:
                bad.append(r['name'])
                continue
            cache[k] = h
        r['h'] = cache[k]
        if i % 1000 == 0:
            print(f'  hashed {i}/{len(recs)}', flush=True)
    keys = sorted(cache)
    np.savez_compressed(cf, keys=np.array(keys), h64=np.array([cache[k][0] for k in keys]),
                        h64f=np.array([cache[k][1] for k in keys]), h256=np.array([cache[k][2] for k in keys]),
                        h256f=np.array([cache[k][3] for k in keys]))
    return [r for r in recs if 'h' in r], bad


# ----------------------------------------------------------------------------------------------- hashing
_POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def close_pairs(A, AF, thr, B=None, chunk=256):
    """pairs (i,j) with min(d(A_i,B_j), d(A_i,flip B_j)) <= thr; if B is None, i<j within A."""
    Ap = np.packbits(A.astype(bool), axis=1)
    same = B is None
    Bp = Ap if same else np.packbits(B[0].astype(bool), axis=1)
    BFp = np.packbits((AF if same else B[1]).astype(bool), axis=1)
    out = []
    for s in range(0, len(Ap), chunk):
        a = Ap[s:s + chunk, None, :]
        d = np.minimum(_POP[a ^ Bp[None]].sum(-1, dtype=np.int16), _POP[a ^ BFp[None]].sum(-1, dtype=np.int16))
        ii, jj = np.where(d <= thr)
        ii = ii + s
        if same:
            k = ii < jj
            ii, jj = ii[k], jj[k]
        out.extend(zip(ii.tolist(), jj.tolist()))
    return out


def low_info(h64):
    pc = h64.sum(1)
    return (pc < 4) | (pc > 60)


def hash_arrays(recs):
    H = np.array([r['h'][0] for r in recs], dtype=np.uint8)
    HF = np.array([r['h'][1] for r in recs], dtype=np.uint8)
    H2 = np.array([r['h'][2] for r in recs], dtype=np.uint8)
    H2F = np.array([r['h'][3] for r in recs], dtype=np.uint8)
    return H, HF, H2, H2F


def group(recs):
    n = len(recs)
    uf = v1.UF(n)
    why = collections.Counter()
    first = {}
    for i, r in enumerate(recs):  # stem (per namespace)
        k = (r['ns'], r['stem'])
        if k in first:
            why['stem'] += uf.union(first[k], i)
        else:
            first[k] = i
    first = {}
    for i, r in enumerate(recs):  # v2 domain groups
        if r['src'] == 'dom':
            k = r['v2group']
            if k in first:
                why['v2group'] += uf.union(first[k], i)
            else:
                first[k] = i
    H, HF, H2, H2F = hash_arrays(recs)
    li = low_info(H)
    for tag, A, AF, thr in (('dhash64', H, HF, T64), ('dhash256', H2, H2F, T256)):
        for i, j in close_pairs(A, AF, thr):
            if not (li[i] or li[j]):
                why[tag] += uf.union(i, j)
    # screenshot sessions (v1 namespace stems)
    shots = sorted({(v1.shot_time(r['stem']), r['stem']) for r in recs if r['ns'] == 'v1' and v1.shot_time(r['stem'])})
    fi = {}
    for i, r in enumerate(recs):
        if r['ns'] == 'v1':
            fi.setdefault(r['stem'], i)
    for (t0, s0), (t1, s1) in zip(shots, shots[1:]):
        if t1 - t0 <= SHOT_GAP:
            why['screenshot session'] += uf.union(fi[s0], fi[s1])
    # video-frame series
    for src, key, gap in (('rob', 'frame', 2), ('mob', 'frame', 2), ('trg', 't', SHOT_GAP)):
        seq = sorted((r[key], i) for i, r in enumerate(recs) if r['src'] == src and r.get(key) is not None)
        for (a, i), (b, j) in zip(seq, seq[1:]):
            if b - a <= gap:
                why[f'{src} series'] += uf.union(i, j)
    return uf, why


# ----------------------------------------------------------------------------------------------- split
def assign(groups, key, rng, split_of):
    gl = list(groups)
    rng.shuffle(gl)
    gl.sort(key=lambda g: -g[key])
    total = sum(g[key] for g in gl) or 1
    cur = {s: 0.0 for s in SPLITS}
    for g in gl:
        best = max(SPLITS, key=lambda s: (TARGET[s] * total - cur[s]) / (TARGET[s] * total))
        split_of[g['root']] = best
        cur[best] += g[key]


def main():
    rng = random.Random(SEED)
    print('collecting records ...', flush=True)
    recs, dropped = collect()
    recs, bad = add_hashes(recs)
    if bad:
        print('unreadable (dropped):', bad)
    print(collections.Counter(r['src'] for r in recs))

    # iterate: drop non-domain images that bridge different domain splits, until stable
    bridged = []
    while True:
        uf, why = group(recs)
        comp = collections.defaultdict(list)
        for i in range(len(recs)):
            comp[uf.find(i)].append(i)
        bad_idx = set()
        for idx in comp.values():
            dsp = {recs[i]['split'] for i in idx if recs[i]['src'] == 'dom'}
            if len(dsp) > 1:
                nd = [i for i in idx if recs[i]['src'] != 'dom']
                assert nd, 'domain itself has a cross-split group'
                bad_idx.update(nd)
        if not bad_idx:
            break
        bridged += [recs[i]['name'] for i in bad_idx]
        recs = [r for i, r in enumerate(recs) if i not in bad_idx]
    print('merges:', dict(why), '| bridging images dropped:', len(bridged))

    # --- v4: existing vs new; label disagreement for existing photos
    v4_all = [r for r in recs if r['src'] == 'v4']
    v4_stems_all = {r['stem'] for r in v4_all}
    keep = [True] * len(recs)
    v4_existing_groups = collections.defaultdict(list)
    v4_match_kind = collections.Counter()
    for root, idx in comp.items():
        srcs = {recs[i]['src'] for i in idx}
        if 'v4' in srcs and ({'dom', 'ref'} & srcs):
            for i in idx:
                if recs[i]['src'] == 'v4':
                    keep[i] = False
                    v4_existing_groups[root].append(i)
            v4_match_kind['groups with domain' if 'dom' in srcs else 'groups with raw-v1 only (not in domain)'] += 1
        if 'ref' in srcs and srcs - {'ref', 'dom', 'v4'}:
            v4_match_kind['WARNING new-source image matches raw v1'] += 1
    # disagreement: per photo stem present in both v4 and domain
    dom_by_stem = collections.defaultdict(list)
    v4_by_stem = collections.defaultdict(list)
    for r in recs:
        if r['src'] == 'dom':
            dom_by_stem[r['stem']].append(r)
        elif r['src'] == 'v4':
            v4_by_stem[r['stem']].append(r)

    def sig(rs):
        pres = {c for r in rs for c, *_ in r['boxes']}
        mx = tuple(max(sum(1 for b in r['boxes'] if b[0] == c) for r in rs) for c in (0, 1))
        return pres, mx
    shared = sorted(set(dom_by_stem) & set(v4_by_stem))
    dis_pres, dis_cnt, dis_examples = [], [], []
    for s in shared:
        (pd, md), (pv, mv) = sig(dom_by_stem[s]), sig(v4_by_stem[s])
        if pd != pv:
            dis_pres.append(s)
            dis_examples.append((s, f'ours gun/knife max {md}', f'v4 {mv}'))
        elif md != mv:
            dis_cnt.append(s)
    # same check by group for hash-only matches (v4 stem not in domain stems but group has domain)
    grp_dis = 0
    grp_n = 0
    for root, idx4 in v4_existing_groups.items():
        stems4 = {recs[i]['stem'] for i in idx4}
        if stems4 & set(dom_by_stem):
            continue
        dom_rs = [recs[i] for i in comp[root] if recs[i]['src'] == 'dom']
        if not dom_rs:
            continue
        grp_n += 1
        if sig(dom_rs)[0] != sig([recs[i] for i in idx4])[0]:
            grp_dis += 1

    recs_out = [r for i, r in enumerate(recs) if keep[i] and r['src'] != 'ref']
    v4_new = [r for r in recs_out if r['src'] == 'v4']
    v4_new_stems = sorted({r['stem'] for r in v4_new})

    # --- regroup the output set (ref/v4-existing removed) and split
    uf, why_out = group(recs_out)
    comp = collections.defaultdict(list)
    for i in range(len(recs_out)):
        comp[uf.find(i)].append(i)
    split_of = {}
    new_groups = []
    for root, idx in comp.items():
        dsp = {recs_out[i]['split'] for i in idx if recs_out[i]['src'] == 'dom'}
        assert len(dsp) <= 1
        if dsp:
            split_of[root] = dsp.pop()
            continue
        cnt = collections.Counter(b[0] for i in idx for b in recs_out[i]['boxes'])
        srcs = collections.Counter(recs_out[i]['src'] for i in idx)
        new_groups.append(dict(root=root, knife=cnt[1], gun=cnt[0], nimg=len(idx), src=srcs.most_common(1)[0][0]))
    non_dom_in_dom_groups = [recs_out[i]['name'] for root, idx in comp.items() if root in split_of
                             for i in idx if recs_out[i]['src'] != 'dom']
    forced_train = []
    for src in SRC:
        gs = [g for g in new_groups if g['src'] == src]
        n_src = sum(g['nimg'] for g in gs) or 1
        big = [g for g in gs if g['nimg'] > BIG_FRAC * n_src]  # cannot be held out representatively
        for g in big:
            split_of[g['root']] = 'train'
            forced_train.append((src, g['nimg'], n_src))
        gs = [g for g in gs if g not in big]
        assign([g for g in gs if g['knife'] > 0], 'knife', rng, split_of)
        assign([g for g in gs if g['knife'] == 0 and g['gun'] > 0], 'gun', rng, split_of)
        assign([g for g in gs if g['knife'] == 0 and g['gun'] == 0], 'nimg', rng, split_of)
    for i, r in enumerate(recs_out):
        r['gid'] = int(uf.find(i))
        if r['src'] == 'dom':
            assert split_of[uf.find(i)] == r['split'], r['name']
        r['split'] = split_of[uf.find(i)]

    for src in SRC:
        gs = sorted([g for g in new_groups if g['src'] == src], key=lambda g: -g['nimg'])
        tot = collections.Counter()
        for r in recs_out:
            if r['src'] == src:
                tot[(r['split'], 'img')] += 1
                for b in r['boxes']:
                    tot[(r['split'], NAMES[b[0]])] += 1
        print(f'  {src}: {len(gs)} groups, largest {[g["nimg"] for g in gs[:8]]}; ' +
              ' '.join(f'{s}:{tot[(s, "img")]}img/{tot[(s, "gun")]}g/{tot[(s, "knife")]}k' for s in SPLITS))
    print('  forced-to-train groups (src, size, src total):', forced_train)
    if os.environ.get('DRY'):
        print('DRY run: stopping before write')
        return

    # --- verification
    checks = {}
    gsp = collections.defaultdict(set)
    for r in recs_out:
        gsp[r['gid']].add(r['split'])
    checks['groups in >1 split'] = sum(len(v) > 1 for v in gsp.values())
    ssp = collections.defaultdict(set)
    for r in recs_out:
        ssp[(r['ns'], r['stem'])].add(r['split'])
    checks['normalised stems in >1 split'] = sum(len(v) > 1 for v in ssp.values())
    H, HF, H2, H2F = hash_arrays(recs_out)
    li = low_info(H)
    sp = np.array([r['split'] for r in recs_out])
    for tag, A, AF, thr in (('dhash64', H, HF, T64), ('dhash256', H2, H2F, T256)):
        pr = [(i, j) for i, j in close_pairs(A, AF, thr) if sp[i] != sp[j] and not (li[i] or li[j])]
        checks[f'cross-split {tag} pairs (<= {thr}) all sources'] = len(pr)
        checks[f'cross-split {tag} pairs (<= {thr}) new vs domain'] = sum(
            (recs_out[i]['src'] == 'dom') != (recs_out[j]['src'] == 'dom') for i, j in pr)
    # new-source images vs ALL raw v1 reference images (any split) -> must be 0 for kept non-domain images
    ref = [r for r in recs if r['src'] == 'ref']
    RH, RHF, RH2, RH2F = hash_arrays(ref)
    nd = [i for i, r in enumerate(recs_out) if r['src'] != 'dom']
    rli = low_info(RH)
    c = 0
    for A, AF, B, thr in ((H[nd], HF[nd], (RH, RHF), T64), (H2[nd], H2F[nd], (RH2, RH2F), T256)):
        c += sum(1 for i, j in close_pairs(A, AF, thr, B=B) if not (li[nd[i]] or rli[j]))
    checks['kept new-source images near-duplicate of raw v1 images'] = c
    dom_now = {r['name']: r['split'] for r in recs_out if r['src'] == 'dom'}
    dom_orig = {Path(l).name: sp_ for sp_ in SPLITS for l in (V2 / f'{sp_}_domain.txt').read_text().split()}
    checks['domain images missing or split changed'] = sum(dom_now.get(k) != v for k, v in dom_orig.items())
    for k, v in checks.items():
        print(f'  check {k}: {v}')
    assert all(v == 0 for v in checks.values()), 'LEAK / CONSISTENCY CHECK FAILED'

    # --- write
    for s in SPLITS:
        shutil.rmtree(OUT / s, ignore_errors=True)
        (OUT / s / 'images').mkdir(parents=True)
        (OUT / s / 'labels').mkdir(parents=True)
    names = set()
    lists = collections.defaultdict(list)
    man = []
    for r in recs_out:
        name = r['name']
        assert name not in names, name
        names.add(name)
        dst = OUT / r['split'] / 'images' / name
        try:
            os.link(r['path'], dst)
        except OSError:
            shutil.copy2(r['path'], dst)
        with open(OUT / r['split'] / 'labels' / (Path(name).stem + '.txt'), 'w') as fh:
            for c_, cx, cy, w, h in r['boxes']:
                fh.write(f'{int(c_)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        lists[(r['split'], r['src'])].append(str(dst))
        man.append(dict(name=name, source=r['src'], split=r['split'], group=r['gid'], stem=r['stem'],
                        n_gun=sum(b[0] == 0 for b in r['boxes']), n_knife=sum(b[0] == 1 for b in r['boxes']),
                        src_path=str(r['path'])))

    def wl(fn, items):
        (OUT / fn).write_text(''.join(sorted(f'{x}\n' for x in items)))
    new = [k for k in SRC]
    wl('train.txt', [x for (s, src), v in lists.items() if s == 'train' for x in v])
    wl('valid.txt', [x for (s, src), v in lists.items() if s == 'valid' for x in v])
    wl('test_domain.txt', lists[('test', 'dom')])
    wl('test_new.txt', [x for src in new for x in lists[('test', src)]])
    for s in SPLITS:
        wl(f'{s}_domain.txt', lists[(s, 'dom')])
        wl(f'{s}_new.txt', [x for src in new for x in lists[(s, src)]])
    for fn, test, comment in (
            ('data_indian_all.yaml', 'test_domain.txt',
             'Indian-only v3: train/val = domain + approved Indian sources; test = DOMAIN test (primary benchmark).'),
            ('data_indian_newtest.yaml', 'test_new.txt',
             'Indian-only v3: train/val = domain + approved Indian sources; test = new-sources test (v4/rob/trg/mob).')):
        (OUT / fn).write_text(f'# {comment}\npath: {OUT}\ntrain: train.txt\nval: valid.txt\ntest: {test}\n'
                              f'nc: 3\nnames: {NAMES}\n')
    with open(OUT / '_manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(man[0]))
        w.writeheader()
        w.writerows(man)

    meta = dict(seed=SEED, thresholds=dict(dhash64=T64, dhash256=T256, shot_gap_s=SHOT_GAP, frame_gap=2),
                merges_all=dict(why), merges_output=dict(why_out), checks=checks,
                forced_train_groups=forced_train,
                bridging_dropped=bridged, non_domain_in_domain_groups=non_dom_in_dom_groups,
                v4=dict(images=len(v4_all), source_photos=len(v4_stems_all),
                        images_dropped_existing=len(v4_all) - len(v4_new), images_new=len(v4_new),
                        new_source_photos=len(v4_new_stems),
                        new_groups=len({r['gid'] for r in v4_new}),
                        match_kind=dict(v4_match_kind),
                        shared_stems_with_domain=len(shared),
                        label_disagree_class_presence=len(dis_pres), label_disagree_counts_only=len(dis_cnt),
                        hash_only_matched_groups=grp_n, hash_only_groups_class_presence_disagree=grp_dis,
                        disagree_examples=dis_examples[:40], new_stems=v4_new_stems),
                dropped={k: v for k, v in dropped.items()}, unreadable=bad)
    (OUT / '_build_meta.json').write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: v for k, v in meta['v4'].items() if k not in ('new_stems', 'disagree_examples')}, indent=1))


if __name__ == '__main__':
    main()

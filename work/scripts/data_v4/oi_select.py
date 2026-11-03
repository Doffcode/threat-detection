#!/usr/bin/env python
"""Select Open Images V7 images + boxes for v4_big. Output: downloads/openimages/selected.csv
(ImageID, subset, kind, boxes json [(cls, cx, cy, w, h)])."""
import collections
import csv
import json
import random
from pathlib import Path

OI = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work/downloads/openimages')
names = dict(l.strip().split(',', 1) for l in open(OI / 'classes.csv'))
mid = {v: k for k, v in names.items()}
KNIFE = {mid[n] for n in ('Knife', 'Kitchen knife', 'Dagger', 'Sword')}
GUN = {mid[n] for n in ('Handgun', 'Rifle', 'Shotgun')}
WEAPONISH = KNIFE | GUN | {mid['Weapon']}
HH_CAP = {'Mobile phone': 650, 'Bottle': 200, 'Baseball bat': 250, 'Spoon': 150, 'Fork': 150, 'Pen': 150,
          'Tool': 180, 'Scissors': 999, 'Remote control': 999, 'Flashlight': 999, 'Hammer': 999,
          'Screwdriver': 999, 'Wrench': 999, 'Spatula': 999, 'Axe': 999, 'Chisel': 999, 'Corded phone': 100,
          'Kitchen utensil': 100, 'Telephone': 50}
HH = {mid[n] for n in HH_CAP}
CLS = {**{m: 1 for m in KNIFE}, **{m: 0 for m in GUN}, **{m: 2 for m in HH}}
MIN_SIDE = 0.08  # weapon must have max(w,h) >= 8% of the image on at least one box


def iou(a, b):
    ix = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    u = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / u if u > 0 else 0


rows = collections.defaultdict(list)
subset = {}
for f, s in (('filt_oidv6-train-annotations-bbox.csv', 'train'), ('filt_validation-annotations-bbox.csv', 'validation'),
             ('filt_test-annotations-bbox.csv', 'test')):
    for r in csv.reader(open(OI / f)):
        iid, m = r[0], r[2]
        x1, x2, y1, y2 = map(float, r[4:8])
        rows[iid].append((m, (x1, x2, y1, y2), r[10] == '1', r[11] == '1'))
        subset[iid] = s
weap_lbl = set()
for f in OI.glob('filt_oidv7-*-imagelabels.csv'):
    for r in csv.reader(open(f)):
        if r[2] in WEAPONISH and r[3] == '1':
            weap_lbl.add(r[0])

stats = collections.Counter()
weapon_imgs, hh_by_class = [], collections.defaultdict(list)
for iid, bs in rows.items():
    wb = [b for b in bs if b[0] in WEAPONISH]
    if wb:
        mapped = [b for b in wb if b[0] in KNIFE | GUN]
        if not mapped:
            stats['weapon: only generic Weapon boxes'] += 1
            continue
        if any(b[2] or b[3] for b in wb):
            stats['weapon: group/depiction weapon box'] += 1
            continue
        unc = [b for b in wb if b[0] == mid['Weapon'] and max((iou(b[1], c[1]) for c in mapped), default=0) < 0.5]
        if unc:
            stats['weapon: unmapped Weapon box'] += 1
            continue
        if max(max(b[1][1] - b[1][0], b[1][3] - b[1][2]) for b in mapped) < MIN_SIDE:
            stats['weapon: all boxes tiny'] += 1
            continue
        out = []
        for m, (x1, x2, y1, y2), g, d in bs:
            if m not in CLS or (CLS[m] == 2 and (g or d)):
                continue
            box = (CLS[m], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
            # dedupe hierarchical duplicates (Knife + Kitchen knife on the same object)
            if any(o[0] == box[0] and iou((o[1] - o[3] / 2, o[1] + o[3] / 2, o[2] - o[4] / 2, o[2] + o[4] / 2),
                                          (x1, x2, y1, y2)) > 0.7 for o in out):
                continue
            out.append(box)
        weapon_imgs.append((iid, 'weapon', out))
    else:
        if iid in weap_lbl:
            stats['hardneg: weapon image-level label'] += 1
            continue
        if any(b[2] or b[3] for b in bs if b[0] in HH):
            stats['hardneg: group/depiction box'] += 1
            continue
        hb = [b for b in bs if b[0] in HH]
        if not hb or max(max(b[1][1] - b[1][0], b[1][3] - b[1][2]) for b in hb) < 0.1:
            stats['hardneg: no box >=10%'] += 1
            continue
        out = [(2, (b[1][0] + b[1][1]) / 2, (b[1][2] + b[1][3]) / 2, b[1][1] - b[1][0], b[1][3] - b[1][2]) for b in hb]
        rare = min(hb, key=lambda b: HH_CAP[names[b[0]]])  # bucket by the rarest class present
        hh_by_class[names[rare[0]]].append((iid, 'hardneg', out))

rng = random.Random(2026)
sel = list(weapon_imgs)
for n, lst in sorted(hh_by_class.items()):
    rng.shuffle(lst)
    sel += lst[:HH_CAP[n]]
    stats[f'hardneg kept {n}'] = min(len(lst), HH_CAP[n])
    stats[f'hardneg avail {n}'] = len(lst)
c = collections.Counter()
for iid, k, out in sel:
    s = {b[0] for b in out}
    c['img_gun'] += 0 in s
    c['img_knife'] += 1 in s
    c['img_hh_only'] += s == {2}
with open(OI / 'selected.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    for iid, k, out in sel:
        w.writerow([iid, subset[iid], k, json.dumps([[b[0]] + [round(x, 6) for x in b[1:]] for b in out])])
print(json.dumps({**stats, **c, 'selected': len(sel), 'weapon_imgs': len(weapon_imgs)}, indent=1))

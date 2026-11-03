#!/usr/bin/env python
"""Previews + STATS.md for work/data/indian_v3 (run after build.py, with python -I)."""
import collections
import csv
import json
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet import tile, sheet, boxes_for  # noqa: E402
import build as B  # noqa: E402

OUT = B.OUT
SPLITS = B.SPLITS
NAMES = B.NAMES
SRCS = ['dom', 'v4', 'rob', 'trg', 'mob']
SRC_DESC = {
    'dom': 'our domain data (work/data/v2 dom_*, v1 = IntProj weapon-detection-two, incl. v2 label fixes)',
    'v4': 'viveks-workspace-85aoc/weapon-detection-two v4 (NEW source photos only)',
    'rob': 'innovatus-gv1rg/anti_robbing v1',
    'trg': 'sathishcrisp2/trail_guard v1',
    'mob': 'mohini-bamankar-esrz6/my-first-project-uffsi v1 (Machete only)',
}


def main():
    rng = random.Random(7)
    man = list(csv.DictReader(open(OUT / '_manifest.csv')))
    meta = json.loads((OUT / '_build_meta.json').read_text())
    path = {r['name']: OUT / r['split'] / 'images' / r['name'] for r in man}

    # --- v4 new photos preview: 24 distinct new stems, one copy each
    v4 = [r for r in man if r['source'] == 'v4']
    by_stem = collections.defaultdict(list)
    for r in v4:
        by_stem[r['stem']].append(r)
    stems = sorted(by_stem)
    pick = rng.sample(stems, min(24, len(stems)))
    tiles = [tile(path[by_stem[s][0]['name']], 300, caption=f"{by_stem[s][0]['split'][:2]} {s[:30]}", names=NAMES)
             for s in pick]
    sheet(OUT / '_v4_new_preview.jpg', tiles, 6)

    # --- 16 random new-source train images
    tr = [r for r in man if r['split'] == 'train' and r['source'] != 'dom']
    pick = rng.sample(tr, 16)
    tiles = [tile(path[r['name']], 320, caption=r['name'][:34], names=NAMES) for r in pick]
    sheet(OUT / '_preview_new.jpg', tiles, 4)

    # --- stats
    inst = collections.Counter()
    imgs = collections.Counter()
    groups = collections.defaultdict(set)
    nimg = collections.Counter()
    for r in man:
        s, src = r['split'], r['source']
        nimg[(s, src)] += 1
        groups[(s, src)].add(r['group'])
        ng, nk = int(r['n_gun']), int(r['n_knife'])
        inst[(s, src, 'gun')] += ng
        inst[(s, src, 'knife')] += nk
        imgs[(s, src, 'gun')] += ng > 0
        imgs[(s, src, 'knife')] += nk > 0
        imgs[(s, src, 'bg')] += (ng + nk) == 0
    L = []
    L.append('# indian_v3 - Indian-only, leak-free gun / knife dataset\n')
    L.append('Built by `work/scripts/indian_v3/build.py` (+ `report.py`), seed %d. Classes: `0 gun`, `1 knife` '
             '(any bladed weapon: knife, machete, sword, aruval, sickle), `2 handheld_object` (kept for '
             'compatibility, 0 instances).\n' % meta['seed'])
    L.append('Images are hardlinks (or copies) of the originals; labels were rewritten into `<split>/labels`. '
             'Domain images keep their v2 split and v2 labels (incl. `_manual_fixes.txt`).\n')
    L.append('## YAMLs\n')
    L.append(f'- `{OUT}/data_indian_all.yaml`: train `train.txt`, val `valid.txt` (all sources), '
             'test `test_domain.txt` (domain test = primary benchmark, identical to v2 `test_domain.txt`).')
    L.append(f'- `{OUT}/data_indian_newtest.yaml`: same train/val, test `test_new.txt` (new-sources test).')
    L.append('- Also: `{train,valid,test}_domain.txt`, `{train,valid,test}_new.txt`, `_manifest.csv` '
             '(name, source, split, group, stem, counts, original path), `_build_meta.json`.\n')
    L.append('## Counts (split x source)\n')
    L.append('| split | source | images | groups | gun inst | gun imgs | knife inst | knife imgs | background imgs |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for s in SPLITS:
        for src in SRCS + ['ALL']:
            ks = SRCS if src == 'ALL' else [src]
            n = sum(nimg[(s, k)] for k in ks)
            if not n:
                continue
            g = len(set().union(*[{(k, x) for x in groups[(s, k)]} for k in ks]))
            row = [s, f'**{src}**' if src == 'ALL' else src, n, g,
                   sum(inst[(s, k, 'gun')] for k in ks), sum(imgs[(s, k, 'gun')] for k in ks),
                   sum(inst[(s, k, 'knife')] for k in ks), sum(imgs[(s, k, 'knife')] for k in ks),
                   sum(imgs[(s, k, 'bg')] for k in ks)]
            L.append('| ' + ' | '.join(map(str, row)) + ' |')
    tot = collections.Counter()
    for r in man:
        tot['img'] += 1
        tot['gun'] += int(r['n_gun'])
        tot['knife'] += int(r['n_knife'])
    L.append(f"\nTotal: {tot['img']} images, {tot['gun']} gun and {tot['knife']} knife instances. "
             "Groups are union-find components, so a group number can span sources (one mixed group, see below).\n")
    tn = sum(nimg[('test', k)] for k in SRCS if k != 'dom')
    L.append(f"New-sources test (`test_new.txt`): {tn} images "
             f"({sum(inst[('test', k, 'gun')] for k in SRCS if k != 'dom')} gun, "
             f"{sum(inst[('test', k, 'knife')] for k in SRCS if k != 'dom')} knife instances). "
             "Domain test (`test_domain.txt`): "
             f"{nimg[('test', 'dom')]} images.\n")

    L.append('## Leak / overlap checks (all asserted == 0 at build time)\n')
    for k, v in meta['checks'].items():
        L.append(f'- {k}: **{v}**')
    L.append(f"\nGrouping: normalised stem (`split.norm_stem`), v2 domain group, dHash64 <= {meta['thresholds']['dhash64']} "
             f"OR border-trimmed dHash256 <= {meta['thresholds']['dhash256']} (both incl. h-flip, low-information "
             f"hashes ignored), screenshot sessions (<= {meta['thresholds']['shot_gap_s']} s), video-frame series "
             f"(anti_robbing / mohini frame numbers within {meta['thresholds']['frame_gap']}, trail_guard photo "
             f"timestamps within {meta['thresholds']['shot_gap_s']} s). Hashes were compared across ALL sources and "
             "ALL splits, including the domain images and the raw v1 sets (work/raw/IntProj_Weapon_Dataset_annot, "
             "work/raw/roboflow_v1) as reference.")
    L.append(f"Union merges (full pool incl. reference): `{json.dumps(meta['merges_all'])}`.\n")
    L.append('Split rule: groups touching the domain inherit the domain split (domain splits unchanged). New groups '
             'were split ~80/10/10 per source, greedily stratified by knife instances, then gun instances, then '
             'image count (background). A new group holding > 30% of its source cannot be held out '
             'representatively and is forced to train:')
    for src, n, ntot in meta['forced_train_groups']:
        L.append(f'- {src}: group of {n} / {ntot} images -> train')
    L.append('\n**trail_guard** is a single 17-minute photo burst (one wall, three actresses rotating the props); '
             'hash grouping alone chains 177 of 458 photos, and with the timestamp series it is one group. It is '
             'therefore train-only (no new-sources test/val portion).')
    L.append('**mohini**: 7 v4 screenshot copies (Screenshot-2026-07-06-at-11-49/11-50) fell into the largest mohini '
             'group through a dHash256 link (visually different scenes); keeping them together is the conservative '
             'direction and both sides are in train.\n')

    v = meta['v4']
    L.append('## v4 (weapon-detection-two v4) vs our domain data\n')
    L.append(f"- v4 images: {v['images']} ({v['source_photos']} distinct normalised stems = source photos).")
    L.append(f"- dropped as copies of photos we already have: **{v['images_dropped_existing']} images** "
             f"({v['match_kind']}); {v['shared_stems_with_domain']} v4 stems are identical to domain stems, the "
             'rest joined domain groups through dHash / screenshot sessions.')
    L.append(f"- kept: **{v['images_new']} images = {v['new_source_photos']} NEW source photos** "
             f"({v['new_groups']} leak groups). All new v4 photos are knife images (0 gun boxes after dropping "
             "'armed person'). Preview: `_v4_new_preview.jpg` (24 new photos, one copy each).")
    L.append(f"- label disagreement on the {v['shared_stems_with_domain']} shared photos (v4 labels vs ours; per "
             f"photo, over all copies): **{v['label_disagree_class_presence']}** photos differ in which classes "
             f"are present (gun vs knife), **{v['label_disagree_counts_only']}** more differ only in the number of "
             'boxes. Our fixed labels are kept (all v4 copies of existing photos are dropped). Class-presence '
             'disagreements:')
    for e in v['disagree_examples']:
        L.append(f'  - `{e[0]}`: {e[1]}, {e[2]}')
    import re
    fams = [('knife_N / knife-N (public knife set)', r'^knife[-_]\d+'), ('Screenshot-*', r'^screenshot'),
            ('YYYYMMDD_HHMMSS phone photos', r'^\d{8}_\d{6}'), ('opencv_frame / openN (CCTV-style)', r'^(opencv_frame|open\d)'),
            ('krav-maga / knife-defence videos', r'(krav|defens)'), ('dsc / img', r'^(dsc|img)'),
            ('random-id web images', r'^[0-9a-z]{10,14}$'), ('pisau* (Indonesian "knife")', r'^pisau'),
            ('plain numbers', r'^\d+$')]
    fc = collections.Counter()
    for s in stems:
        fc[next((k for k, p in fams if re.search(p, s)), 'other')] += 1
    L.append('\n**WARNING - provenance of the new v4 photos.** Unlike our domain data (mostly Indian film/video '
             'screenshots and named Indian photos), most new v4 photos are generic web knife images: product shots, '
             'krav-maga / self-defence videos, non-Indian people (see `_v4_new_preview.jpg`). Only a minority look '
             'like Indian CCTV/office scenes. Stem families of the new photos:')
    for k, c in fc.most_common():
        L.append(f'  - {k}: {c}')
    L.append('They were kept because v4 was explicitly approved. To drop them, filter `_manifest.csv` on '
             '`source == v4`.')
    L.append('')

    L.append('## Dropped images\n')
    d = meta['dropped']
    rob = d.get('rob', [])
    why = collections.defaultdict(list)
    for name, w in rob:
        why[w].append(int(name.split('_')[0]))
    L.append(f'### anti_robbing: {len(rob)} frames with a visible UNLABELLED weapon (of 408)\n')
    L.append('Found by reviewing every frame on contact sheets plus zoomed crops of the desk areas and a gold-hilt '
             'colour scan. Dropped rather than kept with wrong labels:')
    for w, ns in why.items():
        L.append(f"- {w}: {len(ns)} frames: {', '.join(map(str, sorted(ns)))}")
    L.append('- Not weapons (kept): the black-and-gold sword scabbard held / lying on the desk (frames 126-143, '
             '223-290), the Indian desk flags, a bag (164), a plant tray (463-514).\n')
    mob = d.get('mob', [])
    mc = collections.Counter(w for _, w in mob)
    L.append(f'### mohini: {len(mob)} of 1161 images dropped\n')
    for w, c in mc.items():
        L.append(f'- {w}: {c}')
    L.append('- Kept 157 images that contain >= 1 Machete and no Gun box. Images with Machete AND a toy Gun were '
             'dropped too: removing the Gun box would leave an unlabelled (toy) gun, teaching gun = background.\n')
    L.append('### trail_guard\n- Person and Rod boxes dropped; the 105 images with only Person/Rod are kept as '
             'negatives. 24 polygon labels converted to boxes.\n')
    L.append('## Licences and attribution (all CC BY 4.0)\n')
    for k, cfg in B.SRC.items():
        L.append(f"- {SRC_DESC[k]}: CC BY 4.0, {cfg['url']}")
    L.append('- domain v1 (IntProj_Weapon_Dataset_annot) is weapon-detection-two v1: CC BY 4.0, '
             'https://universe.roboflow.com/viveks-workspace-85aoc/weapon-detection-two\n')
    (OUT / 'STATS.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()

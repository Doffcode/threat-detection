#!/usr/bin/env python
"""Write data yamls / train lists for work/data/clean and a stats table (STATS.md).

  data.yaml            train = train/images (real + synknife_*)
  data_nosyn.yaml      train = train_nosyn.txt (real train images only)            -> ablation
  data_oversample.yaml train = train_knife_oversample.txt (all train images, real knife images x3)
"""
import collections
import csv
import json
from pathlib import Path

import numpy as np

WORK = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work')
CLEAN = WORK / 'data' / 'clean'
SPLITS = ['train', 'valid', 'test']
NAMES = ['armed person', 'gun', 'knife']
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def yaml_text(train, comment):
    return (f'# {comment}\n'
            f'path: {CLEAN}\n'
            f'train: {train}\n'
            f'val: valid/images\n'
            f'test: test/images\n'
            f'nc: 3\n'
            f"names: ['armed person', 'gun', 'knife']\n")


def read_labels(p):
    out = []
    if p.exists():
        for l in p.read_text().splitlines():
            q = l.split()
            if q:
                out.append((int(q[0]), *map(float, q[1:])))
    return out


def main():
    imgs = {s: sorted(f for f in (CLEAN / s / 'images').iterdir() if f.suffix.lower() in IMG_EXT) for s in SPLITS}
    labels = {s: {f: read_labels(CLEAN / s / 'labels' / (f.stem + '.txt')) for f in imgs[s]} for s in SPLITS}

    # --- train lists / yamls
    real = [f for f in imgs['train'] if not f.name.startswith('synknife_')]
    syn = [f for f in imgs['train'] if f.name.startswith('synknife_')]
    real_knife = [f for f in real if any(b[0] == 2 for b in labels['train'][f])]
    (CLEAN / 'train_nosyn.txt').write_text(''.join(f'{f}\n' for f in real))
    over = list(imgs['train']) + real_knife * 2
    (CLEAN / 'train_knife_oversample.txt').write_text(''.join(f'{f}\n' for f in over))
    (CLEAN / 'data.yaml').write_text(yaml_text('train/images', 'leak-free group split; train includes synknife_* copy-paste images'))
    (CLEAN / 'data_nosyn.yaml').write_text(yaml_text('train_nosyn.txt', 'ablation: real train images only (no synknife_*)'))
    (CLEAN / 'data_oversample.yaml').write_text(yaml_text(
        'train_knife_oversample.txt', 'all train images (incl. synknife_*), real knife-containing images listed 3x total'))

    # --- label sanity
    bad = []
    for s in SPLITS:
        for f, labs in labels[s].items():
            for b in labs:
                if len(b) != 5 or b[0] not in (0, 1, 2):
                    bad.append(f'{s}/{f.name}: {b}')
                    continue
                _, cx, cy, w, h = b
                if not (0 <= cx - w / 2 + 1e-6 and cx + w / 2 <= 1 + 1e-6 and 0 <= cy - h / 2 + 1e-6
                        and cy + h / 2 <= 1 + 1e-6 and w > 0 and h > 0):
                    bad.append(f'{s}/{f.name}: out of range {b}')
    missing = [f'{s}/{f.name}' for s in SPLITS for f in imgs[s] if not (CLEAN / s / 'labels' / (f.stem + '.txt')).exists()]

    # --- groups + overlaps
    man = list(csv.DictReader(open(CLEAN / '_manifest.csv')))
    g_by = collections.defaultdict(set)
    stem_by = collections.defaultdict(set)
    kg_by = collections.defaultdict(set)
    for r in man:
        g_by[r['split']].add(r['group'])
        stem_by[r['split']].add(r['stem'])
        if int(r['n_knife']) > 0:
            kg_by[r['split']].add(r['group'])
    ov = {}
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        ov[f'{a}/{b}'] = dict(groups=len(g_by[a] & g_by[b]), stems=len(stem_by[a] & stem_by[b]))
    meta = json.loads((CLEAN / '_split_meta.json').read_text())
    for k, v in meta['overlap'].items():
        if 'pairs' in k:
            ov[k] = v
    # filenames shared across splits
    names = {s: {f.name for f in imgs[s]} for s in SPLITS}
    ov['identical filenames'] = len((names['train'] & names['valid']) | (names['train'] & names['test']) | (names['valid'] & names['test']))
    crop_meta = json.loads((CLEAN / '_knife_crops_filter.json').read_text())
    syn_meta = json.loads((CLEAN / '_synthetic_meta.json').read_text())

    # --- table
    def row(name, files, s, groups):
        cls_inst = collections.Counter()
        cls_img = collections.Counter()
        empty = 0
        for f in files:
            labs = labels[s][f]
            if not labs:
                empty += 1
            cs = collections.Counter(b[0] for b in labs)
            cls_inst.update(cs)
            cls_img.update(cs.keys())
        return (f'| {name} | {len(files)} | {groups} | ' + ' | '.join(f'{cls_inst[c]} ({cls_img[c]})' for c in range(3))
                + f' | {empty} |')

    L = ['# work/data/clean — dataset statistics', '',
         'Instances per class, with number of images containing that class in parentheses.', '',
         '| split | images | groups | armed person | gun | knife | no-label imgs |',
         '|---|---|---|---|---|---|---|']
    L.append(row('train (real)', real, 'train', len(g_by['train'])))
    L.append(row('train synknife_*', syn, 'train', '-'))
    L.append(row('train (total)', imgs['train'], 'train', len(g_by['train'])))
    L.append(row('valid', imgs['valid'], 'valid', len(g_by['valid'])))
    L.append(row('test', imgs['test'], 'test', len(g_by['test'])))
    L += ['', f"Knife-containing groups: train {len(kg_by['train'])}, valid {len(kg_by['valid'])}, test {len(kg_by['test'])}.",
          f"Train-list sizes: data.yaml {len(imgs['train'])}, data_nosyn.yaml {len(real)}, "
          f"data_oversample.yaml {len(over)} ({len(real_knife)} real knife images x3).", '',
          '## Split construction', '',
          f"- pooled {meta['n_pooled']} images (IntProj + roboflow_v1), {meta['n_stem_groups']} normalised stems -> "
          f"{meta['n_groups']} groups after {meta['n_phash_merges']} dHash-64 merges (<=4), {meta['n_hash16_merges']} "
          f"trimmed dHash-256 merges (<=32) and {meta['n_shot_session_merges']} screenshot-session merges "
          f"(Screenshot-* files taken <=30 s apart)",
          f"- {meta['dropped_rv1_dups']} roboflow_v1 images dropped as duplicates of IntProj images (same photos, "
          f"no 'armed person' labels); kept only roboflow_v1-only groups",
          f"- seed {meta['seed']}; knife groups split greedily by knife-box count (targets "
          + ', '.join(f"{k} {v:.2f}" for k, v in meta['knife_targets'].items())
          + '), other groups by image count 0.75/0.15/0.10, stratified gun / no-gun', '',
          '## Overlap checks (all must be 0)', '',
          '| check | value |', '|---|---|']
    for k, v in ov.items():
        L.append(f'| {k} | {v} |')
    ex = {k: len(v) for k, v in crop_meta['excluded'].items()}
    L += ['', '## Synthetic knives', '',
          f"- {crop_meta['n_crops']} knife_crops, all {crop_meta['n_verified']} matched back to their source IntProj box",
          f"- excluded: {ex}",
          f"- usable crops: {len(crop_meta['usable'])}; synthetic images {syn_meta['stats']['images']}, pasted knives "
          f"{syn_meta['stats']['knives']}, max uses per crop {syn_meta['stats']['max_uses_per_crop']}",
          '', '## Label sanity', '',
          f'- malformed / out-of-range lines in clean labels: {len(bad)}',
          f'- images without a label file: {len(missing)}',
          '- raw-label issues and gun/knife disagreements between duplicates: see _label_issues.txt']
    if bad:
        L += ['', '```'] + bad[:50] + ['```']
    txt = '\n'.join(L) + '\n'
    (CLEAN / 'STATS.md').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()

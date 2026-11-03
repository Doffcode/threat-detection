"""Build data/v2_testclean from the domain test set + audit notes (data/test_audit/notes.txt).
notes.txt line format: idx|issue_type|details|actions   (actions ';'-separated, referencing ORIGINAL box indices)
  add c x y w h   -> append new box        set j c x y w h -> replace box j
  del j           -> remove box j          cls j c         -> change class of box j
Outputs: data/test_audit/issues.csv, data/v2_testclean/{images,labels}/ (copies), test.txt,
         data_testclean.yaml, CHANGELOG.md. Never writes into data/v2."""
import csv, os, shutil, collections
ROOT = '/home/shiva/Desktop/projects/akriti/weapon-detection/work'
A = f'{ROOT}/data/test_audit'; OUT = f'{ROOT}/data/v2_testclean'
NAMES = ['gun', 'knife', 'handheld_object']
idx = {int(r['idx']): r['image'] for r in csv.DictReader(open(f'{A}/sheets/index.csv'))}
lab = lambda p: os.path.splitext(p.replace('/images/', '/labels/'))[0] + '.txt'
notes = collections.defaultdict(list)
for l in open(f'{A}/notes.txt'):
    l = l.rstrip('\n')
    if not l.strip() or l.startswith('#'): continue
    i, t, d, a = (l.split('|') + [''])[:4]
    notes[int(i)].append((t, d, a.strip()))
assert sorted(notes) == sorted(idx), 'every image must have a note'
for d in ('images', 'labels'): os.makedirs(f'{OUT}/{d}', exist_ok=True)
fmt = lambda r: f'{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}'
desc = lambda r: f'{NAMES[r[0]]} xywh=({r[1]:.3f},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f})'
issues, changes = [], []
UNCERTAIN = '''- dom_099 (3 copies): added second sword - parallel blade partly overlapping the main sword, extent estimated.
- dom_Screenshot-2026-06-07-145207 (3 copies): kept the full-rifle box (barrel re-emerges behind the head) and removed the partial stock box; if you prefer visible-segment boxes, the opposite choice is defensible.
- dom_Screenshot-2026-07-01-at-2-25-16-PM (3 copies): of the two overlapping boxes on the bottom rifle, kept the one starting at the rifle body (y~0.74); the dark mass above may be the stock. Added the guard's long gun (small, low-res).
- dom_WhatsApp-Image-2026-06-07-at-3-16-15-PM (3 copies): added second curved sword held low by the man in white; tightened the main sword box in one copy.
'''
cnt_issue = collections.Counter(); cnt_change = collections.Counter(); n_before = n_after = 0
for i in sorted(idx):
    p = idx[i]; name = os.path.basename(p)
    rows = [[int(v[0])] + list(map(float, v[1:5])) for v in (l.split() for l in open(lab(p))) if len(v) >= 5]
    orig = [r[:] for r in rows]; n_before += len(rows)
    new = [r[:] for r in rows]; dead = set(); adds = []
    for t, d, a in notes[i]:
        cnt_issue[t] += 1
        approx = ''
        for act in [x.strip() for x in a.split(';') if x.strip()]:
            k = act.split(); op = k[0]
            if op == 'add':
                r = [int(k[1])] + list(map(float, k[2:6])); adds.append(r); approx = ' '.join(k[2:6])
                changes.append((name, '(none)', desc(r), t, d)); cnt_change['added'] += 1
            elif op == 'set':
                j = int(k[1]); r = [int(k[2])] + list(map(float, k[3:7])); new[j] = r
                changes.append((name, f'box{j} ' + desc(orig[j]), desc(r), t, d)); cnt_change['box_changed'] += 1
            elif op == 'del':
                j = int(k[1]); dead.add(j)
                changes.append((name, f'box{j} ' + desc(orig[j]), '(removed)', t, d)); cnt_change['removed'] += 1
            elif op == 'cls':
                j = int(k[1]); new[j][0] = int(k[2])
                changes.append((name, f'box{j} ' + desc(orig[j]), desc(new[j]), t, d)); cnt_change['class_changed'] += 1
            else: raise ValueError(act)
        issues.append([name, i, t, d, approx, a])
    final = [r for j, r in enumerate(new) if j not in dead] + adds
    n_after += len(final)
    shutil.copy2(os.path.realpath(p), f'{OUT}/images/{name}')
    with open(f'{OUT}/labels/{os.path.splitext(name)[0]}.txt', 'w') as f:
        f.write(''.join(fmt(r) + '\n' for r in final))
with open(f'{A}/issues.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['image', 'sheet_idx', 'issue_type', 'details', 'approx_box_xywh_norm', 'action'])
    w.writerows(issues)
with open(f'{OUT}/test.txt', 'w') as f:
    f.write(''.join(f'{OUT}/images/{os.path.basename(idx[i])}\n' for i in sorted(idx)))
with open(f'{OUT}/data_testclean.yaml', 'w') as f:
    f.write(f"""# Domain test set with audited/cleaned labels (see CHANGELOG.md). train/val unchanged from data/v2.
path: {OUT}
train: {ROOT}/data/v2/train_domain.txt
val: {ROOT}/data/v2/valid_domain.txt
test: {OUT}/images
nc: 3
names: ['gun', 'knife', 'handheld_object']
""")
imgs_changed = len({c[0] for c in changes})
with open(f'{OUT}/CHANGELOG.md', 'w') as f:
    f.write('# v2_testclean - cleaned domain test set\n\n')
    f.write('Source: `data/v2/test_domain.txt` (144 images, 60 unique photos + Roboflow copies). '
            'Audit done by viewing images vs ground truth only (no model predictions were consulted). '
            'Rule: box = the weapon itself only (not hand/arm), whole visible weapon. Images are byte copies; '
            'only label files differ. Generated by `scripts/label_audit/build_testclean.py` from `data/test_audit/notes.txt`; '
            'per-image audit in `data/test_audit/issues.csv`.\n\n')
    f.write(f'## Summary\n\n- images: {len(idx)}; images with label changes: {imgs_changed}\n'
            f'- boxes before: {n_before}; boxes after: {n_after}\n')
    for k in ('added', 'removed', 'box_changed', 'class_changed'): f.write(f'- {k}: {cnt_change[k]}\n')
    f.write(f'- total label edits: {sum(cnt_change.values())}\n\n### Issue counts (one image can have several)\n\n')
    for k, v in sorted(cnt_issue.items()): f.write(f'- {k}: {v}\n')
    f.write('\n## Changes\n\n| image | before | after | issue | reason |\n|---|---|---|---|---|\n')
    for c in changes: f.write('| ' + ' | '.join(x.replace('|', '/') for x in c) + ' |\n')
with open(f'{OUT}/CHANGELOG.md', 'a') as f:
    f.write('\n## Flagged but NOT changed (uncertain / judged acceptable)\n\n')
    for name, i, t, d, approx, a in issues:
        if not a and (t != 'ok' or 'UNSURE' in d or 'inconsistent' in d):
            f.write(f'- {name} (sheet #{i}): {t} - {d}\n')
    f.write('\n## Applied but moderately uncertain\n\n' + UNCERTAIN)
print('before', n_before, 'after', n_after, dict(cnt_change), dict(cnt_issue), 'imgs changed', imgs_changed)

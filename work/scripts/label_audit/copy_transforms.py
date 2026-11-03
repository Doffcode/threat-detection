"""For each Roboflow group (name before '.rf.'), compare each copy to the first copy: identity vs horizontal flip vs vertical flip (MSE on 64x64 gray).
Writes data/test_audit/copy_groups.csv (idx, image, group, ref_idx, transform, mse)."""
import csv, os, cv2, numpy as np
ROOT = '/home/shiva/Desktop/projects/akriti/weapon-detection/work'; A = f'{ROOT}/data/test_audit'
rows = list(csv.DictReader(open(f'{A}/sheets/index.csv')))
g = lambda p: os.path.basename(p).split('.rf.')[0]
small = lambda p: cv2.resize(cv2.imread(p, 0), (64, 64)).astype(float)
ref = {}; out = []
for r in rows:
    k = g(r['image']); s = small(r['image'])
    if k not in ref: ref[k] = (r['idx'], s); out.append([r['idx'], r['image'], k, r['idx'], 'ref', 0]); continue
    ri, rs = ref[k]
    c = {'identity': s, 'hflip': s[:, ::-1], 'vflip': s[::-1], 'rot180': s[::-1, ::-1]}
    m = {t: float(((v-rs)**2).mean()) for t, v in c.items()}; t = min(m, key=m.get)
    out.append([r['idx'], r['image'], k, ri, t, round(m[t], 1)])
with open(f'{A}/copy_groups.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['idx', 'image', 'group', 'ref_idx', 'transform', 'mse']); w.writerows(out)
for o in out: print(o[0], o[2][:30], o[3], o[4], o[5])

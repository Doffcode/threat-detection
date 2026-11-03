"""Count near-duplicate clusters (dHash16, flip-aware, thr 24) within a downloaded dataset."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review as R
for d in sys.argv[1:]:
    imgs = sorted(p for p in Path(d).rglob('*') if p.suffix.lower() in R.IMG_EXT and p.parent.name == 'images')
    H, HF = [], []
    for p in imgs:
        h = R.hash_img(p); H.append(h[0]); HF.append(h[1])
    D = np.minimum(R.v1.hamming_matrix(np.array(H)), R.v1.hamming_matrix(np.array(H), np.array(HF)))
    par = list(range(len(imgs)))
    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i, j in zip(*np.where(np.triu(D <= R.DUP_THR, 1))):
        par[f(i)] = f(j)
    print(Path(d).name, 'files', len(imgs), 'near-dup clusters', len({f(i) for i in range(len(imgs))}))

"""Point a dataset yaml's train/val lists at data/train_fixed copies (corrected labels) where a same-named image exists.
usage: swap_fixed.py IN.yaml OUT.yaml   (OUT may equal IN; a .bak is kept)"""
import sys, shutil, yaml
from pathlib import Path
W = Path(__file__).resolve().parents[2]; FIX = W / "data/train_fixed/images"
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
fixed = {p.name: p for p in FIX.iterdir()} if FIX.exists() else {}
d = yaml.safe_load(open(src)); root = Path(d.get("path", src.parent))
for k in ("train", "val"):
    v = d.get(k)
    if not isinstance(v, str) or not v.endswith(".txt"): continue
    lp = Path(v) if Path(v).is_absolute() else root / v
    lines = [l.strip() for l in open(lp) if l.strip()]
    new = [str(fixed[Path(l).name]) if Path(l).name in fixed else l for l in lines]
    n = sum(a != b for a, b in zip(lines, new))
    out = dst.parent / f"{dst.stem}_{k}.txt"; out.write_text("\n".join(new) + "\n"); d[k] = str(out)
    print(f"{k}: {n}/{len(lines)} images -> train_fixed")
if src == dst: shutil.copy(src, str(src) + ".bak")
d["path"] = str(root); yaml.safe_dump(d, open(dst, "w"), sort_keys=False)

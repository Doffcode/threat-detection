#!/usr/bin/env python
"""Per-source preview sheets of the built v4_big (mapped class names drawn)."""
import csv, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sheet import draw_sheet, read_yolo
OUT = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work/data/v4_big')
N = ['gun', 'knife', 'handheld_object']
rows = list(csv.DictReader(open(OUT / '_manifest.csv')))
for src in sys.argv[1:]:
    sel = [r for r in rows if r['source'] == src.split(':')[0]]
    if ':' in src:  # e.g. oi:knife / oi:gun / oi:hh
        k = src.split(':')[1]
        sel = [r for r in sel if (k == 'knife' and r['n_knife'] != '0') or (k == 'gun' and r['n_gun'] != '0')
               or (k == 'hh' and r['n_gun'] == '0' and r['n_knife'] == '0')]
    else:
        sel = [r for r in sel if r['n_gun'] != '0' or r['n_knife'] != '0' or r['n_handheld'] != '0']
    random.Random(7).shuffle(sel)
    items = []
    for r in sel[:24]:
        ip = OUT / r['split'] / 'images' / r['name']
        lp = OUT / r['split'] / 'labels' / (ip.stem + '.txt')
        items.append((ip, [(N[b[0]], *b[1:]) for b in read_yolo(lp)], f"{r['split'][:5]} {r['name'][:26]}"))
    draw_sheet(items, OUT / f"_preview_{src.replace(':', '_')}.jpg")
    print(src, len(sel))

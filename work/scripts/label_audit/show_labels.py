"""Print GT label lines (x1,y1,x2,y2 too) for sheet indices. Usage: python -I show_labels.py IDX [IDX...]"""
import sys, csv, os
A = '/home/shiva/Desktop/projects/akriti/weapon-detection/work/data/test_audit'
idx = {r['idx']: r['image'] for r in csv.DictReader(open(f'{A}/sheets/index.csv'))}
for a in sys.argv[1:]:
    p = idx[a]; lp = os.path.splitext(p.replace('/images/', '/labels/'))[0] + '.txt'
    print(f'#{a} {os.path.basename(p)}')
    for j, l in enumerate(open(lp)):
        c, x, y, w, h = l.split()[:5]; x, y, w, h = map(float, (x, y, w, h))
        print(f'   {j}: c={c} xywh=({x:.3f},{y:.3f},{w:.3f},{h:.3f}) xyxy=({x-w/2:.2f},{y-h/2:.2f},{x+w/2:.2f},{y+h/2:.2f})')

"""Zoomed view of one audit image with a labelled 5% grid (normalized coords), for estimating boxes.
usage: zoom.py ID [x0 y0 x1 y1] [extra boxes 'c,x,y,w,h;...']  -> data/train_audit/zoom/ID.jpg
"""
import sys, csv
from PIL import Image, ImageDraw, ImageFont

W = '/home/shiva/Desktop/projects/akriti/weapon-detection/work'
OUT = f'{W}/data/train_audit'
sys.path.insert(0, f'{W}/scripts/label_audit_train')
from make_sheets import read_lbl, COL, NAMES

F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
idx = {r['id']: r for r in csv.DictReader(open(f'{OUT}/index.csv'))}


def zoom(tid, x0=0., y0=0., x1=1., y1=1., extra=''):
    p = idx[tid]['rep_path']
    im = Image.open(p).convert('RGB')
    Wd, Ht = im.size
    crop = im.crop((int(x0 * Wd), int(y0 * Ht), int(x1 * Wd), int(y1 * Ht)))
    S = 900
    sc = S / max(crop.size)
    crop = crop.resize((int(crop.size[0] * sc), int(crop.size[1] * sc)))
    cw, ch = crop.size
    d = ImageDraw.Draw(crop, 'RGBA')
    fx = lambda x: (x - x0) / (x1 - x0) * cw
    fy = lambda y: (y - y0) / (y1 - y0) * ch
    step = 0.05 if (x1 - x0) > 0.4 else 0.02
    v = 0.0
    while v <= 1.0001:
        if x0 <= v <= x1:
            d.line([(fx(v), 0), (fx(v), ch)], fill=(255, 255, 0, 90), width=1)
            d.text((fx(v) + 2, 2), f'{v:.2f}', fill=(255, 255, 0), font=F)
        if y0 <= v <= y1:
            d.line([(0, fy(v)), (cw, fy(v))], fill=(255, 255, 0, 90), width=1)
            d.text((2, fy(v) + 2), f'{v:.2f}', fill=(255, 255, 0), font=F)
        v = round(v + step, 4)
    boxes = [(k, b) for k, b in enumerate(read_lbl(p))]
    ex = []
    if extra:
        for s in extra.split(';'):
            c, x, y, w, h = s.split(',')
            ex.append((int(c), float(x), float(y), float(w), float(h)))
    for k, (c, x, y, w, h) in boxes + [(f'N{i}', b) for i, b in enumerate(ex)]:
        pass
    for tag, (c, x, y, w, h) in [(str(k), b) for k, b in boxes] + [(f'N{i}', b) for i, b in enumerate(ex)]:
        col = COL.get(c, (255, 255, 0)) if not tag.startswith('N') else (255, 0, 255)
        d.rectangle([fx(x - w / 2), fy(y - h / 2), fx(x + w / 2), fy(y + h / 2)], outline=col, width=2)
        d.text((fx(x - w / 2) + 2, fy(y - h / 2) + 2), f'{tag}:{NAMES.get(c, c)}', fill=col, font=F)
    import os
    os.makedirs(f'{OUT}/zoom', exist_ok=True)
    crop.save(f'{OUT}/zoom/{tid}.jpg', quality=88)


if __name__ == '__main__':
    a = sys.argv[1:]
    tid = a[0]
    coords = [float(v) for v in a[1:5]] if len(a) >= 5 else [0, 0, 1, 1]
    extra = a[5] if len(a) >= 6 else (a[1] if len(a) == 2 else '')
    zoom(tid, *coords, extra=extra)

"""Render GT contact sheets, one image (representative copy) per unique Roboflow stem.
Writes data/train_audit/index.csv (id, split, stem, rep_path, n_copies) and sheets/sheet_XXX.jpg.
Order: dom (train), dom (valid), rob, mob, trg.
"""
import os, csv, collections
from PIL import Image, ImageDraw, ImageFont

W = '/home/shiva/Desktop/projects/akriti/weapon-detection/work'
OUT = f'{W}/data/train_audit'
NAMES = {0: 'gun', 1: 'knife', 2: 'hand'}
COL = {0: (255, 0, 0), 1: (0, 120, 255), 2: (0, 220, 0)}
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
FONTB = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
T = 520  # tile size
NC, NR = 3, 2


def lbl(p):
    return p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'


def read_lbl(p):
    out = []
    for l in open(lbl(p)):
        v = l.split()
        if len(v) >= 5:
            out.append((int(float(v[0])), *map(float, v[1:5])))
    return out


def draw_tile(p, tid, grid=True):
    im = Image.open(p).convert('RGB')
    im.thumbnail((T, T - 26))
    w, h = im.size
    tile = Image.new('RGB', (T, T), (30, 30, 30))
    ox, oy = (T - w) // 2, 26
    tile.paste(im, (ox, oy))
    d = ImageDraw.Draw(tile, 'RGBA')
    if grid:
        for i in range(1, 10):
            d.line([(ox + w * i / 10, oy), (ox + w * i / 10, oy + h)], fill=(255, 255, 255, 60), width=1)
            d.line([(ox, oy + h * i / 10), (ox + w, oy + h * i / 10)], fill=(255, 255, 255, 60), width=1)
    for k, (c, x, y, bw, bh) in enumerate(read_lbl(p)):
        x0, y0 = ox + (x - bw / 2) * w, oy + (y - bh / 2) * h
        x1, y1 = ox + (x + bw / 2) * w, oy + (y + bh / 2) * h
        col = COL.get(c, (255, 255, 0))
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        txt = f'{k}:{NAMES.get(c, c)}'
        ty = y0 - 18 if y0 - 18 > oy else y0 + 1
        d.rectangle([x0, ty, x0 + 9 * len(txt) + 4, ty + 17], fill=col + (170,))
        d.text((x0 + 2, ty), txt, fill=(255, 255, 255), font=FONT)
    d.text((4, 2), tid, fill=(255, 255, 0), font=FONTB)
    return tile


def main():
    groups = []
    for split, f in [('train', f'{W}/data/indian_v3/train_nov4.txt'), ('valid', f'{W}/data/v2/valid_domain.txt')]:
        st = collections.OrderedDict()
        for p in open(f):
            p = p.strip()
            if p:
                st.setdefault(os.path.basename(p).split('.rf.')[0], []).append(p)
        for s, ps in st.items():
            groups.append((split, s, sorted(ps)))
    order = [('dom_', 'train', 'D'), ('dom_', 'valid', 'V'), ('rob_', 'train', 'R'), ('mob_', 'train', 'M'), ('trg_', 'train', 'T')]
    rows = []
    for pre, split, letter in order:
        g = [x for x in groups if x[0] == split and x[1].startswith(pre)]
        for i, (sp, s, ps) in enumerate(g):
            rows.append((f'{letter}{i:03d}', sp, s, ps[0], len(ps)))
    with open(f'{OUT}/index.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['id', 'split', 'stem', 'rep_path', 'n_copies'])
        wr.writerows(rows)
    per = NC * NR
    # sheets grouped per letter so ids are contiguous
    sheet_no = 0
    for letter in 'DVRMT':
        rr = [r for r in rows if r[0][0] == letter]
        for j in range(0, len(rr), per):
            sheet = Image.new('RGB', (NC * T, NR * T), (0, 0, 0))
            for k, r in enumerate(rr[j:j + per]):
                sheet.paste(draw_tile(r[3], r[0]), ((k % NC) * T, (k // NC) * T))
            sheet.save(f'{OUT}/sheets/{letter}_{j // per:03d}.jpg', quality=85)
            sheet_no += 1
    print('rows', len(rows), 'sheets', sheet_no)


if __name__ == '__main__':
    main()

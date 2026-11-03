#!/usr/bin/env python
"""Convert OD-WeaponDetection (ari-dasci, CC BY-SA 4.0) detection subsets (Pascal VOC xml) to YOLO.

Run with `python -I` (reads untrusted downloads). Subsets used (classification folders have no boxes):
  Knife_detection           (Images/, annotations/*.xml)                 -> prefix odw_knf_
  Pistol detection          (Weapons/, xmls/*.xml)                       -> prefix odw_pis_
  Sohas_weapon-Detection    (images[_test]/, annotations[_test]/xmls)    -> prefix odw_soh_
  (Sohas_weapon-Detection-YOLOv5 holds the same Sohas images -> not used.)
Class mapping -> v2 ids (0 gun, 1 knife, 2 handheld_object); unknown names abort.
Output: work/data/v2/_staging/odw/{labels/*.txt, manifest.csv, class_map.json}
"""
import collections
import csv
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DOWNLOADS, STAGE, write_yolo  # noqa: E402

from PIL import Image, ImageOps  # noqa: E402

ROOT = DOWNLOADS / 'odw' / 'extracted' / 'OD-WeaponDetection-master'
SOH = ROOT / 'Weapons and similar handled objects' / 'Sohas_weapon-Detection'
SUBSETS = [  # tag, xml dir, image dir
    ('knf', ROOT / 'Knife_detection' / 'annotations', ROOT / 'Knife_detection' / 'Images'),
    ('pis', ROOT / 'Pistol detection' / 'xmls', ROOT / 'Pistol detection' / 'Weapons'),
    ('soh', SOH / 'annotations' / 'xmls', SOH / 'images'),
    ('soh', SOH / 'annotations_test' / 'xmls', SOH / 'images_test'),
]
CLASS_MAP = {  # lower-cased VOC name -> v2 id
    'pistol': 0, 'pistola': 0, 'handgun': 0, 'gun': 0, 'rifle': 0,
    'knife': 1, 'cuchillo': 1,
    'smartphone': 2, 'monedero': 2, 'billete': 2, 'tarjeta': 2,  # phone, wallet/purse, bill, card
}
SAFE = re.compile(r'[^A-Za-z0-9_.-]+')


def main():
    out = STAGE / 'odw'
    if out.exists():
        shutil.rmtree(out)
    (out / 'labels').mkdir(parents=True)
    rows, used = [], set()
    st = collections.Counter()
    names_found = collections.Counter()
    for tag, xdir, idir in SUBSETS:
        imgs = {}
        for p in idir.iterdir():
            imgs.setdefault(p.stem.lower(), p)
            imgs.setdefault(p.name.lower(), p)
        for xf in sorted(xdir.glob('*.xml')):
            r = ET.parse(xf).getroot()
            fn = (r.findtext('filename') or xf.stem).strip()
            img = imgs.get(fn.lower()) or imgs.get(Path(fn).stem.lower()) or imgs.get(xf.stem.lower())
            if img is None:
                st[f'{tag}: xml without image'] += 1
                continue
            try:
                with Image.open(img) as im:
                    W0, H0 = im.size
                    exif_rot = ImageOps.exif_transpose(im).size != im.size
            except Exception:
                st[f'{tag}: unreadable image'] += 1
                continue
            sz = r.find('size')
            xw, xh = (int(float(sz.findtext('width') or 0)), int(float(sz.findtext('height') or 0))) if sz is not None else (0, 0)
            # boxes are in the coordinate frame of the size stored in the xml; scale if it differs
            if (xw, xh) not in ((W0, H0), (0, 0)):
                st[f'{tag}: xml size != image size (scaled)'] += 1
            W, H = (xw, xh) if xw > 0 and xh > 0 else (W0, H0)
            if exif_rot:
                st[f'{tag}: EXIF-rotated image (skipped)'] += 1
                continue
            boxes = []
            for o in r.findall('object'):
                nm = (o.findtext('name') or '').strip()
                names_found[(tag, nm)] += 1
                c = CLASS_MAP.get(nm.lower())
                if c is None:
                    raise SystemExit(f'unknown class name {nm!r} in {xf}')
                b = o.find('bndbox')
                x1, y1, x2, y2 = (float(b.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
                x1, x2 = sorted(min(float(W), max(0.0, v)) for v in (x1, x2))
                y1, y2 = sorted(min(float(H), max(0.0, v)) for v in (y1, y2))
                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    st[f'{tag}: degenerate box dropped'] += 1
                    continue
                boxes.append((c, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
            if not boxes:
                st[f'{tag}: image without boxes'] += 1
            ext = img.suffix.lower() if img.suffix.lower() in ('.jpg', '.jpeg', '.png') else '.jpg'
            name = f'odw_{tag}_' + SAFE.sub('_', img.stem).strip('_') + ext
            k = 2
            while name.lower() in used:
                name = f'odw_{tag}_' + SAFE.sub('_', img.stem).strip('_') + f'_{k}' + ext
                k += 1
            used.add(name.lower())
            lab = out / 'labels' / (Path(name).stem + '.txt')
            write_yolo(lab, boxes)
            rows.append(dict(name=name, src=str(img), label=str(lab), source=f'odw_{tag}'))
            st[f'{tag}: images'] += 1
            st[f'{tag}: boxes'] += len(boxes)
    with open(out / 'manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['name', 'src', 'label', 'source'])
        w.writeheader()
        w.writerows(rows)
    cm = {f'{t}/{n}': dict(count=c, v2=CLASS_MAP[n.lower()]) for (t, n), c in sorted(names_found.items())}
    (out / 'class_map.json').write_text(json.dumps(dict(found=cm, stats=dict(st)), indent=1))
    for k, v in sorted(st.items()):
        print(f'  {k}: {v}')
    for k, v in cm.items():
        print(f'  class {k}: {v}')


if __name__ == '__main__':
    main()

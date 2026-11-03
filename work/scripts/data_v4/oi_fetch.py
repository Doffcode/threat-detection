#!/usr/bin/env python
"""Download selected Open Images (public S3 over HTTPS), resize to max side 1280, write jpg.
Usage: oi_fetch.py  (reads downloads/openimages/selected.csv, writes downloads/openimages/images/<id>.jpg)"""
import csv
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

OI = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work/downloads/openimages')
DST = OI / 'images'
DST.mkdir(exist_ok=True)
MAXS = 1280


def fetch(row):
    iid, sub = row[0], row[1]
    out = DST / f'{iid}.jpg'
    if out.exists() and out.stat().st_size > 0:
        return 'skip'
    try:
        with urllib.request.urlopen(f'https://open-images-dataset.s3.amazonaws.com/{sub}/{iid}.jpg', timeout=60) as r:
            data = r.read()
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            return 'bad'
        h, w = im.shape[:2]
        s = MAXS / max(h, w)
        if s < 1:
            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        tmp = out.with_suffix('.tmp.jpg')
        cv2.imwrite(str(tmp), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tmp.rename(out)
        return 'ok'
    except Exception as e:
        return f'err {type(e).__name__}'


rows = list(csv.reader(open(OI / 'selected.csv')))
cv2.setNumThreads(1)
res = {}
with ThreadPoolExecutor(16) as ex:
    for i, r in enumerate(ex.map(fetch, rows)):
        res[r] = res.get(r, 0) + 1
        if i % 500 == 0:
            print(i, len(rows), res, flush=True)
print('DONE', res, flush=True)

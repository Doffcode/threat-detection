"""Download Roboflow Universe datasets (YOLO format) into work/downloads/roboflow/<ws>__<proj>__v<N>/.
Usage: download.py ws/proj[@N] ...   (default: latest version). Key from ~/.roboflow_key, never printed."""
import os, sys, json
from pathlib import Path
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[2] / 'downloads' / 'roboflow_v4'
KEY = Path.home().joinpath('.roboflow_key').read_text().strip()
rf = Roboflow(api_key=KEY)
for arg in sys.argv[1:]:
    slug, _, ver = arg.partition('@')
    ws, proj = slug.split('/')
    try:
        p = rf.workspace(ws).project(proj)
        vers = sorted(int(v.version) for v in p.versions())
        n = int(ver) if ver else vers[-1]
        dst = ROOT / f'{ws}__{proj}__v{n}'
        if dst.exists() and any(dst.rglob('*.jpg')):
            print('exists', dst); continue
        v = p.version(n)
        ok = False
        for fmt in ('yolov11', 'yolov8'):
            try:
                v.download(fmt, location=str(dst), overwrite=True)
                ok = True
                break
            except Exception as e:
                print('  fmt fail', fmt, type(e).__name__, str(e)[:200])
        print('OK' if ok else 'FAIL', dst)
    except Exception as e:
        print('FAIL', arg, type(e).__name__, str(e)[:300])

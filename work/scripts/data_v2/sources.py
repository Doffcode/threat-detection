"""Optional public sources (hooks). Each fetch_* returns True if the source was fetched + staged.

Staged format (same for every public source): work/data/v2/_staging/<tag>/manifest.csv with columns
  name (output file name incl. prefix), src (image path), label (staged YOLO txt in v2 classes), source
Downloads go to work/downloads/<tag>/ (untrusted: never executed; Python that reads them runs with -I).
"""
import csv
import json
import os
import subprocess
from pathlib import Path

from common import DOWNLOADS, STAGE, WORK

PY = str(WORK / '.venv' / 'bin' / 'python')

# class-name -> v2 id maps for the optional sources (lower-cased names)
KAGGLE_MAP = {'gun': 0, 'guns': 0, 'pistol': 0, 'handgun': 0, 'rifle': 0, 'knife': 1, 'knives': 1}
ROBOFLOW_MAP = {'gun': 0, 'pistol': 0, 'handgun': 0, 'rifle': 0, 'weapon': None, 'knife': 1, 'knives': 1}
ROBOFLOW_PROJECTS = [  # (workspace, project, version) -- 1-2 knife/pistol sets from Roboflow Universe
    ('roboflow-universe-projects', 'knife-detection-bstjz', 1),
    ('joseph-nelson', 'pistols', 1),
]


def kaggle_available():
    return (Path.home() / '.kaggle' / 'kaggle.json').exists() or bool(os.environ.get('KAGGLE_KEY'))


def roboflow_available():
    return bool(os.environ.get('ROBOFLOW_API_KEY'))


def fetch_kaggle(slug='iqmansingh/guns-knives-object-detection'):
    """Download + stage the Kaggle guns/knives set (YOLO or VOC). Skipped without credentials."""
    if not kaggle_available():
        print('[kaggle] ~/.kaggle/kaggle.json not found -> skipped')
        return False
    dst = DOWNLOADS / 'kaggle_gk'
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, '-I', '-m', 'pip', 'install', '-q', 'kaggle'], check=True)
    subprocess.run([PY, '-I', '-m', 'kaggle', 'datasets', 'download', '-d', slug, '-p', str(dst), '--unzip'],
                   check=True)
    return stage_generic(dst, 'kag', KAGGLE_MAP)


def fetch_roboflow(projects=ROBOFLOW_PROJECTS):
    """Download 1-2 Roboflow Universe sets in YOLOv8 format and stage them. Skipped without ROBOFLOW_API_KEY."""
    if not roboflow_available():
        print('[roboflow] ROBOFLOW_API_KEY not set -> skipped')
        return False
    import urllib.request
    import zipfile
    key = os.environ['ROBOFLOW_API_KEY']
    ok = False
    for ws, proj, ver in projects:
        dst = DOWNLOADS / f'rf_{proj}'
        dst.mkdir(parents=True, exist_ok=True)
        meta = json.load(urllib.request.urlopen(
            f'https://api.roboflow.com/{ws}/{proj}/{ver}/yolov8?api_key={key}'))
        zp = dst / 'ds.zip'
        urllib.request.urlretrieve(meta['export']['link'], zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(dst / 'x')
        ok |= stage_generic(dst / 'x', f'rf{len(proj)}', ROBOFLOW_MAP)
    return ok


def stage_generic(root, tag, cmap):
    """Stage a YOLO-format download (data.yaml names + images/labels dirs) into v2 classes."""
    import yaml
    root = Path(root)
    ymls = list(root.rglob('data.yaml'))
    names = yaml.safe_load(ymls[0].read_text())['names'] if ymls else []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    out = STAGE / tag
    (out / 'labels').mkdir(parents=True, exist_ok=True)
    rows = []
    for img in root.rglob('*'):
        if img.suffix.lower() not in ('.jpg', '.jpeg', '.png') or 'images' not in img.parts:
            continue
        lab = Path(str(img.parent).replace('images', 'labels')) / (img.stem + '.txt')
        lines = []
        if lab.exists():
            for ln in lab.read_text().splitlines():
                p = ln.split()
                if len(p) != 5:
                    continue
                new = cmap.get(str(names[int(p[0])]).lower()) if names else None
                if new is not None:
                    lines.append(f'{new} ' + ' '.join(p[1:]))
        name = f'{tag}_{len(rows):06d}{img.suffix.lower()}'
        (out / 'labels' / (Path(name).stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
        rows.append(dict(name=name, src=str(img), label=str(out / 'labels' / (Path(name).stem + '.txt')), source=tag))
    with open(out / 'manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['name', 'src', 'label', 'source'])
        w.writeheader()
        w.writerows(rows)
    print(f'[{tag}] staged {len(rows)} images')
    return bool(rows)

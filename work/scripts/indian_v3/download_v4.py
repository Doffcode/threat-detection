"""Download viveks-workspace-85aoc/weapon-detection-two v4 (YOLO) into
work/downloads/roboflow/viveks-workspace-85aoc__weapon-detection-two__v4/. Key from ~/.roboflow_key, never printed."""
from pathlib import Path
from roboflow import Roboflow

DST = Path('/home/shiva/Desktop/projects/akriti/weapon-detection/work/downloads/roboflow/'
           'viveks-workspace-85aoc__weapon-detection-two__v4')
key = Path.home().joinpath('.roboflow_key').read_text().strip()
v = Roboflow(api_key=key).workspace('viveks-workspace-85aoc').project('weapon-detection-two').version(4)
del key
if DST.exists() and any(DST.rglob('*.jpg')):
    print('exists', DST)
else:
    for fmt in ('yolov11', 'yolov8'):
        try:
            v.download(fmt, location=str(DST), overwrite=True)
            print('OK', fmt)
            break
        except Exception as e:  # noqa
            print('fmt fail', fmt, type(e).__name__, str(e)[:200].replace('api_key', '<k>'))

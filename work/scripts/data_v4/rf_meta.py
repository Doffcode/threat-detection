"""Query Roboflow API metadata for workspace/project slugs. Key read from ~/.roboflow_key, never printed."""
import json, sys, urllib.request, urllib.error
from pathlib import Path
KEY = Path.home().joinpath('.roboflow_key').read_text().strip()

def get(path):
    url = f"https://api.roboflow.com/{path}?api_key={KEY}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {'error': e.code}
    except Exception as e:
        return {'error': type(e).__name__}

if __name__ == '__main__':
    out = {}
    for slug in sys.argv[1:]:
        d = get(slug)
        if 'error' in d:
            print(slug, 'ERROR', d['error']); continue
        p = d.get('project', {})
        vers = d.get('versions', [])
        print(f"== {slug} | name={p.get('name')} | images={p.get('images')} | license={p.get('license')} | annotation={p.get('annotation')}")
        print('   classes:', p.get('classes'))
        for v in vers:
            print(f"   v{v.get('id','').split('/')[-1]} images={v.get('images')} splits={v.get('splits')} aug={bool(v.get('augmentation'))} exports={v.get('exports')}")
        out[slug] = d
    json.dump(out, open(sys.argv[0].replace('rf_meta.py', '_meta_cache_v4.json'), 'w'), default=str)

"""print labels of audit ids: show.py D005 D004"""
import sys, csv
W='/home/shiva/Desktop/projects/akriti/weapon-detection/work'
idx={r['id']:r for r in csv.DictReader(open(f'{W}/data/train_audit/index.csv'))}
for t in sys.argv[1:]:
    p=idx[t]['rep_path']
    print(t, idx[t]['n_copies'])
    for k,l in enumerate(open(p.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt')):
        v=l.split()
        if len(v)>=5:
            c,x,y,w,h=int(v[0]),*map(float,v[1:5]); print(f'  {k}: c{c} xywh {x:.3f} {y:.3f} {w:.3f} {h:.3f}  x0y0x1y1 {x-w/2:.3f} {y-h/2:.3f} {x+w/2:.3f} {y+h/2:.3f}')

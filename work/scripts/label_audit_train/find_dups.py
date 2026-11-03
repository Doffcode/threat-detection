"""Flag same-class nested/overlapping boxes in rep images (likely duplicate / hand-inclusive double labels)."""
import csv
W='/home/shiva/Desktop/projects/akriti/weapon-detection/work'
import sys; sys.path.insert(0,f'{W}/scripts/label_audit_train')
from make_sheets import read_lbl
def xyxy(b): c,x,y,w,h=b; return (x-w/2,y-h/2,x+w/2,y+h/2)
def inter(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def area(a): return (a[2]-a[0])*(a[3]-a[1])
out=[]
for r in csv.DictReader(open(f'{W}/data/train_audit/index.csv')):
    B=read_lbl(r['rep_path'])
    for i in range(len(B)):
        for j in range(len(B)):
            if i==j or B[i][0]!=B[j][0]: continue
            a,b=xyxy(B[i]),xyxy(B[j])
            # j contained in i, i bigger
            if area(a)>=area(b) and inter(a,b)/max(area(b),1e-9)>0.85:
                out.append((r['id'],i,j,round(area(a)/max(area(b),1e-9),2)))
w=csv.writer(open(f'{W}/data/train_audit/dup_candidates.csv','w',newline=''))
w.writerow(['id','big','small','area_ratio']); w.writerows(out)
import collections
print(len(out), collections.Counter(o[0][0] for o in out))
for o in out: print(*o)

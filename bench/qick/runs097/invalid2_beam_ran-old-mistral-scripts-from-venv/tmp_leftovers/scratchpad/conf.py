import h5py, numpy as np

def load(q,name):
    f=h5py.File('ss_ge.h5','r')
    raw=f[q][name][()]
    s=raw[0].decode() if isinstance(raw,np.ndarray) else raw.decode()
    s=s.replace('[','').replace(']','')
    return np.fromstring(s, sep=' ')

q='Q1'
angle=float(h5py.File('ss_ge.h5','r')[q]['Angle'][0])
Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
rg=Ig*np.cos(angle)-Qg*np.sin(angle)
re_=Ie*np.cos(angle)-Qe*np.sin(angle)
allv=np.concatenate([rg,re_]); thr=np.unique(np.sort(allv))
best=None
for t in thr:
    acc=(np.sum(rg<=t)+np.sum(re_>t))/(len(rg)+len(re_))
    if best is None or acc>best[0]:
        best=(acc,t,np.sum(rg<=t),np.sum(re_>t))
acc,t,cg,ce=best
print('opt thr',t,'acc',acc)
print('P(g|g)=',cg/len(rg),'P(e|e)=',ce/len(re_))
print('min=',min(cg/len(rg),ce/len(re_)))
print('geometric=',np.sqrt((cg/len(rg))*(ce/len(re_))))
print('1 - (FPR+FNR)/2?')
# try threshold 0
for t in [0, (rg.mean()+re_.mean())/2, np.median(rg), np.median(re_)]:
    a=(np.sum(rg<=t)+np.sum(re_>t))/(len(rg)+len(re_))
    print('thr',t,'acc',a)
# maybe stored fidelity = accuracy using threshold from config scaled? no
# Try fidelity definition: 0.5*(P(g|g)+P(e|e)) with threshold chosen to equalize? same as acc for equal N
# Check if 0.8365 equals maxacc for Q1 on rotated but with clean-side std? no
print('stored', 0.8364727454575762)

import h5py, numpy as np

def load(q,name):
    f=h5py.File('ss_ge.h5','r')
    raw=f[q][name][()]
    s=raw[0].decode() if isinstance(raw,np.ndarray) else raw.decode()
    s=s.replace('[','').replace(']','')
    return np.fromstring(s, sep=' ')

f=h5py.File('ss_ge.h5','r')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    angle=float(f[q]['Angle'][0])
    Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
    stored=float(f[q]['Fidelity'][0])
    # candidate: maxacc on raw I (no rotation)
    for name,rg,re_ in [('I_raw',Ig,Ie),('Q_raw',Qg,Qe)]:
        allv=np.concatenate([rg,re_]); thr=np.unique(np.sort(allv))
        best=max((np.sum(rg<=t)+np.sum(re_>t))/(len(rg)+len(re_)) for t in thr)
        print(f'{q} {name}: maxacc={best:.4f} (stored {stored:.4f})')
    # candidate: threshold at 0
    for name,rg,re_ in [('I_raw',Ig,Ie)]:
        acc=(np.sum(rg<=0)+np.sum(re_>0))/(len(rg)+len(re_))
        print(f'{q} I_raw thr0: {acc:.4f}')
    # candidate: simple mean threshold (Ig.mean()+Ie.mean())/2 no rotation
    t=(Ig.mean()+Ie.mean())/2
    acc=(np.sum(Ig<=t)+np.sum(Ie>t))/(len(Ig)+len(Ie))
    print(f'{q} I_raw meanthr: {acc:.4f}')

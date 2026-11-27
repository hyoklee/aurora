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
    for sign in [+1,-1]:
        rg=Ig*np.cos(angle)+sign*Qg*np.sin(angle)
        re_=Ie*np.cos(angle)+sign*Qe*np.sin(angle)
        mg=np.median(rg); me=np.median(re_)
        cg=rg[rg<=mg]; ce=re_[re_>=me]
        sep=abs(re_.mean()-rg.mean())  # or median diff
        sep_med=abs(me-mg)
        sg=cg.std(ddof=1); se=ce.std(ddof=1)
        snr=sep_med/np.sqrt((sg*sg+se*se)/2)
        # maxacc
        allv=np.concatenate([rg,re_]); thr=np.unique(np.sort(allv))
        best=0
        for t in thr:
            acc=(np.sum(rg<=t)+np.sum(re_>t))/(len(rg)+len(re_))
            if acc>best: best=acc
        print(f'{q} sign={sign:+d} sep_mean={sep:7.2f} sep_med={sep_med:7.2f} sg={sg:6.2f} se={se:6.2f} SNR={snr:.3f} maxacc={best:.4f} storedF={float(f[q]["Fidelity"][0]):.4f}')

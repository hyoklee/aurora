import h5py, numpy as np

def load(q,name):
    f=h5py.File('ss_ge.h5','r')
    raw=f[q][name][()]
    s=raw[0].decode() if isinstance(raw,np.ndarray) else raw.decode()
    s=s.replace('[','').replace(']','')
    return np.fromstring(s, sep=' ')

q='Q1'
Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
best=[]
for i in range(360):
    th=np.deg2rad(i)
    for sign in [+1,-1]:
        rg=Ig*np.cos(th)+sign*Qg*np.sin(th)
        re_=Ie*np.cos(th)+sign*Qe*np.sin(th)
        mg=np.median(rg); me=np.median(re_)
        cg=rg[rg<=mg]; ce=re_[re_>=me]
        sep=abs(me-mg)
        sg=cg.std(ddof=1); se=ce.std(ddof=1)
        snr=sep/np.sqrt((sg*sg+se*se)/2)
        allv=np.concatenate([rg,re_]); thr=np.unique(np.sort(allv))
        maxacc=max((np.sum(rg<=t)+np.sum(re_>t))/(len(rg)+len(re_)) for t in thr)
        best.append((abs(snr-4.4), i, sign, snr, maxacc, sg, se, sep))
best.sort()
for b in best[:8]:
    print('rank', b[0], 'deg', b[1], 'sign', b[2], 'SNR', round(b[3],3), 'maxacc', round(b[4],4), 'sg', round(b[5],2), 'se', round(b[6],2), 'sep', round(b[7],2))
# also print at stored angle
th=-0.129196
rg=Ig*np.cos(th)-Qg*np.sin(th); re_=Ie*np.cos(th)-Qe*np.sin(th)
print('stored-ish sign- SNR', (abs(np.median(re_)-np.median(rg))/np.sqrt(((rg[rg<=np.median(rg)].std(ddof=1))**2+(re_[re_>=np.median(re_)].std(ddof=1))**2)/2)))

import h5py, numpy as np, re

def parse(g, name):
    raw = g[name][()][0]
    text = raw.decode() if isinstance(raw,(bytes,bytearray)) else str(raw)
    nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', text)
    return np.array([float(x) for x in nums])

f = h5py.File('ss_ge.h5','r')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g=f[q]
    Ig=parse(g,'I_g'); Qg=parse(g,'Q_g'); Ie=parse(g,'I_e'); Qe=parse(g,'Q_e')
    ang=g['Angle'][()][0]
    print('===',q,'stored Fid=',g['Fidelity'][()][0],'angle=',ang,'n=',len(Ig),len(Ie))
    for sign in [+1,-1]:
        Igr = Ig*np.cos(sign*ang) - Qg*np.sin(sign*ang)
        Ier = Ie*np.cos(sign*ang) - Qe*np.sin(sign*ang)
        mg=np.median(Igr); me=np.median(Ier)
        allI=np.concatenate([Igr,Ier])
        xs=np.sort(allI)
        best=0; bestt=None
        for t in xs:
            acc=((Igr<=t).sum()+(Ier>t).sum())/(len(Igr)+len(Ier))
            if acc>best: best=acc; bestt=t
        print(f'  sign={sign} naive best acc={best:.4f} at t={bestt:.2f} medians g/e={mg:.2f}/{me:.2f} sep={abs(me-mg):.2f}')
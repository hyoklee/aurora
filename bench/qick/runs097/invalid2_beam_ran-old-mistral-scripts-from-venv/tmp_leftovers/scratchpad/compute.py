import h5py, numpy as np, re

def load(q,name):
    f=h5py.File('ss_ge.h5','r')
    raw=f[q][name][()]
    s=raw[0].decode() if isinstance(raw,np.ndarray) else raw.decode()
    s=s.replace('[','').replace(']','')
    return np.fromstring(s, sep=' ')

f=h5py.File('ss_ge.h5','r')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g=f[q]
    angle=float(g['Angle'][0])
    Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
    print(q, 'N', Ig.size, Ie.size, Qg.size, Qe.size, 'angle', angle, 'stored F', float(g['Fidelity'][0]))
    for label, Igr, Ier in [
        ('Icos+Qsin', Ig*np.cos(angle)+Qg*np.sin(angle), Ie*np.cos(angle)+Qe*np.sin(angle)),
        ('Icos-Qsin', Ig*np.cos(angle)-Qg*np.sin(angle), Ie*np.cos(angle)-Qe*np.sin(angle)),
    ]:
        allv=np.concatenate([Igr,Ier])
        thr_candidates=np.unique(np.sort(allv))
        best=0; bestt=None
        for t in thr_candidates:
            correct_g=np.sum(Igr<=t)
            correct_e=np.sum(Ier>t)
            acc=(correct_g+correct_e)/(len(Igr)+len(Ier))
            if acc>best:
                best=acc; bestt=t
        print(f'   {label}: maxacc={best:.6f} at t={bestt:.3f}  mean_g={Igr.mean():.2f} mean_e={Ier.mean():.2f}')

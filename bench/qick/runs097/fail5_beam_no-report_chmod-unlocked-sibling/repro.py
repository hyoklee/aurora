import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
def parse(s):
    s=s.strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')
def load(q,k):
    s=f[q][k][()][0]
    if isinstance(s,(bytes,bytearray)): s=s.decode()
    return parse(s)
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
    ang=float(np.asarray(f[q]['Angle'][()]).ravel()[0])
    stored=float(np.asarray(f[q]['Fidelity'][()]).ravel()[0])
    thr=float(np.asarray(f[q]['Syst Config'][()][0] if False else 0).ravel()[0]) if False else None
    # rotation convention x = I cos a - Q sin a
    c=np.cos(ang); s=np.sin(ang)
    xg=Ig*c - Qg*s
    xe=Ie*c - Qe*s
    # max threshold accuracy over all midpoints
    xs=np.sort(np.concatenate([xg,xe]))
    best=None
    for t in xs:
        acc=(np.sum(xg<=t)+np.sum(xe>t))/(len(xg)+len(xe))
        if best is None or acc>best[1]:
            best=(t,acc)
    # also threshold midway? and median-based
    print(f'{q}: storedF={stored:.4f} maxThreshAcc={best[1]:.4f} at t={best[0]:.2f}  med_g={np.median(xg):.1f} med_e={np.median(xe):.1f}')

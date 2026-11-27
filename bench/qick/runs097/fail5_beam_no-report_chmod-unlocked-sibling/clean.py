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
    c=np.cos(ang); s=np.sin(ang)
    xg=Ig*c - Qg*s
    xe=Ie*c - Qe*s
    mg=np.median(xg); me=np.median(xe)
    # clean side: g samples with x <= median_g ; e samples with x >= median_e
    gclean=xg[xg<=mg]
    eclean=xe[xe>=me]
    sg=np.std(gclean,ddof=1)
    se=np.std(eclean,ddof=1)
    dmu=abs(me-mg)
    snr=dmu/np.sqrt((sg*sg+se*se)/2)
    print(f'{q}: dmu={dmu:.2f} sg={sg:.2f} se={se:.2f} snr={snr:.2f} nclean_g={len(gclean)} nclean_e={len(eclean)}')

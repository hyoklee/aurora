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
q='Q1'
Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
ang=float(np.asarray(f[q]['Angle'][()]).ravel()[0])
stored=float(np.asarray(f[q]['Fidelity'][()]).ravel()[0])
print('stored',stored,'ang',ang)
# syst config threshold list
import ast
ec=f[q]['Syst Config'][()][0].decode()
# rotated x with various conventions
for name, (c,s) in [('c+ s-',(np.cos(ang), np.sin(ang))), ('c- s+',(np.cos(-ang), np.sin(-ang)))]:
    pass
# Build rotated coordinate x = I*cos(a) - Q*sin(a) (sep positive)
c=np.cos(ang); s=np.sin(ang)
# try both sign for angle
for anguse in [ang, -ang]:
    ca=np.cos(anguse); sa=np.sin(anguse)
    for formula in ['Ic-Qs','Ic+Qs','Ic-Qs (a)','Ic+Qs (a)']:
        if formula.startswith('Ic-Qs'):
            xg=Ig*ca - Qg*sa; xe=Ie*ca - Qe*sa
        else:
            xg=Ig*ca + Qg*sa; xe=Ie*ca + Qe*sa
        # choose orientation so e mean > g mean
        if xe.mean()<xg.mean():
            xg=-xg; xe=-xe
        mg=np.median(xg); me=np.median(xe)
        # accuracy at midpoint threshold
        tmid=(mg+me)/2
        acc_mid=(np.sum(xg<=tmid)+np.sum(xe>tmid))/(len(xg)+len(xe))
        # accuracy at threshold=0
        acc0=(np.sum(xg<=0)+np.sum(xe>0))/(len(xg)+len(xe))
        # find threshold giving stored accuracy
        xs=np.sort(np.concatenate([xg,xe]))
        accs=[(t,(np.sum(xg<=t)+np.sum(xe>t))/(len(xg)+len(xe))) for t in xs]
        # closest to stored
        best=min(accs,key=lambda z:abs(z[1]-stored))
        print(f'anguse={anguse:.4f} formula={formula}: sep={xe.mean()-xg.mean():.1f} acc_mid={acc_mid:.4f} acc0={acc0:.4f} closest_to_stored={best[1]:.4f} at t={best[0]:.2f}')
# Also unrotated
for coord, (a,b) in [('I', (Ig,Ie)), ('Q',(Qg,Qe))]:
    g=a; e=b
    if e.mean()<g.mean(): g=-g; e=-e
    xs=np.sort(np.concatenate([g,e]))
    accs=[(t,(np.sum(g<=t)+np.sum(e>t))/(len(g)+len(e))) for t in xs]
    best=max(accs,key=lambda z:z[1])
    mid=(np.median(g)+np.median(e))/2
    accmid=(np.sum(g<=mid)+np.sum(e>mid))/(len(g)+len(e))
    closest=min(accs,key=lambda z:abs(z[1]-stored))
    print(f'unrotated {coord}: maxAcc={best[1]:.4f} acc_mid={accmid:.4f} closest_stored={closest[1]:.4f} at t={closest[0]:.2f}')

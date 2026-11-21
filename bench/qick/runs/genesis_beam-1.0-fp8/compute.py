import h5py, numpy as np, re

def parse(g, name):
    raw = g[name][()][0]
    text = raw.decode() if isinstance(raw,(bytes,bytearray)) else str(raw)
    nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', text)
    return np.array([float(x) for x in nums])

f = h5py.File('ss_ge.h5','r')
q='Q1'
g=f[q]
Ig=parse(g,'I_g'); Qg=parse(g,'Q_g'); Ie=parse(g,'I_e'); Qe=parse(g,'Q_e')
ang=g['Angle'][()][0]
print('angle',ang)
Igr = Ig*np.cos(ang) - Qg*np.sin(ang)
Ier = Ie*np.cos(ang) - Qe*np.sin(ang)
mg=np.median(Igr); me=np.median(Ier)
print('medians',mg,me,'sep',abs(me-mg))
gclean = Igr[Igr <= mg]
eclean = Ier[Ier >= me]
sg = np.std(gclean, ddof=1)
se = np.std(eclean, ddof=1)
print('clean counts', len(gclean), len(eclean))
print('sigma_g', sg, 'sigma_e', se, 'avg', (sg+se)/2)
dmu=abs(me-mg)
snr = dmu / np.sqrt((sg**2+se**2)/2)
print('SNR', snr)
allI=np.concatenate([Igr,Ier])
xs=np.sort(allI)
best=0; bestt=None
for t in xs:
    acc=((Igr<=t).sum()+(Ier>t).sum())/(len(Igr)+len(Ier))
    if acc>best: best=acc; bestt=t
print('best acc (fidelity)', best, 'threshold', bestt)
print('stored Fid D', g['Fidelity'][()][0], '(1+D)/2', (1+g['Fidelity'][()][0])/2)

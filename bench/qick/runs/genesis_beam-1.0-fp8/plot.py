import h5py, numpy as np, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
Igr = Ig*np.cos(ang) - Qg*np.sin(ang)
Qgr = Ig*np.sin(ang) + Qg*np.cos(ang)
Ier = Ie*np.cos(ang) - Qe*np.sin(ang)
Qer = Ie*np.sin(ang) + Qe*np.cos(ang)

mg=np.median(Igr); me=np.median(Ier)
gclean = Igr[Igr <= mg]; eclean = Ier[Ier >= me]
sg=np.std(gclean, ddof=1); se=np.std(eclean, ddof=1)
dmu=abs(me-mg)
snr=dmu/np.sqrt((sg**2+se**2)/2)
allI=np.concatenate([Igr,Ier])
xs=np.sort(allI)
best=0; bestt=None
for t in xs:
    acc=((Igr<=t).sum()+(Ier>t).sum())/(len(Igr)+len(Ier))
    if acc>best: best=acc; bestt=t
F=best

fig, axes = plt.subplots(1,2, figsize=(10,4.5))
ax=axes[0]
ax.scatter(Igr, Qgr, s=6, c='tab:blue', alpha=0.4, label='|g>')
ax.scatter(Ier, Qer, s=6, c='tab:red', alpha=0.4, label='|e>')
ax.scatter([mg],[np.median(Qgr)], c='k', marker='x', s=60, label='median g')
ax.scatter([me],[np.median(Qer)], c='k', marker='+', s=60, label='median e')
ax.set_xlabel('I (rotated)')
ax.set_ylabel('Q (rotated)')
ax.set_title('(a) Rotated I-Q scatter')
ax.legend(markerscale=2, fontsize=8)

ax=axes[1]
bins=np.linspace(allI.min(), allI.max(), 61)
ax.hist(Igr, bins=bins, color='tab:blue', alpha=0.5, density=True, label='|g>')
ax.hist(Ier, bins=bins, color='tab:red', alpha=0.5, density=True, label='|e>')
x=np.linspace(allI.min(), allI.max(), 400)
ax.plot(x, (1/(sg*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mg)/sg)**2), 'b-', lw=2)
ax.plot(x, (1/(se*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-me)/se)**2), 'r-', lw=2)
ax.axvline(bestt, color='k', ls='--', lw=1)
ax.set_xlabel('I (rotated)')
ax.set_ylabel('Density')
ax.set_title('(b) Projected histogram')
ax.legend(fontsize=8)

fig.suptitle(f'Q1 single-shot readout: SNR={snr:.1f}, F={F*100:.1f}% (stored D={g["Fidelity"][()][0]:.3f})', fontsize=10)
fig.tight_layout()
fig.savefig('readout.png', dpi=150)
print('saved readout.png; F=', F)

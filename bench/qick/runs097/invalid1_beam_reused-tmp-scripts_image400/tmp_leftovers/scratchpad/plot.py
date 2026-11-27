import h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(g,k):
    raw=g[k][()][0].decode('utf-8').strip().strip('[]')
    return np.fromstring(raw, sep=' ')

path='/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick_runs/coder_beam-1.0-fp8/ss_ge.h5'
f=h5py.File(path,'r')
g=f['Q1']
Ig=load(g,'I_g'); Qg=load(g,'Q_g'); Ie=load(g,'I_e'); Qe=load(g,'Q_e')
ang=float(g['Angle'][()][0])
# rotate so separation is along I (convention A)
xg = Ig*np.cos(ang) - Qg*np.sin(ang)
xe = Ie*np.cos(ang) - Qe*np.sin(ang)
yg = Ig*np.sin(ang) + Qg*np.cos(ang)
ye = Ie*np.sin(ang) + Qe*np.cos(ang)

# max-contrast threshold
vals=np.sort(np.concatenate([xg,xe]))
mids=(vals[1:]+vals[:-1])/2
best=0; thr=None
for t in mids:
    acc=(np.sum(xg<=t)+np.sum(xe>t))/(len(xg)+len(xe))
    if acc>best: best=acc; thr=t

# clean-side Gaussian cores
mg=np.median(xg); me=np.median(xe)
cg=xg[xg<=mg]; ce=xe[xe>=me]

# fit gaussians to clean sides (mean=median of clean? use sample mean/std)
mu_g, sd_g = np.mean(cg), np.std(cg, ddof=1)
mu_e, sd_e = np.mean(ce), np.std(ce, ddof=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
# scatter
ax1.scatter(xg, yg, s=6, alpha=0.4, label='|g>', c='tab:blue')
ax1.scatter(xe, ye, s=6, alpha=0.4, label='|e>', c='tab:red')
ax1.axvline(thr, color='k', linestyle='--', linewidth=1.5, label='threshold')
ax1.set_xlabel('I (rotated, a.u.)')
ax1.set_ylabel('Q (rotated, a.u.)')
ax1.set_title('Q1 I-Q shots (rotated)')
ax1.legend(markerscale=3, fontsize=9)
ax1.grid(True, alpha=0.3)

# histogram projected onto rotated I
bins=np.linspace(min(xg.min(), xe.min()), max(xg.max(), xe.max()), 80)
ax2.hist(xg, bins=bins, density=True, alpha=0.5, color='tab:blue', label='|g>')
ax2.hist(xe, bins=bins, density=True, alpha=0.5, color='tab:red', label='|e>')
xs=np.linspace(bins[0], bins[-1], 400)
ax2.plot(xs, (1/(sd_g*np.sqrt(2*np.pi)))*np.exp(-0.5*((xs-mu_g)/sd_g)**2), color='tab:blue', linewidth=2, label='g Gaussian core')
ax2.plot(xs, (1/(sd_e*np.sqrt(2*np.pi)))*np.exp(-0.5*((xs-mu_e)/sd_e)**2), color='tab:red', linewidth=2, label='e Gaussian core')
ax2.axvline(thr, color='k', linestyle='--', linewidth=1.5, label='threshold')
ax2.set_xlabel('I (rotated, a.u.)')
ax2.set_ylabel('density')
ax2.set_title('Projected histogram (F=%.1f%%)'%(best*100))
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out='/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick_runs/coder_beam-1.0-fp8/readout.png'
fig.savefig(out, dpi=150)
print('saved', out)

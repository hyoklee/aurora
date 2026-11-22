import h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse(raw):
    s = raw.decode('utf-8', errors='replace').replace('\n', ' ')
    return np.fromstring(s.strip().strip('[]'), sep=' ')

def gauss(x, mu, sig):
    return np.exp(-0.5*((x-mu)/sig)**2)/(sig*np.sqrt(2*np.pi))

with h5py.File('ss_ge.h5', 'r') as f:
    g = f['Q1']
    Ig = parse(g['I_g'][0]); Qg = parse(g['Q_g'][0])
    Ie = parse(g['I_e'][0]); Qe = parse(g['Q_e'][0])
    ang = float(g['Angle'][0]); fid_stored = float(g['Fidelity'][0])

c, s = np.cos(ang), np.sin(ang)
rg, qg = Ig*c + Qg*s, -Ig*s + Qg*c
re_, qe = Ie*c + Qe*s, -Ie*s + Qe*c
if np.median(re_) < np.median(rg):
    rg, re_ = -rg, -re_
    qg, qe = -qg, -qe

ts = np.unique(np.concatenate([rg, re_]))
cg = np.searchsorted(np.sort(rg), ts, side='right')/len(rg)
ce = 1.0 - np.searchsorted(np.sort(re_), ts, side='left')/len(re_)
F = float(np.max(0.5*(cg+ce))); dF = np.sqrt(F*(1-F)/(len(rg)+len(re_)))
thr = ts[np.argmax(0.5*(cg+ce))]

mid = 0.5*(np.median(rg)+np.median(re_))
F_mid = 0.5*(np.mean(rg < mid) + np.mean(re_ > mid))
mean_mid = 0.5*(np.mean(rg)+np.mean(re_))
F_meanmid = 0.5*(np.mean(rg < mean_mid) + np.mean(re_ > mean_mid))
F_zero = 0.5*(np.mean(rg < 0) + np.mean(re_ > 0))
print(f'Q1: F_max={F:.6f} +/- {dF:.6f} ({F*100:.2f}% +/- {dF*100:.2f}%) thr={thr:.3f}')
print(f'    F(median-mid)={F_mid:.6f}  F(mean-mid)={F_meanmid:.6f}  F(thr=0)={F_zero:.6f}  stored={fid_stored:.6f}')

mug, mue = np.median(rg), np.median(re_)
sg = rg[rg <= mug]; se = re_[re_ >= mue]
sg_, se_ = sg.std(ddof=1), se.std(ddof=1)
dmu = mue - mug
sbar = np.sqrt(sg_**2+se_**2)/2
print(f'    clean-side sigma_g={sg_:.3f} sigma_e={se_:.3f} dmu={dmu:.3f} SNR={dmu/sbar:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ax = axes[0]
ax.scatter(qg, rg, s=3, alpha=0.35, c='tab:blue', label='|g> (3000 shots)')
ax.scatter(qe, re_, s=3, alpha=0.35, c='tab:red', label='|e> (3000 shots)')
ax.scatter([np.median(qg)], [mug], marker='x', s=120, c='black', linewidths=2.5, zorder=5)
ax.scatter([np.median(qe)], [mue], marker='x', s=120, c='black', linewidths=2.5, zorder=5)
ax.axhline(thr, color='k', ls='--', lw=1, label=f'opt. threshold ({thr:.1f})')
ax.set_xlabel('Q (rotated, a.u.)'); ax.set_ylabel('I (rotated, a.u.)')
ax.set_title('(a) Rotated I-Q scatter, Q1 (ss_ge.h5)')
ax.legend(loc='best', fontsize=8)

ax = axes[1]
bins = np.linspace(min(rg.min(), re_.min()), max(rg.max(), re_.max()), 121)
ax.hist(rg, bins=bins, density=True, alpha=0.55, color='tab:blue', label='|g>')
ax.hist(re_, bins=bins, density=True, alpha=0.55, color='tab:red', label='|e>')
xx = np.linspace(bins[0], bins[-1], 400)
ax.plot(xx, gauss(xx, sg.mean(), sg_), 'k--', lw=1.2, label=f'g clean core, sigma_g={sg_:.1f}')
ax.plot(xx, gauss(xx, se.mean(), se_), 'k-.', lw=1.2, label=f'e clean core, sigma_e={se_:.1f}')
ax.axvline(thr, color='k', ls=':', lw=1)
ax.set_xlabel('I (rotated, a.u.)'); ax.set_ylabel('density')
ax.set_title(f'(b) Projected histogram; F = {F*100:.2f}% +/- {dF*100:.2f}%')
ax.legend(fontsize=8)
fig.suptitle('QICK Box single-shot readout - reproduced from ss_ge.h5 (Q1), no hardware rerun')
fig.tight_layout()
fig.savefig('readout.png', dpi=150)
print('saved readout.png')

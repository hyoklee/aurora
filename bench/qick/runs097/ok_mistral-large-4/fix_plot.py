import h5py, numpy as np, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

f = h5py.File('ss_ge.h5','r')
def parse_arr(b):
    s = b.decode() if isinstance(b, bytes) else b
    return np.fromstring(s.strip()[1:-1].replace('\n',' '), sep=' ')

g = f['Q1']
Ig, Qg = parse_arr(g['I_g'][0]), parse_arr(g['Q_g'][0])
Ie, Qe = parse_arr(g['I_e'][0]), parse_arr(g['Q_e'][0])
a = float(g['Angle'][0])
rg = Ig*np.cos(a)-Qg*np.sin(a); qg2 = Ig*np.sin(a)+Qg*np.cos(a)
re_ = Ie*np.cos(a)-Qe*np.sin(a); qe2 = Ie*np.sin(a)+Qe*np.cos(a)
mg, me = float(np.median(rg)), float(np.median(re_))
dmu = abs(me-mg)
sg = float(np.std(rg[rg<=mg])); se = float(np.std(re_[re_>=me]))
# max cumulative contrast
ts = np.unique(np.concatenate([rg, re_]))
cg = np.searchsorted(np.sort(rg), ts, side='right')/len(rg)
ce = 1 - np.searchsorted(np.sort(re_), ts, side='right')/len(re_)
Fv = 0.5*(cg+ce); i = int(np.argmax(Fv)); F = float(Fv[i]); tbest = float(ts[i])
dF = math.sqrt(F*(1-F)/6000)

def core_fit(x, side):
    xs = x[x<=np.median(x)] if side=='g' else x[x>=np.median(x)]
    h, edges = np.histogram(xs, bins=40)
    c = 0.5*(edges[:-1]+edges[1:])
    popt, _ = curve_fit(lambda c,A,mu,s: A*np.exp(-0.5*((c-mu)/s)**2), c, h,
                        p0=[h.max(), np.mean(xs), np.std(xs)],
                        sigma=np.sqrt(np.maximum(h,1)), maxfev=20000)
    return popt
(Ag, mug, sg_f) = core_fit(rg, 'g')
(Ae, mue, se_f) = core_fit(re_, 'e')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
ax = axes[0]
ax.scatter(rg, qg2, s=4, c='tab:blue', alpha=0.35, label='|g> (3000 shots)')
ax.scatter(re_, qe2, s=4, c='tab:red', alpha=0.35, label='|e> (3000 shots)')
ax.scatter([mg], [np.median(qg2)], marker='X', s=140, c='black', zorder=5, label='median |g>')
ax.scatter([me], [np.median(qe2)], marker='X', s=140, c='black', zorder=5, label='median |e>')
ax.set_xlabel("rotated I = I cos$\\theta$ $-$ Q sin$\\theta$  [a.u.]")
ax.set_ylabel("rotated Q  [a.u.]")
ax.set_title("(a) Rotated I-Q scatter, qubit Q1 (3000 shots/state)")
ax.legend(loc='upper left', fontsize=8)

ax = axes[1]
bins = np.linspace(min(rg.min(), re_.min()), max(rg.max(), re_.max()), 121)
ax.hist(rg, bins=bins, color='tab:blue', alpha=0.55, label='|g> histogram')
ax.hist(re_, bins=bins, color='tab:red', alpha=0.55, label='|e> histogram')
xx = np.linspace(bins[0], bins[-1], 500)
ax.plot(xx, Ag*np.exp(-0.5*((xx-mug)/sg_f)**2), 'k--', lw=1.8, label=f'Gaussian core fit |g>: $\\sigma_g$={sg_f:.1f}')
ax.plot(xx, Ae*np.exp(-0.5*((xx-mue)/se_f)**2), 'k:', lw=1.8, label=f'Gaussian core fit |e>: $\\sigma_e$={se_f:.1f}')
ax.axvline(mg, color='tab:blue', lw=1, alpha=0.8)
ax.axvline(me, color='tab:red', lw=1, alpha=0.8)
ax.axvline(tbest, color='green', ls='--', lw=1.5, label=f'optimal threshold ({tbest:.1f})')
ax.set_xlabel("rotated I  [a.u.]")
ax.set_ylabel("counts")
ax.set_title("(b) Projected histogram along rotated I, qubit Q1")
ax.legend(loc='upper right', fontsize=8)

fig.suptitle(f"QICK Box single-shot readout, qubit Q1 (ss_ge.h5, 2025-10-19_12-46-02):  "
             f"$\\Delta\\mu$={dmu:.1f} a.u., clean-side $\\sigma_{{g,e}}$={sg:.1f}/{se:.1f} a.u.\n"
             f"max-cumulative-contrast fidelity F = {100*F:.2f} $\\pm$ {100*dF:.2f}%   "
             f"(paper Fig. 14: FSSF = 91.7 $\\pm$ 0.4 %)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig('readout.png', dpi=160)
print('F = %.4f +/- %.4f, dmu=%.2f, sg=%.2f, se=%.2f, tbest=%.2f' % (F, dF, dmu, sg, se, tbest))

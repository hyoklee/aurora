import h5py, numpy as np, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

f = h5py.File('ss_ge.h5','r')
def parse_arr(b):
    s = b.decode() if isinstance(b, bytes) else b
    return np.fromstring(s.strip()[1:-1].replace('\n',' '), sep=' ')

def max_cum_contrast(ig, ie):
    ts = np.unique(np.concatenate([ig, ie]))
    cg = np.searchsorted(np.sort(ig), ts, side='right')/len(ig)
    ce = 1 - np.searchsorted(np.sort(ie), ts, side='right')/len(ie)
    F = 0.5*(cg+ce); i = int(np.argmax(F))
    return float(F[i]), float(ts[i])

summary = {}
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g = f[q]
    Ig, Qg = parse_arr(g['I_g'][0]), parse_arr(g['Q_g'][0])
    Ie, Qe = parse_arr(g['I_e'][0]), parse_arr(g['Q_e'][0])
    a = float(g['Angle'][0])
    # rotation: I' = I cos a - Q sin a (maximizes separation, matches stored Angle)
    rg = Ig*np.cos(a)-Qg*np.sin(a); qg2 = Ig*np.sin(a)+Qg*np.cos(a)
    re_ = Ie*np.cos(a)-Qe*np.sin(a); qe2 = Ie*np.sin(a)+Qe*np.cos(a)
    mg, me = float(np.median(rg)), float(np.median(re_))
    dmu = abs(me-mg)
    sg = float(np.std(rg[rg<=mg])); se = float(np.std(re_[re_>=me]))
    sbar = math.sqrt((sg**2+se**2)/2)
    F, tbest = max_cum_contrast(rg, re_)
    summary[q] = dict(angle=a, N=len(rg), med_g=mg, med_e=me, dmu=dmu,
                      sigma_g_clean=sg, sigma_e_clean=se, snr_clean=dmu/sbar,
                      fidelity_maxcontrast=F, threshold=tbest,
                      fidelity_stored=float(g['Fidelity'][0]),
                      deltaF_binom=math.sqrt(F*(1-F)/(2*len(rg))))
    if q == 'Q1':
        np.savez('q1_rot.npz', rg=rg, qg=qg2, re=re_, qe=qe2)

# ---- Fig. 14 style plot for Q1 ----
d = np.load('q1_rot.npz')
rg, qg, re_, qe = d['rg'], d['qg'], d['re'], d['qe']
s = summary['Q1']
mg, me = s['med_g'], s['med_e']

def core_fit(x, side):
    xs = x[x<=np.median(x)] if side=='g' else x[x>=np.median(x)]
    h, edges = np.histogram(xs, bins=40)
    c = 0.5*(edges[:-1]+edges[1:])
    popt, _ = curve_fit(lambda c,A,mu,sg: A*np.exp(-0.5*((c-mu)/sg)**2), c, h,
                        p0=[h.max(), np.mean(xs), np.std(xs)],
                        sigma=np.sqrt(np.maximum(h,1)), maxfev=20000)
    return popt, c, h

(Ag, mug, sg_f), cg, hg = core_fit(rg, 'g')
(Ae, mue, se_f), ce, he = core_fit(re_, 'e')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
ax = axes[0]
ax.scatter(rg, qg, s=4, c='tab:blue', alpha=0.35, label='|g> (3000 shots)')
ax.scatter(re_, qe, s=4, c='tab:red', alpha=0.35, label='|e> (3000 shots)')
ax.scatter([mg], [np.median(qg)], marker='X', s=140, c='black', zorder=5, label='median |g>')
ax.scatter([me], [np.median(qe)], marker='X', s=140, c='black', zorder=5, label='median |e>')
ax.set_xlabel("rotated I  (I cos$\\theta$ $-$ Q sin$\\theta$)  [a.u.]")
ax.set_ylabel("rotated Q  [a.u.]")
ax.set_title("(a) Rotated I-Q scatter, Q1 (3000 shots/state)")
ax.legend(loc='best', fontsize=8)

ax = axes[1]
bins = np.linspace(min(rg.min(), re_.min()), max(rg.max(), re_.max()), 121)
ax.hist(rg, bins=bins, color='tab:blue', alpha=0.55, label='|g> histogram', density=False)
ax.hist(re_, bins=bins, color='tab:red', alpha=0.55, label='|e> histogram', density=False)
xx = np.linspace(bins[0], bins[-1], 500)
ax.plot(xx, Ag*np.exp(-0.5*((xx-mug)/sg_f)**2), 'k--', lw=1.8,
        label=f'Gaussian core fit |g>: $\\sigma_g$={sg_f:.1f}')
ax.plot(xx, Ae*np.exp(-0.5*((xx-mue)/se_f)**2), 'k:', lw=1.8,
        label=f'Gaussian core fit |e>: $\\sigma_e$={se_f:.1f}')
ax.axvline(mg, color='tab:blue', ls='-', lw=1, alpha=0.8)
ax.axvline(me, color='tab:red', ls='-', lw=1, alpha=0.8)
ax.axvline(s['threshold'], color='green', ls='--', lw=1.5, label=f"optimal threshold ({s['threshold']:.1f})")
ax.set_xlabel("rotated I  [a.u.]")
ax.set_ylabel("counts")
ax.set_title("(b) Projected histogram along rotated I, Q1")
ax.legend(loc='best', fontsize=8)

F = s['fidelity_maxcontrast']; dF = s['deltaF_binom']
fig.suptitle(f"QICK Box single-shot readout, qubit Q1 (file ss_ge.h5, 2025-10-19_12-46-02):  "
             f"$\\Delta\\mu$={s['dmu']:.1f}, $\\sigma_{{g,e}}$={s['sigma_g_clean']:.1f}/{s['sigma_e_clean']:.1f} (clean side), "
             f"max-cumulative-contrast fidelity F = {100*F:.2f} $\\pm$ {100*dF:.2f}%   "
             f"[paper Fig. 14: FSSF = (91.7 $\\pm$ 0.4)%]", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('readout.png', dpi=160)
print('saved readout.png')
print(json.dumps(summary, indent=1))
json.dump(summary, open('readout_summary.json','w'), indent=1)

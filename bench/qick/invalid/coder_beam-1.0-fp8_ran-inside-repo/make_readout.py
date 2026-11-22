import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH = 'ss_ge.h5'
Q = 'Q1'

def parse(s):
    if isinstance(s, (bytes, bytearray)):
        s = s.decode()
    return np.fromstring(s.replace('[', '').replace(']', ''), sep=' ')

f = h5py.File(PATH, 'r')
g = f[Q]
angle = float(g['Angle'][0])
Ig = parse(g['I_g'][0]); Ie = parse(g['I_e'][0])
Qg = parse(g['Q_g'][0]); Qe = parse(g['Q_e'][0])

# Rotate so the axis of maximum state separation lies on I.
Ig_r = Ig*np.cos(angle) - Qg*np.sin(angle)
Ie_r = Ie*np.cos(angle) - Qe*np.sin(angle)
Qg_r = Ig*np.sin(angle) + Qg*np.cos(angle)
Qe_r = Ie*np.sin(angle) + Qe*np.cos(angle)

# Threshold maximizing cumulative histogram contrast.
best_n, best_t = 0, None
for t in np.unique(np.concatenate([Ig_r, Ie_r])):
    n = int(np.sum(Ig_r <= t) + np.sum(Ie_r > t))
    if n > best_n:
        best_n, best_t = n, float(t)
F = best_n / (len(Ig_r) + len(Ie_r))

# Clean-side (outer half) Gaussian core estimates.
mg, me = np.median(Ig_r), np.median(Ie_r)
cg = Ig_r[Ig_r <= mg]
ce = Ie_r[Ie_r >= me]
sg_trunc = np.std(cg, ddof=1)
se_trunc = np.std(ce, ddof=1)
factor = 1.0/np.sqrt(1.0 - 2.0/np.pi)
sg = sg_trunc*factor
se = se_trunc*factor
dmu = abs(me - mg)
sbar = np.sqrt((sg*sg + se*se)/2.0)
snr = dmu/sbar

# Figure.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

ax1.scatter(Ig_r, Qg_r, s=6, alpha=0.45, color='tab:blue', label='|g>')
ax1.scatter(Ie_r, Qe_r, s=6, alpha=0.45, color='tab:red', label='|e>')
ax1.plot(mg, np.median(Qg_r), 'ko', ms=7)
ax1.plot(me, np.median(Qe_r), 'ko', ms=7)
ax1.set_xlabel('I (rotated)')
ax1.set_ylabel('Q (rotated)')
ax1.set_title(f'{Q} (qubit 0): rotated I-Q')
ax1.legend(loc='best', fontsize=9)
ax1.grid(alpha=0.25)

# Histogram.
bins = np.linspace(min(Ig_r.min(), Ie_r.min()), max(Ig_r.max(), Ie_r.max()), 70)
ax2.hist(Ig_r, bins=bins, density=True, alpha=0.35, color='tab:blue', label='|g>')
ax2.hist(Ie_r, bins=bins, density=True, alpha=0.35, color='tab:red', label='|e>')

xs = np.linspace(bins[0], bins[-1], 400)
# Gaussian core overlays (corrected widths).
ax2.plot(xs, (1/(sg*np.sqrt(2*np.pi)))*np.exp(-0.5*((xs-mg)/sg)**2),
         color='tab:blue', lw=2, label=f'g core (sigma={sg:.1f})')
ax2.plot(xs, (1/(se*np.sqrt(2*np.pi)))*np.exp(-0.5*((xs-me)/se)**2),
         color='tab:red', lw=2, label=f'e core (sigma={se:.1f})')
ax2.axvline(best_t, color='k', ls='--', lw=1.5, label=f'threshold={best_t:.1f}')
ax2.set_xlabel('I (rotated)')
ax2.set_ylabel('density')
ax2.set_title('Projected histogram + Gaussian cores')
ax2.legend(loc='best', fontsize=8)
ax2.grid(alpha=0.25)

fig.suptitle(
    f'QICK single-shot readout (ss_ge.h5, {Q}): '
    f'F={F*100:.2f}% ({best_n}/{len(Ig_r)+len(Ie_r)}), '
    f'SNR={snr:.2f}, dmu={dmu:.1f}, sigma_g/e={sg:.1f}/{se:.1f}',
    fontsize=11)
fig.tight_layout()
fig.savefig('readout.png', dpi=160)
print(f'Q={Q} angle={angle:.6f} F={F:.6f} count={best_n} threshold={best_t:.3f}')
print(f'dmu={dmu:.3f} sg={sg:.3f} se={se:.3f} snr={snr:.3f}')
print(f'raw-I F={maxF(Ig,Ie) if False else None}')

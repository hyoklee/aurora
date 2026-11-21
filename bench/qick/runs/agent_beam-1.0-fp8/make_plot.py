import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse(raw):
    s = raw.decode('ascii') if isinstance(raw, (bytes, np.bytes_)) else str(raw)
    s = s.strip()
    if s.startswith('['):
        s = s[1:]
    if s.endswith(']'):
        s = s[:-1]
    return np.fromstring(s, sep=' ')


def max_contrast(xg, xe):
    xs = np.sort(np.unique(np.concatenate([xg, xe])))
    ts = 0.5 * (xs[:-1] + xs[1:])
    sg = np.sort(xg)
    se = np.sort(xe)
    cg = np.searchsorted(sg, ts, side='right') / len(xg)
    ce = np.searchsorted(se, ts, side='right') / len(xe)
    F = 0.5 * (cg + (1.0 - ce))
    idx = int(np.argmax(F))
    D = float(np.max(cg - ce))
    return float(F[idx]), float(ts[idx]), D


f = h5py.File('ss_ge.h5', 'r')
q = 'Q1'
g = f[q]
Ig = parse(g['I_g'][0])
Qg = parse(g['Q_g'][0])
Ie = parse(g['I_e'][0])
Qe = parse(g['Q_e'][0])
theta = float(g['Angle'][0])

Xg = Ig * np.cos(theta) - Qg * np.sin(theta)
Yg = Ig * np.sin(theta) + Qg * np.cos(theta)
Xe = Ie * np.cos(theta) - Qe * np.sin(theta)
Ye = Ie * np.sin(theta) + Qe * np.cos(theta)

medg = float(np.median(Xg))
mede = float(np.median(Xe))
clean_g = Xg[Xg <= medg]
clean_e = Xe[Xe >= mede]
sig_g = float(np.sqrt(np.mean((clean_g - medg) ** 2)))
sig_e = float(np.sqrt(np.mean((clean_e - mede) ** 2)))

F, thr, D = max_contrast(Xg, Xe)
SNR = abs(mede - medg) / np.sqrt(0.5 * (sig_g ** 2 + sig_e ** 2))

fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

ax_scatter.scatter(Xg, Yg, s=4, alpha=0.35, c='tab:blue', label='|g>')
ax_scatter.scatter(Xe, Ye, s=4, alpha=0.35, c='tab:red', label='|e>')
ax_scatter.plot(medg, np.median(Yg), 'k+', ms=12, mew=2, label='medians')
ax_scatter.plot(mede, np.median(Ye), 'k+', ms=12, mew=2)
ax_scatter.set_xlabel('Rotated I (a.u.)')
ax_scatter.set_ylabel('Rotated Q (a.u.)')
ax_scatter.set_title('Q1 rotated I-Q scatter')
ax_scatter.legend(loc='best', fontsize=8, frameon=False)
ax_scatter.set_aspect('equal', adjustable='box')

bins = np.linspace(min(Xg.min(), Xe.min()), max(Xg.max(), Xe.max()), 71)
ax_hist.hist(Xg, bins=bins, density=True, alpha=0.45, color='tab:blue', label='|g>')
ax_hist.hist(Xe, bins=bins, density=True, alpha=0.45, color='tab:red', label='|e>')
x = np.linspace(bins[0], bins[-1], 500)
ax_hist.plot(x, (1.0 / (sig_g * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - medg) / sig_g) ** 2),
             'b--', lw=1.8, label='Gaussian core |g>')
ax_hist.plot(x, (1.0 / (sig_e * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mede) / sig_e) ** 2),
             'r--', lw=1.8, label='Gaussian core |e>')
ax_hist.axvline(thr, color='k', ls=':', lw=1.2, label='threshold = %.1f' % thr)
ax_hist.set_xlabel('Rotated I (a.u.)')
ax_hist.set_ylabel('Probability density')
ax_hist.set_title('Q1 projected histogram (F = %.2f%%)' % (100.0 * F))
ax_hist.legend(loc='best', fontsize=8, frameon=False)

fig.suptitle('QICK Box single-shot readout, qubit Q1 (qubit index 0)')
fig.savefig('readout.png', dpi=150)

print('q=%s  N_g=%d  N_e=%d' % (q, len(Xg), len(Xe)))
print('median_g=%.3f  median_e=%.3f  Delta_mu=%.3f' % (medg, mede, abs(mede - medg)))
print('clean N_g=%d  clean N_e=%d' % (len(clean_g), len(clean_e)))
print('sigma_g=%.3f  sigma_e=%.3f  SNR=%.3f' % (sig_g, sig_e, SNR))
print('F_max_contrast=%.6f  threshold=%.3f  KS_D=%.6f' % (F, thr, D))
print('stored Fidelity (KS D) = %.6f' % float(g['Fidelity'][0]))

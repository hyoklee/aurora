"""Reproduce the QICK Box single-shot readout plot (Fig. 14) and fidelity
from the saved shots in ss_ge.h5, without touching the quantum hardware.

Method follows the paper (FERMILAB-PUB-26-0193, Sec. III C):
  * rotate each shot's (I, Q) by the stored Angle so the separation axis
    aligns with the I quadrature,
  * project onto I,
  * fidelity = maximum cumulative histogram contrast:
      F = (1 + max_t [F_g(t) - F_e(t)]) / 2     (equal priors, g lower, e upper)
  * clean-side Gaussian cores: outer half of each distribution (|g|: I <= med_g,
    |e|: I >= med_e), sigma reconstructed for a full Gaussian from the half
    sample std (std_half = sigma*sqrt(1-2/pi)).
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PY = '/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/venv/bin/python'

def decode(raw):
    if isinstance(raw, np.ndarray):
        raw = raw.ravel()[0]
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return np.array(raw.strip().strip('[]').split(), dtype=float)

HALF_STD_FACTOR = np.sqrt(1.0 - 2.0 / np.pi)  # std of one half of a Gaussian

def load(q, f):
    g = f[q]
    theta = float(np.ravel(g['Angle'][()])[0])
    stored = float(np.ravel(g['Fidelity'][()])[0])
    Ig, Ie, Qg, Qe = (decode(g[k][()]) for k in ['I_g', 'I_e', 'Q_g', 'Q_e'])
    return theta, stored, Ig, Ie, Qg, Qe

def rotate(I, Q, theta):
    c, s = np.cos(theta), np.sin(theta)
    return I * c - Q * s, I * s + Q * c

def max_contrast(xg, xe, ngrid=20001):
    """One-sided max cumulative histogram contrast, g assumed lower than e."""
    lo = min(xg.min(), xe.min())
    hi = max(xg.max(), xe.max())
    grid = np.linspace(lo, hi, ngrid)
    # CDF of g at grid via searchsorted; CDF of e likewise
    fg = np.searchsorted(np.sort(xg), grid, side='right') / xg.size
    fe = np.searchsorted(np.sort(xe), grid, side='right') / xe.size
    contrast = fg - fe
    i = int(np.argmax(contrast))
    return float(contrast[i]), float(grid[i])

def clean_core(x, med, side):
    """Outer-half (clean-side) samples; return Gaussian sigma estimate."""
    if side == 'le':
        core = x[x <= med]
    else:
        core = x[x >= med]
    return core, float(np.std(core)) / HALF_STD_FACTOR

def analyze(q, f):
    theta, stored, Ig, Ie, Qg, Qe = load(q, f)
    Irg, Qrg = rotate(Ig, Qg, theta)
    Ire, Qre = rotate(Ie, Qe, theta)
    # ensure e is the upper distribution
    if np.median(Ire) < np.median(Irg):
        Irg, Ire = Ire, Irg
        Qrg, Qre = Qre, Qrg
    med_g, med_e = np.median(Irg), np.median(Ire)
    D, thr = max_contrast(Irg, Ire)
    F = 0.5 * (1.0 + D)
    N = Irg.size + Ire.size
    dF = np.sqrt(F * (1.0 - F) / N)
    core_g, sg = clean_core(Irg, med_g, 'le')
    core_e, se = clean_core(Ire, med_e, 'ge')
    dmu = abs(med_e - med_g)
    sbar = np.sqrt((sg ** 2 + se ** 2) / 2.0)
    snr = dmu / sbar
    return dict(
        q=q, theta=theta, stored=stored, F=F, D=D, thr=thr, dF=dF, N=N,
        med_g=med_g, med_e=med_e, dmu=dmu, sg=sg, se=se, sbar=sbar, snr=snr,
        n_g=core_g.size, n_e=core_e.size,
        Irg=Irg, Qrg=Qrg, Ire=Ire, Qre=Qre,
    )

def main():
    f = h5py.File('ss_ge.h5', 'r')
    results = {}
    for q in sorted(f.keys(), key=lambda k: int(k[1:])):
        results[q] = analyze(q, f)
    f.close()

    print('%-4s %-12s %-12s %-10s %-8s %-8s %-6s %-6s %-7s %-7s' % (
        'qubit', 'stored field', 'F=(1+D)/2', 'delta F', 'D', 'thr',
        'dmu', 'sg', 'se', 'SNR'))
    for q, r in results.items():
        print('%-4s %+.6f   %+.6f   %+.6f   %+.6f  %+8.1f  %+7.1f %+6.1f %+6.1f %+6.2f' % (
            q, r['stored'], r['F'], r['dF'], r['D'], r['thr'],
            r['dmu'], r['sg'], r['se'], r['snr']))

    r = results['Q1']
    print()
    print('Q1 (file label) = physical qubit 0 (Syst Config: list_of_all_qubits [0],')
    print('      f01 = 4194.7 MHz, readout resonator ~ 6226.9 MHz)')
    print('stored "Fidelity" field = %.6f  ->  this is the max cumulative histogram' % r['stored'])
    print('      contrast D, so the readout fidelity is F = (1 + D)/2 = %.6f' % r['F'])
    print('paper: (91.7 +/- 0.4)%%; binomial uncertainty here = %.4f (%.2f%%)' % (r['dF'], 100 * r['dF']))
    print('clean-side cores: n_g = %d, n_e = %d (paper: clean ~ 1500 per state)' % (r['n_g'], r['n_e']))

    # ---- Fig. 14 style plot ----
    fig = plt.figure(figsize=(9, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(r['Qrg'], r['Irg'], c='tab:blue', s=6, alpha=0.5, label='|g>', marker='.')
    ax.scatter(r['Qre'], r['Ire'], c='tab:red', s=6, alpha=0.5, label='|e>', marker='.')
    ax.scatter([r['med_g']], [np.median(r['Qrg'])], c='black', s=40, marker='o', zorder=5)
    ax.scatter([r['med_e']], [np.median(r['Qre'])], c='black', s=40, marker='o', zorder=5)
    ax.set_xlabel('Q (a.u.)')
    ax.set_ylabel('I (a.u.)')
    ax.set_title('Q1: rotated I-Q scatter (3000 shots per state)')
    ax.legend(frameon=False, fontsize=9)
    ax.axis('equal')

    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(min(r['Irg'].min(), r['Ire'].min()),
                       max(r['Irg'].max(), r['Ire'].max()), 60)
    ax.hist(r['Irg'], bins=bins, density=True, alpha=0.55, color='tab:blue',
            label='|g>', histtype='stepfilled')
    ax.hist(r['Ire'], bins=bins, density=True, alpha=0.55, color='tab:red',
            label='|e>', histtype='stepfilled')
    xx = np.linspace(bins[0], bins[-1], 400)
    for med, sig, col, lab in [(r['med_g'], r['sg'], 'tab:blue', '|g> core'),
                               (r['med_e'], r['se'], 'tab:red', '|e> core')]:
        gg = np.exp(-0.5 * ((xx - med) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        ax.plot(xx, gg, ls='--', color=col, label=lab)
    ax.axvline(r['thr'], color='black', ls=':', label='threshold')
    ax.set_xlabel('I after rotation (a.u.)')
    ax.set_ylabel('Probability density (a.u.)')
    ax.set_title('Q1: projected histogram with Gaussian cores')
    ax.legend(frameon=False, fontsize=8)
    txt = ('F = %.3f +/- %.3f\n' % (r['F'], r['dF']) +
           r'$\Delta\mu$ = %.1f, $\sigma_{\bar{}}$ = %.1f' % (r['dmu'], r['sbar']) +
           '\nSNR = %.2f' % r['snr'])
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    fig.suptitle('Single-shot qubit readout, QICK Box at QUIET (reproduced from ss_ge.h5)',
                 fontsize=11)
    fig.savefig('readout.png', dpi=150, bbox_inches='tight')
    print('saved readout.png')

if __name__ == '__main__':
    main()

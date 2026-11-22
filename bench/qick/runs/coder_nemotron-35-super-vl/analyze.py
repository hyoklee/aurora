import h5py
import numpy as np

f = h5py.File('ss_ge.h5', 'r')

def decode(raw):
    if isinstance(raw, np.ndarray):
        raw = raw.ravel()[0]
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return np.array(raw.strip().strip('[]').split(), dtype=float)

def rotate(I, Q, theta):
    # rotation aligning separation axis with I: convention A
    c, s = np.cos(theta), np.sin(theta)
    return I * c + Q * s, -I * s + Q * c

def empirical_fidelity(xg, xe):
    """Max cumulative histogram contrast: best threshold accuracy, equal priors."""
    # make sure e is the upper distribution (flip if needed)
    if np.median(xe) < np.median(xg):
        xg, xe = xe, xg
    thresholds = np.unique(np.concatenate([xg, xe]))
    # midpoints between consecutive unique values are enough; include endpoints
    best = 0.0; best_t = None
    # use fine grid of midpoints
    mids = (thresholds[:-1] + thresholds[1:]) / 2.0
    cand = np.concatenate([[thresholds[0] - 1], mids, [thresholds[-1] + 1]])
    for t in cand:
        acc = 0.5 * (np.mean(xg < t) + np.mean(xe >= t))
        if acc > best:
            best = acc; best_t = t
    return best, best_t

def clean_side_stats(xg, xe):
    mg, me = np.median(xg), np.median(xe)
    # g outer side: I <= mg ; e outer side: I >= me
    gc = xg[xg <= mg]
    ec = xe[xe >= me]
    sg = np.std(gc); se = np.std(ec)
    sbar = np.sqrt((sg**2 + se**2) / 2.0)
    dmu = abs(me - mg)
    return dmu, sg, se, sbar, dmu / sbar, len(gc), len(ec)

for q in sorted(f.keys(), key=lambda k: int(k[1:])):
    g = f[q]
    stored = float(np.ravel(g['Fidelity'][()])[0])
    theta = float(np.ravel(g['Angle'][()])[0])
    Ig, Ie, Qg, Qe = (decode(g[k][()]) for k in ['I_g', 'I_e', 'Q_g', 'Q_e'])

    print('=' * 60)
    print(q, 'stored Fidelity = %.4f' % stored, 'Angle = %.4f' % theta)
    for sign, name in [(+1, 'rot Ic+Qs'), (-1, 'rot Ic-Qs')]:
        Irg, Qrg = rotate(Ig, Qg, sign * theta)
        Ire, Qre = rotate(Ie, Qe, sign * theta)
        fid, t = empirical_fidelity(Irg, Ire)
        dmu, sg, se, sbar, snr, ng, ne = clean_side_stats(Irg, Ire)
        print('  [%s] empirical max-contrast fidelity = %.4f (threshold=%.1f)' % (name, fid, t))
        print('         medians g=%.1f e=%.1f | dmu=%.2f sg=%.2f se=%.2f sbar=%.2f SNR=%.2f clean n=(%d,%d)' % (
            np.median(Irg), np.median(Ire), dmu, sg, se, sbar, snr, ng, ne))
    # unrotated
    fid, t = empirical_fidelity(Ig, Ie)
    print('  [unrotated] empirical max-contrast fidelity = %.4f (threshold=%.1f)' % (fid, t))
    # simple midpoint threshold on rotated (Ic+Qs)
    Irg, _ = rotate(Ig, Qg, theta)
    Ire, _ = rotate(Ie, Qe, theta)
    if np.median(Ire) < np.median(Irg):
        Irg, Ire = Ire, Irg
    tmid = (np.mean(Irg) + np.mean(Ire)) / 2
    acc = 0.5 * (np.mean(Irg < tmid) + np.mean(Ire >= tmid))
    print('  [midpoint threshold] acc = %.4f' % acc)

f.close()

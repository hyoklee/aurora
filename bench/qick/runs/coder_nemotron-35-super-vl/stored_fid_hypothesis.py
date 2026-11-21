import h5py
import numpy as np

f = h5py.File('ss_ge.h5', 'r')

def decode(raw):
    if isinstance(raw, np.ndarray):
        raw = raw.ravel()[0]
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return np.array(raw.strip().strip('[]').split(), dtype=float)

q = 'Q1'
g = f[q]
stored = float(np.ravel(g['Fidelity'][()])[0])
theta = float(np.ravel(g['Angle'][()])[0])
Ig, Ie, Qg, Qe = (decode(g[k][()]) for k in ['I_g', 'I_e', 'Q_g', 'Q_e'])
print('stored Fidelity =', stored)

def acc(xg, xe, t):
    return 0.5 * (np.mean(xg < t) + np.mean(xe >= t))

def rot(I, Q, th):
    c, s = np.cos(th), np.sin(th)
    return I * c - Q * s, I * s + Q * c

Irg, Qrg = rot(Ig, Qg, theta)
Ire, Qre = rot(Ie, Qe, theta)
for name, (xg, xe) in [
    ('unrot I', (Ig, Ie)),
    ('unrot Q', (Qg, Qe)),
    ('rot I', (Irg, Ire)),
]:
    med_g, med_e = np.median(xg), np.median(xe)
    print('%s: medians g=%.2f e=%.2f | acc(t=0)=%.4f | acc(t=midmeans)=%.4f | acc(t=midmeds)=%.4f | best=%.4f' % (
        name, med_g, med_e,
        acc(xg, xe, 0.0),
        acc(xg, xe, (np.mean(xg) + np.mean(xe)) / 2),
        acc(xg, xe, (med_g + med_e) / 2),
        max(acc(xg, xe, t) for t in np.unique(np.concatenate([xg, xe])))))

Irg, Qrg = rot(Ig, Qg, theta)
Ire, Qre = rot(Ie, Qe, theta)
# make e upper
if np.median(Ire) < np.median(Irg):
    Irg, Ire = Ire, Irg
print('rot I (Ic-Qs), e-upper: medians g=%.2f e=%.2f' % (np.median(Irg), np.median(Ire)))
print('  acc(t=0)=%.4f | acc(t=midmeans)=%.4f | acc(t=midmeds)=%.4f' % (
    acc(Irg, Ire, 0.0),
    acc(Irg, Ire, (np.mean(Irg) + np.mean(Ire)) / 2),
    acc(Irg, Ire, (np.median(Irg) + np.median(Ire)) / 2)))

# Gaussian-overlap fidelity estimates (full-distribution stds)
from math import erfc, sqrt
for name, (xg, xe) in [('unrot I', (Ig, Ie)), ('rot I', (Irg, Ire)), ('unrot Q', (Qg, Qe))]:
    dmu = abs(np.mean(xe) - np.mean(xg))
    sbar = sqrt((np.std(xg)**2 + np.std(xe)**2) / 2)
    snr = dmu / sbar
    F = 1 - 0.5 * erfc(snr / (2 * sqrt(2)))
    print('%s Gaussian-overlap F=%.4f (SNR=%.2f)' % (name, F, snr))

f.close()

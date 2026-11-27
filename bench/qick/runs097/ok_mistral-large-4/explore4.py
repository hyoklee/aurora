import h5py, numpy as np
from scipy.optimize import curve_fit
f = h5py.File('ss_ge.h5','r')
def parse_arr(b):
    s = b.decode() if isinstance(b, bytes) else b
    return np.fromstring(s.strip()[1:-1].replace('\n',' '), sep=' ')

g = f['Q1']
Ig, Qg = parse_arr(g['I_g'][0]), parse_arr(g['Q_g'][0])
Ie, Qe = parse_arr(g['I_e'][0]), parse_arr(g['Q_e'][0])
a = g['Angle'][0]
rg = Ig*np.cos(a)-Qg*np.sin(a); re_ = Ie*np.cos(a)-Qe*np.sin(a)
mg, me = np.median(rg), np.median(re_)
print('medians:', mg, me, 'dmu=', abs(me-mg))

def gauss_core_fit(x, side):
    xs = x[x<=np.median(x)] if side=='g' else x[x>=np.median(x)]
    n = 40
    h, edges = np.histogram(xs, bins=n)
    c = 0.5*(edges[:-1]+edges[1:])
    # fit gaussian to counts
    def model(c, A, mu, s): return A*np.exp(-0.5*((c-mu)/s)**2)
    p0 = [h.max(), np.mean(xs), np.std(xs)]
    popt, _ = curve_fit(model, c, h, p0=p0, sigma=np.sqrt(np.maximum(h,1)), maxfev=20000)
    return popt

for name, x in [('g', rg), ('e', re_)]:
    A, mu, s = gauss_core_fit(x, name)
    print(f'{name}: core-fit mu={mu:.2f} sigma={s:.3f} (sample std clean side={np.std(x[x<=np.median(x)] if name=="g" else x[x>=np.median(x)]):.3f}, N_clean={(x<=np.median(x)).sum() if name=="g" else (x>=np.median(x)).sum()})')

Ag, mug, sg = gauss_core_fit(rg, 'g')
Ae, mue, se = gauss_core_fit(re_, 'e')
dmu = abs(me-mg)
sbar = np.sqrt((sg**2+se**2)/2)
print(f'Gaussian-core-fit: dmu(median)={dmu:.2f} sg={sg:.3f} se={se:.3f} sbar={sbar:.3f} SNR={dmu/sbar:.3f}')
print(f'scale check: if dmu=28 a.u., 1 a.u. = {dmu/28:.3f} raw -> sg={sg/(dmu/28):.3f} a.u., se={se/(dmu/28):.3f} a.u.')
# uncertainties per paper eq.4,5 with N=3000, Nclean~1500
import math
N=3000
print('delta_mu =', sg*math.sqrt(math.pi/(2*N)), 'delta_sigma =', sg*math.sqrt(1/(2*1500)))
# binomial fidelity uncertainty
F = 0.9181666666666667
print('delta F binomial N=6000:', math.sqrt(F*(1-F)/6000))

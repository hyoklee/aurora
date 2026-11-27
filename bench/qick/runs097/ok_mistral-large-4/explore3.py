import h5py, numpy as np
f = h5py.File('ss_ge.h5','r')
def parse_arr(b):
    s = b.decode() if isinstance(b, bytes) else b
    return np.fromstring(s.strip()[1:-1].replace('\n',' '), sep=' ')

def max_cum(ig, ie):
    allv = np.concatenate([ig, ie]); ts = np.unique(allv)
    cg = np.searchsorted(np.sort(ig), ts, side='right')/len(ig)
    ce = 1 - np.searchsorted(np.sort(ie), ts, side='right')/len(ie)
    F = 0.5*(cg+ce); i = np.argmax(F); return F[i], ts[i]

print('--- rotation conventions, Q1 ---')
g = f['Q1']
Ig, Qg = parse_arr(g['I_g'][0]), parse_arr(g['Q_g'][0])
Ie, Qe = parse_arr(g['I_e'][0]), parse_arr(g['Q_e'][0])
a = g['Angle'][0]
for name, rg, re_ in [
    ('cos - sin (a)', Ig*np.cos(a)-Qg*np.sin(a), Ie*np.cos(a)-Qe*np.sin(a)),
    ('cos + sin (a)', Ig*np.cos(a)+Qg*np.sin(a), Ie*np.cos(a)+Qe*np.sin(a)),
    ('cos - sin (-a)', Ig*np.cos(a)+Qg*np.sin(a), Ie*np.cos(a)+Qe*np.sin(a)),
    ('raw I', Ig, Ie),
    ('raw Q', Qg, Qe),
]:
    Fm, t = max_cum(rg, re_)
    mg, me = np.median(rg), np.median(re_)
    tm = 0.5*(mg+me)
    Fmid = 0.5*(np.mean(rg<=tm)+np.mean(re_>tm))
    print(f"{name:16s} Fmax={Fm:.6f} Fmid={Fmid:.6f} dmu={abs(me-mg):8.2f}")

print()
print('--- try to reproduce stored Fidelity values ---')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g = f[q]
    Ig, Qg = parse_arr(g['I_g'][0]), parse_arr(g['Q_g'][0])
    Ie, Qe = parse_arr(g['I_e'][0]), parse_arr(g['Q_e'][0])
    a = g['Angle'][0]; fst = g['Fidelity'][0]
    rg = Ig*np.cos(a)-Qg*np.sin(a); re_ = Ie*np.cos(a)-Qe*np.sin(a)
    mg, me = np.median(rg), np.median(re_)
    mug, mue = np.mean(rg), np.mean(re_)
    variants = {
      'Fmax': max_cum(rg, re_)[0],
      'Fmid_med': 0.5*(np.mean(rg<=0.5*(mg+me))+np.mean(re_>0.5*(mg+me))),
      'Fmid_mean': 0.5*(np.mean(rg<=0.5*(mug+mue))+np.mean(re_>0.5*(mug+mue))),
      'FrawI_midmed': 0.5*(np.mean(Ig<=0.5*(np.median(Ig)+np.median(Ie)))+np.mean(Ie>0.5*(np.median(Ig)+np.median(Ie)))),
      'FrawI_max': max_cum(Ig, Ie)[0],
      'FrawQ_max': max_cum(Qg, Qe)[0],
    }
    print(q, 'stored=%.10f' % fst, {k: round(v,6) for k,v in variants.items()})
    # also: maybe stored fidelity = mean of per-shot 0/1 with threshold from config? check fraction formats
    print('   stored*6000=%.3f stored*3000=%.3f' % (fst*6000, fst*3000))

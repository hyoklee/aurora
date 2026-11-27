import h5py, numpy as np

f = h5py.File('ss_ge.h5','r')

def parse_arr(b):
    s = b.decode() if isinstance(b, bytes) else b
    return np.fromstring(s.strip()[1:-1].replace('\n',' '), sep=' ')

def max_cum_contrast(ig, ie):
    # F(t) = 0.5*P(g<=t) + 0.5*P(e>t); maximize over all sample thresholds
    allv = np.concatenate([ig, ie])
    ts = np.unique(allv)
    cg = np.searchsorted(np.sort(ig), ts, side='right')/len(ig)
    ce = 1 - np.searchsorted(np.sort(ie), ts, side='right')/len(ie)
    F = 0.5*(cg+ce)
    i = np.argmax(F)
    return F[i], ts[i]

def mid_fid(ig, ie):
    t = 0.5*(np.median(ig)+np.median(ie))
    return 0.5*(np.mean(ig<=t)+np.mean(ie>t)), t

print(f"{'Q':3s} {'N_g':>5s} {'N_e':>5s} {'angle':>8s} {'sep(cos+sin)':>12s} {'sep(cos-sin)':>12s} {'med_g':>9s} {'med_e':>9s} {'dmu':>8s} {'sig_g':>7s} {'sig_e':>7s} {'SNR':>6s} {'Fmax':>7s} {'Fmid':>7s} {'Fstored':>8s}")
results = {}
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g = f[q]
    Ig, Qg = parse_arr(g['I_g'][0]), parse_arr(g['Q_g'][0])
    Ie, Qe = parse_arr(g['I_e'][0]), parse_arr(g['Q_e'][0])
    a = g['Angle'][0]; fst = g['Fidelity'][0]
    # try both rotation conventions
    sep_plus  = abs(np.mean(Ie*np.cos(a)+Qe*np.sin(a)) - np.mean(Ig*np.cos(a)+Qg*np.sin(a)))
    sep_minus = abs(np.mean(Ie*np.cos(a)-Qe*np.sin(a)) - np.mean(Ig*np.cos(a)-Qg*np.sin(a)))
    # pick convention with larger separation
    if sep_plus >= sep_minus:
        rg = Ig*np.cos(a)+Qg*np.sin(a); re_ = Ie*np.cos(a)+Qe*np.sin(a)
    else:
        rg = Ig*np.cos(a)-Qg*np.sin(a); re_ = Ie*np.cos(a)-Qe*np.sin(a)
    mg, me = np.median(rg), np.median(re_)
    dmu = abs(me-mg)
    sg = np.std(rg[rg<=mg]); se = np.std(re_[re_>=me])
    sbar = np.sqrt((sg**2+se**2)/2)
    snr = dmu/sbar
    Fmax, tbest = max_cum_contrast(rg, re_)
    Fmid, tmid = mid_fid(rg, re_)
    results[q] = dict(rg=rg, re=re_, Qg=Qg, Qe=Qe, Ig=Ig, Ie=Ie, a=a, mg=mg, me=me, dmu=dmu, sg=sg, se=se, sbar=sbar, snr=snr, Fmax=Fmax, tbest=tbest, Fmid=Fmid, tmid=tmid, fst=fst)
    print(f"{q:3s} {len(rg):5d} {len(re_):5d} {a:8.4f} {sep_plus:12.2f} {sep_minus:12.2f} {mg:9.2f} {me:9.2f} {dmu:8.2f} {sg:7.3f} {se:7.3f} {snr:6.3f} {Fmax:7.4f} {Fmid:7.4f} {fst:8.4f}")

import json
# save summary
summ = {q: {k: (v if not isinstance(v, np.ndarray) else None) for k,v in r.items()} for q,r in results.items()}
print(json.dumps({q:{k:float(v) for k,v in r.items() if np.isscalar(v)} for q,r in results.items()}, indent=1))
np.savez('results.npz', **{f"{q}_{k}": v for q,r in results.items() for k,v in r.items() if isinstance(v,np.ndarray)})

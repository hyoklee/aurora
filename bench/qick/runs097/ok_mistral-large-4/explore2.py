import h5py, numpy as np
f = h5py.File('ss_ge.h5','r')
def parse_arr(b):
    s = b.decode() if isinstance(b, bytes) else b
    return np.fromstring(s.strip()[1:-1].replace('\n',' '), sep=' ')

for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g = f[q]
    Ig, Qg = parse_arr(g['I_g'][0]), parse_arr(g['Q_g'][0])
    Ie, Qe = parse_arr(g['I_e'][0]), parse_arr(g['Q_e'][0])
    a = g['Angle'][0]
    # optimal rotation angle from mean difference (PCA-like)
    d = np.array([np.mean(Ie)-np.mean(Ig), np.mean(Qe)-np.mean(Qg)])
    theta_opt = np.arctan2(d[1], d[0])
    for name, ang in [('stored', a), ('stored_neg', -a), ('opt', theta_opt)]:
        rg = Ig*np.cos(ang) - Qg*np.sin(ang)
        re_ = Ie*np.cos(ang) - Qe*np.sin(ang)
        mg, me = np.median(rg), np.median(re_)
        dmu = abs(me-mg)
        # clean side by median
        sg_c = np.std(rg[rg<=mg]); se_c = np.std(re_[re_>=me])
        # full std
        sg_f = np.std(rg); se_f = np.std(re_)
        # clean side by mean
        mug, mue = np.mean(rg), np.mean(re_)
        sg_m = np.std(rg[rg<=mug]); se_m = np.std(re_[re_>=mue])
        # gaussian fit to clean-side histogram (least squares on binned pdf)
        def gaussfit(x, lo, hi):
            # fit gaussian to histogram of x in [lo,hi] range samples
            xs = x[(x>=lo)&(x<=hi)]
            if len(xs)<50: return np.nan, np.nan
            h, edges = np.histogram(xs, bins=60)
            c = 0.5*(edges[:-1]+edges[1:])
            w = np.sqrt(np.maximum(h,1))
            # fit log(h) = A - (c-mu)^2/(2 s^2)
            A = np.polyfit((c-np.mean(xs))**2, np.log(np.maximum(h,0.5)), 1)
            s2 = -1/(2*A[0]); s = np.sqrt(max(s2,1e-9))
            mu = np.mean(xs)
            return mu, s
        mug2, sg_fg = gaussfit(rg, -np.inf, mg)
        mue2, se_fg = gaussfit(re_, me, np.inf)
        sbar_c = np.sqrt((sg_c**2+se_c**2)/2)
        sbar_f = np.sqrt((sg_f**2+se_f**2)/2)
        sbar_m = np.sqrt((sg_m**2+se_m**2)/2)
        sbar_fg = np.sqrt((sg_fg**2+se_fg**2)/2)
        print(f"{q} {name:10s} ang={ang:+.4f} dmu={dmu:8.2f} | clean-med: sg={sg_c:6.2f} se={se_c:6.2f} SNR={dmu/sbar_c:5.2f} | full: SNR={dmu/sbar_f:5.2f} | clean-mean: SNR={dmu/sbar_m:5.2f} | gaussfit-clean: sg={sg_fg:6.2f} se={se_fg:6.2f} SNR={dmu/sbar_fg:5.2f}")
    print()

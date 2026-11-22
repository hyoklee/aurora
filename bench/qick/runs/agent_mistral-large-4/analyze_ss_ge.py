import h5py, numpy as np

def parse(raw):
    s = raw.decode('utf-8', errors='replace').replace('\n', ' ')
    return np.fromstring(s.strip().strip('[]'), sep=' ')

results = {}
with h5py.File('ss_ge.h5', 'r') as f:
    for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
        g = f[q]
        Ig = parse(g['I_g'][0]); Qg = parse(g['Q_g'][0])
        Ie = parse(g['I_e'][0]); Qe = parse(g['Q_e'][0])
        fid_stored = float(g['Fidelity'][0]); ang_stored = float(g['Angle'][0])
        print(f'== {q}: stored Fidelity={fid_stored:.6f} ({fid_stored*100:.2f}%)  stored Angle={ang_stored:.6f} rad')
        print(f'   shots: Ng={len(Ig)} (Q {len(Qg)})  Ne={len(Ie)} (Q {len(Qe)})')
        mg = np.array([np.median(Ig), np.median(Qg)])
        me = np.array([np.median(Ie), np.median(Qe)])
        d = me - mg
        ang_data = np.arctan2(d[1], d[0])
        print(f'   median g=({mg[0]:.3f},{mg[1]:.3f})  median e=({me[0]:.3f},{me[1]:.3f})  sep={np.hypot(*d):.3f}  angle_from_medians={ang_data:.6f}')
        for label, ang in [('stored', ang_stored), ('data', ang_data)]:
            c, s = np.cos(ang), np.sin(ang)
            rg = Ig*c + Qg*s
            re_ = Ie*c + Qe*s
            ts = np.unique(np.concatenate([rg, re_]))
            cg = np.searchsorted(np.sort(rg), ts, side='right')/len(rg)
            ce = 1.0 - np.searchsorted(np.sort(re_), ts, side='left')/len(re_)
            F = np.max(0.5*(cg+ce))
            ntot = len(rg)+len(re_)
            dF = np.sqrt(F*(1-F)/ntot)
            print(f'   [{label} ang] F_max={F:.6f} = {F*100:.2f}% +/- {dF*100:.2f}%')
        # clean-side gaussian core widths (paper method)
        c, s = np.cos(ang_data), np.sin(ang_data)
        rg = Ig*c + Qg*s; re_ = Ie*c + Qe*s
        mug, mue = np.median(rg), np.median(re_)
        sg = rg[rg <= mug]; se = re_[re_ >= mue]
        sg_, se_ = sg.std(ddof=1), se.std(ddof=1)
        dmu = abs(np.median(re_) - np.median(rg))
        sbar = np.sqrt(sg_**2 + se_**2)/2
        print(f'   clean-side: sigma_g={sg_:.3f} (n={len(sg)}), sigma_e={se_:.3f} (n={len(se)}), dmu={dmu:.3f}, SNR={dmu/sbar:.3f}')
        results[q] = dict(fid_stored=fid_stored, ang_stored=ang_stored, ang_data=ang_data,
                          Ig=Ig, Qg=Qg, Ie=Ie, Qe=Qe)
        print()

import pickle
with open('parsed_shots.pkl','wb') as fh:
    pickle.dump(results, fh)
print('saved parsed_shots.pkl')

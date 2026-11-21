import h5py, numpy as np, math

def parse_arr(ds):
    s = ds[0]
    if isinstance(s, bytes):
        s = s.decode()
    s = s.strip()
    return np.fromstring(s[1:-1], sep=' ')

def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def max_threshold_fid(xg, xe):
    xs = np.sort(np.concatenate([xg, xe]))
    c = (xs[:-1] + xs[1:]) / 2
    cdf_g = np.searchsorted(np.sort(xg), c, side='right') / len(xg)
    sf_e = 1 - np.searchsorted(np.sort(xe), c, side='right') / len(xe)
    F = (cdf_g + sf_e) / 2
    k = int(np.argmax(F))
    return F[k], c[k]

with h5py.File("ss_ge.h5", "r") as f:
    for i in range(1, 7):
        q = "Q%d" % i
        g = f[q]
        Ig, Qg, Ie, Qe = (parse_arr(g[k]) for k in ("I_g", "Q_g", "I_e", "Q_e"))
        ang = float(g["Angle"][0])
        fid = float(g["Fidelity"][0])
        variants = {
            "rawI": (Ig, Ie),
            "rawQ": (Qg, Qe),
            "rotB": (Ig * np.cos(ang) - Qg * np.sin(ang), Ie * np.cos(ang) - Qe * np.sin(ang)),
            "rotA": (Ig * np.cos(ang) + Qg * np.sin(ang), Ie * np.cos(ang) + Qe * np.sin(ang)),
        }
        d = np.array([np.median(Ie) - np.median(Ig), np.median(Qe) - np.median(Qg)])
        u = d / np.hypot(*d)
        xgC = Ig * u[0] + Qg * u[1]
        xeC = Ie * u[0] + Qe * u[1]
        if np.median(xeC) < np.median(xgC):
            xgC, xeC = -xgC, -xeC
        variants["sepaxis"] = (xgC, xeC)
        print("=== %s  stored=%r" % (q, fid))
        for name, (xg, xe) in variants.items():
            Fmax, t = max_threshold_fid(xg, xe)
            mug, mue = xg.mean(), xe.mean()
            sg, se = xg.std(), xe.std()
            Fg_emp_t = 0.5 * (Phi((t - mug) / sg) + Phi((mue - t) / se))
            tm = (mug + mue) / 2
            Fg_mid = 0.5 * (Phi((tm - mug) / sg) + Phi((mue - tm) / se))
            mg, me = np.median(xg), np.median(xe)
            sgc = xg[xg <= mg].std()
            sec = xe[xe >= me].std()
            Fg_clean_mid = 0.5 * (Phi((tm - mug) / sgc) + Phi((mue - tm) / sec))
            Fmid = ((xg <= (mg + me) / 2).mean() + (xe > (mg + me) / 2).mean()) / 2
            print("  %-8s Fmax=%.6f Fmid(med)=%.6f | gauss@tmp=%.6f gauss@mid=%.6f gaussClean@mid=%.6f"
                  % (name, Fmax, Fmid, Fg_emp_t, Fg_mid, Fg_clean_mid))

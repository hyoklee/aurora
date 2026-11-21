import h5py, numpy as np, math

def parse_arr(ds):
    s = ds[0]
    if isinstance(s, bytes):
        s = s.decode()
    s = s.strip()
    return np.fromstring(s[1:-1], sep=' ')

with h5py.File("ss_ge.h5", "r") as f:
    for i in range(1, 7):
        q = "Q%d" % i
        g = f[q]
        arrs = {k: parse_arr(g[k]) for k in ("I_g", "Q_g", "I_e", "Q_e")}
        naninfo = {k: int(np.isnan(v).sum()) for k, v in arrs.items()}
        ang = float(g["Angle"][0])
        fid = float(g["Fidelity"][0])
        print("=== %s stored=%.16f nan=%s" % (q, fid, naninfo))
        # candidate projections
        Ig, Qg, Ie, Qe = arrs["I_g"], arrs["Q_g"], arrs["I_e"], arrs["Q_e"]
        variants = {
            "rawI": (Ig, Ie), "rawQ": (Qg, Qe),
            "rotB": (Ig * np.cos(ang) - Qg * np.sin(ang), Ie * np.cos(ang) - Qe * np.sin(ang)),
            "rotA": (Ig * np.cos(ang) + Qg * np.sin(ang), Ie * np.cos(ang) + Qe * np.sin(ang)),
        }
        dd = np.array([np.median(Ie) - np.median(Ig), np.median(Qe) - np.median(Qg)])
        u = dd / np.hypot(*dd)
        xg, xe = Ig * u[0] + Qg * u[1], Ie * u[0] + Qe * u[1]
        if np.median(xe) < np.median(xg):
            xg, xe = -xg, -xe
        variants["sepaxis"] = (xg, xe)
        # find N and count
        Ntot = None
        for N in range(2, 12001):
            v = fid * N
            if abs(v - round(v)) < 1e-7 and N >= 5000:
                Ntot, cnt = N, round(v)
                break
        print("   Ntot=%s count=%s" % (Ntot, cnt))
        if Ntot is None:
            continue
        for name, (a, b) in variants.items():
            a2, b2 = a[~np.isnan(a)], b[~np.isnan(b)]
            xs = np.sort(np.concatenate([a2, b2]))
            c = (xs[:-1] + xs[1:]) / 2
            cnt_t = (np.searchsorted(np.sort(a2), c, side='right')
                     + (len(b2) - np.searchsorted(np.sort(b2), c, side='right')))
            hits = np.where(cnt_t == cnt)[0]
            if len(hits):
                # special thresholds within hit range
                med_a, med_b = np.median(a2), np.median(b2)
                specials = {"t=0": 0.0, "mid(med)": (med_a + med_b) / 2,
                            "mid(mean)": (a2.mean() + b2.mean()) / 2,
                            "med_a": med_a, "med_b": med_b, "pooled_med": np.median(np.concatenate([a2, b2]))}
                inside = [k for k, tv in specials.items() if hits[0] < np.searchsorted(c, tv) < hits[-1] + 1 and cnt_t[np.searchsorted(c, tv)] == cnt] if len(c) else []
                print("   %-8s count==%d for t in [%.4f, %.4f]; special thresholds hitting: %s"
                      % (name, cnt, c[hits[0]], c[hits[-1]], inside))

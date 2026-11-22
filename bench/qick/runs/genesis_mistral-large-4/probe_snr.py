import h5py, numpy as np, math

def parse_arr(ds):
    s = ds[0]
    if isinstance(s, bytes):
        s = s.decode()
    s = s.strip()
    return np.fromstring(s[1:-1], sep=' ')

def max_threshold_fid(xg, xe):
    xs = np.sort(np.concatenate([xg, xe]))
    c = (xs[:-1] + xs[1:]) / 2
    cdf_g = np.searchsorted(np.sort(xg), c, side='right') / len(xg)
    sf_e = 1 - np.searchsorted(np.sort(xe), c, side='right') / len(xe)
    F = (cdf_g + sf_e) / 2
    return F.max()

with h5py.File("ss_ge.h5", "r") as f:
    for i in range(1, 7):
        q = "Q%d" % i
        g = f[q]
        Ig, Qg, Ie, Qe = (parse_arr(g[k]) for k in ("I_g", "Q_g", "I_e", "Q_e"))
        ang = float(g["Angle"][0])
        xg = Ig * np.cos(ang) - Qg * np.sin(ang)
        xe = Ie * np.cos(ang) - Qe * np.sin(ang)
        if np.median(xe) < np.median(xg):
            xg, xe = -xg, -xe
        mg, me = np.median(xg), np.median(xe)
        # clean sides (outer halves relative to medians)
        sgc = xg[xg <= mg].std()
        sec = xe[xe >= me].std()
        sbar_c = math.sqrt((sgc ** 2 + sec ** 2) / 2)
        sga, sea = xg.std(), xe.std()
        sbar_a = math.sqrt((sga ** 2 + sea ** 2) / 2)
        dmed = abs(me - mg)
        dmean = abs(xe.mean() - xg.mean())
        F = max_threshold_fid(xg, xe)
        print("%s: dmed=%.2f dmean=%.2f | clean: sg=%.2f se=%.2f sbar=%.2f SNR=%.3f | all: sbar=%.2f SNR=%.3f | Fmax=%.4f"
              % (q, dmed, dmean, sgc, sec, sbar_c, dmed / sbar_c, sbar_a, dmed / sbar_a, F))
        # also try clean = outer 25% and outer 75%
        for frac, lbl in [(0.25, "outer25"), (0.75, "outer75")]:
            n_g = int(len(xg) * frac)
            n_e = int(len(xe) * frac)
            sg = np.sort(xg)[:n_g].std() if me > mg else np.sort(xg)[-n_g:].std()
            se = np.sort(xe)[-n_e:].std() if me > mg else np.sort(xe)[:n_e].std()
            sb = math.sqrt((sg ** 2 + se ** 2) / 2)
            print("     %s: sg=%.2f se=%.2f SNR=%.3f" % (lbl, sg, se, dmed / sb))

import h5py, numpy as np, math

def parse_arr(ds):
    s = ds[0]
    if isinstance(s, bytes):
        s = s.decode()
    s = s.strip()
    return np.fromstring(s[1:-1], sep=' ')

def gauss_fit(x, nbins=60):
    """Fit A*exp(-(x-mu)^2/(2 s^2)) to the histogram of x via simple least squares on log? Use moment-free grid fit."""
    hist, edges = np.histogram(x, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    # grid search over mu, s minimizing squared error vs hist (amplitude free)
    best = None
    mu0, s0 = x.mean(), x.std()
    for mu in np.linspace(mu0 - s0, mu0 + s0, 121):
        for s in np.linspace(0.2 * s0, 2.0 * s0, 181):
            g = np.exp(-(centers - mu) ** 2 / (2 * s ** 2))
            A = (hist @ g) / (g @ g)
            err = np.sum((hist - A * g) ** 2)
            if best is None or err < best[0]:
                best = (err, mu, s, A)
    return best[1], best[2]

def max_threshold_fid(xg, xe):
    xs = np.sort(np.concatenate([xg, xe]))
    c = (xs[:-1] + xs[1:]) / 2
    cdf_g = np.searchsorted(np.sort(xg), c, side='right') / len(xg)
    sf_e = 1 - np.searchsorted(np.sort(xe), c, side='right') / len(xe)
    return ((cdf_g + sf_e) / 2).max()

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
        mug, sg = gauss_fit(xg)
        mue, se = gauss_fit(xe)
        d = abs(mue - mug)
        sbar = math.sqrt((sg ** 2 + se ** 2) / 2)
        F = max_threshold_fid(xg, xe)
        print("%s: gaussfit mug=%.1f sg=%.2f mue=%.1f se=%.2f dmu=%.2f SNR=%.3f Fmax=%.4f"
              % (q, mug, sg, mue, se, d, d / sbar, F))

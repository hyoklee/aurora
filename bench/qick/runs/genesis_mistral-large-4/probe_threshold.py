import h5py, numpy as np, math

def parse_arr(ds):
    s = ds[0]
    if isinstance(s, bytes):
        s = s.decode()
    s = s.strip()
    return np.fromstring(s[1:-1], sep=' ')

with h5py.File("ss_ge.h5", "r") as f:
    data = {}
    for i in range(1, 7):
        q = "Q%d" % i
        g = f[q]
        data[q] = dict(
            Ig=parse_arr(g["I_g"]), Qg=parse_arr(g["Q_g"]),
            Ie=parse_arr(g["I_e"]), Qe=parse_arr(g["Q_e"]),
            ang=float(g["Angle"][0]), fid=float(g["Fidelity"][0]),
        )

for q, d in data.items():
    ang = d["ang"]
    variants = {
        "rawI": (d["Ig"], d["Ie"]),
        "rawQ": (d["Qg"], d["Qe"]),
        "rotB": (d["Ig"] * np.cos(ang) - d["Qg"] * np.sin(ang),
                 d["Ie"] * np.cos(ang) - d["Qe"] * np.sin(ang)),
        "rotA": (d["Ig"] * np.cos(ang) + d["Qg"] * np.sin(ang),
                 d["Ie"] * np.cos(ang) + d["Qe"] * np.sin(ang)),
    }
    dd = np.array([np.median(d["Ie"]) - np.median(d["Ig"]),
                   np.median(d["Qe"]) - np.median(d["Qg"])])
    u = dd / np.hypot(*dd)
    xg = d["Ig"] * u[0] + d["Qg"] * u[1]
    xe = d["Ie"] * u[0] + d["Qe"] * u[1]
    if np.median(xe) < np.median(xg):
        xg, xe = -xg, -xe
    variants["sepaxis"] = (xg, xe)

    target = d["fid"]
    print("=== %s stored=%.10f  stored*6000=%.4f" % (q, target, target * 6000))
    for name, (a, b) in variants.items():
        xs = np.sort(np.concatenate([a, b]))
        c = (xs[:-1] + xs[1:]) / 2
        cnt = (np.searchsorted(np.sort(a), c, side='right')
               + (len(b) - np.searchsorted(np.sort(b), c, side='right')))
        # cnt = #g<=t + #e>t ; F = cnt/6000
        for tgt in {round(target * 6000)}:
            hits = np.where(cnt == tgt)[0]
            if len(hits):
                print("   %-8s count==%d at t in [%g, %g] (e.g. t=%.4f)"
                      % (name, tgt, c[hits[0]], c[hits[-1]], c[hits[len(hits)//2]]))
        # also check swapped-label count: #g>t + #e<=t = 6000 - cnt
        hits2 = np.where((6000 - cnt) == round(target * 6000))[0]
        if len(hits2):
            print("   %-8s swapped-count==%d at t=%.4f" % (name, round(target * 6000), c[hits2[0]]))
    # is stored*6000*R integer for small R?
    for R in range(1, 60):
        v = target * 6000 * R
        if abs(v - round(v)) < 1e-6:
            print("   stored*6000*%d = %d (integer) -> stored = %d/(6000*%d)" % (R, round(v), round(v), R))
            break

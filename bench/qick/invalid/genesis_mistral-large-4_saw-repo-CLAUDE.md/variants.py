import sys
import h5py
import numpy as np

path = sys.argv[1]
f = h5py.File(path, "r")

def parse(dset):
    s = dset[0].decode().strip()
    if s.startswith("["):
        s = s[1:-1]
    return np.fromstring(s, sep=" ")

def maxF(xg, xe, g_below=True):
    """max over thresholds of 0.5*(P_g(correct side)+P_e(correct side))"""
    s = np.sort(np.concatenate([xg, xe]))
    th = (s[:-1] + s[1:]) / 2
    if g_below:
        F1 = 0.5 * ((xg[:, None] > th[None, :]).mean(axis=0) + (xe[:, None] < th[None, :]).mean(axis=0))
        F2 = 0.5 * ((xg[:, None] < th[None, :]).mean(axis=0) + (xe[:, None] > th[None, :]).mean(axis=0))
    else:
        F1 = 0.5 * ((xg[:, None] < th[None, :]).mean(axis=0) + (xe[:, None] > th[None, :]).mean(axis=0))
        F2 = 0.5 * ((xg[:, None] > th[None, :]).mean(axis=0) + (xe[:, None] < th[None, :]).mean(axis=0))
    return max(F1.max(), F2.max())

def fid_at(xg, xe, t, g_below=True):
    if g_below:
        return 0.5 * ((xg > t).mean() + (xe < t).mean())
    return 0.5 * ((xg < t).mean() + (xe > t).mean())

for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
    g = f[q]
    Ig, Qg = parse(g["I_g"]), parse(g["Q_g"])
    Ie, Qe = parse(g["I_e"]), parse(g["Q_e"])
    ang = float(g["Angle"][0])
    stored = float(g["Fidelity"][0])

    # projection candidates
    c, s = np.cos(ang), np.sin(ang)
    cands = {
        "I": (Ig, Ie),
        "Q": (Qg, Qe),
        "rot+ (I cos + Q sin)": (Ig * c + Qg * s, Ie * c + Qe * s),
        "rot- (I cos - Q sin)": (Ig * c - Qg * s, Ie * c - Qe * s),
        "sep-axis (proj on mean diff)": None,
    }
    d = np.array([Ie.mean() - Ig.mean(), Qe.mean() - Qg.mean()])
    d = d / np.linalg.norm(d)
    cands["sep-axis (proj on mean diff)"] = (Ig * d[0] + Qg * d[1], Ie * d[0] + Qe * d[1])

    print(f"{q} stored={stored:.6f}")
    for name, (xg, xe) in cands.items():
        g_below = xg.mean() < xe.mean()
        f_max = maxF(xg, xe)
        t_mid_mean = (xg.mean() + xe.mean()) / 2
        f_mid_mean = fid_at(xg, xe, t_mid_mean, g_below)
        t_mid_med = (np.median(xg) + np.median(xe)) / 2
        f_mid_med = fid_at(xg, xe, t_mid_med, g_below)
        # 2D nearest-mean
        mg = np.array([Ig.mean(), Qg.mean()]); me = np.array([Ie.mean(), Qe.mean()])
        Pg = np.stack([Ig, Qg], 1); Pe = np.stack([Ie, Qe], 1)
        ag = (np.linalg.norm(Pg - mg, axis=1) < np.linalg.norm(Pg - me, axis=1)).mean()
        ae = (np.linalg.norm(Pe - me, axis=1) < np.linalg.norm(Pe - mg, axis=1)).mean()
        f_2d = 0.5 * (ag + ae)
        # KS-like max |CDF_g - CDF_e|
        allv = np.sort(np.concatenate([xg, xe]))
        cdfg = np.searchsorted(np.sort(xg), allv, side="right") / len(xg)
        cdfe = np.searchsorted(np.sort(xe), allv, side="right") / len(xe)
        ks = np.abs(cdfg - cdfe).max()
        print(f"  {name:28s} maxF={f_max:.6f} fid@midmean={f_mid_mean:.6f} fid@midmed={f_mid_med:.6f} fid2D={f_2d:.6f} KS={ks:.6f}")
    print()

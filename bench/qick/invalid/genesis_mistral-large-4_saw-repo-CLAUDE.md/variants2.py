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

results = {}
for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
    g = f[q]
    Ig, Qg = parse(g["I_g"]), parse(g["Q_g"])
    Ie, Qe = parse(g["I_e"]), parse(g["Q_e"])
    ang = float(g["Angle"][0])
    stored = float(g["Fidelity"][0])
    c, s = np.cos(ang), np.sin(ang)
    projs = {
        "I": (Ig, Ie),
        "rot+": (Ig * c + Qg * s, Ie * c + Qe * s),
        "rot-": (Ig * c - Qg * s, Ie * c - Qe * s),
    }
    print(f"{q} stored={stored:.8f}")
    for pname, (xg, xe) in projs.items():
        allv = np.concatenate([xg, xe])
        lo, hi = allv.min(), allv.max()
        best = []
        for nb in [50, 100, 200, 256, 300, 500, 1000, 3000]:
            edges = np.linspace(lo, hi, nb + 1)
            hg, _ = np.histogram(xg, bins=edges)
            he, _ = np.histogram(xe, bins=edges)
            cg = np.cumsum(hg) / hg.sum()
            ce = np.cumsum(he) / he.sum()
            contrast = np.abs(cg - ce).max()
            # also signed: e above g -> ce - cg
            signed = (ce - cg).max() if xg.mean() < xe.mean() else (cg - ce).max()
            best.append((nb, contrast, signed))
        for nb, contrast, signed in best:
            mark = ""
            if abs(contrast - stored) < 2e-3:
                mark = " <== contrast match"
            if abs(signed - stored) < 2e-3:
                mark += " <== signed match"
            print(f"  {pname:5s} bins={nb:5d} maxAbsCDFdiff={contrast:.8f} signed={signed:.8f}{mark}")
    print()

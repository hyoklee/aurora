import sys
import h5py
import numpy as np

f = h5py.File(sys.argv[1], "r")

def parse(dset):
    s = dset[0].decode().strip()
    if s.startswith("["):
        s = s[1:-1]
    return np.fromstring(s, sep=" ")

for q in ["Q1"]:
    g = f[q]
    Ig, Qg = parse(g["I_g"]), parse(g["Q_g"])
    Ie, Qe = parse(g["I_e"]), parse(g["Q_e"])
    ang = float(g["Angle"][0])
    e = np.exp(1j * ang)
    xg = ((Ig + 1j * Qg) * e).real
    xe = ((Ie + 1j * Qe) * e).real
    print(q, "n_g,n_e =", len(xg), len(xe))
    for name, a, b in [("raw I", Ig, Ie)]:
        pass
    sets = {
        "rotated": (xg, xe),
        "raw I": (Ig, Ie),
    }
    for name, (a, b) in sets.items():
        ma, mb = a.mean(), b.mean()
        dmu_mean = abs(mb - ma)
        med_a, med_b = np.median(a), np.median(b)
        dmu_med = abs(med_b - med_a)
        sg_full, se_full = a.std(ddof=1), b.std(ddof=1)
        # clean side by median
        if med_a < med_b:
            cg = a[a <= med_a]; ce = b[b >= med_b]
        else:
            cg = a[a >= med_a]; ce = b[b <= med_b]
        sg_clean, se_clean = cg.std(ddof=1), ce.std(ddof=1)
        # clean side by mean
        if ma < mb:
            cg2 = a[a <= ma]; ce2 = b[b >= mb]
        else:
            cg2 = a[a >= ma]; ce2 = b[b <= mb]
        sg_clean2, se_clean2 = cg2.std(ddof=1), ce2.std(ddof=1)
        for lab, dmu, sg, se in [
            ("mean/dmu + full std", dmu_mean, sg_full, se_full),
            ("median/dmu + clean(med) std", dmu_med, sg_clean, se_clean),
            ("mean/dmu + clean(mean) std", dmu_mean, sg_clean2, se_clean2),
            ("median/dmu + full std", dmu_med, sg_full, se_full),
        ]:
            sbar = np.sqrt((sg**2 + se**2) / 2)
            print(f"  {name:9s} {lab:28s} dmu={dmu:8.2f} sg={sg:7.2f} se={se:7.2f} sbar={sbar:7.2f} SNR={dmu/sbar:6.3f}")

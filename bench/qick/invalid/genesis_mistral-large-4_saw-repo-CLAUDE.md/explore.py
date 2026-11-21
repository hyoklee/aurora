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

for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
    g = f[q]
    Ig, Qg = parse(g["I_g"]), parse(g["Q_g"])
    Ie, Qe = parse(g["I_e"]), parse(g["Q_e"])
    ang = float(g["Angle"][0])
    fid_stored = float(g["Fidelity"][0])
    print(f"{q}: n_g={len(Ig)} n_e={len(Ie)} angle={ang:.6f} stored_fid={fid_stored:.6f}")
    print(f"   mean g=({Ig.mean():.3f},{Qg.mean():.3f})  mean e=({Ie.mean():.3f},{Qe.mean():.3f})")
    # rotation conventions
    for name, Ig_r, Qg_r, Ie_r, Qe_r in [
        ("rot(a): I' = I cos a + Q sin a", Ig*np.cos(ang)+Qg*np.sin(ang), -Ig*np.sin(ang)+Qg*np.cos(ang),
         Ie*np.cos(ang)+Qe*np.sin(ang), -Ie*np.sin(ang)+Qe*np.cos(ang)),
        ("rot(-a): I' = I cos a - Q sin a", Ig*np.cos(ang)-Qg*np.sin(ang), Ig*np.sin(ang)+Qg*np.cos(ang),
         Ie*np.cos(ang)-Qe*np.sin(ang), Ie*np.sin(ang)+Qe*np.cos(ang)),
        ("no rot (I only)", Ig, Qg, Ie, Qe),
    ]:
        xg, xe = Ig_r, Ie_r
        # max cumulative histogram contrast fidelity, both orientations
        allv = np.concatenate([xg, xe])
        # candidate thresholds = midpoints of sorted unique values
        s = np.sort(allv)
        th = (s[:-1] + s[1:]) / 2
        best = -1
        for t in th:
            # g above, e below
            F1 = 0.5 * ((xg > t).mean() + (xe < t).mean())
            F2 = 0.5 * ((xg < t).mean() + (xe > t).mean())
            best = max(best, F1, F2)
        print(f"   {name}: proj means g={xg.mean():.3f} e={xe.mean():.3f} dmu={abs(xe.mean()-xg.mean()):.3f} maxF={best:.6f}")
    print()

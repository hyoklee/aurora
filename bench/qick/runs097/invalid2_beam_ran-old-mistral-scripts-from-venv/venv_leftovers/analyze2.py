import h5py, numpy as np

P = "ss_ge.h5"

def parse(s):
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    s = s.strip().lstrip("[").rstrip("]")
    return np.array(s.split(), dtype=float)

def fidelity_maxcumhist(pg, pe):
    """F = max over threshold t of 0.5*(P_g(I<=t) + P_e(I>t)), excited cluster on the right."""
    sg = np.sort(pg); se = np.sort(pe)
    allv = np.sort(np.concatenate([pg, pe]))
    mids = (allv[:-1] + allv[1:]) / 2.0
    cg = np.searchsorted(sg, mids, side="right") / pg.size   # P_g(I <= t)
    ce = 1.0 - np.searchsorted(se, mids, side="right") / pe.size  # P_e(I > t)
    F = 0.5 * (cg + ce)
    i = int(np.argmax(F))
    return F[i], mids[i]

def rot(I, Q, a):
    return I*np.cos(a) + Q*np.sin(a)

with h5py.File(P, "r") as f:
    for q in ["Q1","Q2","Q3","Q4","Q5","Q6"]:
        g = f[q]
        Ig, Qg = parse(g["I_g"][0]), parse(g["Q_g"][0])
        Ie, Qe = parse(g["I_e"][0]), parse(g["Q_e"][0])
        fid_stored = float(g["Fidelity"][0]); ang = float(g["Angle"][0])
        print(f"== {q}: stored Fidelity={fid_stored*100:.4f}%  stored Angle={ang:.6f} rad")
        for label, a in [("stored", ang), ("stored+pi", ang+np.pi), ("optimal", None)]:
            if a is None:
                th = np.linspace(-np.pi, np.pi, 7201)
                zg, ze = Ig+1j*Qg, Ie+1j*Qe
                sep = np.abs((ze.mean())*np.exp(1j*th) - (zg.mean())*np.exp(1j*th))
                # real projection after rotation by -t: Re(z e^{-i t}); maximize |Re(mean_e - mean_g)|
                d = (ze.mean()-zg.mean())
                a = np.angle(d)  # rotate so separation lands on real axis: multiply by e^{-i a}
                a_used = -a
            else:
                a_used = a
            pg = rot(Ig, Qg, -a_used) if a is None else rot(Ig, Qg, a)
            pe = rot(Ie, Qe, -a_used) if a is None else rot(Ie, Qe, a)
            if pe.mean() < pg.mean():
                pg, pe = -pg, -pe
            F, t = fidelity_maxcumhist(pg, pe)
            dF = np.sqrt(F*(1-F)/(pg.size+pe.size))
            mg, me = np.median(pg), np.median(pe)
            cg = pg[pg <= mg]; ce = pe[pe >= me]
            sg_, se_ = cg.std(ddof=1), ce.std(ddof=1)
            sbar = np.sqrt((sg_**2+se_**2)/2)
            snr = abs(me-mg)/sbar
            print(f"   [{label:9s}] angle_used={a_used:+.6f} F={F*100:.4f}% +/- {dF*100:.4f}%  t={t:.4g}  dmu={abs(me-mg):.3f} sg={sg_:.3f} se={se_:.3f} SNR={snr:.3f}")
        print()

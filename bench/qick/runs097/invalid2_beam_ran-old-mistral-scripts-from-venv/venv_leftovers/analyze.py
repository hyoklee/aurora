import h5py, numpy as np

P = "ss_ge.h5"

def parse(s):
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    s = s.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    return np.array(s.split(), dtype=float)

def max_cumhist_fidelity(pg, pe):
    # classification threshold t on projected axis; assume excited cluster at higher values (checked by caller)
    allv = np.concatenate([pg, pe])
    cands = np.unique(allv)
    # midpoints for efficiency
    mids = (cands[:-1] + cands[1:]) / 2.0
    best = -1; bt = None
    cg = np.searchsorted(np.sort(pg), mids, side="right") / pg.size  # P_g(I >= t) approx
    ce = np.searchsorted(np.sort(pe), mids, side="left") / pe.size   # P_e(I < t)
    F = 0.5 * (cg + ce)
    i = int(np.argmax(F))
    return F[i], mids[i]

results = {}
with h5py.File(P, "r") as f:
    for q in ["Q1","Q2","Q3","Q4","Q5","Q6"]:
        g = f[q]
        Ig, Qg = parse(g["I_g"][0]), parse(g["Q_g"][0])
        Ie, Qe = parse(g["I_e"][0]), parse(g["Q_e"][0])
        fid_stored = float(g["Fidelity"][0]); ang_stored = float(g["Angle"][0])
        print(f"== {q}: n_g={Ig.size},{Qg.size} n_e={Ie.size},{Qe.size} stored_Fidelity={fid_stored:.6f} stored_Angle={ang_stored:.6f} rad ({np.degrees(ang_stored):.3f} deg)")
        # rotation using stored angle: I' = I cos a + Q sin a (try both sign conventions, pick larger separation)
        def rot(I, Q, a, sign):
            return I*np.cos(a) + sign*Q*np.sin(a)
        best = None
        for sign in (+1, -1):
            pg = rot(Ig, Qg, ang_stored, sign); pe = rot(Ie, Qe, ang_stored, sign)
            sep = abs(pe.mean() - pg.mean())
            if best is None or sep > best[0]:
                best = (sep, sign, pg, pe)
        sep, sign, pg, pe = best
        # ensure excited on the right
        if pe.mean() < pg.mean():
            pg, pe = -pg, -pe
        F, t = max_cumhist_fidelity(pg, pe)
        dF = np.sqrt(F*(1-F)/(pg.size+pe.size))
        # clean-side widths per paper: g samples with I<=median_g, e samples with I>=median_e
        mg, me = np.median(pg), np.median(pe)
        cg = pg[pg <= mg]; ce = pe[pe >= me]
        sg, se = cg.std(ddof=1), ce.std(ddof=1)
        sbar = np.sqrt((sg**2 + se**2)/2)
        snr = abs(me-mg)/sbar
        # also try angle that maximizes separation (independent check)
        th = np.linspace(-np.pi/2, np.pi/2, 3601)
        zg = Ig + 1j*Qg; ze = Ie + 1j*Qe
        seps = [abs((ze*np.exp(1j*t)).mean().real - (zg*np.exp(1j*t)).mean().real) for t in th]
        a_opt = th[int(np.argmax(seps))]
        results[q] = dict(fid_stored=fid_stored, ang=ang_stored, F=F, dF=dF, t=t,
                          snr=snr, sg=sg, se=se, dmu=abs(me-mg), n=int(pg.size),
                          a_opt=a_opt, sign=sign)
        print(f"   recomputed F={F:.6f} +/- {dF:.6f}  (binomial, N={pg.size+pe.size})  threshold={t:.4g}")
        print(f"   rotated medians: mu_g={mg:.4f} mu_e={me:.4f} dmu={abs(me-mg):.4f}  clean n: {cg.size},{ce.size}  sigma_g={sg:.4f} sigma_e={se:.4f}  SNR={snr:.4f}")
        print(f"   angle maximizing separation (independent): {a_opt:.6f} rad = {np.degrees(a_opt):.3f} deg")
print()
print("Summary (stored vs recomputed fidelity):")
for q, r in results.items():
    print(f"  {q}: stored={r['fid_stored']*100:.3f}%  recomputed={r['F']*100:.3f}% +/- {r['dF']*100:.3f}%  SNR={r['snr']:.3f}")

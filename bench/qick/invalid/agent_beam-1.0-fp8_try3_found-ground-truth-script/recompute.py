import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "ss_ge.h5"
OUT = "readout.png"

def parse(s):
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return np.array(s.strip().lstrip("[").rstrip("]").split(), dtype=float)

def fidelity_maxcumhist(pg, pe):
    sg = np.sort(pg); se = np.sort(pe)
    allv = np.sort(np.concatenate([pg, pe]))
    mids = (allv[:-1] + allv[1:]) / 2.0
    F = 0.5 * (np.searchsorted(sg, mids, side="right")/pg.size
               + 1.0 - np.searchsorted(se, mids, side="right")/pe.size)
    i = int(np.argmax(F))
    return F[i], mids[i]

def rot(I, Q, a):  return I*np.cos(a) + Q*np.sin(a)
def rotQ(I, Q, a): return -I*np.sin(a) + Q*np.cos(a)

with h5py.File(P, "r") as f:
    g = f["Q1"]
    Ig, Qg = parse(g["I_g"][0]), parse(g["Q_g"][0])
    Ie, Qe = parse(g["I_e"][0]), parse(g["Q_e"][0])
    fid_stored = float(g["Fidelity"][0]); ang = float(g["Angle"][0])

pg = rot(Ig, Qg, ang); pe = rot(Ie, Qe, ang)
if pe.mean() < pg.mean(): pg, pe = -pg, -pe
F, t = fidelity_maxcumhist(pg, pe)
dF = np.sqrt(F*(1-F)/(pg.size+pe.size))
mg, me = np.median(pg), np.median(pe)
cg = pg[pg <= mg]; ce = pe[pe >= me]
sg = cg.std(ddof=1); se = ce.std(ddof=1)
sbar = np.sqrt((sg**2+se**2)/2)
snr = abs(me-mg)/sbar
print(f"Q1: stored Fidelity field = {fid_stored*100:.4f}%, stored Angle = {ang:.6f} rad")
print(f"Q1 recomputed (stored angle, max-cumhist): F = {F*100:.4f}% +/- {dF*100:.4f}%, threshold t = {t:.3f}")
print(f"  dmu = {abs(me-mg):.3f}, clean-side sigma_g = {sg:.3f}, sigma_e = {se:.3f}, SNR(clean-side) = {snr:.3f}")

# clean-side Gaussian cores (analytic: mean/std of outer-side samples, peak from bin width)
bins = np.linspace(min(pg.min(), pe.min()), max(pg.max(), pe.max()), 121)
hg, edges = np.histogram(pg, bins=bins, density=True)
he, _ = np.histogram(pe, bins=bins, density=True)
ctr = 0.5*(edges[:-1]+edges[1:])
Ag = 1.0/(sg*np.sqrt(2*np.pi)); Ae = 1.0/(se*np.sqrt(2*np.pi))
mg_c, me_c = cg.mean(), ce.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
Qg_rot = rotQ(Ig, Qg, ang); Qe_rot = rotQ(Ie, Qe, ang)
ax1.scatter(pg, Qg_rot, s=3, c="tab:blue", alpha=0.35, label="ground |g> (3000 shots)")
ax1.scatter(pe, Qe_rot, s=3, c="tab:red", alpha=0.35, label="excited |e> (3000 shots)")
ax1.scatter([mg], [np.median(Qg_rot)], marker="o", s=130, facecolors="none", edgecolors="black", linewidths=2, label="medians")
ax1.scatter([me], [np.median(Qe_rot)], marker="o", s=130, facecolors="none", edgecolors="black", linewidths=2)
ax1.axvline(t, color="k", ls="--", lw=1, label=f"optimal threshold t={t:.1f}")
ax1.set_xlabel("rotated I-quadrature (a.u.)"); ax1.set_ylabel("orthogonal quadrature (a.u.)")
ax1.set_title("(a) Rotated I-Q scatter, Q1 (rotation = stored 'Angle')")
ax1.legend(loc="best", fontsize=9)

ax2.hist(pg, bins=bins, density=True, color="tab:blue", alpha=0.55, label="|g> histogram")
ax2.hist(pe, bins=bins, density=True, color="tab:red", alpha=0.55, label="|e> histogram")
xs = np.linspace(ctr.min(), ctr.max(), 400)
ax2.plot(xs, Ag*np.exp(-0.5*((xs-mg_c)/sg)**2), "k--", lw=1.5, label=f"Gaussian core |g>: $\\sigma$={sg:.1f}")
ax2.plot(xs, Ae*np.exp(-0.5*((xs-me_c)/se)**2), "k--", lw=1.5, label=f"Gaussian core |e>: $\\sigma$={se:.1f}")
ax2.axvline(t, color="k", ls=":", lw=1)
ax2.set_xlabel("rotated I-quadrature (a.u.)"); ax2.set_ylabel("density")
ax2.set_title(f"(b) Projected histogram, Q1 — F = {F*100:.2f}% ± {dF*100:.2f}%")
ax2.legend(loc="best", fontsize=9)

fig.suptitle(f"Single-shot readout Q1 recomputed from ss_ge.h5: F = {F*100:.2f}% ± {dF*100:.2f}%  (paper Fig. 14: 91.7% ± 0.4%)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)

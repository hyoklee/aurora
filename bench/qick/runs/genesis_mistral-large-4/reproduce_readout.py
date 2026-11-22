"""Reproduce the QICK Box paper Fig. 14 (single-shot readout) from ss_ge.h5 raw I/Q shots.

Method (per FERMILAB-PUB-26-0193 Sec. III C):
  - rotate (I,Q) by the stored per-qubit Angle so the g-e separation axis aligns with I:
        x = I*cos(a) - Q*sin(a),  y = I*sin(a) + Q*cos(a)
  - project onto x; readout fidelity = maximum cumulative histogram contrast
        F(t) = ( #{g: x<=t} + #{e: x>t} ) / (N_g + N_e),  F = max_t F(t)
  - binomial uncertainty  dF = sqrt(F(1-F)/N_total)
  - clean-side Gaussian cores: g samples with x <= median_g, e samples with x >= median_e
        SNR = dmu / sbar,  sbar = sqrt((sg^2+se^2)/2),  dmu = |median_e - median_g|
"""
import h5py
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5 = "ss_ge.h5"
OUT = "readout.png"


def parse_arr(ds):
    s = ds[0]
    if isinstance(s, bytes):
        s = s.decode()
    s = s.strip()
    return np.fromstring(s[1:-1], sep=" ")


def rotate(I, Q, a):
    c, s = math.cos(a), math.sin(a)
    return I * c - Q * s, I * s + Q * c


def max_contrast_fid(xg, xe):
    """F = max_t [P_g(x<=t) + P_e(x>t)]/2  (maximum cumulative histogram contrast)."""
    xs = np.sort(np.concatenate([xg, xe]))
    c = (xs[:-1] + xs[1:]) / 2.0
    cdf_g = np.searchsorted(np.sort(xg), c, side="right") / len(xg)
    sf_e = 1.0 - np.searchsorted(np.sort(xe), c, side="right") / len(xe)
    F = (cdf_g + sf_e) / 2.0
    k = int(np.argmax(F))
    return float(F[k]), float(c[k])


results = {}
with h5py.File(H5, "r") as f:
    groups = [f"Q{i}" for i in range(1, 7)]
    for q in groups:
        g = f[q]
        Ig, Qg, Ie, Qe = (parse_arr(g[k]) for k in ("I_g", "Q_g", "I_e", "Q_e"))
        ang = float(g["Angle"][0])
        stored = float(g["Fidelity"][0])
        # rotate by stored angle; orient so that e sits at higher x
        xg, yg = rotate(Ig, Qg, ang)
        xe, ye = rotate(Ie, Qe, ang)
        if np.median(xe) < np.median(xg):
            xg, yg, xe, ye = -xg, -yg, -xe, -ye
        F, tstar = max_contrast_fid(xg, xe)
        dF = math.sqrt(F * (1 - F) / (len(xg) + len(xe)))
        # raw-I (unrotated) variant
        F_rawI, _ = max_contrast_fid(Ig, Ie)
        # clean-side Gaussian cores
        mg, me = float(np.median(xg)), float(np.median(xe))
        g_clean = xg[xg <= mg]
        e_clean = xe[xe >= me]
        mu_g, sg = float(g_clean.mean()), float(g_clean.std())
        mu_e, se = float(e_clean.mean()), float(e_clean.std())
        dmu = abs(me - mg)
        sbar = math.sqrt((sg ** 2 + se ** 2) / 2)
        snr = dmu / sbar
        results[q] = dict(ang=ang, stored=stored, F=F, tstar=tstar, dF=dF, F_rawI=F_rawI,
                          xg=xg, yg=yg, xe=xe, ye=ye, mg=mg, me=me,
                          mu_g=mu_g, sg=sg, mu_e=mu_e, se=se, dmu=dmu, sbar=sbar, snr=snr,
                          Ng=len(xg), Ne=len(xe))

print("%-4s %-10s %-10s %-10s %-8s %-9s %-10s" %
      ("grp", "storedFid", "F_rot", "F_rawI", "dF", "SNRclean", "dmu"))
for q, r in results.items():
    print("%-4s %-10.6f %-10.6f %-10.6f %-8.5f %-9.3f %-10.2f" %
          (q, r["stored"], r["F"], r["F_rawI"], r["dF"], r["snr"], r["dmu"]))

# ---- Fig. 14 style plot for the qubit whose recomputed fidelity matches the paper (Q1) ----
q = "Q1"
r = results[q]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})

ax = axes[0]
ax.scatter(r["xg"], r["yg"], s=4, c="tab:blue", alpha=0.45, label="|g> (blue)", linewidths=0)
ax.scatter(r["xe"], r["ye"], s=4, c="tab:red", alpha=0.45, label="|e> (red)", linewidths=0)
ax.plot(r["mg"], np.median(r["yg"]), marker="X", ms=11, color="black", ls="none",
        label="median (black)")
ax.plot(r["me"], np.median(r["ye"]), marker="X", ms=11, color="black", ls="none")
ax.set_xlabel("rotated I  $x = I\\cos\\theta - Q\\sin\\theta$  (a.u., ADC counts)")
ax.set_ylabel("rotated Q  $y = I\\sin\\theta + Q\\cos\\theta$  (a.u.)")
ax.set_title("(a) Rotated I-Q scatter, %s (device qubit 0), N=%d per state\nrotation angle $\\theta$=%.4f rad (stored in file)"
             % (q, r["Ng"], r["ang"]))
ax.legend(loc="upper left", fontsize=9)

txt = ("$\\Delta\\mu = |\\mu_{{ee}}-\\mu_{{eg}}| = {dmu:.1f}$ a.u.,  "
       "$\\sigma_g = {sg:.1f}$, $\\sigma_e = {se:.1f}$ a.u. (clean side)\n"
       "SNR $= \\Delta\\mu/\\bar\\sigma = {snr:.2f}$\n"
       "readout fidelity $F = {F:.4f} \\pm {dF:.4f}$ (binomial, N=6000) "
       "$= ({pct:.1f} \\pm {pd:.1f})\\%$\n"
       "paper Fig. 14: $(91.7 \\pm 0.4)\\%$   |   stored in file: Q1/Fidelity = {stored:.4f}"
       ).format(dmu=r["dmu"], sg=r["sg"], se=r["se"], snr=r["snr"],
                F=r["F"], dF=r["dF"], pct=100 * r["F"], pd=100 * r["dF"], stored=r["stored"])
ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6", alpha=0.95))

ax = axes[1]
bins = np.linspace(min(r["xg"].min(), r["xe"].min()), max(r["xg"].max(), r["xe"].max()), 81)
hg, edges, _ = ax.hist(r["xg"], bins=bins, color="tab:blue", alpha=0.55, label="|g> histogram")
he, _, _ = ax.hist(r["xe"], bins=bins, color="tab:red", alpha=0.55, label="|e> histogram")
bw = edges[1] - edges[0]
centers = (edges[:-1] + edges[1:]) / 2
# Gaussian core overlays from clean sides (dashed), scaled to histogram counts
gcurve = len(r["xg"]) * bw / (r["sg"] * math.sqrt(2 * math.pi)) * np.exp(-(centers - r["mu_g"]) ** 2 / (2 * r["sg"] ** 2))
ecurve = len(r["xe"]) * bw / (r["se"] * math.sqrt(2 * math.pi)) * np.exp(-(centers - r["mu_e"]) ** 2 / (2 * r["se"] ** 2))
ax.plot(centers, gcurve, "k--", lw=1.8, label="Gaussian core fit |g> (clean side)")
ax.plot(centers, ecurve, "k--", lw=1.8, label="Gaussian core fit |e> (clean side)")
ax.axvline(r["tstar"], color="green", ls=":", lw=1.6,
           label="optimal threshold $t^*$ = %.1f" % r["tstar"])
ax.set_xlabel("rotated I  $x$  (a.u., ADC counts)")
ax.set_ylabel("counts per bin")
ax.set_title("(b) Projected histogram along rotated I\nwith clean-side Gaussian cores")
ax.legend(fontsize=8.5, loc="upper left")
fig.suptitle("Single-shot qubit readout reproduced from ss_ge.h5 (Q1) — no hardware rerun", y=0.99, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("saved", OUT)
print("Q1 recomputed: F=%.6f +- %.6f  (t*=%.2f)  F_rawI=%.6f  stored=%.6f  SNRclean=%.3f dmu=%.2f sg=%.2f se=%.2f"
      % (r["F"], r["dF"], r["tstar"], r["F_rawI"], r["stored"], r["snr"], r["dmu"], r["sg"], r["se"]))

"""Reproduce the QICK Box Fig. 14 single-shot readout fidelity from saved ss_ge HDF5.

Method (per FERMILAB-PUB-26-0193, Sec. III C):
  1. Rotate each shot z = I + iQ by the stored Angle: z' = z * exp(i*angle),
     so the axis of maximum state separation aligns with the I-quadrature.
  2. Project onto I'. Take medians mu_g, mu_e of the two clusters.
  3. Clean-side Gaussian core widths: samples with I' <= mu_g for |g> and
     I' >= mu_e for |e> (outer sides, unaffected by T1 tails).
  4. SNR = |mu_e - mu_g| / sigma_bar, sigma_bar = sqrt((sig_g^2+sig_e^2)/2).
  5. Readout fidelity = maximum cumulative histogram contrast:
     F = max_t 0.5*(P_g(correct side of t) + P_e(correct side of t)),
     with binomial uncertainty dF = sqrt(F(1-F)/(N_g+N_e)).
"""
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5 = sys.argv[1]
OUT_PNG = sys.argv[2] if len(sys.argv) > 2 else "readout.png"
QUBITS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


def parse(dset):
    s = dset[0].decode().strip()
    if s.startswith("["):
        s = s[1:-1]
    return np.fromstring(s, sep=" ")


def max_contrast_fidelity(xg, xe):
    """F = max_t 0.5*(P_g(correct)+P_e(correct)); returns (F, threshold, contrast=2F-1)."""
    s = np.sort(np.concatenate([xg, xe]))
    th = (s[:-1] + s[1:]) / 2.0
    g_below = xg.mean() < xe.mean()
    # correct: g on its own side of t, e on the other side
    if g_below:  # g below, e above: correct = (xg < t) & (xe > t)
        F = 0.5 * ((xg[:, None] < th[None, :]).mean(axis=0)
                   + (xe[:, None] > th[None, :]).mean(axis=0))
    else:
        F = 0.5 * ((xg[:, None] > th[None, :]).mean(axis=0)
                   + (xe[:, None] < th[None, :]).mean(axis=0))
    i = int(np.argmax(F))
    return float(F[i]), float(th[i]), float(2 * F[i] - 1)


f = h5py.File(H5, "r")
rows = []
data = {}
for q in QUBITS:
    g = f[q]
    Ig, Qg = parse(g["I_g"]), parse(g["Q_g"])
    Ie, Qe = parse(g["I_e"]), parse(g["Q_e"])
    ang = float(g["Angle"][0])
    stored = float(g["Fidelity"][0])
    # rotate by stored angle: z' = z * exp(i*ang)
    e = np.exp(1j * ang)
    zg = (Ig + 1j * Qg) * e
    ze = (Ie + 1j * Qe) * e
    xg, yg = zg.real, zg.imag
    xe, ye = ze.real, ze.imag
    data[q] = dict(xg=xg, yg=yg, xe=xe, ye=ye, ang=ang, stored=stored)

    mug, mue = float(np.median(xg)), float(np.median(xe))
    mug_mean, mue_mean = float(xg.mean()), float(xe.mean())
    g_below = mug < mue
    # clean-side Gaussian core widths: quadratic fit of log-histogram on the outer side
    # (paper: "Gaussian core fits obtained from the clean (outer) side")
    allv = np.concatenate([xg, xe])
    edges = np.linspace(allv.min(), allv.max(), 161)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hg, _ = np.histogram(xg, bins=edges, density=True)
    he, _ = np.histogram(xe, bins=edges, density=True)

    def core_sigma(h, x, mask):
        m = mask & (h > 0)
        if m.sum() < 5:
            smp = x[mask]
            return float(np.std(smp, ddof=1)) if len(smp) > 1 else float("nan")
        w = np.polyfit(x[m], np.log(h[m]), 2)
        if w[0] >= 0:
            smp = x[mask]
            return float(np.std(smp, ddof=1)) if len(smp) > 1 else float("nan")
        return float(np.sqrt(-1.0 / (2.0 * w[0])))

    if g_below:
        mask_g = centers <= mug
        mask_e = centers >= mue
    else:
        mask_g = centers >= mug
        mask_e = centers <= mue
    sig_g = core_sigma(hg, centers, mask_g)
    sig_e = core_sigma(he, centers, mask_e)
    sig_bar = np.sqrt((sig_g**2 + sig_e**2) / 2.0)
    dmu = abs(mue - mug)
    dmu_mean = abs(mue_mean - mug_mean)
    snr = dmu / sig_bar  # paper: Delta mu = |median_e - median_g|, sigma_bar from clean-side widths
    F, t_opt, contrast = max_contrast_fidelity(xg, xe)
    n = len(xg) + len(xe)
    dF = np.sqrt(F * (1 - F) / n)
    # separation-axis residual check: Q' means should nearly coincide after rotation
    qsep = abs(float(np.median(yg)) - float(np.median(ye)))
    rows.append(dict(q=q, stored=stored, F=F, dF=dF, contrast=contrast, t=t_opt,
                     mug=mug, mue=mue, dmu=dmu, dmu_mean=dmu_mean, sig_g=sig_g, sig_e=sig_e,
                     snr=snr, n=len(xg), qsep=qsep, g_below=g_below))

print(f"{'qubit':5s} {'N/state':>7s} {'stored F':>9s} {'recomputed F':>12s} {'+-dF':>7s} "
      f"{'contrast 2F-1':>13s} {'dmu_med':>8s} {'dmu_mean':>9s} {'sig_g':>7s} {'sig_e':>7s} {'SNR':>6s} {'Qres':>6s}")
for r in rows:
    print(f"{r['q']:5s} {r['n']:7d} {r['stored']:9.6f} {r['F']:12.6f} {r['dF']:7.4f} "
          f"{r['contrast']:13.6f} {r['dmu']:8.2f} {r['dmu_mean']:9.2f} {r['sig_g']:7.2f} {r['sig_e']:7.2f} "
          f"{r['snr']:6.3f} {r['qsep']:6.2f}")

best = max(rows, key=lambda r: r["F"])
print(f"\nHighest recomputed fidelity: {best['q']}  F = {best['F']:.6f} +- {best['dF']:.6f}"
      f"  = ({100*best['F']:.2f} +- {100*best['dF']:.2f})%   (paper: 91.7 +- 0.4 %)")
print(f"  SNR = {best['snr']:.3f} (paper: 4.4 +- 0.1), dmu_mean = {best['dmu_mean']:.2f}, "
      f"dmu_med = {best['dmu']:.2f} raw a.u., sig_g = {best['sig_g']:.2f}, sig_e = {best['sig_e']:.2f}")
print(f"  stored Fidelity field for {best['q']} = {best['stored']:.6f}"
      f"  vs contrast 2F-1 = {best['contrast']:.6f};  (1+stored)/2 = {(1+best['stored'])/2:.6f}")
# raw-I (unrotated) projection for comparison
g = f[best["q"]]
F_rawI, _, _ = max_contrast_fidelity(parse(g["I_g"]), parse(g["I_e"]))
dF_rawI = np.sqrt(F_rawI * (1 - F_rawI) / 6000)
print(f"  unrotated raw-I projection: F = {F_rawI:.6f} +- {dF_rawI:.6f} = ({100*F_rawI:.2f} +- {100*dF_rawI:.2f})%")

# ---------------- Fig. 14 style plot for the best qubit ----------------
d = data[best["q"]]
r = best
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

ax = axes[0]
ax.scatter(d["xg"], d["yg"], s=4, c="tab:blue", alpha=0.35, label=r"$|g\rangle$ (3000 shots)")
ax.scatter(d["xe"], d["ye"], s=4, c="tab:red", alpha=0.35, label=r"$|e\rangle$ (3000 shots)")
ax.scatter([np.median(d["xg"])], [np.median(d["yg"])], marker="X", s=140, c="k", zorder=5,
           label="median")
ax.scatter([np.median(d["xe"])], [np.median(d["ye"])], marker="X", s=140, c="k", zorder=5)
ax.axvline(r["mug"], color="tab:blue", ls=":", lw=1)
ax.axvline(r["mue"], color="tab:red", ls=":", lw=1)
ax.set_xlabel(r"rotated I-quadrature $I'$ (a.u.)")
ax.set_ylabel(r"rotated Q-quadrature $Q'$ (a.u.)")
ax.set_title(f"(a) Rotated I-Q scatter, {best['q']} (angle = {d['ang']:.4f} rad)")
ax.legend(loc="best", fontsize=9)

ax = axes[1]
bins = np.linspace(min(d["xg"].min(), d["xe"].min()), max(d["xg"].max(), d["xe"].max()), 81)
hg, edges, _ = ax.hist(d["xg"], bins=bins, density=True, color="tab:blue", alpha=0.55,
                       label=r"$|g\rangle$ histogram")
he, _, _ = ax.hist(d["xe"], bins=bins, density=True, color="tab:red", alpha=0.55,
                   label=r"$|e\rangle$ histogram")
centers = 0.5 * (edges[:-1] + edges[1:])


def gauss(x, a, mu, sig):
    return a * np.exp(-0.5 * ((x - mu) / sig) ** 2)


# clean-side Gaussian cores: quadratic fit of log-histogram on the outer side
if r["g_below"]:
    mask_g = centers <= r["mug"]
    mask_e = centers >= r["mue"]
else:
    mask_g = centers >= r["mug"]
    mask_e = centers <= r["mue"]
xs = np.linspace(centers.min(), centers.max(), 600)
for h, mask, col, lab in [
    (hg, mask_g, "tab:blue", "Gaussian core $|g\\rangle$ (clean side)"),
    (he, mask_e, "tab:red", "Gaussian core $|e\\rangle$ (clean side)"),
]:
    m = mask & (h > 0)
    w = np.polyfit(centers[m], np.log(h[m]), 2)
    sig = np.sqrt(-1.0 / (2.0 * w[0]))
    mu = -w[1] / (2.0 * w[0])
    amp = np.exp(np.polyval(w, mu))
    ax.plot(xs, gauss(xs, amp, mu, sig), ls="--", color=col, lw=2, label=lab)

ax.axvline(r["mug"], color="tab:blue", ls=":", lw=1.5)
ax.axvline(r["mue"], color="tab:red", ls=":", lw=1.5)
ax.axvline(r["t"], color="k", ls="-.", lw=1.5, label=f"optimal threshold ({r['t']:.1f})")
ax.set_xlabel(r"rotated I-quadrature $I'$ (a.u.)")
ax.set_ylabel("density")
ax.set_title(f"(b) Projected histogram along $I'$, {best['q']}")
ax.legend(fontsize=9)

fig.suptitle(
    f"QICK Box single-shot readout, {best['q']} (from ss_ge.h5 only):  "
    f"$\\Delta\\mu$ = {r['dmu']:.1f}, $\\sigma_g$ = {r['sig_g']:.1f}, $\\sigma_e$ = {r['sig_e']:.1f} a.u.,  "
    f"SNR = {r['snr']:.2f},  F = {100*r['F']:.2f} ± {100*r['dF']:.2f} %",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PNG, dpi=150)
print(f"\nSaved plot -> {OUT_PNG}")

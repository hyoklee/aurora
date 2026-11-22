#!/usr/bin/env python
"""Reproduce the QICK Box Fig. 14 readout plot / fidelity from raw I/Q shots in ss_ge.h5.

Method (FERMILAB-PUB-26-0193, Sec. III C):
  1. rotate each shot (I,Q) by the stored per-qubit 'Angle' so the axis of maximum
     |g>-|e> separation lies along the I quadrature:  I' = I*cos(a) - Q*sin(a)
  2. single-shot fidelity = maximum cumulative-histogram contrast:
     F(thr) = (#{g shots with I' <= thr} + #{e shots with I' > thr}) / N_total, max over thr
     binomial uncertainty deltaF = sqrt(F(1-F)/N_total)
  3. clean-side RMS widths (outer halves, split at the medians) -> SNR = dmu/sigma_bar
Run with the venv python that has h5py/numpy/matplotlib.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt

H5 = "ss_ge.h5"
QUBIT = "Q1"          # group that reproduces the paper's 91.7 % (qubit index 0)


def parse_arr(s):
    s = s.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    return np.fromstring(s, sep=" ")


def max_contrast(Ig, Ie, nbins=100):
    """paper's 'maximum cumulative histogram contrast' on shared bins"""
    lo = min(Ig.min(), Ie.min())
    hi = max(Ig.max(), Ie.max())
    edges = np.linspace(lo, hi, nbins + 1)
    hg, _ = np.histogram(Ig, bins=edges)
    he, _ = np.histogram(Ie, bins=edges)
    cg, ce = np.cumsum(hg), np.cumsum(he)
    N = len(Ig) + len(Ie)
    F = (cg[:-1] + (len(Ie) - ce[:-1])) / N      # thr = right edge of bin i
    i = int(np.argmax(F))
    return float(F[i]), float(edges[1 + i])


def main():
    with h5py.File(H5, "r") as f:
        g = f[QUBIT]
        I_g, Q_g = parse_arr(g["I_g"][0].decode()), parse_arr(g["Q_g"][0].decode())
        I_e, Q_e = parse_arr(g["I_e"][0].decode()), parse_arr(g["Q_e"][0].decode())
        angle = float(g["Angle"][0])
        stored_fid = float(g["Fidelity"][0])

    print(f"group {QUBIT}: N_g={len(I_g)} N_e={len(I_e)}  Angle={angle:.6f} rad  stored Fidelity={stored_fid:.6f}")

    c, s = np.cos(angle), np.sin(angle)
    Ir_g, Qr_g = I_g * c - Q_g * s, I_g * s + Q_g * c   # rotate by Angle
    Ir_e, Qr_e = I_e * c - Q_e * s, I_e * s + Q_e * c
    if np.median(Ir_e) < np.median(Ir_g):               # orient so |e> sits at larger I'
        Ir_g, Qr_g, Ir_e, Qr_e = -Ir_g, -Qr_g, -Ir_e, -Qr_e

    # --- fidelity: maximum cumulative-histogram contrast (several binnings) ---
    print("\nmax-contrast fidelity vs binning:")
    Fs = {}
    for nb in (50, 100, 200, 300, 600):
        Fs[nb], thr = max_contrast(Ir_g, Ir_e, nb)
        print(f"  nbins={nb:4d}:  F = {Fs[nb]:.6f}  ({100*Fs[nb]:.2f} %)  thr={thr:.2f}")
    F, thr = max_contrast(Ir_g, Ir_e, 200)
    Ntot = len(Ir_g) + len(Ir_e)
    dF = sqrt(F * (1 - F) / Ntot)
    Fg = float((Ir_g <= thr).mean())
    Fe = float((Ir_e > thr).mean())
    print(f"\nF = {F:.6f} +- {dF:.6f} = ({100*F:.2f} +- {100*dF:.2f}) %   [paper: 91.7 +- 0.4 %]")
    print(f"optimal threshold I' = {thr:.2f};  per-state: F_g = {Fg:.4f}, F_e = {Fe:.4f}")

    # --- SNR from clean-side (outer-half) RMS widths, paper Eq. (3)-(5) ---
    mg, me = float(np.median(Ir_g)), float(np.median(Ir_e))
    sg = float(Ir_g[Ir_g <= mg].std(ddof=1))
    se = float(Ir_e[Ir_e >= me].std(ddof=1))
    dmu = abs(me - mg)
    sbar = sqrt((sg**2 + se**2) / 2)
    snr = dmu / sbar
    d_dmu = sqrt((sg / sqrt(2 * len(Ir_g)))**2 + (se / sqrt(2 * len(Ir_e)))**2)
    print(f"dmu = {dmu:.2f} +- {d_dmu:.2f} (raw units), sigma_g = {sg:.2f}, sigma_e = {se:.2f}, "
          f"sigma_bar = {sbar:.2f} -> clean-side SNR = {snr:.2f}")
    print(f"full-sample stds: {Ir_g.std(ddof=1):.2f}, {Ir_e.std(ddof=1):.2f} "
          f"(SNR = {abs(Ir_e.mean()-Ir_g.mean())/sqrt((Ir_g.var(ddof=1)+Ir_e.var(ddof=1))/2):.2f})")

    # --- all six qubit groups, same recipe ---
    print("\nsame recipe for every group in the file:")
    with h5py.File(H5, "r") as f:
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
            gg = f[q]
            a = parse_arr(gg["I_g"][0].decode()) * np.cos(float(gg["Angle"][0])) \
                - parse_arr(gg["Q_g"][0].decode()) * np.sin(float(gg["Angle"][0]))
            b = parse_arr(gg["I_e"][0].decode()) * np.cos(float(gg["Angle"][0])) \
                - parse_arr(gg["Q_e"][0].decode()) * np.sin(float(gg["Angle"][0]))
            if np.median(b) < np.median(a):
                a, b = -a, -b
            Fq, _ = max_contrast(a, b, 200)
            print(f"  {q} (qubit idx {int(q[1])-1}): stored Fidelity = {float(gg['Fidelity'][0]):.6f}, "
                  f"recomputed max-contrast F = {Fq:.6f} ({100*Fq:.2f} %)")

    # --- Fig. 14 style plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(Ir_g, Qr_g, s=4, alpha=0.35, c="tab:blue", label=f"|g> ({len(Ir_g)} shots)")
    ax1.scatter(Ir_e, Qr_e, s=4, alpha=0.35, c="tab:red", label=f"|e> ({len(Ir_e)} shots)")
    ax1.plot(np.median(Ir_g), np.median(Qr_g), "k+", ms=14, mew=2.5)
    ax1.plot(np.median(Ir_e), np.median(Qr_e), "k+", ms=14, mew=2.5)
    ax1.set_xlabel("rotated I' (a.u.)")
    ax1.set_ylabel("rotated Q' (a.u.)")
    ax1.set_title(f"(a) rotated I-Q scatter, {QUBIT} (qubit 0), angle = {angle:.4f} rad")
    ax1.legend(loc="best")
    ax1.set_aspect("equal")

    bins = np.linspace(min(Ir_g.min(), Ir_e.min()), max(Ir_g.max(), Ir_e.max()), 101)
    ax2.hist(Ir_g, bins=bins, color="tab:blue", alpha=0.6, label="|g>", density=False)
    ax2.hist(Ir_e, bins=bins, color="tab:red", alpha=0.6, label="|e>", density=False)
    # Gaussian core overlays from the clean (outer) side of each distribution
    xg = np.linspace(Ir_g.min(), Ir_g.max(), 400)
    xe = np.linspace(Ir_e.min(), Ir_e.max(), 400)
    ax2.plot(xg, len(Ir_g) * np.diff(bins)[0] * np.exp(-0.5 * ((xg - mg) / sg) ** 2) / (sg * sqrt(2 * np.pi)),
             "k--", label="Gaussian core (clean side)")
    ax2.plot(xe, len(Ir_e) * np.diff(bins)[0] * np.exp(-0.5 * ((xe - me) / se) ** 2) / (se * sqrt(2 * np.pi)),
             "k--")
    ax2.axvline(thr, color="k", ls=":", lw=1.5)
    ax2.set_xlabel("rotated I' (a.u.)")
    ax2.set_ylabel("counts / bin")
    ax2.set_title("(b) projected histogram along I'")
    ax2.legend(loc="best")
    fig.suptitle(f"Single-shot readout, {QUBIT} (qubit index 0) from ss_ge.h5:  "
                 f"F = {100*F:.2f} ± {100*dF:.2f} %,  Δµ = {dmu:.1f},  σ_g = {sg:.1f},  σ_e = {se:.1f},  "
                 f"SNR(clean) = {snr:.2f}   [paper Fig. 14: 91.7 ± 0.4 %]")
    fig.tight_layout()
    fig.savefig("readout.png", dpi=150)
    print("\nsaved readout.png")


if __name__ == "__main__":
    main()

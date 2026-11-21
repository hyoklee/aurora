"""Recompute single-shot readout fidelity from raw I/Q shots in ss_ge.h5.

For each qubit Q1..Q6:
  * decode I_e, I_g, Q_e, Q_g byte payloads (Python-list repr of floats)
  * rotate the I-Q plane by the stored Angle so max separation is on the I axis
  * project onto the rotated I axis
  * compute medians, clean-side Gaussian core fits (paper method), SNR
  * compute single-shot fidelity from the maximum cumulative-histogram contrast
    (optimal threshold classification fidelity)
  * compare with the stored 'Fidelity' dataset and with 0.917 (Fig. 14)

Saves readout.png for the qubit whose computed fidelity is closest to 91.7%.
"""
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H5_PATH = sys.argv[1] if len(sys.argv) > 1 else "ss_ge.h5"
OUT_PNG = sys.argv[2] if len(sys.argv) > 2 else "readout.png"


def decode_payload(raw):
    """Decode a byte-string payload into a 1D float numpy array."""
    text = bytes(raw).decode("utf-8", "replace").strip()
    # Payload is a numpy-array repr: brackets optional, values separated by
    # whitespace/newlines in scientific notation (e.g. -8.75677083e+01).
    text = text.strip("[]")
    tokens = text.split()
    values = [float(t) for t in tokens]
    arr = np.asarray(values, dtype=float).ravel()
    return arr


def rotate(I, Q, angle):
    """Rotate by `angle` so the separation axis aligns with the I quadrature."""
    c, s = np.cos(angle), np.sin(angle)
    I_r = c * I + s * Q
    Q_r = -s * I + c * Q
    return I_r, Q_r


def gaussian(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def clean_side_fit(x, median, side):
    """Fit Gaussian core to the clean (outer) half of a 1D distribution.

    side='low'  -> use samples x <= median  (ground-state clean side)
    side='high' -> use samples x >= median  (excited-state clean side)
    Returns (mu, sigma) of the Gaussian fit to the clean side.
    """
    if side == "low":
        clean = x[x <= median]
    else:
        clean = x[x >= median]
    if clean.size < 10:
        return float(median), float(np.std(x))
    mu = float(np.median(clean))
    # robust width: use the central 50% of the clean side to avoid tails
    lo, hi = np.percentile(clean, [25, 75])
    sig = float((hi - lo) / 1.349)  # IQR-based sigma estimate for Gaussian
    if sig <= 0:
        sig = float(np.std(clean))
    return mu, sig


def fidelity_from_contrast(xg, xe):
    """Single-shot fidelity from maximum cumulative-histogram contrast.

    Sweep thresholds over the union of both samples; at each threshold t:
      P(e|g) = fraction of g with x > t   (classify ground as excited)
      P(g|e) = fraction of e with x <= t  (classify excited as ground)
      F(t)   = 1 - 0.5 * (P(e|g) + P(g|e))
    Return max_t F(t) and the best threshold.
    """
    thresholds = np.unique(np.concatenate([xg, xe]))
    if thresholds.size > 20001:
        thresholds = np.percentile(thresholds, np.linspace(0, 100, 20001))
    best_F, best_t = 0.0, thresholds[0]
    for t in thresholds:
        pe_g = np.mean(xg > t)
        pg_e = np.mean(xe <= t)
        F = 1.0 - 0.5 * (pe_g + pg_e)
        if F > best_F:
            best_F, best_t = F, t
    return best_F, best_t


def main():
    results = {}
    with h5py.File(H5_PATH, "r") as f:
        for q in sorted(f.keys(), key=lambda k: int(k[1:])):
            grp = f[q]
            angle = float(grp["Angle"][()].ravel()[0])
            stored_fid = float(grp["Fidelity"][()].ravel()[0])

            Ie = decode_payload(grp["I_e"][()])
            Ig = decode_payload(grp["I_g"][()])
            Qe = decode_payload(grp["Q_e"][()])
            Qg = decode_payload(grp["Q_g"][()])

            Ie_r, Qe_r = rotate(Ie, Qe, angle)
            Ig_r, Qg_r = rotate(Ig, Qg, angle)

            # Project onto rotated I axis (axis of max separation)
            xg, xe = Ig_r, Ie_r

            med_g, med_e = np.median(xg), np.median(xe)
            dmu = abs(med_e - med_g)
            mu_g, sig_g = clean_side_fit(xg, med_g, "low" if med_g < med_e else "high")
            mu_e, sig_e = clean_side_fit(xe, med_e, "high" if med_e > med_g else "low")
            # align clean-side fits with separation direction: g lower, e higher
            if med_g > med_e:
                mu_g, mu_e = mu_e, mu_g
                sig_g, sig_e = sig_e, sig_g
            sig_bar = 0.5 * (sig_g + sig_e)
            snr = dmu / sig_bar if sig_bar > 0 else np.nan

            F_contrast, t_best = fidelity_from_contrast(xg, xe)

            results[q] = dict(
                n_g=xg.size,
                n_e=xe.size,
                angle=angle,
                stored_fid=stored_fid,
                med_g=med_g,
                med_e=med_e,
                dmu=dmu,
                sig_g=sig_g,
                sig_e=sig_e,
                sig_bar=sig_bar,
                snr=snr,
                F_contrast=F_contrast,
                t_best=t_best,
                Ie_r=Ie_r, Qe_r=Qe_r, Ig_r=Ig_r, Qg_r=Qg_r,
            )

    # ---- Report ----
    print(f"{'Qubit':6} {'N_g':>6} {'N_e':>6} {'angle':>9} {'storedF':>9} {'dmu':>8} "
          f"{'sig_g':>7} {'sig_e':>7} {'SNR':>7} {'F_contrast':>10} {'closest_to_91.7':>9}")
    for q, r in results.items():
        flag = " <-- closest to 91.7%" if q == min(results, key=lambda k: abs(results[k]["F_contrast"] - 0.917)) else ""
        print(f"{q:6} {r['n_g']:6d} {r['n_e']:6d} {r['angle']:+9.4f} {r['stored_fid']:9.4f} "
              f"{r['dmu']:8.2f} {r['sig_g']:7.2f} {r['sig_e']:7.2f} {r['snr']:7.2f} "
              f"{r['F_contrast']:10.4f}{flag}")

    print("\nTarget from paper (Fig. 14): F = 0.917 +- 0.004")
    best_q = min(results, key=lambda k: abs(results[k]["F_contrast"] - 0.917))
    print(f"Qubit closest to 91.7% by recomputed contrast fidelity: {best_q} "
          f"(F={results[best_q]['F_contrast']:.4f}, stored={results[best_q]['stored_fid']:.4f})")

    # ---- Plot: rotated I-Q scatter for the best-matching qubit (Fig. 14 style) ----
    r = results[best_q]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.scatter(r["Ig_r"], r["Qg_r"], s=6, c="tab:blue", alpha=0.5, label="|g>", marker="o")
    ax.scatter(r["Ie_r"], r["Qe_r"], s=6, c="tab:red", alpha=0.5, label="|e>", marker="o")
    ax.axvline(r["t_best"], color="black", ls="--", lw=1, label=f"threshold = {r['t_best']:.2f}")
    ax.scatter([r["med_g"]], [np.median(r["Qg_r"])], c="black", marker="|", s=120, label="medians")
    ax.scatter([r["med_e"]], [np.median(r["Qe_r"])], c="black", marker="|", s=120)
    ax.set_xlabel("I (rotated, a.u.)")
    ax.set_ylabel("Q (rotated, a.u.)")
    ax.set_title(f"{best_q} single-shot readout (rotated I-Q), N={r['n_g']} shots/state")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = np.linspace(min(r["Ig_r"].min(), r["Ie_r"].min()),
                       max(r["Ig_r"].max(), r["Ie_r"].max()), 60)
    ax.hist(r["Ig_r"], bins=bins, alpha=0.6, label="|g>", color="tab:blue", density=True)
    ax.hist(r["Ie_r"], bins=bins, alpha=0.6, label="|e>", color="tab:red", density=True)
    ax.axvline(r["t_best"], color="black", ls="--", lw=1)
    ax.set_xlabel("I (rotated, a.u.)")
    ax.set_ylabel("Density")
    ax.set_title(f"Projected I histogram; F = {r['F_contrast']*100:.2f}% (contrast), "
                 f"stored = {r['stored_fid']*100:.2f}%")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved plot to {OUT_PNG} (best-matching qubit: {best_q})")

    # Also dump per-qubit numbers to a small text report
    with open("readout_report.txt", "w") as fh:
        fh.write("Qubit readout fidelity recomputation from ss_ge.h5\n")
        fh.write("=" * 55 + "\n\n")
        fh.write("Paper target (Fig. 14, FERMILAB-PUB-26-0193): F = 91.7 +- 0.4%\n\n")
        fh.write(f"{'Qubit':6} {'N_g':>6} {'N_e':>6} {'angle':>9} {'storedF':>9} "
                 f"{'dmu':>8} {'sig_g':>7} {'sig_e':>7} {'SNR':>7} {'F_contrast':>10}\n")
        for q, r in results.items():
            fh.write(f"{q:6} {r['n_g']:6d} {r['n_e']:6d} {r['angle']:+9.4f} {r['stored_fid']:9.4f} "
                     f"{r['dmu']:8.2f} {r['sig_g']:7.2f} {r['sig_e']:7.2f} {r['snr']:7.2f} "
                     f"{r['F_contrast']:10.4f}\n")
    print("Saved numeric summary to readout_report.txt")


if __name__ == "__main__":
    main()

"""Reproduce QICK Box paper Fig. 14 readout fidelity from raw I/Q shots in ss_ge.h5.

Method (FERMILAB-PUB-26-0193, Sec. III C):
  1. Rotate I-Q so the axis of maximum state separation aligns with I.
  2. Project to 1D: x = I*cos(theta) + Q*sin(theta).
  3. Medians mu_g, mu_e of the two clusters; separation dmu = |mu_e - mu_g|.
  4. Clean-side widths: std of {x_g <= mu_g} (ground) and {x_e >= mu_e} (excited).
     sigma_bar = sqrt((sig_g^2 + sig_e^2)/2); SNR = dmu / sigma_bar.
  5. Single-shot fidelity = maximum cumulative-histogram contrast:
     F(t) = 0.5*(P(x_g <= t) + P(x_e > t)), maximized over threshold t,
     delta F = sqrt(F(1-F)/N_total), N_total = N_g + N_e.
"""
import h5py
import numpy as np
from math import erf, sqrt, pi

H5 = "/home/hyoklee/aurora/bench/qick/runs/coder_mistral-large-4/ss_ge.h5"
QUBITS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


def parse(raw):
    s = raw.decode().strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    return np.array(s.replace(",", " ").split(), dtype=float)


def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def max_contrast_fidelity(xg, xe):
    """Maximum cumulative-histogram contrast (empirical, threshold optimized)."""
    allv = np.sort(np.concatenate([xg, xe]))
    ths = 0.5 * (allv[:-1] + allv[1:])
    sg = np.searchsorted(np.sort(xg), ths, side="right") / xg.size  # P(x_g <= t)
    se = 1.0 - np.searchsorted(np.sort(xe), ths, side="right") / xe.size  # P(x_e > t)
    F = 0.5 * (sg + se)
    i = int(np.argmax(F))
    return float(F[i]), float(ths[i])


def analyze_qubit(g):
    Ig, Qg = parse(g["I_g"][0]), parse(g["Q_g"][0])
    Ie, Qe = parse(g["I_e"][0]), parse(g["Q_e"][0])
    angle = float(g["Angle"][0])
    stored_fid = float(g["Fidelity"][0])

    out = {"angle": angle, "stored": stored_fid, "Ng": Ig.size, "Ne": Ie.size}

    variants = {}
    for name, th in [("stored+", angle), ("stored-", -angle)]:
        xg = Ig * np.cos(th) + Qg * np.sin(th)
        xe = Ie * np.cos(th) + Qe * np.sin(th)
        variants[name] = (xg, xe, th)
    dI, dQ = Ie.mean() - Ig.mean(), Qe.mean() - Qg.mean()
    th_opt = float(np.arctan2(dQ, dI))
    xg = Ig * np.cos(th_opt) + Qg * np.sin(th_opt)
    xe = Ie * np.cos(th_opt) + Qe * np.sin(th_opt)
    variants["optimal"] = (xg, xe, th_opt)
    variants["none"] = (Ig.copy(), Ie.copy(), 0.0)

    results = {}
    for name, (xg, xe, th) in variants.items():
        mu_g, mu_e = float(np.median(xg)), float(np.median(xe))
        dmu = abs(mu_e - mu_g)
        cg = xg[xg <= mu_g] if mu_g < mu_e else xg[xg >= mu_g]
        ce = xe[xe >= mu_e] if mu_e > mu_g else xe[xe <= mu_e]
        sig_g, sig_e = float(np.std(cg, ddof=1)), float(np.std(ce, ddof=1))
        sig_bar = sqrt((sig_g**2 + sig_e**2) / 2.0)
        snr = dmu / sig_bar
        dmu_err = sqrt((sig_g**2 * pi / 2 / xg.size) + (sig_e**2 * pi / 2 / xe.size))
        sig_bar_err = sig_bar * sqrt(0.5 * (1.0 / (2 * cg.size) + 1.0 / (2 * ce.size)))
        snr_err = snr * sqrt((dmu_err / dmu) ** 2 + (sig_bar_err / sig_bar) ** 2)
        F, tstar = max_contrast_fidelity(xg, xe)
        n_tot = xg.size + xe.size
        F_err = sqrt(F * (1 - F) / n_tot)
        F_gauss = norm_cdf(dmu / (2 * sig_bar))
        results[name] = dict(
            theta=th, mu_g=mu_g, mu_e=mu_e, dmu=dmu, dmu_err=dmu_err,
            sig_g=sig_g, sig_e=sig_e, sig_bar=sig_bar,
            n_clean_g=cg.size, n_clean_e=ce.size,
            snr=snr, snr_err=snr_err,
            F=F, F_err=F_err, tstar=tstar, F_gauss=F_gauss,
        )
    out["variants"] = results
    return out


def main():
    with h5py.File(H5, "r") as f:
        for q in QUBITS:
            r = analyze_qubit(f[q])
            print(f"===== {q}  stored Angle={r['angle']:+.6f} rad  stored Fidelity={r['stored']*100:.2f}%  N={r['Ng']}/{r['Ne']}")
            for name, v in r["variants"].items():
                print(
                    f"  [{name:8s}] theta={v['theta']:+.6f}  dmu={v['dmu']:8.3f}+-{v['dmu_err']:.3f}  "
                    f"sig_g={v['sig_g']:7.3f} sig_e={v['sig_e']:7.3f} (clean n={v['n_clean_g']}/{v['n_clean_e']})  "
                    f"SNR={v['snr']:.3f}+-{v['snr_err']:.3f}  "
                    f"F_maxcontrast={v['F']*100:6.2f}% +-{v['F_err']*100:.2f}% (thr={v['tstar']:.2f})  "
                    f"F_gauss={v['F_gauss']*100:.2f}%"
                )
            print()


if __name__ == "__main__":
    main()

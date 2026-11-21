"""Reproduce Fig. 14 of FERMILAB-PUB-26-0193 from ss_ge.h5 raw I/Q shots.

Panel (a): rotated I-Q scatter (g blue, e red) with median markers.
Panel (b): 1D histogram of the rotated I projection with clean-side Gaussian
           core overlays, discrimination threshold, and the annotated
           delta_mu, sigma_g, sigma_e, SNR, and max-cumulative-contrast fidelity.
Saves readout.png in the workspace root.
"""
import h5py
import numpy as np
from math import erf, sqrt, pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5 = "/home/hyoklee/aurora/bench/qick/runs/coder_mistral-large-4/ss_ge.h5"
OUT = "/home/hyoklee/aurora/bench/qick/runs/coder_mistral-large-4/readout.png"
QUBITS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


def parse(raw):
    s = raw.decode().strip().lstrip("[").rstrip("]")
    return np.array(s.replace(",", " ").split(), dtype=float)


def Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def max_contrast(xg, xe):
    allv = np.sort(np.concatenate([xg, xe]))
    ths = 0.5 * (allv[:-1] + allv[1:])
    F = 0.5 * (np.searchsorted(np.sort(xg), ths, side="right") / xg.size
               + 1.0 - np.searchsorted(np.sort(xe), ths, side="right") / xe.size)
    i = int(np.argmax(F))
    return float(F[i]), float(ths[i])


def gaussian_core_fit(x, bins=120):
    """Fit A*exp(-(x-mu)^2/2sig^2) to a histogram of x via weighted log-parabola."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return x.mean(), 0.0, 1.0
    n, edges = np.histogram(x, bins=bins, range=(lo, hi))
    c = 0.5 * (edges[:-1] + edges[1:])
    m = n > max(3, 0.02 * n.max())
    if m.sum() < 5:
        return float(np.mean(x)), float(np.std(x)), 1.0
    w = n[m].astype(float)
    y = np.log(w)
    # weighted quadratic fit y = a + b x + c2 x^2 ; sigma = sqrt(-1/(2 c2))
    W = np.sqrt(w)
    A = np.vstack([np.ones(m.sum()), c[m], c[m] ** 2]).T * W[:, None]
    coef, *_ = np.linalg.lstsq(A, y * W, rcond=None)
    a, b, c2 = coef
    if c2 >= 0:
        return float(np.mean(x)), float(np.std(x)), 1.0
    sig = sqrt(-1.0 / (2.0 * c2))
    mu = -b / (2.0 * c2)
    amp = float(np.exp(a - c2 * mu ** 2))
    return mu, sig, amp


def analyze(g):
    Ig, Qg = parse(g["I_g"][0]), parse(g["Q_g"][0])
    Ie, Qe = parse(g["I_e"][0]), parse(g["Q_e"][0])
    stored_angle = float(g["Angle"][0])
    stored_fid = float(g["Fidelity"][0])
    # rotation aligning mean-difference with I axis (compare both signs of stored angle too)
    cands = {"stored+": stored_angle, "stored-": -stored_angle,
             "optimal": float(np.arctan2(Qe.mean() - Qg.mean(), Ie.mean() - Ig.mean()))}
    best = max(cands, key=lambda k: abs((Ie * np.cos(cands[k]) + Qe * np.sin(cands[k])).mean()
                                       - (Ig * np.cos(cands[k]) + Qg * np.sin(cands[k])).mean()))
    th = cands[best]
    c, s = np.cos(th), np.sin(th)
    xg, yg = Ig * c + Qg * s, -Ig * s + Qg * c
    xe, ye = Ie * c + Qe * s, -Ie * s + Qe * c
    mu_g, mu_e = float(np.median(xg)), float(np.median(xe))
    lo, hi = (mu_g, mu_e) if mu_g < mu_e else (mu_e, mu_g)
    dmu = abs(mu_e - mu_g)
    cg = xg[xg <= mu_g] if mu_g < mu_e else xg[xg >= mu_g]
    ce = xe[xe >= mu_e] if mu_e > mu_g else xe[xe <= mu_e]
    sig_g, sig_e = float(np.std(cg, ddof=1)), float(np.std(ce, ddof=1))
    # Gaussian core fits on the clean-side histograms
    mug_fit, sigg_fit, ag = gaussian_core_fit(cg)
    mue_fit, sige_fit, ae = gaussian_core_fit(ce)
    sig_bar = sqrt((sig_g ** 2 + sig_e ** 2) / 2.0)
    snr = dmu / sig_bar
    F, tstar = max_contrast(xg, xe)
    n_tot = xg.size + xe.size
    F_err = sqrt(F * (1 - F) / n_tot)
    pg = float(np.mean(xg <= tstar))
    pe = float(np.mean(xe > tstar))
    return dict(theta=th, best=best, stored_angle=stored_angle, stored_fid=stored_fid,
                xg=xg, yg=yg, xe=xe, ye=ye, mu_g=mu_g, mu_e=mu_e, dmu=dmu,
                sig_g=sig_g, sig_e=sig_e, sig_bar=sig_bar, snr=snr,
                cg=cg, ce=ce, sigg_fit=sigg_fit, sige_fit=sige_fit,
                mug_fit=mug_fit, mue_fit=mue_fit, ag=ag, ae=ae,
                F=F, F_err=F_err, tstar=tstar, pg=pg, pe=pe, n_tot=n_tot,
                std_full_g=float(np.std(xg, ddof=1)), std_full_e=float(np.std(xe, ddof=1)))


def main():
    with h5py.File(H5, "r") as f:
        results = {q: analyze(f[q]) for q in QUBITS}
        cfg_q1 = None
        import ast, re
        sc = re.sub(r"np\.float64\(([^)]*)\)", r"\1", f["Q1/Syst Config"][0].decode())
        cfg = ast.literal_eval(sc)
        cfg_q1 = (cfg["list_of_all_qubits"], cfg["qubit_freq_ge"], cfg["res_freq_ge"])

    print("Per-qubit summary (rotation = sign of stored Angle maximizing |dmu|, plus optimal angle):")
    for q, r in results.items():
        print(f"{q}: best={r['best']} theta={r['theta']:+.6f} (stored Angle={r['stored_angle']:+.6f}) "
              f"dmu={r['dmu']:.3f} sig_g={r['sig_g']:.3f} sig_e={r['sig_e']:.3f} "
              f"(core-fit sig: {r['sigg_fit']:.3f}/{r['sige_fit']:.3f}, full std {r['std_full_g']:.2f}/{r['std_full_e']:.2f}) "
              f"SNR={r['snr']:.3f} F={r['F']*100:.2f}%+-{r['F_err']*100:.2f}% "
              f"(Pg={r['pg']*100:.2f}% Pe={r['pe']*100:.2f}% at thr={r['tstar']:.2f}) stored={r['stored_fid']*100:.2f}%")
    print("Q1 syst config:", cfg_q1)

    r = results["Q1"]  # Q1 reproduces the paper's 91.7%
    xg, xe, yg, ye = r["xg"], r["xe"], r["yg"], r["ye"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) rotated I-Q scatter
    ax1.scatter(xg, yg, s=6, c="tab:blue", alpha=0.45, label=f"|g> ({xg.size} shots)", edgecolors="none")
    ax1.scatter(xe, ye, s=6, c="tab:red", alpha=0.45, label=f"|e> ({xe.size} shots)", edgecolors="none")
    ax1.plot(r["mu_g"], np.median(yg), marker="X", ms=12, color="black", ls="none",
             label="median |g>")
    ax1.plot(r["mu_e"], np.median(ye), marker="X", ms=12, color="black", ls="none",
             label="median |e>")
    ax1.set_xlabel("rotated I (a.u.)")
    ax1.set_ylabel("rotated Q (a.u.)")
    ax1.set_title("(a) Rotated I-Q scatter, Q1 (3000 shots/state)")
    ax1.legend(loc="best", fontsize=8)

    # (b) projected histogram with clean-side Gaussian cores
    bins = np.linspace(min(xg.min(), xe.min()), max(xg.max(), xe.max()), 121)
    ax2.hist(xg, bins=bins, color="tab:blue", alpha=0.55, label="|g>", density=False)
    ax2.hist(xe, bins=bins, color="tab:red", alpha=0.55, label="|e>", density=False)
    # scale Gaussian core fits to histogram counts
    binw = bins[1] - bins[0]
    xs = np.linspace(bins[0], bins[-1], 600)
    # clean-side cores: g -> x <= mu_g ; e -> x >= mu_e
    gmask = xs <= r["mu_g"]
    emask = xs >= r["mu_e"]
    ng = len(r["cg"]); ne = len(r["ce"])
    ax2.plot(xs[gmask], r["ag"] * ng * binw * np.exp(-(xs[gmask] - r["mug_fit"]) ** 2 / (2 * r["sigg_fit"] ** 2)),
             "k--", lw=1.8, label="Gaussian core fit (clean side)")
    ax2.plot(xs[emask], r["ae"] * ne * binw * np.exp(-(xs[emask] - r["mue_fit"]) ** 2 / (2 * r["sige_fit"] ** 2)),
             "k--", lw=1.8)
    ax2.axvline(r["mu_g"], color="tab:blue", ls=":", lw=1.2)
    ax2.axvline(r["mu_e"], color="tab:red", ls=":", lw=1.2)
    ax2.axvline(r["tstar"], color="green", ls="-.", lw=1.5, label=f"optimal threshold ({r['tstar']:.1f})")
    ax2.set_xlabel("rotated I (a.u.)")
    ax2.set_ylabel("counts")
    ax2.set_title("(b) Projected histogram along rotated I")
    ax2.legend(loc="best", fontsize=8)

    txt = (f"Q1 (qubit index {cfg_q1[0][0]}, f_ge={cfg_q1[1]:.1f} MHz)\n"
           f"$\\Delta\\mu$ = {r['dmu']:.1f} a.u. (medians {r['mu_g']:.1f} / {r['mu_e']:.1f})\n"
           f"clean-side RMS: $\\sigma_g$ = {r['sig_g']:.1f}, $\\sigma_e$ = {r['sig_e']:.1f} a.u. "
           f"(Gaussian core fits: {r['sigg_fit']:.1f} / {r['sige_fit']:.1f})\n"
           f"SNR = {r['snr']:.2f}   "
           f"$F_{{SSF}}$ = {r['F']*100:.1f} $\\pm$ {r['F_err']*100:.2f}% "
           f"(paper: 91.7 $\\pm$ 0.4%, SNR 4.4 $\\pm$ 0.1)\n"
           f"stored file Fidelity = {r['stored_fid']*100:.2f}%")
    fig.text(0.5, -0.04, txt, ha="center", fontsize=9)
    fig.suptitle("Single-shot readout of Q1 recomputed from ss_ge.h5 (Fig. 14 of FERMILAB-PUB-26-0193)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()

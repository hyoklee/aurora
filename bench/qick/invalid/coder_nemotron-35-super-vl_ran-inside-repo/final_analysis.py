"""Recompute single-shot readout fidelity from ss_ge.h5 using the paper's method.

Method (per fermilab-pub-26-0193, Sec. III.C):
  1. Rotate each shot (I,Q) so the axis of maximum state separation lies on I.
  2. Project onto I; take medians mu_g, mu_e; dmu = |mu_e - mu_g|.
  3. Clean-side Gaussian cores: g shots with I <= mu_g, e shots with I >= mu_e;
     sigma_g/e = RMS widths; sigma_bar = sqrt((sg^2+se^2)/2); SNR = dmu/sigma_bar.
  4. Single-shot fidelity = max over threshold of
     (frac_g(I < th) + frac_e(I > th)) / 2  (max cumulative histogram contrast).
"""
import re
import ast
import json
import h5py
import numpy as np
from math import erfc, sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NUM_RE = re.compile(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
PYTHON = "/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/venv/bin/python"


def parse_bytes(raw):
    if isinstance(raw, np.ndarray):
        raw = raw[0]
    s = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    return np.array([float(x) for x in NUM_RE.findall(s)])


def parse_syst(raw):
    if isinstance(raw, np.ndarray):
        raw = raw[0]
    s = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    s = re.sub(r"np\.float64\(([^)]*)\)", r"\1", s)
    return ast.literal_eval(s)


def max_contrast_fidelity(xg, xe):
    """Max cumulative-histogram-contrast fidelity; excited cluster on the right."""
    if np.median(xe) < np.median(xg):
        xg, xe = xe, xg
    pool = np.unique(np.concatenate([xg, xe]))
    thr = (pool[:-1] + pool[1:]) / 2.0
    sg_s = np.sort(xg)
    se_s = np.sort(xe)
    F = ((np.searchsorted(sg_s, thr, side="right") / xg.size)
         + (1.0 - np.searchsorted(se_s, thr, side="right") / xe.size)) / 2.0
    i = int(np.argmax(F))
    return float(F[i]), float(thr[i])


def analyze(q, g):
    angle = float(np.asarray(g["Angle"][()]).ravel()[0])
    stored_fid = float(np.asarray(g["Fidelity"][()]).ravel()[0])
    Ig = parse_bytes(g["I_g"][()])
    Ie = parse_bytes(g["I_e"][()])
    Qg = parse_bytes(g["Q_g"][()])
    Qe = parse_bytes(g["Q_e"][()])
    syst = parse_syst(g["Syst Config"][()])

    Zg = Ig + 1j * Qg
    Ze = Ie + 1j * Qe

    best = None
    for sgn in [+1.0, -1.0]:
        rot = np.exp(-1j * sgn * angle)
        xg = np.real(Zg * rot)
        xe = np.real(Ze * rot)
        if np.median(xe) < np.median(xg):
            xg, xe = xe, xg
        dmu = abs(np.median(xe) - np.median(xg))
        if best is None or dmu > best[0]:
            best = (dmu, sgn, xg, xe)
    dmu, sgn, xg, xe = best

    muc, mue = np.median(xg), np.median(xe)
    dmu = abs(mue - muc)
    g_clean = xg[xg <= muc]
    e_clean = xe[xe >= mue]
    sg = float(np.std(g_clean))
    se = float(np.std(e_clean))
    sbar = sqrt((sg**2 + se**2) / 2.0)
    snr = dmu / sbar

    F_data, thr = max_contrast_fidelity(xg, xe)
    n_total = xg.size + xe.size
    dF = sqrt(F_data * (1 - F_data) / n_total)  # binomial std err
    F_gauss = 1.0 - 0.5 * erfc(dmu / (2 * sqrt(2) * sbar))

    return dict(
        q=q, n_g=int(xg.size), n_e=int(xe.size), angle=angle, sgn=float(sgn),
        stored_fid=stored_fid, muc=muc, mue=mue, dmu=dmu, sg=sg, se=se,
        sbar=sbar, snr=snr, F_data=F_data, dF=dF, F_gauss=F_gauss,
        thr=thr, xg=xg, xe=xe,
        res_freq=syst.get("res_freq_ge"), qubit_freq=syst.get("qubit_freq_ge"),
        qubit_index=syst.get("list_of_all_qubits"),
        res_gain=syst.get("res_gain_ge"),
    )


def main():
    rows = []
    with h5py.File("ss_ge.h5", "r") as f:
        for q in sorted(f.keys(), key=lambda k: int(k[1:])):
            rows.append(analyze(q, f[q]))

    print(f"{'q':3} {'n_g':>5} {'n_e':>5} {'stored':>8} {'F_data':>8} {'+-dF':>6} {'F_gauss':>8} {'dmu':>8} {'sg':>7} {'se':>7} {'sbar':>7} {'SNR':>7}")
    for r in rows:
        print(f"{r['q']:3} {r['n_g']:5d} {r['n_e']:5d} {r['stored_fid']:8.4f} {r['F_data']:8.4f} "
              f"{r['dF']:6.4f} {r['F_gauss']:8.4f} {r['dmu']:8.2f} {r['sg']:7.2f} {r['se']:7.2f} "
              f"{r['sbar']:7.2f} {r['snr']:7.2f}")
    print()
    for r in rows:
        print("=" * 70)
        print(r["q"], "| stored Fidelity in file:", f"{r['stored_fid']:.4f}")
        print("  qubit index:", r["qubit_index"], "| qubit_freq_ge (MHz):", r["qubit_freq"],
              "| res_freq_ge[0] (MHz):", r["res_freq"][0] if r["res_freq"] else None)
        print("  medians (rotated I): g=%.3f e=%.3f  dmu=%.3f" % (r["muc"], r["mue"], r["dmu"]))
        print("  clean-side sigma: g=%.3f e=%.3f  sbar=%.3f  SNR=%.3f" % (r["sg"], r["se"], r["sbar"], r["snr"]))
        print("  F(max contrast, data) = %.4f +/- %.4f  (%.2f%%)" % (r["F_data"], r["dF"], 100 * r["F_data"]))
        print("  F(Gaussian clean-core overlap) = %.4f" % r["F_gauss"])
        print("  rotation angle stored=%.5f, chosen sign=%+.0f" % (r["angle"], r["sgn"]))

    # ---- Fig. 14-style plot for the qubit matching 91.7% (Q1) ----
    r = [x for x in rows if x["q"] == "Q1"][0]
    xg, xe = r["xg"], r["xe"]
    # recompute y (Q quadrature) with the chosen rotation for plotting
    with h5py.File("ss_ge.h5", "r") as f:
        g = f["Q1"]
        angle = float(np.asarray(g["Angle"][()]).ravel()[0])
        Ig = parse_bytes(g["I_g"][()]); Ie = parse_bytes(g["I_e"][()])
        Qg = parse_bytes(g["Q_g"][()]); Qe = parse_bytes(g["Q_e"][()])
    Zg = Ig + 1j * Qg; Ze = Ie + 1j * Qe
    rot = np.exp(-1j * r["sgn"] * angle)
    # make sure e is on the right in x as in analysis
    if np.median(np.real(Ze * rot)) < np.median(np.real(Zg * rot)):
        Zg, Ze = Ze, Zg
    xg_p = np.real(Zg * rot); xe_p = np.real(Ze * rot)
    yg_p = np.imag(Zg * rot); ye_p = np.imag(Ze * rot)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.scatter(xg_p, yg_p, s=6, c="tab:blue", alpha=0.5, label="|g>", edgecolors="none")
    ax.scatter(xe_p, ye_p, s=6, c="tab:red", alpha=0.5, label="|e>", edgecolors="none")
    ax.axvline(r["thr"], color="black", ls="--", lw=1, label=f"threshold = {r['thr']:.1f}")
    ax.plot([np.median(xg_p)], [np.median(yg_p)], "ko", ms=6, label="medians")
    ax.plot([np.median(xe_p)], [np.median(ye_p)], "ko", ms=6)
    ax.set_xlabel("I (rotated, a.u.)")
    ax.set_ylabel("Q (a.u.)")
    ax.set_title(f"Q1 (qubit 0): {100*r['F_data']:.1f}% fidelity, SNR = {r['snr']:.2f}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    bins = np.linspace(min(xg_p.min(), xe_p.min()), max(xg_p.max(), xe_p.max()), 60)
    ax.hist(xg_p, bins=bins, density=True, alpha=0.5, color="tab:blue", label="|g>")
    ax.hist(xe_p, bins=bins, density=True, alpha=0.5, color="tab:red", label="|e>")
    # clean-side Gaussian core fits (dashed)
    muc, mue = np.median(xg_p), np.median(xe_p)
    g_clean = xg_p[xg_p <= muc]
    e_clean = xe_p[xe_p >= mue]
    sg, se = np.std(g_clean), np.std(e_clean)
    xs = np.linspace(bins[0], bins[-1], 400)
    def gauss(x, mu, sig):
        return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    ax.plot(xs, gauss(xs, muc, sg), "k--", lw=1.4, label=f"Gaussian core g (sigma={sg:.1f})")
    ax.plot(xs, gauss(xs, mue, se), "k--", lw=1.4, label=f"Gaussian core e (sigma={se:.1f})")
    ax.axvline(r["thr"], color="black", ls=":", lw=1, label=f"threshold = {r['thr']:.1f}")
    ax.set_xlabel("I (rotated, a.u.)")
    ax.set_ylabel("Density (a.u.)")
    ax.set_title(f"Projected histogram: dmu = {r['dmu']:.1f}, SNR = {r['snr']:.2f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig("readout.png", dpi=150)
    print("saved readout.png")

    with open("readout_results.txt", "w") as fh:
        fh.write("Qubit readout fidelity recomputation from ss_ge.h5\n")
        fh.write("Method: fermilab-pub-26-0193 Sec. III.C (clean-side Gaussian cores + max contrast)\n\n")
        for r in rows:
            fh.write(f"{r['q']}: stored={r['stored_fid']:.4f} F_data={r['F_data']:.4f} "
                     f"+/-{r['dF']:.4f} SNR={r['snr']:.3f} dmu={r['dmu']:.2f}\n")
        fh.write(f"\nPaper Fig. 14 value: (91.7 +/- 0.4)%. Matching qubit in file: Q1 (qubit index 0).\n")

    import pickle
    with open("analysis_results.pkl", "wb") as fh:
        pickle.dump(rows, fh)


if __name__ == "__main__":
    main()

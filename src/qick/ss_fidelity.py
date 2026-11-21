#!/usr/bin/env python3
"""Recompute QICK single-shot (g/e) readout fidelity from a saved ss_ge HDF5 file.

Reproduces the standard QICK/qubit-tuning `hist()` analysis without hardware:
rotate IQ blobs so the g->e separation lies on I, then pick the threshold that
maximises |CDF_g - CDF_e|.  QICK stores that maximum contrast C in "Fidelity";
the QICK Box paper (FERMILAB-PUB-26-0193, Fig. 14) quotes the assignment
fidelity F = 1 - (P(e|g) + P(g|e))/2 = (1 + C)/2.

usage: ss_fidelity.py FILE.h5 [--plot OUT.png] [--bins N]
"""
import argparse, ast
import h5py, numpy as np


def arr(ds):
    """IQ shots are stored as numpy repr strings; parse them back to floats."""
    s = ds[0].decode()
    if "..." in s:
        raise ValueError("array was saved truncated by numpy repr")
    return np.array(s.strip("[]").split(), dtype=float)


def hist_fidelity(ig, qg, ie, qe, bins=200):
    xg, yg, xe, ye = map(np.median, (ig, qg, ie, qe))
    theta = -np.arctan2(ye - yg, xe - xg)
    c, s = np.cos(theta), np.sin(theta)
    ig_r = ig * c - qg * s
    ie_r = ie * c - qe * s
    lo, hi = min(ig_r.min(), ie_r.min()), max(ig_r.max(), ie_r.max())
    ng, edges = np.histogram(ig_r, bins=bins, range=(lo, hi))
    ne, _ = np.histogram(ie_r, bins=bins, range=(lo, hi))
    contrast = np.abs(np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())
    k = contrast.argmax()
    return contrast[k], theta, edges[k], (ig_r, ie_r, edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--plot")
    ap.add_argument("--bins", type=int, default=200)
    a = ap.parse_args()
    f = h5py.File(a.h5, "r")
    rows = []
    for q in sorted(f):
        g = f[q]
        ig, qg, ie, qe = (arr(g[k]) for k in ("I_g", "Q_g", "I_e", "Q_e"))
        fid, th, thr, extra = hist_fidelity(ig, qg, ie, qe, a.bins)
        rows.append((q, len(ig), len(ie), g["Fidelity"][0], fid, g["Angle"][0], th, thr, extra,
                     (ig, qg, ie, qe)))
        print(f"{q}: shots g={len(ig)} e={len(ie)}  stored C={g['Fidelity'][0]:.4f}  "
              f"recomputed C={fid:.4f}  F_assign=(1+C)/2={100*(1+fid)/2:.2f}%  stored angle={g['Angle'][0]:+.4f}  "
              f"recomputed angle={th:+.4f}  threshold={thr:.2f}")
    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, len(rows), figsize=(3.2 * len(rows), 6))
        for j, (q, _, _, fs, fr, _, _, thr, (igr, ier, edges), (ig, qg, ie, qe)) in enumerate(rows):
            ax[0, j].scatter(ig, qg, s=1, alpha=.3, label="g")
            ax[0, j].scatter(ie, qe, s=1, alpha=.3, label="e")
            ax[0, j].set_title(f"{q} IQ"); ax[0, j].legend(markerscale=8)
            ax[1, j].hist(igr, bins=edges, alpha=.5, label="g")
            ax[1, j].hist(ier, bins=edges, alpha=.5, label="e")
            ax[1, j].axvline(thr, color="k", ls="--")
            ax[1, j].set_title(f"F={100*(1+fr)/2:.1f}%  C={100*fr:.1f}% (stored {100*fs:.1f}%)", fontsize=9)
        fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print("wrote", a.plot)


if __name__ == "__main__":
    main()

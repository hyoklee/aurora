#!/usr/bin/env python3
"""
Regenerate Figure 10 of the IceCube 7-year point-source paper:

    "A Search for Neutrino Point-Source Populations in 7 yr of IceCube Data
     with Neutrino-count Statistics"
    ApJ 893 (2020) 102, arXiv:1909.08623
    IceCube Collaboration

Figure 10 shows the pointwise likelihood ratio M(d; phi) over the space of
standard candle luminosity functions, parameterized by the effective
luminosity L_eff (erg yr^-1) and the local source density rho_0 (Mpc^-3),
with the electromagnetic source classes (FSRQ, BL LAC, Galaxy Clusters,
FR-II) marked.

All data and parameters are read from clio-core via cte_ice. Nothing is
downloaded and no local copy of posterior.h5 is used.
"""
import os
import sys
import time

# ---- clio-core runtime environment (mirrors clio_env.sh) ----
B = "/home/hyoklee/clio-core.dataverse/build/bin"
os.environ["LD_LIBRARY_PATH"] = (
    B + ":/opt/aurora/26.181.0/oneapi/compiler/latest/lib:/home/hyoklee/"
    "miniconda3/envs/iowarp/lib" + os.environ.get("LD_LIBRARY_PATH", ""))
os.environ["PATH"] = B + ":" + os.environ.get("PATH", "")
os.environ["PYTHONPATH"] = B
os.environ["CLIO_SERVER_CONF"] = "/home/hyoklee/aurora/bench/icecube7/clio_ice.yaml"
os.environ["CAE_DATAVERSE_CACHE"] = (
    "/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-"
    "00dce7042087/scratchpad/ice/dvcache")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, B)

import numpy as np
from scipy.stats import gaussian_kde

import cte_ice as ci

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Tag prefix for the ingested release zip (see ice_omni.yaml / ci.manifest()).
ZIP = ("20190904_Bayesian_posterior_for_IceCube_7-year_point-source_"
       "search_with_neutrino-count_statistics.zip")
BASE = "ice/" + ZIP + "/"


def main():
    # ---------------- Read all data and parameters from clio-core ----------------
    t_read0 = time.time()

    # Posterior samples for the Isotropic spatial template (used for Fig. 10).
    samples = ci.read_table(BASE + "posterior.h5/Isotropic")
    # Table attributes: P_log_evidence, NP_log_evidence.
    table_attrs = ci.read_attributes(BASE + "posterior.h5/Isotropic")
    # Root attributes: priors and units.
    root_attrs = ci.read_attributes(BASE + "posterior.h5")
    t_read1 = time.time()
    read_time = t_read1 - t_read0

    # ---------------- Build posterior kernel-density estimate and prior ----------------
    sample_array = np.vstack([
        samples["ln_A"],
        samples["ln_Fb"],
        samples["n1"],
        samples["n2"],
    ])
    posterior_estimate = gaussian_kde(sample_array)

    prior = (root_attrs["prior_ln_A"] * root_attrs["prior_ln_Fb"] *
             root_attrs["prior_n1"] * root_attrs["prior_n2"])

    NP_log_evidence = table_attrs["NP_log_evidence"]
    P_log_evidence = table_attrs["P_log_evidence"]

    # ---------------- Bayes factor and pointwise likelihood ratio ----------------
    def ln_model_evidence(params):
        """ln p(d | model) = ln NP_evidence + ln posterior - ln prior.

        Eq.: p(d|theta) = p(d) p(theta|d) / p(theta), with p(d) the
        non-Poissonian evidence.
        """
        params = np.asarray(params, dtype=float)
        if params.ndim == 1:
            params = params.reshape(4, 1)          # single point -> (4, 1)
        return (NP_log_evidence
                + posterior_estimate.logpdf(params)
                - np.log(prior))

    def compute_bayes_factor(pd):
        """Log Bayes factor ln[p(d|model)/p(d|Poisson)] at a parameter point."""
        params = np.array([[pd["ln_A"]], [pd["ln_Fb"]],
                           [pd["n1"]], [pd["n2"]]])
        return float(ln_model_evidence(params)[0] - P_log_evidence)

    def compute_M_factor(params, model_prior=0.5):
        """Pointwise likelihood ratio M(d; phi), Eq. 27 of the paper.

        ln M = ln p(d|model) + ln p(non-Poisson)
               - ln[ p(d|Poisson) p(Poisson) + p(d|non-Poisson) p(non-Poisson) ]
        """
        params = np.asarray(params, dtype=float)
        if params.ndim == 1:
            params = params.reshape(4, 1)
        lme = ln_model_evidence(params)
        ln_model_space = np.logaddexp(
            P_log_evidence + np.log(1.0 - model_prior),
            NP_log_evidence + np.log(model_prior),
        )
        return lme + np.log(model_prior) - ln_model_space

    # ---------------- Log Bayes factors for the two example models ----------------
    # (These are the two models evaluated in the release's example.py.)
    model_params_example = dict(ln_A=np.log(1e16), ln_Fb=np.log(1e-12),
                                n1=1.9, n2=-2)
    model_params_example2 = dict(ln_A=np.log(1e17), ln_Fb=np.log(1e-13),
                                 n1=1.9, n2=-2)

    ln_bf_1 = compute_bayes_factor(model_params_example)
    ln_bf_2 = compute_bayes_factor(model_params_example2)

    print(f"[fig10] clio-core data read time: {read_time:.4f} s")
    print(f"[fig10] Log Bayes Factor model 1 "
          f"(ln_A=ln(1e16), ln_Fb=ln(1e-12), n1=1.9, n2=-2): {ln_bf_1:.6f}")
    print(f"[fig10] Log Bayes Factor model 2 "
          f"(ln_A=ln(1e17), ln_Fb=ln(1e-13), n1=1.9, n2=-2): {ln_bf_2:.6f}")

    # ---------------- Figure 10: standard-candle (L_eff, rho_0) grid ----------------
    # Grid bounds (base-10 logs).
    Leff_lims = np.array([52, 55.5])
    rho0_lims = np.array([-12, -5])
    log10_Leff_points = np.linspace(*Leff_lims, num=100)
    log10_rho0_points = np.linspace(*rho0_lims, num=100)

    # FIRESONG-derived scaling: A <-> rho_0, F_b <-> L_eff.
    # log10 F_b = log10 L_eff + (-13 - 52)
    # log10 A   = log10 rho_0 + (18 - (-7))
    log10_Fb_points = log10_Leff_points + (-13 - 52)
    log10_A_points = log10_rho0_points + (18 - (-7))

    ln_Fb_points = log10_Fb_points * np.log(10)
    ln_A_points = log10_A_points * np.log(10)

    ln_A_mesh, ln_Fb_mesh = np.meshgrid(ln_A_points, ln_Fb_points)
    log10_rho0_mesh, log10_Leff_mesh = np.meshgrid(log10_rho0_points,
                                                   log10_Leff_points)

    N_eval = ln_A_mesh.size
    posterior_eval = np.vstack([
        ln_A_mesh.ravel(),
        ln_Fb_mesh.ravel(),
        1.9 * np.ones(N_eval),        # n1 fixed
        -2.0 * np.ones(N_eval),       # n2 fixed
    ])
    ln_M_mesh = compute_M_factor(posterior_eval, model_prior=0.5).reshape(
        ln_A_mesh.shape)

    # ---------------- Plot ----------------
    mpl.rcParams["figure.figsize"] = 10 / 1.1, 8 / 1.1
    mpl.rcParams["xtick.labelsize"] = 22
    mpl.rcParams["ytick.labelsize"] = 22
    mpl.rcParams["axes.labelsize"] = 24
    mpl.rcParams["legend.fontsize"] = 20

    cont_levels = [-2 * np.log(10), -1 * np.log(10), 1000]
    cont_level_labels = [r"10^{-2}", r"10^{-1}", "max"]

    fig, ax = plt.subplots()

    ax.contour(10 ** log10_Leff_mesh, 10 ** log10_rho0_mesh,
               ln_M_mesh, levels=cont_levels, colors=["k", "w", "w"])
    contf = ax.contourf(10 ** log10_Leff_mesh, 10 ** log10_rho0_mesh,
                        ln_M_mesh, levels=cont_levels,
                        colors=["indigo", "mediumseagreen", "b"])

    # Legend: M(d; phi) > 10^-2 and M(d; phi) > 10^-1.
    proxy = [Rectangle((0, 0), 1, 1, fc=pc)
             for pc in list(contf.colors)[:2]]
    legend_labels = [r"$\mathcal{M}(d; \boldsymbol{\phi}) > $" + lbl + r"$"
                     for lbl in cont_level_labels[:-1]]
    ax.legend(proxy, legend_labels, loc=1, fancybox=False)

    # Standard-candle (electromagnetic source class) points.
    candle_points_rho0, candle_points_Leff = np.array([
        [1.000000000000004e-9, 1.0642092440647181e+55],   # FSRQ
        [6.866488450043026e-8, 1e+54],                      # BL LAC
        [0.000001346640589965585, 1.545927736419466e+53],   # Galaxy Clusters
        [5.179474679231223e-8, 1.282649830528052e+52]]).T   # FR II

    ax.scatter(candle_points_Leff[:-1], candle_points_rho0[:-1],
               marker="x", c="black", s=60)
    ax.scatter(candle_points_Leff[-1], candle_points_rho0[-1],
               marker="x", c="white", s=60)

    for i, txt in enumerate(["FSRQ", "BL LAC", "Galaxy Clusters", "FR-II"]):
        if txt == "FSRQ":
            ax.annotate(txt, (candle_points_Leff[i] * 0.9,
                              candle_points_rho0[i] * 0.35), fontsize=22)
        elif txt == "BL LAC":
            ax.annotate(txt, (candle_points_Leff[i] * 0.9,
                              candle_points_rho0[i] * 0.35), fontsize=22)
        elif txt == "Galaxy Clusters":
            ax.annotate("Galaxy", (candle_points_Leff[i] * 0.305,
                                   candle_points_rho0[i] * 0.35),
                        fontsize=22, color="white")
            ax.annotate("Clusters", (candle_points_Leff[i] * 0.85,
                                     candle_points_rho0[i] * 0.34),
                        fontsize=22)
        elif txt == "FR-II":
            ax.annotate(txt, (candle_points_Leff[i] * 0.9,
                              candle_points_rho0[i] * 0.35),
                        fontsize=22, color="white")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*(10.0 ** Leff_lims))
    ax.set_ylim(*(10.0 ** rho0_lims))
    ax.set_ylabel(r"$\rho_0$ (Mpc$^{-3}$)")
    ax.set_xlabel(r"$L_{\mathrm{eff}}$ (erg yr$^{-1}$)")
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "fig10.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[fig10] Saved {out_png}")


if __name__ == "__main__":
    main()

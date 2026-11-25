#!/usr/bin/env python
"""
Regenerate Figure 10 of the IceCube 7-year point-source paper
(Aartsen et al. 2020, ApJ 893, 102; arXiv:1909.08623).

Figure 10 shows the pointwise likelihood ratio M(d; phi) over the space of
standard candle luminosity functions, parameterized by the effective
luminosity Leff and the source density rho0, with the electromagnetic
source classes (FSRQ, BL LAC, Galaxy Clusters, FR-II) marked.

All data and parameters are read from clio-core via cte_ice (the posterior
samples, prior densities, and log evidences stored in posterior.h5).
No download is performed; the clio-core runtime must already be running.

Usage:
    source ./clio_env.sh
    /path/to/venv/bin/python fig10.py
"""

import time
import numpy as np
import scipy.stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cte_ice as ci


# ---------------------------------------------------------------------------
# Tag names for the ingested release (from ice_omni.yaml / ci.manifest())
# ---------------------------------------------------------------------------
ZIP = ("ice/20190904_Bayesian_posterior_for_IceCube_7-year_"
       "point-source_search_with_neutrino-count_statistics.zip")
POST = ZIP + "/posterior.h5"
ISO_TAG = POST + "/Isotropic"   # spatial template used for Figure 10


def main():
    # ------------------------------------------------------------------
    # 1. Read everything from clio-core (timed)
    # ------------------------------------------------------------------
    t0 = time.time()
    root_attrs = ci.read_attributes(POST)           # prior densities, units
    table_attrs = ci.read_attributes(ISO_TAG)       # P_/NP_log_evidence
    samples = ci.read_table(ISO_TAG)                # posterior samples
    t1 = time.time()
    read_time = t1 - t0
    print("clio-core read time: {:.3f} s".format(read_time))
    print("  root attributes : {}".format(list(root_attrs.keys())))
    print("  table attributes: {}".format(list(table_attrs.keys())))
    print("  posterior samples: {} rows".format(samples.shape[0]))

    # ------------------------------------------------------------------
    # 2. Build the posterior estimate (KDE) and the prior
    # ------------------------------------------------------------------
    ln_A = samples["ln_A"]
    ln_Fb = samples["ln_Fb"]
    n1 = samples["n1"]
    n2 = samples["n2"]

    # Stack the four model parameters (natural logs for A and Fb)
    sample_array = np.vstack([ln_A, ln_Fb, n1, n2])

    # Kernel-density estimate of the posterior from the equally-weighted samples
    posterior_estimate = scipy.stats.gaussian_kde(sample_array)

    # Product of the uniform prior densities (read from clio-core root attributes)
    prior = (root_attrs["prior_ln_A"]
             * root_attrs["prior_ln_Fb"]
             * root_attrs["prior_n1"]
             * root_attrs["prior_n2"])

    NP_log_evidence = table_attrs["NP_log_evidence"]
    P_log_evidence = table_attrs["P_log_evidence"]

    # ------------------------------------------------------------------
    # 3. Bayes factor and pointwise likelihood ratio (M) functions
    #    (same formulas as the release's example.py)
    # ------------------------------------------------------------------
    def compute_bayes_factor(params, post_estimate, param_prior):
        """ln BF = ln p(d|model, non-Poiss) - ln p(d|Poiss)."""
        if isinstance(params, dict):
            params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
        ln_model_evidence = (NP_log_evidence
                             + post_estimate.logpdf(params)
                             - np.log(param_prior))
        ln_alternative_evidence = P_log_evidence
        return ln_model_evidence - ln_alternative_evidence

    def compute_M_factor(params, post_estimate, param_prior, model_prior):
        """Pointwise likelihood ratio M(d; phi) from Eq. (27) of the paper."""
        if isinstance(params, dict):
            params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
        ln_model_evidence = (NP_log_evidence
                             + post_estimate.logpdf(params)
                             - np.log(param_prior))
        ln_model_space = np.logaddexp(
            P_log_evidence + np.log(1.0 - model_prior),
            NP_log_evidence + np.log(model_prior),
        )
        return ln_model_evidence + np.log(model_prior) - ln_model_space

    # ------------------------------------------------------------------
    # 4. Log Bayes factors for the two example models from example.py
    # ------------------------------------------------------------------
    model_params_example = dict(ln_A=np.log(1e16), ln_Fb=np.log(1e-12),
                                n1=1.9, n2=-2)
    model_params_example2 = dict(ln_A=np.log(1e17), ln_Fb=np.log(1e-13),
                                 n1=1.9, n2=-2)

    lbf1 = compute_bayes_factor(model_params_example, posterior_estimate, prior)[0]
    lbf2 = compute_bayes_factor(model_params_example2, posterior_estimate, prior)[0]
    print("Log Bayes Factor for model 1 (ln_A=ln(1e16), ln_Fb=ln(1e-12)): {:.6f}".format(lbf1))
    print("Log Bayes Factor for model 2 (ln_A=ln(1e17), ln_Fb=ln(1e-13)): {:.6f}".format(lbf2))

    # ------------------------------------------------------------------
    # 5. Figure 10: pointwise likelihood ratio over (Leff, rho0)
    # ------------------------------------------------------------------
    Leff_lims = np.array([52, 55.5])      # log10 Leff [erg yr^-1]
    rho0_lims = np.array([-12, -5])       # log10 rho0 [Mpc^-3]
    log10_Leff_points = np.linspace(*Leff_lims, num=100)
    log10_rho0_points = np.linspace(*rho0_lims, num=100)

    # Convert Leff -> Fb and rho0 -> A using factors derived from FIRESONG
    # (as in the release's example.py)
    log10_Fb_points = log10_Leff_points + (-13 - 52)
    log10_A_points = log10_rho0_points + (18 - -7)

    # Switch to natural logarithms (matching the stored posterior coordinates)
    ln_Fb_points = log10_Fb_points * np.log(10)
    ln_A_points = log10_A_points * np.log(10)

    # Setup the evaluation grid
    ln_A_meshpoints, ln_Fb_meshpoints = np.meshgrid(ln_A_points, ln_Fb_points)
    log10_rho0_meshpoints, log10_Leff_meshpoints = np.meshgrid(
        log10_rho0_points, log10_Leff_points)
    N_evalpoints = len(ln_A_meshpoints.ravel())

    # Evaluate M on the grid with n1 = 1.9, n2 = -2 fixed
    posterior_evalpoints = np.vstack([
        ln_A_meshpoints.ravel(),
        ln_Fb_meshpoints.ravel(),
        1.9 * np.ones(N_evalpoints),
        -2.0 * np.ones(N_evalpoints),
    ])
    ln_M_meshpoints = compute_M_factor(
        posterior_evalpoints, posterior_estimate, prior, 0.5
    ).reshape(ln_A_meshpoints.shape)

    # ------------------------------------------------------------------
    # 6. Plot (same style as the release's example.py, but without LaTeX)
    # ------------------------------------------------------------------
    matplotlib.rcParams["figure.figsize"] = 10 / 1.1, 8 / 1.1
    matplotlib.rcParams["xtick.labelsize"] = 22
    matplotlib.rcParams["ytick.labelsize"] = 22
    matplotlib.rcParams["axes.labelsize"] = 24
    matplotlib.rcParams["legend.fontsize"] = 20
    # LaTeX is not available in this environment; use mathtext instead.
    plt.rc("text", usetex=False)

    cont_levels = [-2.0 * np.log(10), -1.0 * np.log(10), 1000.0]
    # Labels without math delimiters; the wrapper adds a single $...$ pair.
    cont_level_labels = [r"10^{-2}", r"10^{-1}", "max"]

    fig, ax = plt.subplots()

    # Contour lines
    cont = ax.contour(
        10 ** log10_Leff_meshpoints,
        10 ** log10_rho0_meshpoints,
        ln_M_meshpoints,
        levels=cont_levels,
        colors=["k", "w", "w"],
    )

    # Filled contour regions
    contf = ax.contourf(
        10 ** log10_Leff_meshpoints,
        10 ** log10_rho0_meshpoints,
        ln_M_meshpoints,
        levels=cont_levels,
        colors=["indigo", "mediumseagreen", "b"],
    )

    # Legend for the filled regions (M > 10^-2 and M > 10^-1).
    # matplotlib >= 3.1 removed QuadContourSet.collections; use the
    # face colors of the filled levels directly.
    face_colors = contf.get_facecolors()
    n_regions = len(cont_level_labels[:-1])  # number of legend entries
    proxy = [Rectangle((0, 0), 1, 1, fc=face_colors[i])
             for i in range(n_regions)]
    # Single math expression (no extra $ in the middle).
    legend_labels = [r"$\mathcal{M}(d; \phi) > " + lbl + r"$"
                     for lbl in cont_level_labels[:-1]]
    ax.legend(proxy, legend_labels, loc=1, fancybox=False)

    # ------------------------------------------------------------------
    # 7. Overplot the electromagnetic standard candle source classes
    #    (parameters taken from the release's example.py, originally
    #    from arXiv:1411.4385 via FIRESONG)
    # ------------------------------------------------------------------
    candle_points = np.array([
        [1.000000000000004e-9, 1.0642092440647181e+55],   # FSRQ
        [6.866488450043026e-8, 1e+54],                      # BL LAC
        [0.000001346640589965585, 1.545927736419466e+53],   # Gal. Clusters
        [5.179474679231223e-8, 1.282649830528052e+52],      # FR II
    ]).T
    candle_points_rho0 = candle_points[0]
    candle_points_Leff = candle_points[1]

    # First three candles in black, last (FR-II) in white
    ax.scatter(candle_points_Leff[:-1], candle_points_rho0[:-1],
               marker="x", c="black", s=60)
    ax.scatter(candle_points_Leff[-1], candle_points_rho0[-1],
               marker="x", c="white", s=60)

    for i, txt in enumerate(["FSRQ", "BL LAC", "Galaxy Clusters", "FR-II"]):
        if txt == "FSRQ":
            ax.annotate(txt,
                        (candle_points_Leff[i] * 0.9, candle_points_rho0[i] * 0.35),
                        fontsize=22)
        elif txt == "BL LAC":
            ax.annotate(txt,
                        (candle_points_Leff[i] * 0.9, candle_points_rho0[i] * 0.35),
                        fontsize=22)
        elif txt == "Galaxy Clusters":
            ax.annotate("Galaxy",
                        (candle_points_Leff[i] * 0.305, candle_points_rho0[i] * 0.35),
                        fontsize=22, color="white")
            ax.annotate("Clusters",
                        (candle_points_Leff[i] * 0.85, candle_points_rho0[i] * 0.34),
                        fontsize=22)
        elif txt == "FR-II":
            ax.annotate(txt,
                        (candle_points_Leff[i] * 0.9, candle_points_rho0[i] * 0.35),
                        fontsize=22, color="white")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*10.0 ** Leff_lims)
    ax.set_ylim(*10.0 ** rho0_lims)
    ax.set_ylabel(r"$\rho_0$ (Mpc$^{-3}$)")
    ax.set_xlabel(r"$L_{\rm eff}$ (erg yr$^{-1}$)")
    plt.tight_layout()
    plt.savefig("fig10.png")
    plt.close()
    print("Figure saved to fig10.png")


if __name__ == "__main__":
    main()

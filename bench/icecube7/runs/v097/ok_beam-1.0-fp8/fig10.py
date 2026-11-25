#!/usr/bin/env python
"""Regenerate Figure 10 of arXiv:1909.08623 (ApJ 893, 102) from the IceCube
7-year point-source posterior stored in clio-core.

All data and parameters are read from clio-core via cte_ice (no downloads,
no local posterior.h5).  Run with:
    source ./clio_env.sh
    python fig10.py
"""
import os
import sys
import time

import numpy as np
import scipy.stats

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cte_ice as ci


def main():
    t_start = time.time()

    # ---- Locate the ingested release in clio-core ----
    root_tag = None
    for tag in ci.tags(r"ice/.*/posterior\.h5"):
        root_tag = tag
        break
    if root_tag is None:
        raise RuntimeError("posterior.h5 tag not found in clio-core")
    zip_name = root_tag.split("/")[1]
    table_tag = root_tag + "/Isotropic"   # template used in the release example

    # ---- Read all data and parameters from clio-core ----
    t_read = time.time()
    root_attrs = ci.read_attributes(root_tag)
    table_attrs = ci.read_attributes(table_tag)
    posterior_samples = ci.read_table(table_tag)
    read_elapsed = time.time() - t_read

    # Priors (from root attributes)
    prior_ln_A = root_attrs["prior_ln_A"]
    prior_ln_Fb = root_attrs["prior_ln_Fb"]
    prior_n1 = root_attrs["prior_n1"]
    prior_n2 = root_attrs["prior_n2"]
    prior = prior_ln_A * prior_ln_Fb * prior_n1 * prior_n2

    # Evidence (from table attributes)
    NP_log_evidence = table_attrs["NP_log_evidence"]
    P_log_evidence = table_attrs["P_log_evidence"]

    # Posterior samples
    sample_array = np.vstack([
        posterior_samples["ln_A"],
        posterior_samples["ln_Fb"],
        posterior_samples["n1"],
        posterior_samples["n2"],
    ])
    posterior_estimate = scipy.stats.gaussian_kde(sample_array)

    # ---- Bayes factor / M factor (from the release example.py) ----
    def compute_bayes_factor(params):
        if isinstance(params, dict):
            params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
        ln_model_evidence = (NP_log_evidence
                             + posterior_estimate.logpdf(params)
                             - np.log(prior))
        ln_alternative_evidence = P_log_evidence
        return float((ln_model_evidence - ln_alternative_evidence)[0] if np.ndim(ln_model_evidence) else
                     (ln_model_evidence - ln_alternative_evidence))

    def compute_M_factor(params, model_prior=0.5):
        if isinstance(params, dict):
            params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
        ln_model_evidence = (NP_log_evidence
                             + posterior_estimate.logpdf(params)
                             - np.log(prior))
        ln_model_space = np.logaddexp(
            P_log_evidence + np.log(1.0 - model_prior),
            NP_log_evidence + np.log(model_prior))
        ln_M = ln_model_evidence + np.log(model_prior) - ln_model_space
        return np.asarray(ln_M).ravel()

    # Example models from the release
    model_params_example = dict(ln_A=np.log(1e16), ln_Fb=np.log(1e-12),
                               n1=1.9, n2=-2)
    model_params_example2 = dict(ln_A=np.log(1e17), ln_Fb=np.log(1e-13),
                                 n1=1.9, n2=-2)
    bf1 = compute_bayes_factor(model_params_example)
    bf2 = compute_bayes_factor(model_params_example2)
    print("Log Bayes Factor for model {} = {}".format(model_params_example, bf1))
    print("Log Bayes Factor for model {} = {}".format(model_params_example2, bf2))

    # ---- Reproduce Fig. 10 ----
    Leff_lims = np.array([52.0, 55.5])
    rho0_lims = np.array([-12.0, -5.0])
    log10_Leff_points = np.linspace(*Leff_lims, num=100)
    log10_rho0_points = np.linspace(*rho0_lims, num=100)

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
        1.9 * np.ones(N_eval),
        -2.0 * np.ones(N_eval),
    ])
    ln_M_mesh = compute_M_factor(posterior_eval, 0.5).reshape(ln_A_mesh.shape)

    cont_levels = [-2 * np.log(10), -1 * np.log(10), 1000]
    cont_level_labels = [r"10^{-2}", r"10^{-1}", "max"]

    mpl.rcParams["figure.figsize"] = (10 / 1.1, 8 / 1.1)
    mpl.rcParams["xtick.labelsize"] = 22
    mpl.rcParams["ytick.labelsize"] = 22
    mpl.rcParams["axes.labelsize"] = 24
    mpl.rcParams["legend.fontsize"] = 20
    plt.rc("text", usetex=False)

    cont = plt.contour(10 ** log10_Leff_mesh, 10 ** log10_rho0_mesh,
                       ln_M_mesh, levels=cont_levels,
                       colors=["k", "w", "w"])
    contf = plt.contourf(10 ** log10_Leff_mesh, 10 ** log10_rho0_mesh,
                          ln_M_mesh, levels=cont_levels,
                          colors=["indigo", "mediumseagreen", "b"])

    fc = contf.get_facecolors()
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=fc[i]) for i in range(len(fc))]
    legend_labels = [r"$\mathcal{M}(d; \phi) > " + lbl + r"$"
                     for lbl in cont_level_labels[:-1]]
    plt.legend(proxy, legend_labels, loc=1, fancybox=False)

    candle = np.array([
        [1.000000000000004e-9, 1.0642092440647181e+55],   # FSRQ
        [6.866488450043026e-8, 1e54],                      # BL LAC
        [0.000001346640589965585, 1.545927736419466e53], # Galaxy Clusters
        [5.179474679231223e-8, 1.282649830528052e52],     # FR II
    ]).T
    plt.scatter(candle[1][:-1], candle[0][:-1], marker="x", c="black", s=60)
    plt.scatter(candle[1][-1], candle[0][-1], marker="x", c="white", s=60)
    names = ["FSRQ", "BL LAC", "Galaxy Clusters", "FR-II"]
    for i, txt in enumerate(names):
        if txt == "FSRQ":
            plt.annotate(txt, (candle[1][i] * 0.9, candle[0][i] * 0.35),
                         fontsize=22)
        if txt == "BL LAC":
            plt.annotate(txt, (candle[1][i] * 0.9, candle[0][i] * 0.35),
                         fontsize=22)
        if txt == "Galaxy Clusters":
            plt.annotate("Galaxy", (candle[1][i] * 0.305, candle[0][i] * 0.35),
                         fontsize=22, color="white")
            plt.annotate("Clusters", (candle[1][i] * 0.85, candle[0][i] * 0.34),
                         fontsize=22)
        if txt == "FR-II":
            plt.annotate(txt, (candle[1][i] * 0.9, candle[0][i] * 0.35),
                         fontsize=22, color="white")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(*10.0 ** Leff_lims)
    plt.ylim(*10.0 ** rho0_lims)
    plt.ylabel(r"$\rho_0$ (Mpc${}^{-3}$)")
    plt.xlabel(r"$L_{\mathrm{eff}}$ (erg yr${}^{-1}$)")
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fig10.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    total = time.time() - t_start
    print("Read from clio-core: {:.3f} s (data read {:.3f} s)".format(
        total, read_elapsed))
    print("Wrote", out_png)


if __name__ == "__main__":
    main()

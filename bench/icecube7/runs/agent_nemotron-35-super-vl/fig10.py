#!/usr/bin/env python3
"""Regenerate Figure 10 of the IceCube 7-year point-source paper

Reproduces Fig. 10: pointwise likelihood ratio over the space of standard
candle luminosity functions, with electromagnetic source classes marked.

All posterior data and parameters are read from clio-core (CTE) via cte_ice,
NOT from a local posterior.h5 file. Nothing is downloaded.
"""

import time
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")

# LaTeX is not installed in this environment; use mathtext instead of usetex
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/ice/runs/agent_nemotron-35-super-vl")
import cte_ice as ci


# ---------------------------------------------------------------------------
# 1. Connect to clio-core and time the data read
# ---------------------------------------------------------------------------
t0 = time.time()
ci.connect()

ZIP = "20190904_Bayesian_posterior_for_IceCube_7-year_point-source_search_with_neutrino-count_statistics.zip"
BASE = f"ice/{ZIP}"
POST_TAG = f"{BASE}/posterior.h5/Isotropic"

# Read posterior samples from clio-core
posterior_samples = ci.read_table(POST_TAG)  # numpy structured array

# Read table attributes (evidences) and root attributes (priors)
table_attrs = ci.read_attributes(POST_TAG)
root_attrs = ci.read_attributes(f"{BASE}/posterior.h5")

t_read = time.time() - t0
print(f"clio-core data read took {t_read:.3f} s")
print(f"  posterior samples: {len(posterior_samples)}")

P_log_evidence = table_attrs["P_log_evidence"]
NP_log_evidence = table_attrs["NP_log_evidence"]

# Prior cube = product of uniform prior densities (from root attributes)
prior = (root_attrs["prior_ln_A"] * root_attrs["prior_ln_Fb"]
         * root_attrs["prior_n1"] * root_attrs["prior_n2"])
print(f"  P_log_evidence = {P_log_evidence}")
print(f"  NP_log_evidence = {NP_log_evidence}")
print(f"  prior (density product) = {prior:.6e}")


# ---------------------------------------------------------------------------
# 2. Estimate posterior via kernel density estimation
# ---------------------------------------------------------------------------
# Stack ln_A, ln_Fb, n1, n2 (natural logs; n1, n2 in linear space)
sample_array = np.vstack([
    posterior_samples["ln_A"],
    posterior_samples["ln_Fb"],
    posterior_samples["n1"],
    posterior_samples["n2"],
])
posterior_estimate = scipy.stats.gaussian_kde(sample_array)


# ---------------------------------------------------------------------------
# 3. Bayes Factor and pointwise likelihood ratio (M) functions
# ---------------------------------------------------------------------------
def _as_list(params):
    if isinstance(params, dict):
        params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
    return params


def compute_bayes_factor(params, post_estimate, param_prior):
    params = _as_list(params)
    # ln p(d|A) = ln(evidence) + ln(posterior) - ln(prior)
    ln_model_evidence = (NP_log_evidence
                         + post_estimate.logpdf(params)
                         - np.log(param_prior))
    ln_alternative_evidence = P_log_evidence
    return ln_model_evidence - ln_alternative_evidence


def compute_M_factor(params, post_estimate, param_prior, model_prior):
    params = _as_list(params)
    ln_model_evidence = (NP_log_evidence
                         + post_estimate.logpdf(params)
                         - np.log(param_prior))
    # Numerically stable logaddexp for the model-space mixture
    ln_model_space = np.logaddexp(
        P_log_evidence + np.log(1.0 - model_prior),
        NP_log_evidence + np.log(model_prior),
    )
    # ln M = ln p(d|A, non-Poiss) + ln p(non-Poiss) - ln(model space)
    ln_M_factor = ln_model_evidence + np.log(model_prior) - ln_model_space
    return ln_M_factor


# ---------------------------------------------------------------------------
# 4. Bayes Factors for the two example models in the release's example code
# ---------------------------------------------------------------------------
model_params_example = dict(ln_A=np.log(1e16), ln_Fb=np.log(1e-12), n1=1.9, n2=-2)
model_params_example2 = dict(ln_A=np.log(1e17), ln_Fb=np.log(1e-13), n1=1.9, n2=-2)

bf1 = compute_bayes_factor(model_params_example, posterior_estimate, prior)[0]
bf2 = compute_bayes_factor(model_params_example2, posterior_estimate, prior)[0]
print(f"Log Bayes Factor for model 1 (ln_A={np.log(1e16):.3f}, "
      f"ln_Fb={np.log(1e-12):.3f}, n1=1.9, n2=-2): {bf1:.6f}")
print(f"Log Bayes Factor for model 2 (ln_A={np.log(1e17):.3f}, "
      f"ln_Fb={np.log(1e-13):.3f}, n1=1.9, n2=-2): {bf2:.6f}")


# ---------------------------------------------------------------------------
# 5. Reproduce Figure 10
# ---------------------------------------------------------------------------
# Figure bounds and parameter space (same as the release's example.py)
Leff_lims = np.array([52, 55.5])
rho0_lims = np.array([-12, -5])
log10_Leff_points = np.linspace(*Leff_lims, num=100)
log10_rho0_points = np.linspace(*rho0_lims, num=100)

# Convert L_eff -> F_b, and rho_0 -> A using factor derived from FIRESONG
log10_Fb_points = log10_Leff_points + (-13 - 52)
log10_A_points = log10_rho0_points + (18 - -7)

# Switch to natural logarithms
ln_Fb_points = log10_Fb_points * np.log(10)
ln_A_points = log10_A_points * np.log(10)

# Setup a grid of coordinates
ln_A_meshpoints, ln_Fb_meshpoints = np.meshgrid(ln_A_points, ln_Fb_points)
log10_rho0_meshpoints, log10_Leff_meshpoints = np.meshgrid(
    log10_rho0_points, log10_Leff_points)
N_evalpoints = len(ln_A_meshpoints.ravel())

# Compute the M factor on this grid (n1=1.9, n2=-2 fixed, standard candles)
posterior_evalpoints = np.vstack([
    ln_A_meshpoints.ravel(),
    ln_Fb_meshpoints.ravel(),
    1.9 * np.ones(N_evalpoints),
    -2 * np.ones(N_evalpoints),
])
t_m = time.time()
ln_M_meshpoints = compute_M_factor(
    posterior_evalpoints, posterior_estimate, prior, 0.5).reshape(
    ln_A_meshpoints.shape)
t_compute = time.time() - t_m
print(f"  M-factor grid computation took {t_compute:.3f} s")

# Contour levels and labels
cont_levels = [-2 * np.log(10), -1 * np.log(10), 1000]
cont_level_labels = ["$10^{-2}$", "$10^{-1}$", "max"]

# Plotting style (matching the paper as closely as possible without LaTeX)
matplotlib.rcParams["figure.figsize"] = 10 / 1.1, 8 / 1.1
matplotlib.rcParams["xtick.labelsize"] = 22
matplotlib.rcParams["ytick.labelsize"] = 22
matplotlib.rcParams["axes.labelsize"] = 24
matplotlib.rcParams["legend.fontsize"] = 20

# Plot and fill the contours
cont = plt.contour(
    10**log10_Leff_meshpoints, 10**log10_rho0_meshpoints,
    ln_M_meshpoints, levels=cont_levels, colors=["k", "w", "w"])
contf = plt.contourf(
    10**log10_Leff_meshpoints, 10**log10_rho0_meshpoints,
    ln_M_meshpoints, levels=cont_levels,
    colors=["indigo", "mediumseagreen", "b"])

# Create legend
contf_colors = ["indigo", "mediumseagreen", "b"]
proxy = [plt.Rectangle((0, 0), 1, 1, fc=c)
         for c in contf_colors[: len(cont_level_labels) - 1]]
legend_labels = [r"$\mathcal{M}(d; \boldsymbol{\phi}) > " + lbl.strip("$") + r"$"
                 for lbl in cont_level_labels[:-1]]
plt.legend(proxy, legend_labels, loc=1, fancybox=False)

# Overplot 4 electromagnetic standard candle parameters, taken from 1411.4385
candle_points_rho0, candle_points_Leff = np.array([
    [1.000000000000004e-9, 1.0642092440647181e+55],  # FSRQ
    [6.866488450043026e-8, 1e+54],                     # BL LAC
    [0.000001346640589965585, 1.545927736419466e+53],  # Gal. Clusters
    [5.179474679231223e-8, 1.282649830528052e+52]]).T  # FR II

# Place the standard candles as points on the plot
plt.scatter(candle_points_Leff[:-1], candle_points_rho0[:-1],
            marker="x", c="black", s=60)
plt.scatter(candle_points_Leff[-1], candle_points_rho0[-1],
            marker="x", c="white", s=60)
for i, txt in enumerate(["FSRQ", "BL LAC", "Galaxy Clusters", "FR-II"]):
    if txt == "FSRQ":
        plt.annotate(txt, (candle_points_Leff[i] * 0.9,
                           candle_points_rho0[i] * 0.35), fontsize=22)
    if txt == "BL LAC":
        plt.annotate(txt, (candle_points_Leff[i] * 0.9,
                           candle_points_rho0[i] * 0.35), fontsize=22)
    if txt == "Galaxy Clusters":
        plt.annotate("Galaxy", (candle_points_Leff[i] * 0.305,
                               candle_points_rho0[i] * 0.35),
                     fontsize=22, color="white")
        plt.annotate("Clusters", (candle_points_Leff[i] * 0.85,
                                  candle_points_rho0[i] * 0.34),
                     fontsize=22)
    if txt == "FR-II":
        plt.annotate(txt, (candle_points_Leff[i] * 0.9,
                           candle_points_rho0[i] * 0.35),
                     fontsize=22, color="white")

plt.xscale("log")
plt.yscale("log")
plt.xlim(*10.0**Leff_lims)
plt.ylim(*10.0**rho0_lims)
plt.ylabel(r"$\rho_0$ (Mpc$^{-3}$)")
plt.xlabel("$L_{\text{eff}}$ (erg yr$^{-1}$)")
plt.tight_layout()

OUT = "/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/ice/runs/agent_nemotron-35-super-vl/fig10.png"
plt.savefig(OUT, dpi=150)
plt.close()
print(f"Saved figure to {OUT}")
print(f"Total script time (including data read): {time.time()-t0:.3f} s")

#!/usr/bin/env python
"""Regenerate Figure 10 of arXiv:1909.08623 (ApJ 893 (2020) 102) from the
IceCube 7-yr point-source posterior stored in clio-core.

All data and parameters are read from clio-core via cte_ice; nothing is
downloaded and no local copy of posterior.h5 is used.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cte_ice as ci

# ---------------------------------------------------------------- reading data
t_read0 = time.perf_counter()

rows = ci.manifest()
# Find the posterior.h5 and example.py tags from the manifest.
post_tag = None
example_tag = None
readme_tag = None
for r in rows:
    if r["member"] == "posterior.h5":
        post_tag = r["tag"]
    elif r["member"] == "example.py":
        example_tag = r["tag"]
    elif r["member"] == "readme.txt":
        readme_tag = r["tag"]
assert post_tag is not None, "posterior.h5 not found in clio-core manifest"
assert example_tag is not None, "example.py not found in clio-core manifest"

root_attrs = ci.read_attributes(post_tag)
posterior_samples = ci.read_table(post_tag + "/Isotropic")
example_code = ci.read_text(example_tag)
readme_text = ci.read_text(readme_tag)

t_read1 = time.perf_counter()
read_time_s = t_read1 - t_read0

# Parameters / priors come from the HDF5 attributes (clio-core), not hard-coded.
prior_ln_A = float(root_attrs["prior_ln_A"])
prior_ln_Fb = float(root_attrs["prior_ln_Fb"])
prior_n1 = float(root_attrs["prior_n1"])
prior_n2 = float(root_attrs["prior_n2"])

# Attributes of the table are stored under clio-core as a separate blob.
table_attrs = ci.read_attributes(post_tag + "/Isotropic")
NP_log_evidence = float(table_attrs["NP_log_evidence"])
P_log_evidence = float(table_attrs["P_log_evidence"])

# ---------------------------------------------------------------- KDE posterior
import scipy.stats

sample_array = np.vstack([
    posterior_samples["ln_A"],
    posterior_samples["ln_Fb"],
    posterior_samples["n1"],
    posterior_samples["n2"],
])
posterior_estimate = scipy.stats.gaussian_kde(sample_array)

prior = prior_ln_A * prior_ln_Fb * prior_n1 * prior_n2


def compute_bayes_factor(params, post_estimate, param_prior):
    if isinstance(params, dict):
        params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
    ln_model_evidence = NP_log_evidence + post_estimate.logpdf(params) - np.log(param_prior)
    ln_alternative_evidence = P_log_evidence
    return ln_model_evidence - ln_alternative_evidence


def compute_M_factor(params, post_estimate, param_prior, model_prior):
    if isinstance(params, dict):
        params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
    ln_model_evidence = NP_log_evidence + post_estimate.logpdf(params) - np.log(param_prior)
    ln_model_space = np.logaddexp(
        P_log_evidence + np.log(1.0 - model_prior),
        NP_log_evidence + np.log(model_prior),
    )
    return ln_model_evidence + np.log(model_prior) - ln_model_space


# ---------------------------------------------------------------- example models
model_params_example = dict(ln_A=np.log(1e16), ln_Fb=np.log(1e-12), n1=1.9, n2=-2)
model_params_example2 = dict(ln_A=np.log(1e17), ln_Fb=np.log(1e-13), n1=1.9, n2=-2)

bf1 = float(compute_bayes_factor(model_params_example, posterior_estimate, prior)[0])
bf2 = float(compute_bayes_factor(model_params_example2, posterior_estimate, prior)[0])
print("Log Bayes Factor for model {} = {}".format(model_params_example, bf1))
print("Log Bayes Factor for model {} = {}".format(model_params_example2, bf2))

# ---------------------------------------------------------------- Figure 10
Leff_lims = np.array([52.0, 55.5])
rho0_lims = np.array([-12.0, -5.0])
log10_Leff_points = np.linspace(*Leff_lims, num=100)
log10_rho0_points = np.linspace(*rho0_lims, num=100)

log10_Fb_points = log10_Leff_points + (-13 - 52)
log10_A_points = log10_rho0_points + (18 - (-7))

ln_Fb_points = log10_Fb_points * np.log(10)
ln_A_points = log10_A_points * np.log(10)

ln_A_meshpoints, ln_Fb_meshpoints = np.meshgrid(ln_A_points, ln_Fb_points)
log10_rho0_meshpoints, log10_Leff_meshpoints = np.meshgrid(
    log10_rho0_points, log10_Leff_points
)
N_evalpoints = ln_A_meshpoints.size

posterior_evalpoints = np.vstack([
    ln_A_meshpoints.ravel(),
    ln_Fb_meshpoints.ravel(),
    1.9 * np.ones(N_evalpoints),
    -2.0 * np.ones(N_evalpoints),
])
ln_M_meshpoints = compute_M_factor(
    posterior_evalpoints, posterior_estimate, prior, 0.5
).reshape(ln_A_meshpoints.shape)

cont_levels = [-2 * np.log(10), -1 * np.log(10), 1000]
cont_level_labels = [r"10^{-2}", r"10^{-1}", "max"]

import matplotlib as mpl
mpl.rcParams["figure.figsize"] = (10 / 1.1, 8 / 1.1)
mpl.rcParams["xtick.labelsize"] = 22
mpl.rcParams["ytick.labelsize"] = 22
mpl.rcParams["axes.labelsize"] = 24
mpl.rcParams["legend.fontsize"] = 20
import matplotlib.pyplot as plt

# LaTeX is not available in this environment; use matplotlib's mathtext.
plt.rc("text", usetex=False)

cont = plt.contour(
    10 ** log10_Leff_meshpoints,
    10 ** log10_rho0_meshpoints,
    ln_M_meshpoints,
    levels=cont_levels,
    colors=["k", "w", "w"],
)
contf = plt.contourf(
    10 ** log10_Leff_meshpoints,
    10 ** log10_rho0_meshpoints,
    ln_M_meshpoints,
    levels=cont_levels,
    colors=["indigo", "mediumseagreen", "b"],
)

proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in contf.collections]
legend_labels = [
    r"$\mathcal{M}(d; \boldsymbol{\phi}) > " + lbl + r"$"
    for lbl in cont_level_labels[:-1]
]
plt.legend(proxy, legend_labels, loc=1, fancybox=False)

candle_points_rho0, candle_points_Leff = np.array([
    [1.000000000000004e-9, 1.0642092440647181e55],   # FSRQ
    [6.866488450043026e-8, 1e54],                     # BL LAC
    [0.000001346640589965585, 1.545927736419466e53], # Galaxy Clusters
    [5.179474679231223e-8, 1.282649830528052e52],    # FR II
]).T

plt.scatter(candle_points_Leff[:-1], candle_points_rho0[:-1], marker="x", c="black", s=60)
plt.scatter(candle_points_Leff[-1], candle_points_rho0[-1], marker="x", c="white", s=60)
for i, txt in enumerate(["FSRQ", "BL LAC", "Galaxy Clusters", "FR-II"]):
    if txt == "FSRQ":
        plt.annotate(txt, (candle_points_Leff[i] * 0.9, candle_points_rho0[i] * 0.35), fontsize=22)
    if txt == "BL LAC":
        plt.annotate(txt, (candle_points_Leff[i] * 0.9, candle_points_rho0[i] * 0.35), fontsize=22)
    if txt == "Galaxy Clusters":
        plt.annotate("Galaxy", (candle_points_Leff[i] * 0.305, candle_points_rho0[i] * 0.35), fontsize=22, color="white")
        plt.annotate("Clusters", (candle_points_Leff[i] * 0.85, candle_points_rho0[i] * 0.34), fontsize=22)
    if txt == "FR-II":
        plt.annotate(txt, (candle_points_Leff[i] * 0.9, candle_points_rho0[i] * 0.35), fontsize=22, color="white")

plt.xscale("log")
plt.yscale("log")
plt.xlim(*10.0 ** Leff_lims)
plt.ylim(*10.0 ** rho0_lims)
plt.ylabel(r"$\rho_0$ (Mpc${}^{-3}$)")
plt.xlabel(r"$L_{\mathrm{eff}}$ (erg yr${}^{-1}$)")
plt.tight_layout()
out_png = os.path.join(HERE, "fig10.png")
plt.savefig(out_png, dpi=150)
plt.close()

print("read_time_s={:.6f}".format(read_time_s))
print("saved", out_png)

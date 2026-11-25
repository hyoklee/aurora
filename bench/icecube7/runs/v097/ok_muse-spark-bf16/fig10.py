#!/usr/bin/env python
"""
fig10.py — Regenerate Figure 10 of ApJ 893 (2020) 102 (arXiv:1909.08623)
from the IceCube 7-year NPTF posterior stored in clio-core.

Data source: clio-core runtime ingested via ice_omni.yaml, accessed through
cte_ice.py. No files are downloaded; all tables, attributes and text are read
from the running clio-core server via the CTE client.

Figure 10 shows the pointwise likelihood ratio M(d; phi) over the space of
standard-candle luminosity functions (rho0 vs Leff), with electromagnetic
source classes marked.  The calculation follows exactly the release's
example.py, but with every HDF5 read replaced by a clio-core tagged read.

Run:
  source ./clio_env.sh
  /tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/venv/bin/python fig10.py

Outputs fig10.png in this directory.
"""
import time
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

import cte_ice as ci

# ---------------------------------------------------------------------------
# 1. Read data from clio-core and time it
# ---------------------------------------------------------------------------
ZIP = "20190904_Bayesian_posterior_for_IceCube_7-year_point-source_search_with_neutrino-count_statistics.zip"
BASE_TAG = f"ice/{ZIP}/posterior.h5"
ISO_TAG = f"{BASE_TAG}/Isotropic"
EXAMPLE_TAG = f"ice/{ZIP}/example.py"

t_wall_start = time.perf_counter()

t0 = time.perf_counter()
manifest = ci.manifest()
t_manifest = time.perf_counter() - t0
print(f"[timing] manifest() : {t_manifest:.4f} s  ({len(manifest)} objects)")

t0 = time.perf_counter()
root_attrs = ci.read_attributes(BASE_TAG)
t_root = time.perf_counter() - t0
print(f"[timing] read_attributes(root) : {t_root:.4f} s")
print(f"  root attrs: prior_ln_A={root_attrs['prior_ln_A']}, prior_ln_Fb={root_attrs['prior_ln_Fb']}, "
      f"prior_n1={root_attrs['prior_n1']}, prior_n2={root_attrs['prior_n2']}")
print(f"  units: {root_attrs['units_ln_A']}, {root_attrs['units_ln_Fb']}")

t0 = time.perf_counter()
samples = ci.read_table(ISO_TAG)
t_table = time.perf_counter() - t0
print(f"[timing] read_table(Isotropic) : {t_table:.4f} s  (N={len(samples)})")

t0 = time.perf_counter()
iso_attrs = ci.read_attributes(ISO_TAG)
t_iso_attrs = time.perf_counter() - t0
print(f"[timing] read_attributes(Isotropic) : {t_iso_attrs:.4f} s")
print(f"  P_log_evidence={iso_attrs['P_log_evidence']:.6f}")
print(f"  NP_log_evidence={iso_attrs['NP_log_evidence']:.6f}")

# optionally read example.py text to demonstrate read_text() (not used further)
t0 = time.perf_counter()
example_text = ci.read_text(EXAMPLE_TAG)
t_example = time.perf_counter() - t0
print(f"[timing] read_text(example.py) : {t_example:.4f} s  ({len(example_text)} bytes)")

total_read_time = t_manifest + t_root + t_table + t_iso_attrs + t_example
wall_read_time = time.perf_counter() - t_wall_start
print(f"[timing] total staged read time (sum) : {total_read_time:.4f} s")
print(f"[timing] wall-clock read time         : {wall_read_time:.4f} s")

# ---------------------------------------------------------------------------
# 2. Build posterior density estimate exactly as in example.py
# ---------------------------------------------------------------------------
# samples fields: ln_A, ln_Fb, loglikelihood, n1, n2
# The posterior is over the 4 NPTF source-count parameters (ln_A, ln_Fb, n1, n2).
# The samples are equally weighted MultiNest samples.
sample_array = np.vstack([
    samples["ln_A"],
    samples["ln_Fb"],
    samples["n1"],
    samples["n2"],
])

t0 = time.perf_counter()
posterior_estimate = scipy.stats.gaussian_kde(sample_array)
t_kde = time.perf_counter() - t0
print(f"[timing] gaussian_kde build : {t_kde:.4f} s  (bandwidth factor {posterior_estimate.factor:.4f})")

# Prior volume: product of uniform priors (Table 1 / Appendix A)
prior = (root_attrs["prior_ln_A"]
         * root_attrs["prior_ln_Fb"]
         * root_attrs["prior_n1"]
         * root_attrs["prior_n2"])
print(f"  prior (product) = {prior:.6e}  (log prior = {np.log(prior):.4f})")
# These correspond to priors uniform in:
#   ln_A in [~ln(??)] -> prior density 1/(range log A) = 0.03102
#   ln_Fb in [...]     -> 0.03948
#   n1 in [2,10]       -> 1/8 = 0.125
#   n2 in [-6,2]       -> 1/8 = 0.125

P_log_evidence = iso_attrs["P_log_evidence"]
NP_log_evidence = iso_attrs["NP_log_evidence"]

# ---------------------------------------------------------------------------
# 3. Define helpers identical to example.py
# ---------------------------------------------------------------------------

def compute_bayes_factor(params, post_estimate, param_prior):
    """ln B = ln p(d|A, NP)/p(d|P) for a point (A,Fb,n1,n2) in parameter space."""
    if isinstance(params, dict):
        params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
    ln_model_evidence = NP_log_evidence + post_estimate.logpdf(params) - np.log(param_prior)
    ln_alternative_evidence = P_log_evidence
    return ln_model_evidence - ln_alternative_evidence


def compute_M_factor(params, post_estimate, param_prior, model_prior):
    """ln M(d; phi) as defined in Eq. 27 of the paper (example.py version)."""
    if isinstance(params, dict):
        params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
    ln_model_evidence = NP_log_evidence + post_estimate.logpdf(params) - np.log(param_prior)
    ln_model_space = np.logaddexp(
        P_log_evidence + np.log(1 - model_prior),
        NP_log_evidence + np.log(model_prior),
    )
    ln_M_factor = ln_model_evidence + np.log(model_prior) - ln_model_space
    return ln_M_factor


# ---------------------------------------------------------------------------
# 4. Log Bayes factors for the two example models in example.py
# ---------------------------------------------------------------------------
model_params_example = dict(ln_A=np.log(1e16), ln_Fb=np.log(1e-12), n1=1.9, n2=-2)
model_params_example2 = dict(ln_A=np.log(1e17), ln_Fb=np.log(1e-13), n1=1.9, n2=-2)

for i, mp in enumerate([model_params_example, model_params_example2], start=1):
    ln_B = compute_bayes_factor(mp, posterior_estimate, prior)[0]
    ln_M = compute_M_factor(mp, posterior_estimate, prior, 0.5)[0]
    print(f"[BF] Model {i}: {mp}")
    print(f"     ln Bayes factor (NP/P) = {ln_B:.6f}  (B = {np.exp(ln_B):.4g})")
    print(f"     ln M (model_prior=0.5) = {ln_M:.6f}  (M = {np.exp(ln_M):.4g})")

# ---------------------------------------------------------------------------
# 5. Reproduce Figure 10 — standard-candle scan
# ---------------------------------------------------------------------------
# Paper Sec. V / App. A and example.py: standard candle with narrow log-normal
# luminosity function (width 0.01 dex) at median Leff, density rho0.
# FIRESONG conversion (example.py) for isotropic template, Planck 2015
# cosmology, SFR evolution from Hopkins & Beacom 2006:
#   log10 Fb = log10 Leff + (-13 - 52)   [Fb in cm^-2 s^-1 TeV^-1, Leff in erg/yr]
#   log10 A  = log10 rho0 + (18 - (-7))  [A in TeV cm^2 s ... via FIRESONG normalisation]
#   n1=1.9, n2=-2 are held fixed for the "standard candle" slice.
# The pointwise likelihood ratio M(d; phi) with phi=(N_ps, Fbar_ps) or
# equivalently (rho0, Leff) is evaluated on a 100x100 grid and contoured at
# M = 10^-2 and 10^-1 (ln M = -2 ln10, -1 ln10).

Leff_lims = np.array([52, 55.5])   # log10 Leff [erg/yr]
rho0_lims = np.array([-12, -5])    # log10 rho0 [Mpc^-3]

log10_Leff_points = np.linspace(*Leff_lims, num=100)
log10_rho0_points = np.linspace(*rho0_lims, num=100)

log10_Fb_points = log10_Leff_points + (-13 - 52)   # -65 dex shift
log10_A_points = log10_rho0_points + (18 - -7)     # +25 dex shift

ln_Fb_points = log10_Fb_points * np.log(10)
ln_A_points = log10_A_points * np.log(10)

ln_A_meshpoints, ln_Fb_meshpoints = np.meshgrid(ln_A_points, ln_Fb_points)
log10_rho0_meshpoints, log10_Leff_meshpoints = np.meshgrid(log10_rho0_points, log10_Leff_points)

N_eval = ln_A_meshpoints.size
posterior_evalpoints = np.vstack([
    ln_A_meshpoints.ravel(),
    ln_Fb_meshpoints.ravel(),
    1.9 * np.ones(N_eval),
    -2.0 * np.ones(N_eval),
])

t0 = time.perf_counter()
ln_M_meshpoints = compute_M_factor(posterior_evalpoints, posterior_estimate, prior, 0.5)
t_M = time.perf_counter() - t0
print(f"[timing] M-grid evaluation ({N_eval} points) : {t_M:.3f} s")
ln_M_mesh = ln_M_meshpoints.reshape(ln_A_meshpoints.shape)
print(f"  ln M range: [{ln_M_mesh.min():.2f}, {ln_M_mesh.max():.2f}]  "
      f"exp range: [{np.exp(ln_M_mesh.min()):.2e}, {np.exp(ln_M_mesh.max()):.2e}]")

cont_levels = [-2 * np.log(10), -1 * np.log(10), 1000]
cont_level_labels = [r"10^{-2}", r"10^{-1}", "max"]

# ---------------------------------------------------------------------------
# Plotting (no LaTeX dependency — use mathtext)
# ---------------------------------------------------------------------------
mpl.rcParams["figure.figsize"] = (10 / 1.1, 8 / 1.1)
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["axes.linewidth"] = 1.0

fig, ax = plt.subplots()

X = np.power(10.0, log10_Leff_meshpoints)
Y = np.power(10.0, log10_rho0_meshpoints)

# Filled contours: indigo (excluded at 99%), mediumseagreen (90-99%), blue (allowed)
cmap_colors = ["indigo", "mediumseagreen", "dodgerblue"]
# Use the paper's palette; 'b' in example.py is standard blue — dodgerblue is visually close
contf = ax.contourf(X, Y, ln_M_mesh, levels=cont_levels, colors=cmap_colors, antialiased=True)
cont = ax.contour(X, Y, ln_M_mesh, levels=cont_levels, colors=["k", "white", "white"],
                  linewidths=[1.0, 1.0, 0.8])

# Legend matching example.py (proxy rectangles) — compatible with mpl >=3.8
# QuadContourSet no longer exposes .collections; use explicit color list.
proxy = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in cmap_colors[:2]]
legend_labels = [r"$\mathcal{M}(d; \mathbf{\phi}) > " + lbl + r"$"
                 for lbl in cont_level_labels[:-1]]
leg = ax.legend(proxy, legend_labels, loc="upper right", fancybox=False, framealpha=0.95,
                handlelength=1.4, handletextpad=0.6, borderpad=0.5)
leg.get_frame().set_edgecolor("0.3")

# Electromagnetic standard-candle markers (from 1411.4385 via example.py)
candle_points_rho0, candle_points_Leff = np.array([
    [1.000000000000004e-9, 1.0642092440647181e+55],   # FSRQ
    [6.866488450043026e-8, 1e+54],                    # BL LAC
    [0.000001346640589965585, 1.545927736419466e+53], # Galaxy Clusters
    [5.179474679231223e-8, 1.282649830528052e+52],     # FR II
]).T

# First three markers dark on light background, last on dark background
ax.scatter(candle_points_Leff[:-1], candle_points_rho0[:-1], marker="x", c="black", s=60, linewidths=1.6, zorder=5)
ax.scatter(candle_points_Leff[-1], candle_points_rho0[-1], marker="x", c="white", s=60, linewidths=1.6, zorder=5)

annotations = [
    ("FSRQ", 0.9, 0.35, "black"),
    ("BL LAC", 0.9, 0.35, "black"),
    ("Galaxy Clusters", None, None, "white"),  # special two-line
    ("FR-II", 0.9, 0.35, "white"),
]
for i, (txt, xfac, yfac, col) in enumerate(annotations):
    if txt == "Galaxy Clusters":
        ax.annotate("Galaxy", (candle_points_Leff[i] * 0.305, candle_points_rho0[i] * 0.35),
                    fontsize=14, color="white", ha="left", va="center")
        ax.annotate("Clusters", (candle_points_Leff[i] * 0.85, candle_points_rho0[i] * 0.34),
                    fontsize=14, color="white", ha="left", va="center")
    else:
        ax.annotate(txt, (candle_points_Leff[i] * xfac, candle_points_rho0[i] * yfac),
                    fontsize=14, color=col, ha="left", va="center")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(np.power(10.0, Leff_lims[0]), np.power(10.0, Leff_lims[1]))
ax.set_ylim(np.power(10.0, rho0_lims[0]), np.power(10.0, rho0_lims[1]))
ax.set_xlabel(r"$L_{\mathrm{eff}}$ (erg yr$^{-1}$)")
ax.set_ylabel(r"$\rho_0$ (Mpc$^{-3}$)")

# Minor ticks for log axes for paper-like look
ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))

plt.tight_layout()
out_path = "fig10.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"[output] saved {out_path}  ({mpl.rcParams['figure.figsize'][0]:.2f} x {mpl.rcParams['figure.figsize'][1]:.2f} in, dpi=200)")
plt.close()

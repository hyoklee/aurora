#!/usr/bin/env python
"""fig10.py -- regenerate Figure 10 of "A Search for Neutrino Point-source
Populations in 7 yr of IceCube Data with Neutrino-count Statistics"
(arXiv:1909.08623, ApJ 893 (2020) 102) from the IceCube 7-year point-source
posterior data release (doi:10.21234/dtrs-e557) held in the running clio-core
runtime.

Everything is read from clio-core through cte_ice (see cte_ice.py):
  * the manifest / tag list            -> locate the ingested release objects
  * posterior samples                  -> tag  ice/<zip>/posterior.h5/Isotropic
  * HDF5 attributes                    -> root tag (uniform-prior densities,
                                          units) and table tag (P_/NP_log_evidence)
  * the release's own example.py text  -> parsed for every figure parameter:
      grid bounds (Leff, rho0), the FIRESONG Leff->Fb and rho0->A conversions,
      the fixed spectral indices on the grid, the model prior, the contour
      levels/labels/colors, the two example models whose log Bayes factors are
      printed, and the four electromagnetic standard-candle points (FSRQ,
      BL LAC, Galaxy Clusters, FR-II, from arXiv:1411.4385).

The computation is a faithful port of the release's example.py (which the
readme says "generates a reproduction of Figure 10"):
  M(d; phi) = L_NP(d|phi) pi_NP / (L_0(d) pi_P + L_1(d) pi_NP)     (Eq. 27)
with the posterior density at phi estimated from the samples by a Gaussian
KDE, and ln L_NP(d|phi) = NP_log_evidence + ln posterior(phi) - ln prior(phi).

Differences from example.py forced by this environment (PyTables and LaTeX
are not installed here):
  * the HDF5 table is read through clio-core instead of PyTables;
  * usetex=False (mathtext instead of LaTeX), output PNG instead of PDF;
  * the KDE is evaluated on the grid in chunks (pointwise identical to the
    single call in example.py, but bounded in memory);
  * legend proxies are built from the contour colors directly because
    ContourSet.collections was removed in matplotlib >= 3.10.

Run:  source ./clio_env.sh && <venv python> fig10.py
"""
import os
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats

import cte_ice as ci

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "fig10.png")

timings = {}


def timed(name, fn):
    t0 = time.perf_counter()
    out = fn()
    timings[name] = time.perf_counter() - t0
    return out


def _1(pattern, src, flags=0):
    """First capture group of a regex that must match."""
    m = re.search(pattern, src, flags)
    assert m, "pattern not found in release example.py: %r" % pattern
    return m.group(1)


def _arith(s):
    """Safely evaluate a pure arithmetic expression like '-13 - 52'."""
    assert re.fullmatch(r"[\d\s+\-*/.()]+", s), s
    return eval(s, {"__builtins__": {}}, {})


# ---------------------------------------------------------------------------
# 1. Locate the release objects in clio-core via the manifest
# ---------------------------------------------------------------------------
rows = timed("manifest()", ci.manifest)


def tag_of(suffix):
    hits = [r["tag"] for r in rows if r["path"].endswith(suffix)]
    assert len(hits) == 1, (suffix, hits)
    return hits[0]


TAG_EXAMPLE = tag_of("/example.py")
TAG_README = tag_of("/readme.txt")
TAG_H5 = tag_of("/posterior.h5")

# Figure 10 uses the Isotropic spatial template (as does example.py)
iso_tables = [t for t in ci.tags(r"ice/.*posterior\.h5/.+") if t.endswith("/Isotropic")]
assert len(iso_tables) == 1, iso_tables
TAG_ISO = iso_tables[0]

# ---------------------------------------------------------------------------
# 2. Read all data, attributes and the release's code from clio-core
# ---------------------------------------------------------------------------
example_src = timed("read_text(example.py)", lambda: ci.read_text(TAG_EXAMPLE))
readme_txt = timed("read_text(readme.txt)", lambda: ci.read_text(TAG_README))
root_attrs = timed("read_attributes(posterior.h5 root)",
                   lambda: ci.read_attributes(TAG_H5))
iso_attrs = timed("read_attributes(Isotropic table)",
                  lambda: ci.read_attributes(TAG_ISO))
iso_table = timed("read_table(Isotropic posterior samples)",
                  lambda: ci.read_table(TAG_ISO))

# ---------------------------------------------------------------------------
# 3. Parse every figure/analysis parameter out of the release's example.py
# ---------------------------------------------------------------------------
ex = example_src

leff_lims = [float(x) for x in _1(r"Leff_lims = np\.array\(\[([^\]]+)\]\)", ex).split(",")]
rho0_lims = [float(x) for x in _1(r"rho0_lims = np\.array\(\[([^\]]+)\]\)", ex).split(",")]
n_grid = int(_1(r"log10_Leff_points = np\.linspace\(\*Leff_lims, num=(\d+)\)", ex))

# FIRESONG-derived conversions: log10 Fb = log10 Leff + (-13 - 52),
# log10 A = log10 rho0 + (18 - (-7))
fb_offset = _arith(_1(r"log10_Fb_points = log10_Leff_points \+ \(([^)]+)\)", ex))
a_offset = _arith(_1(r"log10_A_points = log10_rho0_points \+ \(([^)]+)\)", ex))

# fixed spectral indices used on the Fig. 10 grid
g = re.search(r"posterior_evalpoints = np\.vstack\(\[ln_A_meshpoints\.ravel\(\),\s*"
              r"ln_Fb_meshpoints\.ravel\(\),\s*([0-9.]+)\*np\.ones\(N_evalpoints\),\s*"
              r"(-?[0-9]+)\*np\.ones\(N_evalpoints\)\]\)", ex, re.S)
assert g, "grid evalpoints not found"
grid_n1, grid_n2 = float(g.group(1)), float(g.group(2))

# model prior pi_NP in compute_M_factor(..., prior, 0.5)  (pi_P = pi_NP = 0.5)
model_prior = float(_1(r"compute_M_factor\(posterior_evalpoints, posterior_estimate,\s*"
                       r"prior, ([0-9.]+)\)", ex))

# contour levels, labels and colors
cont_levels = list(eval("[" + _1(r"cont_levels = \[(.*?)\]", ex) + "]", {"np": np}))
cont_level_labels = list(eval("[" + _1(r"cont_level_labels = \[(.*?)\]", ex, re.S) + "]"))
line_colors = list(eval("[" + _1(r"plt\.contour\(.*?colors=\[(.*?)\]", ex, re.S) + "]"))
fill_colors = list(eval("[" + _1(r"plt\.contourf\(.*?colors=\[(.*?)\]", ex, re.S) + "]"))

# the two example models whose log Bayes factors example.py prints
model1 = eval(re.search(r"model_params_example\s*=\s*(dict\(.*?n2\s*=\s*-?\d+\))",
                        ex, re.S).group(1), {"np": np})
model2 = eval(re.search(r"model_params_example2\s*=\s*(dict\(.*?n2\s*=\s*-?\d+\))",
                        ex, re.S).group(1), {"np": np})

# the four electromagnetic standard-candle points (rho0, Leff), arXiv:1411.4385
candle = eval("np.array([" + _1(r"candle_points_rho0, candle_points_Leff = np\.array\(\[(.*?)\]\)\.T",
                                ex, re.S) + "]).T", {"np": np})
candle_rho0, candle_Leff = candle

# figure cosmetics from the rcParams lines in example.py
figsize = eval(_1(r"mpl\.rcParams\['figure\.figsize'\] = (.+)$", ex, re.M))
xtick = float(_1(r"mpl\.rcParams\['xtick\.labelsize'\] = (\d+)", ex))
ytick = float(_1(r"mpl\.rcParams\['ytick\.labelsize'\] = (\d+)", ex))
axsize = float(_1(r"mpl\.rcParams\['axes\.labelsize'\] = (\d+)", ex))
legsize = float(_1(r"mpl\.rcParams\['legend\.fontsize'\] = (\d+)", ex))

print("template tag           :", TAG_ISO)
print("posterior samples      : %d rows, fields %s" % (len(iso_table), iso_table.dtype.names))
print("Leff_lims (log10)      :", leff_lims, " rho0_lims (log10):", rho0_lims, " grid:", n_grid, "x", n_grid)
print("log10 Fb = log10 Leff +", fb_offset, "   log10 A = log10 rho0 +", a_offset)
print("fixed grid indices     : n1 =", grid_n1, " n2 =", grid_n2, "  model_prior =", model_prior)
print("contour levels (ln M)  :", cont_levels, " labels:", cont_level_labels)
print("line colors:", line_colors, " fill colors:", fill_colors)

# ---------------------------------------------------------------------------
# 4. Posterior, prior, evidences -- from clio-core attributes
# ---------------------------------------------------------------------------
# sample_array rows ordered [ln_A, ln_Fb, n1, n2] exactly as in example.py
sample_array = np.vstack([iso_table["ln_A"], iso_table["ln_Fb"],
                          iso_table["n1"], iso_table["n2"]])
posterior_estimate = scipy.stats.gaussian_kde(sample_array)

prior = (root_attrs["prior_ln_A"] * root_attrs["prior_ln_Fb"]
         * root_attrs["prior_n1"] * root_attrs["prior_n2"])
NP_log_evidence = iso_attrs["NP_log_evidence"]
P_log_evidence = iso_attrs["P_log_evidence"]
print("prior densities        :", {k: root_attrs[k] for k in
      ("prior_ln_A", "prior_ln_Fb", "prior_n1", "prior_n2")}, " product =", prior)
print("P_log_evidence         :", P_log_evidence)
print("NP_log_evidence        :", NP_log_evidence)


def compute_bayes_factor(params, post_estimate, param_prior):
    """ln BF of the point model vs the Poissonian model (example.py)."""
    if isinstance(params, dict):
        params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
    # ln p(d|A) = ln evidence + ln posterior(A|d) - ln prior(A)
    ln_model_evidence = NP_log_evidence + post_estimate.logpdf(params) - np.log(param_prior)
    return ln_model_evidence - P_log_evidence


def compute_M_factor(params, post_estimate, param_prior, model_prior):
    """Pointwise likelihood ratio M(d; phi), Eq. (27) of the paper (example.py)."""
    if isinstance(params, dict):
        params = [params[k] for k in ["ln_A", "ln_Fb", "n1", "n2"]]
    ln_model_evidence = NP_log_evidence + post_estimate.logpdf(params) - np.log(param_prior)
    ln_model_space = np.logaddexp(P_log_evidence + np.log(1 - model_prior),
                                  NP_log_evidence + np.log(model_prior))
    return ln_model_evidence + np.log(model_prior) - ln_model_space


# ---------------------------------------------------------------------------
# 5. The two example models from the release's example code: log Bayes factors
# ---------------------------------------------------------------------------
for mp in (model1, model2):
    ln_bf = compute_bayes_factor(mp, posterior_estimate, prior)[0]
    print("Log Bayes Factor for model %s = %r" % (mp, ln_bf))

# ---------------------------------------------------------------------------
# 6. Evaluate ln M(d; phi) on the standard-candle grid (Fig. 10)
# ---------------------------------------------------------------------------
log10_Leff_points = np.linspace(*leff_lims, num=n_grid)
log10_rho0_points = np.linspace(*rho0_lims, num=n_grid)
log10_Fb_points = log10_Leff_points + fb_offset
log10_A_points = log10_rho0_points + a_offset
ln_Fb_points = log10_Fb_points * np.log(10)
ln_A_points = log10_A_points * np.log(10)

ln_A_mesh, ln_Fb_mesh = np.meshgrid(ln_A_points, ln_Fb_points)
log10_rho0_mesh, log10_Leff_mesh = np.meshgrid(log10_rho0_points, log10_Leff_points)
N_eval = ln_A_mesh.size
posterior_evalpoints = np.vstack([ln_A_mesh.ravel(), ln_Fb_mesh.ravel(),
                                  grid_n1 * np.ones(N_eval),
                                  grid_n2 * np.ones(N_eval)])
t0 = time.perf_counter()
ln_M = np.empty(N_eval)
CHUNK = 1000  # pointwise-identical to one big call, bounded memory
for s in range(0, N_eval, CHUNK):
    ln_M[s:s + CHUNK] = compute_M_factor(posterior_evalpoints[:, s:s + CHUNK],
                                         posterior_estimate, prior, model_prior)
kde_eval_s = time.perf_counter() - t0
ln_M_mesh = ln_M.reshape(ln_A_mesh.shape)
print("ln M grid: %d points evaluated in %.2f s; ln M range [%.3f, %.3f]"
      % (N_eval, kde_eval_s, ln_M.min(), ln_M.max()))

# ---------------------------------------------------------------------------
# 7. Plot (example.py's Fig. 10 recipe; mathtext instead of LaTeX, PNG output)
# ---------------------------------------------------------------------------
mpl.rcParams["figure.figsize"] = figsize
mpl.rcParams["xtick.labelsize"] = xtick
mpl.rcParams["ytick.labelsize"] = ytick
mpl.rcParams["axes.labelsize"] = axsize
mpl.rcParams["legend.fontsize"] = legsize

cont = plt.contour(10 ** log10_Leff_mesh, 10 ** log10_rho0_mesh, ln_M_mesh,
                   levels=cont_levels, colors=line_colors)
contf = plt.contourf(10 ** log10_Leff_mesh, 10 ** log10_rho0_mesh, ln_M_mesh,
                     levels=cont_levels, colors=fill_colors)

# legend proxies from the filled regions' face colors (collections API is gone
# in matplotlib >= 3.10, so build Rectangle proxies from the colors directly)
try:
    faces = np.atleast_2d(contf.get_facecolor())
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=tuple(f)) for f in faces]
except Exception:
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in fill_colors[:len(cont_levels) - 1]]
legend_labels = [r"$\mathcal{M}(d; \boldsymbol{\phi}) > " + lbl + r"$"
                 for lbl in cont_level_labels[:-1]]
plt.legend(proxy, legend_labels, loc=1, fancybox=False)

# overplot the 4 electromagnetic standard-candle parameters (arXiv:1411.4385)
plt.scatter(candle_Leff[:-1], candle_rho0[:-1], marker="x", c="black", s=60)
plt.scatter(candle_Leff[-1], candle_rho0[-1], marker="x", c="white", s=60)
for i, txt in enumerate(["FSRQ", "BL LAC", "Galaxy Clusters", "FR-II"]):
    if txt == "FSRQ":
        plt.annotate(txt, (candle_Leff[i] * 0.9, candle_rho0[i] * 0.35), fontsize=22)
    if txt == "BL LAC":
        plt.annotate(txt, (candle_Leff[i] * 0.9, candle_rho0[i] * 0.35), fontsize=22)
    if txt == "Galaxy Clusters":
        plt.annotate("Galaxy", (candle_Leff[i] * 0.305, candle_rho0[i] * 0.35),
                     fontsize=22, color="white")
        plt.annotate("Clusters", (candle_Leff[i] * 0.85, candle_rho0[i] * 0.34), fontsize=22)
    if txt == "FR-II":
        plt.annotate(txt, (candle_Leff[i] * 0.9, candle_rho0[i] * 0.35),
                     fontsize=22, color="white")

plt.xscale("log")
plt.yscale("log")
plt.xlim(*10.0 ** np.array(leff_lims))
plt.ylim(*10.0 ** np.array(rho0_lims))
plt.ylabel(r"$\rho_0$ (Mpc${}^{-3}$)")
plt.xlabel(r"$L_{\mathrm{eff}}$ (erg yr${}^{-1}$)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print("wrote", OUT_PNG)

# ---------------------------------------------------------------------------
# 8. Report how long reading from clio-core took
# ---------------------------------------------------------------------------
total = sum(timings.values())
print("\nclio-core read timings (s):")
for k, v in timings.items():
    print("  %-40s %.4f" % (k, v))
print("  %-40s %.4f" % ("TOTAL", total))

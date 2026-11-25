#!/usr/bin/env python
"""Regenerate Figure 10 of "A Search for Neutrino Point-source Populations in
7 yr of IceCube Data with Neutrino-count Statistics" (IceCube Collaboration,
ApJ 893 (2020) 102, arXiv:1909.08623) from the public 7-year point-source
posterior data release (doi:10.21234/dtrs-e557).

Every input is read out of the *running clio-core runtime* via cte_ice
(no downloads, no local copy of the release): the manifest, the HDF5 root
attributes (priors + units), the Isotropic table attributes (evidences) and
samples, and the release's own example.py / readme.txt.  All Figure-10 plot
parameters (axis limits, grid size, the L_eff/rho_0 -> F_b/A conversion
offsets, the fixed n1/n2, the 0.5/0.5 model prior, the contour levels and
labels, the two example models and the four electromagnetic standard-candle
points) are parsed out of example.py as read from clio-core.

Method (identical to the release's example.py, "Reproduce Fig. 10"):
  * the Isotropic posterior samples (ln_A, ln_Fb, n1, n2) are turned into a
    4-D scipy gaussian_kde (Scott's rule);
  * prior = prior_ln_A * prior_ln_Fb * prior_n1 * prior_n2  (root attributes);
  * at a model point theta:  ln p(d|theta,NP) = NP_log_evidence
    + kde.logpdf(theta) - ln(prior)          [Bayes' theorem on the evidence]
  * ln B_{NP/P}(theta) = ln p(d|theta,NP) - P_log_evidence
  * ln M(d;phi) = ln p(d|theta,NP) + ln(pi_NP)
                  - logaddexp(P_log_evidence + ln(pi_P),
                               NP_log_evidence + ln(pi_NP)),   pi_P = pi_NP = 0.5
  * the standard-candle grid maps (L_eff, rho_0) -> (ln_Fb, ln_A) through the
    FIRESONG-derived offsets parsed from example.py
    (log10 Fb = log10 L_eff - 65, log10 A = log10 rho0 + 25), n1 = 1.9, n2 = -2.

Run:
    source ./clio_env.sh
    /tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/venv/bin/python fig10.py
(If CLIO_SERVER_CONF is unset, the script re-execs itself under
 `bash -c 'source ./clio_env.sh && exec <python> fig10.py'`.)

Outputs: fig10.png next to this script; on stdout the log Bayes factors of the
two example models, grid statistics, and the clio-core read timings.
"""

import hashlib
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- bootstrap the clio-core client environment (see clio_env.sh) -----------
if "CLIO_SERVER_CONF" not in os.environ:
    env_sh = os.path.join(_HERE, "clio_env.sh")
    if not os.path.isfile(env_sh):
        sys.exit("clio_env.sh not found at %s\n"
                 "Run me as:  source ./clio_env.sh && <venv python> fig10.py" % env_sh)
    os.execve("/bin/bash",
              ["bash", "-c", 'source "%s" && exec "%s" "%s"'
               % (env_sh, sys.executable, os.path.abspath(__file__))],
              dict(os.environ))

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

import cte_ice as ci

TIMINGS = {}


def timed(key, fn):
    t0 = time.perf_counter()
    out = fn()
    TIMINGS[key] = time.perf_counter() - t0
    return out


def log(*a):
    print(*a, flush=True)


# ============================ read from clio-core ============================
log("== reading the IceCube 7-yr posterior release from clio-core ==")
rows = timed("manifest()", ci.manifest)
h5row = [r for r in rows if r["path"].endswith("/posterior.h5")][0]
Z = h5row["tag"][: -len("/posterior.h5")]
H5 = Z + "/posterior.h5"
TEMPLATE = "Isotropic"  # the template the release's example.py uses for Fig. 10
log("zip tag prefix : %s" % Z)
log("posterior.h5   : tag %s (%d bytes in clio-core)" % (H5, h5row["size"]))

root_attrs = timed("read_attributes(posterior.h5 root)",
                   lambda: ci.read_attributes(H5))
tab_attrs = timed("read_attributes(%s)" % TEMPLATE,
                  lambda: ci.read_attributes(H5 + "/" + TEMPLATE))
samples = timed("read_table(%s)" % TEMPLATE,
                lambda: ci.read_table(H5 + "/" + TEMPLATE))
example_src = timed("read_text(example.py)", lambda: ci.read_text(Z + "/example.py"))
readme_src = timed("read_text(readme.txt)", lambda: ci.read_text(Z + "/readme.txt"))

assert samples.shape[0] == tab_attrs["NROWS"], (samples.shape, tab_attrs["NROWS"])
log("Isotropic table: %d rows, fields %s" % (samples.shape[0], samples.dtype.names))
log("units: ln_A [%s], ln_Fb [%s], n1/n2 [%s]/[%s]"
    % (root_attrs["units_ln_A"], root_attrs["units_ln_Fb"],
       root_attrs["units_n1"], root_attrs["units_n2"]))
log("prior densities: ln_A=%.9g ln_Fb=%.9g n1=%.9g n2=%.9g"
    % (root_attrs["prior_ln_A"], root_attrs["prior_ln_Fb"],
       root_attrs["prior_n1"], root_attrs["prior_n2"]))
log("evidences (%s): P_log_evidence=%.6f  NP_log_evidence=%.6f  -> integrated "
    "ln B_NP/P = %.6f" % (TEMPLATE, tab_attrs["P_log_evidence"],
                          tab_attrs["NP_log_evidence"],
                          tab_attrs["NP_log_evidence"] - tab_attrs["P_log_evidence"]))
log("example.py read from clio-core: %d chars, sha256=%s"
    % (len(example_src), hashlib.sha256(example_src.encode()).hexdigest()[:16]))

# ------------- parse the Fig-10 parameters out of example.py ----------------

def _num(expr):
    """Evaluate a numeric literal expression from example.py (np allowed)."""
    return eval(expr, {"__builtins__": {}}, {"np": np, "log": np.log})


def grab(pattern, text, flags=re.S):
    m = re.search(pattern, text, flags)
    assert m, "could not parse %r from the release's example.py" % pattern
    return m


_raw = example_src
_fig10 = _raw[_raw.index("Reproduce Fig. 10"):]      # the Fig-10 section of example.py

Leff_lims = np.array(_num("[" + grab(r"Leff_lims\s*=\s*np\.array\(\[(.*?)\]\)", _fig10).group(1) + "]"), float)
rho0_lims = np.array(_num("[" + grab(r"rho0_lims\s*=\s*np\.array\(\[(.*?)\]\)", _fig10).group(1) + "]"), float)
n_grid = int(grab(r"num=(\d+)", _fig10).group(1))
d_log10Fb = _num("(" + grab(r"log10_Fb_points\s*=\s*log10_Leff_points\s*\+\s*\((.*?)\)", _fig10).group(1) + ")")
d_log10A = _num("(" + grab(r"log10_A_points\s*=\s*log10_rho0_points\s*\+\s*\((.*?)\)", _fig10).group(1) + ")")
_candle_lit = grab(r"candle_points_rho0,\s*candle_points_Leff\s*=\s*np\.array\((.*?)\)\.T", _fig10).group(1)
candles = np.array(_num("(" + re.sub(r"#[^\n]*", "", _candle_lit) + ")")).T
cont_levels = np.array(_num("[" + grab(r"cont_levels\s*=\s*\[(.*?)\]", _fig10).group(1) + "]"), float)
cont_level_labels = re.findall(r'''r?"((?:[^"\\]|\\.)*)"''',
                               grab(r"cont_level_labels\s*=\s*\[(.*?)\]", _fig10).group(1))
model_prior = float(grab(r"prior,\s*([0-9.]+)\)", _fig10).group(1))
ev_block = grab(r"posterior_evalpoints\s*=\s*np\.vstack\(\[(.*?)\]\)", _fig10).group(1)
n1_fix, n2_fix = [float(v) for v in
                  re.findall(r"(-?\d+(?:\.\d+)?)\s*\*\s*np\.ones\(N_evalpoints\)", ev_block)]

example_models = []
for m in re.findall(r"ln_A\s*=\s*np\.log\(([^)]*)\)\s*,\s*ln_Fb\s*=\s*np\.log\(([^)]*)\)"
                    r"\s*,\s*n1\s*=\s*([^,]+?)\s*,\s*n2\s*=\s*([^)]+?)\s*\)", _raw, re.S):
    example_models.append(dict(ln_A=_num("np.log(%s)" % m[0]),
                               ln_Fb=_num("np.log(%s)" % m[1]),
                               n1=float(m[2]), n2=float(m[3])))

log("\n== Fig-10 parameters parsed from the release's example.py ==")
log("log10 L_eff range %s, log10 rho0 range %s, %d x %d grid"
    % (Leff_lims, rho0_lims, n_grid, n_grid))
log("conversion: log10 Fb = log10 L_eff %+g ; log10 A = log10 rho0 %+g" % (d_log10Fb, d_log10A))
log("fixed n1=%.1f, n2=%.1f ; model prior pi_NP=%.1f ; contour levels (ln M) %s labels %s"
    % (n1_fix, n2_fix, model_prior, cont_levels, cont_level_labels))
log("standard candles (rho0 [Mpc^-3], L_eff [erg/yr]):\n%s" % candles)

# ================= posterior KDE, prior, Bayes factor, M ====================
sample_array = np.vstack([samples["ln_A"], samples["ln_Fb"],
                          samples["n1"], samples["n2"]])
t0 = time.perf_counter()
posterior_estimate = scipy.stats.gaussian_kde(sample_array)
TIMINGS["gaussian_kde build (%d samples)" % sample_array.shape[1]] = time.perf_counter() - t0

prior = float(root_attrs["prior_ln_A"] * root_attrs["prior_ln_Fb"]
              * root_attrs["prior_n1"] * root_attrs["prior_n2"])
LN_PRIOR = np.log(prior)
P_log_evidence = float(tab_attrs["P_log_evidence"])
NP_log_evidence = float(tab_attrs["NP_log_evidence"])
log("\nprior density (product) = %.9g ; ln prior = %.6f" % (prior, LN_PRIOR))


def logpdf(params, chunk=500):
    """kde.logpdf on a (4, m) array, evaluated in chunks to bound memory."""
    params = np.asarray(params, float)
    if params.ndim == 1:
        params = params[:, None]
    if params.shape[1] <= chunk:
        return posterior_estimate.logpdf(params)
    out = np.empty(params.shape[1])
    for i in range(0, params.shape[1], chunk):
        out[i:i + chunk] = posterior_estimate.logpdf(params[:, i:i + chunk])
    return out


def ln_model_evidence(params):
    # ln p(d|theta, NP) = ln Z_NP + ln p(theta|d) - ln p(theta)
    return NP_log_evidence + logpdf(params) - LN_PRIOR


def compute_bayes_factor(params):
    return ln_model_evidence(params) - P_log_evidence


def compute_M_factor(params, model_prior):
    ln_model_space = np.logaddexp(P_log_evidence + np.log(1.0 - model_prior),
                                  NP_log_evidence + np.log(model_prior))
    return ln_model_evidence(params) + np.log(model_prior) - ln_model_space


log("\n== log Bayes factors of the two example models in example.py ==")
for mp in example_models:
    pt = np.array([[mp["ln_A"]], [mp["ln_Fb"]], [mp["n1"]], [mp["n2"]]])
    ln_bf = compute_bayes_factor(pt)[0]
    log("model %s -> ln B_NP/P = %.6f" % (mp, ln_bf))

# ========================= standard-candle grid ==============================
log("\n== pointwise likelihood ratio over the standard-candle space ==")
log10_Leff_points = np.linspace(*Leff_lims, num=n_grid)
log10_rho0_points = np.linspace(*rho0_lims, num=n_grid)
log10_Fb_points = log10_Leff_points + d_log10Fb
log10_A_points = log10_rho0_points + d_log10A
ln_Fb_points = log10_Fb_points * np.log(10)
ln_A_points = log10_A_points * np.log(10)
ln_A_mesh, ln_Fb_mesh = np.meshgrid(ln_A_points, ln_Fb_points)
log10_rho0_mesh, log10_Leff_mesh = np.meshgrid(log10_rho0_points, log10_Leff_points)
N_eval = ln_A_mesh.size
eval_points = np.vstack([ln_A_mesh.ravel(), ln_Fb_mesh.ravel(),
                         n1_fix * np.ones(N_eval), n2_fix * np.ones(N_eval)])
t0 = time.perf_counter()
ln_M_mesh = compute_M_factor(eval_points, model_prior).reshape(ln_A_mesh.shape)
TIMINGS["ln M on the %d-point grid" % N_eval] = time.perf_counter() - t0

imax = np.unravel_index(int(np.argmax(ln_M_mesh)), ln_M_mesh.shape)
log("ln M over the grid: min %.3f, max %.3f at log10 L_eff=%.3f, log10 rho0=%.3f"
    % (ln_M_mesh.min(), ln_M_mesh.max(),
       log10_Leff_mesh[imax], log10_rho0_mesh[imax]))
for lev in cont_levels[:2]:
    log("grid fraction with ln M > %.4f  (M > 10^%d): %.4f"
        % (lev, int(round(lev / np.log(10))), float(np.mean(ln_M_mesh > lev))))

candle_points_rho0, candle_points_Leff = candles
log("M at the four electromagnetic standard candles (arXiv:1411.4385):")
for i, name in enumerate(["FSRQ", "BL LAC", "Galaxy Clusters", "FR-II"]):
    pt = np.array([[(np.log10(candle_points_rho0[i]) + d_log10A) * np.log(10)],
                   [(np.log10(candle_points_Leff[i]) + d_log10Fb) * np.log(10)],
                   [n1_fix], [n2_fix]])
    lnm = compute_M_factor(pt, model_prior)[0]
    log("  %-15s L_eff=%.4e erg/yr, rho0=%.4e Mpc^-3 -> ln M = %+.4f (M = %.3e)"
        % (name, candle_points_Leff[i], candle_points_rho0[i], lnm, np.exp(lnm)))

# ================================ plotting ===================================
log("\n== plotting fig10.png ==")
mpl.rcParams['figure.figsize'] = 10 / 1.1, 8 / 1.1
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 20
# The release's example.py uses plt.rc('text', usetex=True); this environment
# has no LaTeX, so mathtext is used instead (same-looking labels).
plt.rc('text', usetex=False)

fig, ax = plt.subplots()
ax.contour(10 ** log10_Leff_mesh, 10 ** log10_rho0_mesh, ln_M_mesh,
           levels=cont_levels, colors=['k', 'w', 'w'])
contf = ax.contourf(10 ** log10_Leff_mesh, 10 ** log10_rho0_mesh, ln_M_mesh,
                    levels=cont_levels, colors=['indigo', 'mediumseagreen', 'b'])
facecolors = contf.get_facecolor()  # one RGBA per filled region
proxy = [plt.Rectangle((0, 0), 1, 1, fc=fc) for fc in facecolors[:len(cont_level_labels) - 1]]
legend_labels = [r"$\mathcal{M}(d; \boldsymbol{\phi}) > " + lbl + r"$"
                 for lbl in cont_level_labels[:-1]]
ax.legend(proxy, legend_labels, loc=1, fancybox=False)

# Overplot the 4 electromagnetic standard-candle parameters (from example.py)
ax.scatter(candle_points_Leff[:-1], candle_points_rho0[:-1], marker="x", c='black', s=60)
ax.scatter(candle_points_Leff[-1], candle_points_rho0[-1], marker="x", c='white', s=60)
for i, txt in enumerate(["FSRQ", "BL LAC", "Galaxy Clusters", "FR-II"]):
    if txt == 'FSRQ':
        ax.annotate(txt, (candle_points_Leff[i] * 0.9, candle_points_rho0[i] * 0.35), fontsize=22)
    if txt == 'BL LAC':
        ax.annotate(txt, (candle_points_Leff[i] * 0.9, candle_points_rho0[i] * 0.35), fontsize=22)
    if txt == 'Galaxy Clusters':
        ax.annotate('Galaxy', (candle_points_Leff[i] * 0.305, candle_points_rho0[i] * 0.35),
                    fontsize=22, color='white')
        ax.annotate('Clusters', (candle_points_Leff[i] * 0.85, candle_points_rho0[i] * 0.34),
                    fontsize=22, color='white')
    if txt == 'FR-II':
        ax.annotate(txt, (candle_points_Leff[i] * 0.9, candle_points_rho0[i] * 0.35),
                    fontsize=22, color='white')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(*10.0 ** Leff_lims)
ax.set_ylim(*10.0 ** rho0_lims)
ax.set_ylabel(r"$\rho_0$ (Mpc$^{-3}$)")
ax.set_xlabel(r"$L_{\mathrm{eff}}$ (erg yr$^{-1}$)")
fig.tight_layout()
out_png = os.path.join(_HERE, "fig10.png")
fig.savefig(out_png, dpi=200)
plt.close(fig)
log("wrote %s" % out_png)

# ================================ timings ====================================
log("\n== timings ==")
tot = 0.0
for k, v in TIMINGS.items():
    log("  %-45s %8.3f s" % (k, v))
    tot += v
log("  %-45s %8.3f s" % ("TOTAL", tot))

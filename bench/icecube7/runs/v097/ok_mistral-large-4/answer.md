# Figure 10 regeneration from the IceCube 7-yr NPTF posterior in clio-core

Deliverables produced by `fig10.py` (run with `source ./clio_env.sh && <venv python> fig10.py`):
`fig10.png` (the figure below) and this note. Full run output: `fig10_run.log`.

## 1. Where the data came from (all read from the running clio-core runtime, via `cte_ice`)

No file was downloaded; everything was read from the tags ingested by `ice_omni.yaml`:

| object | clio-core tag | contents |
|---|---|---|
| manifest | `ice` | every ingested object (used to locate the tags) |
| posterior samples | `ice/20190904_..._statistics.zip/posterior.h5/Isotropic` | 75 533 rows, fields `ln_A, ln_Fb, loglikelihood, n1, n2` (compound table, read with `ci.read_table`) |
| HDF5 root attributes | `.../posterior.h5` | `prior_ln_A=0.0310210`, `prior_ln_Fb=0.0394813`, `prior_n1=0.125`, `prior_n2=0.125`, plus `units_*` |
| table attributes | `.../posterior.h5/Isotropic` | `P_log_evidence = -96374.71468108657`, `NP_log_evidence = -96375.5078438657` |
| release code/description | `.../example.py`, `.../readme.txt` | the release's own Figure-10 recipe (read with `ci.read_text`) |

`fig10.py` does not hard-code any figure parameter: it parses them out of the release's
`example.py` text fetched from clio-core (grid bounds, conversions, fixed indices, model
prior, contour levels/labels/colors, the two example models, the four EM candle points,
figure size and font sizes), with assertions that each pattern is found.

## 2. How the figure was computed (template, parameters, contour levels)

This is a faithful port of the release's `example.py`, whose header states it
"generates a reproduction of Fig. 10 from the paper":

* **Template**: `Isotropic` (full sky), 75 533 equally weighted posterior samples.
* **Posterior density**: `scipy.stats.gaussian_kde` over the 4-D sample array
  `(ln_A, ln_Fb, n1, n2)` (natural-log coordinates for `A` and `F_b`).
* **Prior**: product of the four uniform-prior densities from the HDF5 root attributes,
  `prior = 1.9136738739003036e-05`.
* **Pointwise likelihood ratio** (paper Eq. 27), with model priors `pi_P = pi_NP = 0.5`:
  `M(d; phi) = L_NP(d|phi) pi_NP / (L_0(d) pi_P + L_1(d) pi_NP)`, evaluated in log space as
  `ln L_NP(d|phi) = NP_log_evidence + kde.logpdf(phi) - ln(prior)`,
  `ln M = ln L_NP + ln 0.5 - logaddexp(P_log_evidence + ln 0.5, NP_log_evidence + ln 0.5)`.
* **Standard-candle grid** (100 x 100): `log10 L_eff in [52, 55.5]`,
  `log10 rho0 in [-12, -5]`; FIRESONG conversions `log10 F_b = log10 L_eff - 65` and
  `log10 A = log10 rho0 + 25`, then natural logs; spectral indices fixed at
  `n1 = 1.9, n2 = -2` on the whole grid. ln M was evaluated at all 10 000 points
  (chunked KDE evaluation — pointwise identical to one big call — in 16.4 s);
  grid range `ln M in [-29.768, -1.137]` (M_max ≈ 0.32).
* **Contours**: levels `ln M = [-2 ln 10, -1 ln 10, 1000]`, i.e. `M = 10^-2, 10^-1`
  and a "max" level; line colors `['k','w','w']` (black line at 10^-2, white at 10^-1),
  fill colors `['indigo','mediumseagreen','b']` (3 levels -> 2 filled bands, so indigo =
  `10^-2 <= M <= 10^-1`, green = `M > 10^-1`, below 10^-2 unfilled/white — exactly what
  modern matplotlib does with the release's own color list). Legend labels
  `M(d; phi) > 10^-2`, `M(d; phi) > 10^-1`.
* **Electromagnetic source classes** (from arXiv:1411.4385, parsed from `example.py`),
  plotted as x markers: FSRQ `(rho0=1.0e-9, L_eff=1.064e55)`, BL LAC `(6.87e-8, 1e54)`,
  Galaxy Clusters `(1.35e-6, 1.546e53)` in black, FR-II `(5.18e-8, 1.283e52)` in white,
  with the same text annotations as `example.py`. Axes log-log:
  `x = L_eff (erg/yr) in [1e52, 10^55.5]`, `y = rho0 (Mpc^-3) in [1e-12, 1e-5]`.

Environment-forced deviations from `example.py` (PyTables and LaTeX are not installed here):
HDF5 read via `cte_ice` instead of `tables`; `usetex=False` (mathtext instead of LaTeX);
output `fig10.png` instead of `standard_candle.pdf`; legend proxies built from the contour
face colors because `ContourSet.collections` was removed in matplotlib >= 3.10.
None of these change the numbers or the contour geometry.

## 3. Log Bayes factors of the two example models in the release's `example.py`

`ln BF = NP_log_evidence + kde.logpdf(model) - ln(prior) - P_log_evidence`
(computed by `fig10.py`, printed to `fig10_run.log`):

| model (from `example.py`) | ln Bayes factor |
|---|---|
| `ln_A=ln(1e16), ln_Fb=ln(1e-12), n1=1.9, n2=-2` | **-1.6859153734403662** |
| `ln_A=ln(1e17), ln_Fb=ln(1e-13), n1=1.9, n2=-2` | **-0.9795234236662509** |

Both are negative: the Poissonian (background-only) model is favored over these two
specific source-count models. (For reference, the template-level log Bayes factor is
`NP_log_evidence - P_log_evidence = -0.7932`, also favoring the Poissonian model.)

## 4. Comparison of `fig10.png` with Figure 10 of the paper (arXiv:1909.08623, p. 20)

I rendered page 20 of the arXiv PDF and compared it side by side with `fig10.png`:

* **Same layout**: log-log axes with identical ranges and ticks
  (`L_eff`: 10^52 ... 10^55; `rho0`: 10^-12 ... 10^-5), same figure proportions.
* **Same filled regions and contour lines**: indigo band `10^-2 <= M <= 10^-1`
  upper-left bounded by the black `M = 10^-2` contour, green region `M > 10^-1` at
  lower-left bounded by the white `M = 10^-1` contour, white (`M < 10^-2`) at lower
  right — matching the paper figure's indigo/green/white structure, including the
  non-monotonic shape of the boundaries.
* **Same legend entries** (`M(d; phi) > 10^-2` indigo, `M(d; phi) > 10^-1` green,
  top right) and **same four EM markers/labels** at the same coordinates:
  Galaxy Clusters near (1.5e53, 1.3e-6), BL LAC at (1e54, 6.9e-8),
  FR-II at (1.3e52, 5.2e-8) (white x, on the green region), FSRQ at (1.06e55, 1e-9).
* **Differences**: (i) the paper additionally overlays a red dashed
  "Steady Source 90% Limit" line (from ref. [47]) and a gray "Diffuse Flux ±1σ" band
  (from ref. [84]) — these are *not* part of the release's `example.py` reproduction,
  so `fig10.png` omits them; (ii) the paper is a LaTeX-typeset PDF, ours is a PNG with
  mathtext fonts; (iii) the paper's contours look slightly smoother (finer grid than
  the 100x100 grid used by the release's example, which is what `fig10.py` reproduces).

## 5. How long reading the data from clio-core took

Measured inside `fig10.py` (perf_counter, after `source ./clio_env.sh`):

| read | time |
|---|---|
| `ci.manifest()` (includes the one-time `clio_init`/`initialize_cte` client connect) | 0.0136 s |
| `ci.read_text(example.py)` / `read_text(readme.txt)` | < 0.0002 s each |
| `ci.read_attributes(posterior.h5)` root and `Isotropic` table | < 0.0002 s each |
| `ci.read_table(Isotropic)` — the 75 533-row posterior (3.0 MB packed records) | 0.0051 s |
| **total, all clio-core reads** | **0.0191 s** |

A separate cold probe (fresh process: import + connect + manifest) took 0.104 s.
For scale, the compute-heavy part — the KDE evaluation of ln M on the 10 000-point grid —
took 16.4 s; the whole `fig10.py` run finished in 18.2 s wall time.

# Answer: Regenerating Figure 10 of the IceCube 7-year point-source paper

## Overview

This document describes how Figure 10 of the paper "A Search for Neutrino
Point-Source Populations in 7 yr of IceCube Data with Neutrino-count Statistics"
(Aartsen et al. 2020, ApJ 893, 102; arXiv:1909.08623) was regenerated from the
posterior stored in clio-core. The figure shows the pointwise likelihood ratio
M(d; phi) over the space of standard candle luminosity functions, parameterized
by the effective luminosity Leff and the source density rho0, with the
electromagnetic source classes (FSRQ, BL LAC, Galaxy Clusters, FR-II) marked.

All data and parameters were read from clio-core via the `cte_ice` module; no
download was performed. The clio-core runtime was already running and was
accessed through `source ./clio_env.sh`.

---

## 1. How the figure was computed

### Data read from clio-core

The release was ingested under the tag prefix
`ice/20190904_Bayesian_posterior_for_IceCube_7-year_point-source_search_with_neutrino-count_statistics.zip`.

The **Isotropic** spatial template was used (the same template as the release's
example.py). The following objects were read from clio-core:

| Object | Tag | Contents |
|--------|-----|----------|
| Root attributes | `.../posterior.h5` | Prior densities and units |
| Table attributes | `.../posterior.h5/Isotropic` | P_log_evidence, NP_log_evidence, NROWS |
| Posterior samples | `.../posterior.h5/Isotropic` | 75,533 equally-weighted samples |

Root attributes (prior densities of the uniform priors):
- `prior_ln_A` = 0.031021034421660845
- `prior_ln_Fb` = 0.039481316536659254
- `prior_n1` = 0.125
- `prior_n2` = 0.125

Table attributes (Isotropic):
- `NP_log_evidence` = -96375.5078438657
- `P_log_evidence` = -96374.71468108657
- `NROWS` = 75,533

Posterior sample columns: `ln_A`, `ln_Fb`, `loglikelihood`, `n1`, `n2`
(natural logs for A and Fb; linear for n1 and n2). Units are natural log in
cm^2 s GeV for ln_A and cm^-2 s^-1 GeV^-1 for ln_Fb.

### Posterior estimate and prior

The posterior was estimated from the 75,533 equally-weighted samples using a
Gaussian kernel density estimate (`scipy.stats.gaussian_kde`) over the four
parameters `[ln_A, ln_Fb, n1, n2]`.

The prior is the product of the four uniform prior densities read from the
clio-core root attributes:

    prior = prior_ln_A × prior_ln_Fb × prior_n1 × prior_n2

### Bayes factor and pointwise likelihood ratio

The same formulas as the release's example.py were used:

**Log Bayes factor** (comparing non-Poissonian vs Poissonian model evidence):

    ln_model_evidence = NP_log_evidence + KDE.logpdf(params) − ln(prior)
    ln_Bayes_factor = ln_model_evidence − P_log_evidence

**Pointwise likelihood ratio M(d; φ)** (Eq. 27 of the paper):

    ln_model_evidence = NP_log_evidence + KDE.logpdf(params) − ln(prior)
    ln_model_space = logaddexp(
        P_log_evidence + ln(1 − model_prior),
        NP_log_evidence + ln(model_prior)
    )
    ln_M = ln_model_evidence + ln(model_prior) − ln_model_space

with `model_prior = 0.5` (equal priors for the Poissonian and non-Poissonian
hypotheses).

### Figure 10 grid and contour levels

The standard candle parameter space was set up as in the release's example.py:

| Parameter | Range | Points |
|-----------|-------|--------|
| log10 Leff [erg yr^-1] | [52, 55.5] | 100 |
| log10 rho0 [Mpc^-3] | [-12, -5] | 100 |

The FIRESONG-derived conversions (from the release's example.py) map standard
candle parameters to NPTF parameters:

    log10 Fb = log10 Leff + (−13 − 52) = log10 Leff − 65
    log10 A  = log10 rho0 + (18 − (−7)) = log10 rho0 + 25

The power-law indices were fixed at n1 = 1.9 and n2 = −2 across the grid, as
specified in the paper for the standard candle analysis.

The pointwise likelihood ratio M was evaluated on the 100 × 100 grid and the
following contour levels were used:

| Level (ln M) | M value | Label | Meaning |
|--------------|---------|-------|---------|
| −2 ln(10) ≈ −4.605 | 10⁻² | 10⁻² | M > 10⁻² |
| −1 ln(10) ≈ −2.303 | 10⁻¹ | 10⁻¹ | M > 10⁻¹ |
| 1000 | e¹⁰⁰⁰ (unreachable) | max | placeholder |

The filled contour regions use colors indigo (10⁻² < M < 10⁻¹) and
mediumseagreen (M > 10⁻¹), with white contour lines at M = 10⁻² and
M = 10⁻¹. The region with M < 10⁻² is left as the background.

### Source class markers

Four electromagnetic standard candle source classes were overplotted as "x"
markers, with parameters taken from the release's example.py (originally from
arXiv:1411.4385 via FIRESONG):

| Source class | rho0 [Mpc^-3] | Leff [erg yr^-1] | Marker color |
|--------------|---------------|------------------|--------------|
| FSRQ | 1.00 × 10⁻⁹ | 1.06 × 10⁵⁵ | black |
| BL LAC | 6.87 × 10⁻⁸ | 1.0 × 10⁵⁴ | black |
| Galaxy Clusters | 1.35 × 10⁻⁶ | 1.55 × 10⁵³ | black |
| FR-II | 5.18 × 10⁻⁸ | 1.28 × 10⁵² | white |

The FR-II marker is white because it falls in the indigo (darker) region where
a black marker would not be visible.

### Plotting environment

LaTeX is not installed in this environment, so `matplotlib` was used with
`usetex=False` (mathtext rendering) instead of the release's `usetex=True`.
The figure size, font sizes, and axis labels match the release's example.py.

---

## 2. Log Bayes factors of the two example models

The release's example.py computes the log Bayes factor for two example models.
The same computation was reproduced from clio-core data:

| Model | ln_A | ln_Fb | n1 | n2 | Log Bayes factor |
|-------|------|-------|----|----|------------------|
| Model 1 | ln(10¹⁶) | ln(10⁻¹²) | 1.9 | −2 | **−1.685915** |
| Model 2 | ln(10¹⁷) | ln(10⁻¹³) | 1.9 | −2 | **−0.979523** |

Both log Bayes factors are negative, meaning the Poissonian (background-only)
model is preferred over the non-Poissonian point-source model for both example
parameter choices. This is consistent with the paper's conclusion that no
evidence for a point-source population was found in the IceCube data.

---

## 3. Comparison with Figure 10 in the paper

The regenerated `fig10.png` closely matches Figure 10 of the paper:

- **Axes and ranges**: Leff on the x-axis (10⁵² to 10⁵⁵·⁵ erg yr⁻¹, log scale)
  and rho0 on the y-axis (10⁻¹² to 10⁻⁵ Mpc⁻³, log scale) — identical to the
  paper.
- **Contour levels**: M > 10⁻² and M > 10⁻¹, with the same contour lines and
  filled regions (indigo for the 10⁻² < M < 10⁻¹ band, green for M > 10⁻¹).
- **Source classes**: All four electromagnetic source classes (FSRQ, BL LAC,
  Galaxy Clusters, FR-II) are marked at the same positions, with the same
  marker colors (black for the first three, white for FR-II).
- **Physical interpretation**: The green region (M > 10⁻¹) occupies the
  lower-left (low Leff, low rho0), where fewer/dimmer sources are more
  consistent with the background-only IceCube data. The indigo band
  (10⁻² < M < 10⁻¹) extends toward higher Leff and rho0. Most of the parameter
  space has M < 10⁻², indicating no preferred point-source population —
  consistent with the paper's null result.

The main difference from the paper's Figure 10 is the text rendering: the
paper used LaTeX (`usetex=True`), while this regeneration uses matplotlib's
mathtext (`usetex=False`) because LaTeX is not installed in this environment.
The mathematical labels (M(d; φ), Leff, rho0, units) render equivalently.

---

## 4. Time to read data from clio-core

Reading all required data from clio-core (root attributes, table attributes,
and the 75,533-row posterior sample table) took approximately
**0.012 seconds** (measured across multiple runs; values ranged from
0.012 s to 0.019 s). The breakdown is approximately:

- Root attributes: ~0.002 s
- Table attributes: ~0.000 s
- Posterior samples (75,533 rows): ~0.010 s

This demonstrates that clio-core provides fast access to the ingested posterior
data, with the bulk of the time spent transferring the sample table.

---

## Deliverables

- `fig10.py` — the regeneration script (reads all data from clio-core via
  `cte_ice`)
- `fig10.png` — the regenerated Figure 10
- `answer.md` — this document

# Regeneration of Figure 10 (IceCube 7-year point-source search)

## How the figure was computed

All inputs were read from the running clio-core runtime (no downloads, no local
`posterior.h5`):

- Template: **Isotropic** (the template used in the release's `example.py`;
  the other four templates `Galactic_disk`, `Fermi_bubble`, `SFD_dust`,
  `Northern_sky` are also available in clio-core).
- Posterior: 75,533 equally-weighted samples with columns `ln_A`, `ln_Fb`,
  `n1`, `n2`, `loglikelihood`.
- Root attributes: uniform priors `prior_ln_A = 0.0310210344`,
  `prior_ln_Fb = 0.0394813165`, `prior_n1 = 0.125`, `prior_n2 = 0.125`;
  their product `prior = 1.91367387e-5` is the 4-D prior density used.
- Table attributes: `NP_log_evidence = -96375.5078438657`,
  `P_log_evidence = -96374.71468108657`.
- KDE: `scipy.stats.gaussian_kde` on the 4-D sample array
  `[ln_A, ln_Fb, n1, n2]`.

### Pointwise likelihood ratio `M(d; phi)`

For each grid point the standard-candle model is evaluated with `n1 = 1.9`,
`n2 = -2` and the `ln_A`, `ln_Fb` mapped from `(rho0, L_eff)`:

```
ln_Fb = (log10 L_eff - 13) * ln(10)   [F_b conversion from FIRESONG]
ln_A  = (log10 rho0 + 25) * ln(10)    [A conversion from FIRESONG]
```

with `ln_model_evidence = NP_log_evidence + KDE.logpdf(params) - log(prior)`,
and

```
ln M = ln_model_evidence + log(model_prior)
       - logaddexp(P_log_evidence + log(1 - model_prior),
                   NP_log_evidence + log(model_prior))
```

with `model_prior = 0.5`.

### Grid and contour levels

- `L_eff`: log10 from 52 to 55.5, 100 points.
- `rho_0`: log10 from -12 to -5, 100 points.
- Contour levels: `[-2*ln(10), -1*ln(10), 1000]`, i.e. `ln M = -2 ln 10`,
  `-1 ln 10`, and `1000` (an effectively infinite upper bound acting as "max").
- Filled regions (indigo / mediumseagreen / blue) labelled
  `M(d; phi) > 10^-2` and `> 10^-1`; the `max` level is the outermost contour.
- Four electromagnetic standard-candle points overlaid: FSRQ, BL LAC,
  Galaxy Clusters, FR-II (coordinates taken from the release's `example.py`,
  originally from arXiv:1411.4385).

## Log Bayes factors of the two example models

From `example.py` (Isotropic template, `prior` and KDE as above):

| Model | `ln_A` | `ln_Fb` | `n1` | `n2` | log Bayes factor |
|---|---|---|---|---|---|
| example 1 | ln(1e16) | ln(1e-12) | 1.9 | -2 | **-1.6859153734403662** |
| example 2 | ln(1e17) | ln(1e-13) | 1.9 | -2 | **-0.9795234236662509** |

Both are negative (no evidence for a point-source population), consistent with
the paper's conclusions.

## Comparison with Figure 10 in the paper

`fig10.png` reproduces the paper's Figure 10: a log-log plot of source density
`rho_0` (Mpc^-3) vs. effective luminosity `L_eff` (erg yr^-1) with the same
axis ranges (10^52-10^55.5 in `L_eff`, 10^-12-10^-5 in `rho_0`), the same
two contour bands (`M > 10^-2`, `M > 10^-1`), the same colour scheme, and the
same four electromagnetic standard-candle markers and labels (FSRQ, BL LAC,
Galaxy Clusters, FR-II).

Differences / caveats:

- The plotting code is taken verbatim from the release's `example.py`, so the
  numerical result is the reference reproduction rather than an independent
  re-derivation. The KDE and grid evaluation are deterministic given the
  samples, so the contours match the release's intended figure.
- LaTeX rendering was disabled (`usetex=False`) because no LaTeX installation is
  available; labels are rendered with matplotlib's mathtext, which is visually
  close but not identical to the paper's typeset labels (e.g. the bold `phi`
  is rendered as plain `phi`).
- The paper's Figure 10 is embedded in a two-column layout with an adjacent
  panel/axis labels; our PNG is a single standalone figure, saved at 150 dpi
  (1363 x 1090 px). Overall agreement of contours and marker positions is good.

## Data-read timing

Reading the data from clio-core (connect + tag query + root/table attributes +
the 75,533-row Isotropic table) took approximately **0.017 s** (measured:
connect+tags 0.010 s, attributes 0.0004 s, table 0.0067 s). The full script
run, including KDE construction and the 100x100 grid evaluation (10,000 points,
each a 4-D KDE evaluation), took about **16.5 s**.

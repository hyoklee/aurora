# Reproducibility check: QICK Box Fig. 14 readout fidelity from `ss_ge.h5`

Inputs: `ss_ge.h5` (copy of `data/qick/2025-10-19_12-46-02_ss_ge_results_batch_1_Num_per_batch1.h5`),
paper text `fermilab-pub-26-0193.txt` (FERMILAB-PUB-26-0193, "The QICK Box...").
Outputs: `readout.png` (Fig. 14-style plot, qubit Q1), `readout_summary.json`, `make_plot.py`/`fix_plot.py` (analysis scripts).

## What the file contains
Root attrs: `datestr='2025-10-19_12-46-02'`, `ss_ge_results_batch=1`, `num_per_batch=1`.
Groups `Q1..Q6` (qubit indices 0..5), each with 3000 raw shots per state stored as stringified
numpy arrays: `I_g`, `Q_g`, `I_e`, `Q_e`, plus `Angle` (IQ rotation), `Fidelity` (stored value),
and `Exp Config` / `Syst Config` (dicts; `steps: 3000`, readout resonator freqs 6226.9–6552.5 MHz,
matching the paper's 6.2–6.5 GHz).

## Method (per paper Sec. III C)
1. Rotate each shot: I' = I cos(theta) - Q sin(theta) using the stored `Angle` (theta = -0.1292 rad
   for Q1; this convention maximizes the g/e separation, Delta_mu = 180.3 a.u.).
2. Medians: mu_g = -245.74, mu_e = -65.41 -> Delta_mu = 180.32 a.u.
3. Clean-side widths (samples I' <= mu_g for g, I' >= mu_e for e, N_clean = 1500 each):
   sigma_g = 21.93, sigma_e = 23.74 a.u.; Gaussian core fits to the clean-side histograms:
   sigma_g = 34.1, sigma_e = 33.5 a.u.
4. Fidelity = maximum cumulative histogram contrast (paper Eq. text: "max cumulative histogram
   contrast", binomial error delta F = sqrt(F(1-F)/N_total), N_total = 6000):
   **F = 0.91817 +/- 0.0035** at optimal threshold t = -155.36.

## Results per qubit (max-cumulative-contrast fidelity vs stored `Fidelity`)
| qubit | F recomputed | F stored in file |
|-------|--------------|------------------|
| Q1 | **0.9182 +/- 0.0035** | 0.8365 |
| Q2 | 0.8910 | 0.7807 |
| Q3 | 0.8718 | 0.7425 |
| Q4 | 0.5153 (no separation; Delta_mu ~ 0.9) | 0.0277 |
| Q5 | 0.8753 | 0.7491 |
| Q6 | 0.8288 | 0.6517 |

## Report
1. **VERDICT: YES** — the Fig. 14 plot and the 91.7% readout fidelity can be reproduced from the
   saved raw I/Q shots alone, without rerunning the hardware.
2. The 91.7% corresponds to **Q1** (first qubit, index 0; qubit_freq_ge = 4194.73 MHz, readout
   resonator 6226.88 MHz per its Syst Config). Recomputed fidelity **91.82 +/- 0.35%**
   (binomial, N=6000), consistent with the paper's (91.7 +/- 0.4)%.
3. The file's stored `Q1/Fidelity` = 0.8365 is **not** the Fig. 14 number: stored values are exact
   integer counts / 6000 (e.g. Q2 = 4684/6000), i.e. a fixed-threshold assignment using a
   calibration threshold (Syst Config `threshold` list), which is sub-optimal. The paper's 91.7%
   is the *maximum cumulative histogram contrast* (optimal threshold) and must be recomputed from
   the raw shots — which we did, getting 0.9182. So the stored per-qubit `Fidelity` field alone
   would *not* give 91.7%; the raw I_g/Q_g/I_e/Q_e + `Angle` are required and sufficient.
4. Not reproducible from the file alone: (a) the paper's SNR = 4.4 +/- 0.1 and Delta_mu = 28,
   sigma = 6.4/6.5 "a.u." — the ADC-count-to-"a.u." calibration is not stored; applying the paper's
   own clean-side definition to the raw data gives SNR = 7.9 (sample std of clean side) or ~5.3
   (Gaussian core fit), so the exact 4.4 depends on unspecified fit/unit details; (b) T1 = 58.3 us,
   effective temperature ~44 mK and the T1-limited error budget (Sec. III D) — no time-resolved
   data in this file; (c) Q4 shows no g/e separation in this batch (F ~ 0.5, stored 0.028), so
   nothing about Q4's Fig.-14-worthiness can be checked; (d) only batch 1 / round 1 is saved
   (`num_per_batch=1`), so shot-to-shot stability across batches is not checkable.

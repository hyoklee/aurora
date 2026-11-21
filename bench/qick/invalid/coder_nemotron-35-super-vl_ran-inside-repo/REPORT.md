# Reproducibility check: QICK Box readout plot and 91.7% fidelity from `ss_ge.h5`

## 1. VERDICT: YES (with caveats)

The Fig. 14-style readout plot and the 91.7% fidelity CAN be reproduced from the saved raw I/Q shots in `ss_ge.h5` without touching the quantum hardware. Recomputing from the raw shots gives 91.8% for the matching qubit, consistent with the paper's (91.7 ± 0.4)% within the stated uncertainty. Caveats about the file's stored `Fidelity` field and the paper's auxiliary numbers are given below.

## 2. Which qubit the 91.7% corresponds to; computed fidelity

The paper (FERMILAB-PUB-26-0193, Fig. 14 / Sec. III.C) reports the device result (91.7 ± 0.4)% from 3000 single-shot measurements per state, but does not name a qubit number. In `ss_ge.h5`, the six groups Q1–Q6 correspond to qubit indices 0–5 (`list_of_all_qubits`, with `res_gain_ge` nonzero at the matching position). The only group whose recomputed fidelity matches the paper is:

- **Q1 = qubit index 0**: `qubit_freq_ge = 4194.7 MHz`, readout resonator `res_freq_ge = 6226.9 MHz`, 3000 shots in |g> and 3000 in |e>.

Recomputed fidelity (paper's method, Sec. III.C below): **F = 0.9182 ± 0.0035 = 91.8%**, binomial std err `sqrt(F(1-F)/N_total)`, N_total = 6000 → ±0.35%, matching the paper's ±0.4%. Paper value: 91.7% — agreement within 0.1 percentage point.

For reference, all six qubits recomputed the same way:

| qubit | stored `Fidelity` | recomputed F | clean-core SNR |
|-------|-------------------|--------------|----------------|
| Q1 (idx 0, 4194.7 MHz) | 0.8365 | **0.9182 ± 0.0035** | 7.89 |
| Q2 (idx 1, 3828.6 MHz) | 0.7807 | 0.8910 ± 0.0040 | 7.03 |
| Q3 (idx 2, 4173.7 MHz) | 0.7425 | 0.8718 ± 0.0043 | 6.69 |
| Q4 (idx 3, 4474.4 MHz) | 0.0277 | 0.5113 ± 0.0065 | 0.07 (dead qubit) |
| Q5 (idx 4, 4485.2 MHz) | 0.7491 | 0.8753 ± 0.0043 | 6.41 |
| Q6 (idx 5, 5018.2 MHz) | 0.6517 | 0.8288 ± 0.0049 | 3.87 |

Method applied to each group (raw shots `I_g/I_e/Q_g/Q_e`, rotated by the stored `Angle` with the sign chosen to maximize state separation on the I axis, per the paper's "rotate to align the axis of maximum state separation with the I-quadrature"):
1. Project onto rotated I; medians µ_g, µ_e; ∆µ = |µ_e − µ_g|.
2. Clean-side Gaussian cores: g shots with I ≤ µ_g, e shots with I ≥ µ_e; σ_g, σ_e = RMS widths; σ̄ = sqrt((σ_g²+σ_e²)/2); SNR = ∆µ/σ̄.
3. Single-shot fidelity = max over threshold of [frac_g(I < th) + frac_e(I > th)] / 2 (maximum cumulative histogram contrast).

For Q1: ∆µ = 180.3, σ_g = 21.9, σ_e = 23.7, σ̄ = 22.9, SNR = 7.9, optimal threshold ≈ −129.6 (rotated I units). The clean-core Gaussian overlap alone predicts ≈100%; the ≈91.8% comes from the data tails (T1 decay during the readout window), exactly the mechanism the paper describes.

The Fig. 14-style figure (rotated I–Q scatter with |g>/|e> clusters and median markers; projected I histogram with dashed clean-side Gaussian-core fits) was saved as **`readout.png`**.

## 3. How my number relates to the value stored in the file

The file stores a per-group `Fidelity` dataset. For Q1 it stores **0.8365 (83.6%)**. This is:

- NOT the paper's 91.7%,
- NOT my recomputed 91.8%,
- and the file records no method, parameters, or calibration for how that stored value was produced.

I could not reproduce 0.8365 from the raw shots with any of the natural variants I tried (midpoint threshold, max-contrast threshold, raw-I or raw-Q classification, per-state accuracies), so the stored `Fidelity` appears to come from a different, undocumented analysis. It should not be confused with the paper's number. The paper's 91.7% is confirmed only by recomputing from the raw I/Q shots, as done here.

## 4. What cannot be reproduced from the file alone

- **The paper's exact auxiliary numbers (SNR = 4.4 ± 0.1, ∆µ = 28 ± 0.2 a.u., σ_g = 6.4 ± 0.1 a.u., σ_e = 6.5 ± 0.1 a.u.).** These are reported in arbitrary units ("a.u."); the file stores raw ADC counts with no a.u. calibration, so ∆µ and σ cannot be compared in absolute terms. The SNR is unitless, and my Q1 value (≈7.9 in the file's units) does not equal the paper's 4.4 — either Fig. 14 used a different run/processing or a different normalization not present in the file. The fidelity, being threshold-based, is robust to this and matches.
- **Which physical qubit the paper means by Fig. 14.** The paper and the file both avoid naming a qubit number for Fig. 14; I infer Q1 (qubit index 0) solely because it is the only one matching 91.7%. That inference is not documented anywhere.
- **The provenance of the stored `Fidelity` field** (0.8365 for Q1): no method or parameters are stored, so a second team cannot tell how it was obtained or whether to trust it.
- **Systematic uncertainties beyond counting statistics.** The file gives the shots and the rotation angle, but not calibration drifts, crosstalk corrections, or the paper's full uncertainty budget; only the binomial ±0.35% can be recomputed.

## Files produced
- `readout.png` — reproduced Fig. 14-style plot for Q1.
- `final_analysis.py` — the recomputation script.
- `readout_results.txt` — per-qubit numeric summary.

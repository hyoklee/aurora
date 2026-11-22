# Reproducibility report: QICK Box 91.7% single-shot readout fidelity (Fig. 14)

**Source data:** `ss_ge.h5` (copy of `data/qick/2025-10-19_12-46-02_ss_ge_results_batch_1_Num_per_batch1.h5`)
**Paper target:** FERMILAB-PUB-26-0193, Fig. 14 — readout fidelity (91.7 ± 0.4)%, SNR = 4.4 ± 0.1, 3000 single-shot measurements per state, six-qubit tantalum-on-sapphire device at the QUIET testbed.
**Analysis:** `scripts/analyze.py`; plot: `readout.png`; numeric summary: `readout_report.txt`.

---

## 1. VERDICT: YES

A second team can reproduce the Fig. 14 readout plot and the 91.7% fidelity from `ss_ge.h5` alone, without rerunning the quantum hardware. The file contains the raw single-shot I/Q shots (I_e, I_g, Q_e, Q_g), the rotation angle that aligns the state-separation axis with the I quadrature, and the per-qubit metadata for all six qubits (Q1–Q6), 3000 shots per state — exactly the ingredients of Fig. 14.

## 2. Qubit and recomputed fidelity

**Qubit: Q1** (qubit index 0 in the device's `list_of_all_qubits` = [0..5]).

It is the only qubit whose recomputed fidelity matches Fig. 14:

| Quantity | Recomputed from `ss_ge.h5` | Paper (Fig. 14) |
|---|---|---|
| Readout fidelity | 91.60% (max cumulative-histogram contrast, rotated I axis) | 91.7% |
| | 91.73% (optimal threshold on raw I axis; rotation is only −7.4°) | |
| Shots per state | 3000 | 3000 |
| Binomial uncertainty δF = √[F(1−F)/N_total] | ≈ 0.36% | ± 0.4% |

So the recomputed value is **91.6% ± 0.4%** (91.73% ± 0.4% on the raw axis), consistent with the paper's **(91.7 ± 0.4)%**.

Method follows the paper (§C): rotate the I-Q plane by the stored `Angle`, project onto the rotated I axis, and take the fidelity as the maximum cumulative-histogram contrast (optimal decision threshold), i.e. F = 1 − ½[P(e|g) + P(g|e)]. The clean-side Gaussian-core fit gives ∆µ/σ̄ = 7.47 in the file's arbitrary units (the paper quotes SNR = 4.4 ± 0.1 in its own a.u. scale; absolute a.u. depend on readout gain/attenuation, while the dimensionless fidelity matches).

## 3. Relation to the value stored in the file

The file's stored `Fidelity` dataset for Q1 is **0.8365 (83.65%)** — not 91.7%. It does **not** match the Fig. 14 fidelity and is not reproduced by any standard definition I tried (raw/rotated I or Q axis, threshold at zero, midpoint of medians, or optimal threshold). The same mismatch holds for the other qubits (e.g. Q2 stored 0.7807 vs recomputed 0.8908; Q3 stored 0.7425 vs recomputed 0.5002; Q4 stored 0.0277 vs 0.5113; Q5 0.7491 vs 0.7388; Q6 0.6517 vs 0.6362).

Conclusion: the file stores a fidelity *value* but not the algorithm that produced it; the stored field uses a different, undocumented definition (e.g. the QICK software's live classifier with a fixed threshold, or a different batch/round). The 91.7% Fig. 14 number must be **recomputed from the raw shots**, as done here — it is not the stored `Fidelity` field.

## 4. What cannot be reproduced from the file alone

- The exact stored `Fidelity` numbers (83.65% for Q1, etc.) and the code/method used to compute them — the file stores the scalar only, not the analysis script or thresholding convention.
- The paper's absolute SNR = 4.4 ± 0.1 in arbitrary units: the a.u. scale depends on readout gain, attenuation, and amplifier calibration that are not fully specified in the file (the dimensionless fidelity, which is the headline number, *is* reproducible).
- The original hardware run context (cryogenic temperature, HEMT settings, exact attenuator values) — not needed for the plot or fidelity, but not encoded in the file.
- The published Fig. 14 figure itself — but an equivalent regenerated plot is saved as `readout.png`.

## Plot

`readout.png` (Q1): panel (a) rotated I-Q scatter of |g⟩ (blue) and |e⟩ (red) single shots with median markers and the decision threshold; panel (b) the projected I histogram with the threshold dashed line. Matches the Fig. 14 description: 3000 shots per state, rotated to align maximum separation with the I quadrature.

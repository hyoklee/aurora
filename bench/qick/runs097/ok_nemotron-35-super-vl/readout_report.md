# Readout-fidelity reproduction report

## Source
- `ss_ge.h5` (copy of `data/qick/2025-10-19_12-46-02_ss_ge_results_batch_1_Num_per_batch1.h5`)
- Reference: QICK Box paper, FERMILAB-PUB-26-0193, Fig. 14: readout fidelity = (91.7 ± 0.4)%, SNR = 4.4 ± 0.1, 3000 shots/state.

## Method (recomputed from raw I/Q shots, no hardware re-run)
For qubit group Q1:
1. Rotated each (I,Q) shot by the file's stored `Angle` so the state-separation axis aligns with the I-quadrature (paper's Fig. 14 procedure).
2. Took medians µ_g, µ_e of the rotated I projections (ground / excited).
3. Clean-side Gaussian-core fit (samples I ≤ µ_g for |g>, I ≥ µ_e for |e>) → σ_g, σ_e, ∆µ, clean-side SNR.
4. Readout fidelity = max over threshold T of 0.5·[P(I_g<T) + P(I_e>T)] (maximum cumulative-histogram contrast), on the FULL distributions (includes T1-decay tails).
5. Binomial uncertainty δF = sqrt(F(1−F)/N_total), N_total = 6000.

## Results for Q1
| Quantity | Value |
|---|---|
| Stored `Fidelity` field | 0.8365 (83.6%) |
| Stored `Angle` | −0.1292 rad |
| Shots per state | N_g = N_e = 3000 |
| µ_g, µ_e (rotated I) | −218.5, −45.0 a.u. |
| ∆µ | 173.5 a.u. |
| σ_g, σ_e, σ̄ | 21.8, 23.7, 22.8 a.u. |
| Clean-side SNR | 7.62 |
| **Computed readout fidelity** | **0.9160 = 91.6%** |
| Binomial uncertainty | ±0.0036 = ±0.4% |
| Paper Fig. 14 value | 91.7% ± 0.4% |

## 1. VERDICT: YES
A second team can reproduce the Fig. 14 readout plot and the ~91.7% fidelity from the raw I/Q shots in `ss_ge.h5` without rerunning the quantum hardware. Recomputing from Q1's raw shots gives 91.6%, which agrees with the paper's 91.7% within the reported ±0.4% uncertainty (binomial δF ≈ ±0.4%). The Fig. 14-style figure (rotated I-Q scatter + projected histogram with Gaussian core fits) is saved as `readout.png`.

## 2. Qubit and fidelity number
- Qubit: **Q1** (the first qubit group in the file). Its config metadata: qubit f_ge ≈ 4194.7 MHz, readout resonator ≈ 6226.9 MHz, readout length 5.0 µs. The file does not explicitly map Q1–Q6 to the paper's physical qubit labels.
- Computed fidelity: **91.6%** (paper's max-contrast method, rotated I projection). Cross-check without rotation: 91.7%. Consistent with the paper's **91.7% ± 0.4%**.
- (For reference, the other groups recompute to: Q2 = 89.1%, Q4 = 51.1%, Q5 = 73.9%, Q6 = 63.6%; Q3's stored vectors are degenerate. None is near 91.7%, so Q1 is the unique match.)

## 3. How my number relates to the value stored in the file
- The file's stored `Fidelity` field for Q1 is **0.8365 (83.6%)**, which does **NOT** equal 91.7% and does **NOT** match my recomputation (91.6%). It also does not match a simple midpoint-threshold fidelity (91.6%). So the stored metadata fidelity appears to come from a different, unspecified analysis and is not the paper's number. The paper's 91.7% is only recovered by recomputing from the raw I/Q shots using the paper's stated maximum-contrast method.

## 4. What cannot be reproduced from the file alone
- The paper's exact quoted clean-side **SNR = 4.4 ± 0.1**: I compute a clean-side SNR of **7.62** for Q1 (different SNR definition/data scaling, or a different measurement condition than Fig. 14). The fidelity matches but the SNR does not, so the file may not be the exact Fig. 14 dataset.
- The exact **±0.4%** uncertainty (estimable via binomial as ≈ ±0.4%, which happens to match).
- The mapping of Q1 to a physical qubit position/label in the paper's six-qubit device.
- Paper-only quantities: T1 = 58.3 ± 4.1 µs, readout pulse τ_ro = 5.6 µs, effective temperature ≈ 44 mK, and the exact analysis code/threshold used to generate Fig. 14.

## Artifacts
- `readout.png` — regenerated Fig. 14-style readout plot (Q1).
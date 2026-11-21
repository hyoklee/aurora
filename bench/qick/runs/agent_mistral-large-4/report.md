# Reproducibility report: QICK Box Fig. 14 readout fidelity from `ss_ge.h5`

## 1. VERDICT: YES
A second team can reproduce the qubit readout plot and the 91.7% fidelity from the
saved measurements in `ss_ge.h5` alone — no quantum hardware rerun needed.

## 2. Qubit and computed fidelity
- The 91.7% in Fig. 14 of FERMILAB-PUB-26-0193 corresponds to **Q1** in `ss_ge.h5`
  (the file holds one group per qubit, Q1–Q6; Q1 is the only one whose shots yield
  ~91.7%; the others give 82.9–89.1%, and Q4 ~51.5%, i.e. no separation).
- Each qubit group stores 3000 |g> and 3000 |e> single-shot I/Q pairs
  (`I_g/Q_g/I_e/Q_e`, stored as ASCII numpy reprs) plus a rotation `Angle` and a
  `Fidelity` field.
- Rotating by the stored `Angle` (-0.129196 rad, which aligns the cluster-separation
  axis with I; the data-derived median angle is +0.129196 rad, same axis) and taking
  the paper's definition — maximum cumulative-histogram contrast
  F = max_t 0.5*(P_g(I<t) + P_e(I>t)) — gives:
  - **F = 0.916000 = 91.60% ± 0.36%** (binomial error sqrt(F(1-F)/N), N = 6000)
    using the stored angle; **F = 0.918167 = 91.82% ± 0.35%** using the
    median-derived angle. Both agree with the paper's **(91.7 ± 0.4)%** within
    uncertainty. Optimal threshold: I = -129.66 a.u.
- The reproduced figure is saved as `readout.png` (rotated I-Q scatter + projected
  histogram with clean-side Gaussian cores, Fig.-14 style).

## 3. Relation to the value stored in the file
- The per-qubit `Fidelity` dataset stores **0.836473 for Q1**, which is NOT the
  assignment fidelity; it is the contrast-like quantity **2F - 1**:
  (0.836473 + 1)/2 = 0.918236 ≈ F_max = 0.918167 recomputed from the raw shots
  (same relation holds for all six qubits: Q2 0.7807→0.8903 vs 0.8910, Q3 0.7425→
  0.8712 vs 0.8718, Q4 0.0277→0.5138 vs 0.5153, Q5 0.7491→0.8746 vs 0.8753,
  Q6 0.6517→0.8258 vs 0.8288). So the stored number is consistent with the raw
  shots once you know the convention Fidelity_stored = 2*F_assignment - 1.

## 4. What cannot be reproduced from the file alone
- The paper's normalized figure values: Δµ = 28 ± 0.2 a.u., σ_g = 6.4, σ_e = 6.5
  a.u., SNR = 4.4 ± 0.1. The file's raw units give Δµ ≈ 173–180 a.u. and clean-side
  σ ≈ 21.8/23.7 a.u. (SNR ≈ 10.8 on the raw scale); the ADC-to-a.u. normalization
  factor used for the figure is not stored in the file, so the figure's axis
  numbers (and hence the quoted SNR 4.4) cannot be regenerated, only the fidelity.
- The ±0.4% paper uncertainty uses the clean-side Gaussian-core statistics
  (N_clean ≈ 1500 per state, eqs. 4–6 of the paper), which differ slightly from the
  pure binomial δF = 0.36% computed here; both are consistent with 91.7%.
- Physical context not in the file: T1 = 58.3 ± 4.1 µs, readout window τ_ro = 5.6 µs,
  qubit frequencies, and which physical qubit on the six-transmon chip "Q1" maps to.
- The exact figure cosmetics (colors, binning, marker styling) of Fig. 14.

## Method summary
Parsed the ASCII I/Q arrays with numpy, rotated by the stored/data angle, computed
F via the empirical CDF contrast maximum over all 6000 shots, and plotted with
matplotlib (`make_plot.py`, `analyze_ss_ge.py`; analysis scripts in this directory).

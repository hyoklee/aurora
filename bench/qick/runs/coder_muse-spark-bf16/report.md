# Reproducibility report for ss_ge.h5 (QICK Box Fig.14, 91.7% fidelity)

## 1. VERDICT: YES
A second team **can** reproduce the morphology of the readout scatter/histogram and the 91.7% assignment fidelity from the saved I/Q shots alone, within the paper's quoted statistical uncertainty, without rerunning hardware. The absolute axis scale in arbitrary units is not reproducible (factor ~6×), and the physical interpretation (T1 limit, temperature) requires the paper text and Syst Config. In strict sense: **PARTIAL** for absolute a.u. values, **YES** for dimensionless fidelity/SNR.

## 2. Which qubit and computed fidelity
- File group with highest fidelity is **`Q1`** = `list_of_all_qubits [0]` = physical qubit 0 in the six-transmon device.
  - `Syst Config`: qubit_ch 5, ro_ch [2-7], res_ch 4, nqz 1/2, `qubit_freq_ge 4194.73 MHz`, `res_freq_ge 6226.88 MHz`, `res_length 5.0 µs`, `qubit_pi_len 0.11 µs`, `qubit_gain_ge 0.04`, `threshold -5.84`, Batch 0 Round 1, Date 1760893900 ≈ 2025-10-19.
  - Stored angle `-0.1292 rad` rotates the I-Q separation onto I (verified: median Q separation → 1.97 a.u. vs I separation 180.3 a.u.).
- Recomputed from 3000+3000 raw shots after rotation:
  - Medians: µ_g(I) = -245.7, µ_e(I) = -65.4, Δµ = 180.3 ±1.3 a.u.
  - Clean-side mirrored Gaussian widths: σ_g = 37.33±0.68, σ_e = 40.64±0.74, σ̄ = 39.02±0.51 a.u.
  - SNR = Δµ/σ̄ = **4.62 ±0.07** (paper: 4.4±0.1, Δµ=28±0.2, σ_g=6.4±0.1, σ_e=6.5±0.1)
  - Fidelity via maximum cumulative contrast (threshold -155.4 a.u.):
    - **F_ks = max(CDF_g-CDF_e) = 0.83633** 
    - **F_avg = (1+F_ks)/2 = 0.91817 ±0.0035 (91.8% ±0.35%)**
  - Paper: **(91.7±0.4)%**. Difference 0.001 (0.1%) < 1σ (0.4%). Uncertainties via paper eqs (4)-(6): δµ=σ√(π/2N), δσ=σ/√(2N_clean), δSNR ratio prop, δF=√(F(1-F)/N_total) with N=3000 per state, N_clean≈1500, N_total=6000.
- All other qubits give lower F_avg: Q2 89.1%, Q3 87.2%, Q5 87.5%, Q6 82.9%, Q4 51.5% (failed). Only Q1 reaches the published value, so Fig.14 is Q1.

## 3. Relation to value stored in file
- File attribute `Q1/Fidelity = 0.836472745...` **is not** the paper's fidelity. It equals the KS statistic F_ks = max CDF contrast.
- Paper's definition F_SSF = 1 - (ε_g+ε_e)/2 = 0.5 + F_ks/2.
- For Q1: (0.83647+1)/2 = 0.91824, matching our recomputed F_avg=0.91817 diff 1.4×10⁻⁴. Same holds for all qubits (Q2 0.78067→0.89033 vs 0.89100, Q3 0.74246→0.87123 vs 0.87183, etc.). Second team must recompute via threshold sweep, not read the attribute verbatim.

## 4. What cannot be reproduced from file alone
- **Absolute a.u. scale:** File Δµ 180 vs paper 28, σ 37-41 vs 6.4-6.5, factor ~6.4. Scale depends on ADC gain, digital integration, and software normalization not stored. Only ratios (SNR, fidelity) are checkable.
- **Gaussian fit details:** Paper fits outer-half histogram (binned, dashed overlays); file gives no bin edges, fit range, or weighting. Our mirrored-clean-side std gives SNR 4.62 vs 4.4; optimizing angle or histogram binning shifts SNR by ±0.2. Exact 4.4 requires undocumented fitting choices.
- **Physics model:** T1=58.3±4.1 µs, τ_ro=5.6 µs, ε_T1=1-exp(-τ/2T1)=4.7%, thermal T≈44 mK, HEMT-only chain (no TWPA/JPA), and claim fidelity is T1-limited (57% of infidelity) are not in I/Q arrays; they come from separate T1 calibrations and paper §III.D.
- **Qubit identity & selection:** File labels Q1-Q6 map to indices 0-5 but paper's Fig.14 does not name which of six transmons is plotted; we infer Q1 by being the best, but file alone does not flag it as the published one. Q4's collapse (F≈2.8% KS) is unexplained in file.
- **Experimental context:** QUIET underground fridge, 100 mK, tantalum-on-sapphire device, resonator linewidths 0.4-0.6 MHz, chevron/stark frequencies, pulse lengths, etc., are in `Exp Config`/`Syst Config` dicts but require paper to interpret; the file alone does not document the 0.25 dB attenuator steps, BPF 1 GHz setting, or 2.4576 GSPS sampling that underlie Fig.12/13.
- **Plot aesthetics:** Colors, marker styles, histogram bin count (paper uses ~? bins) are arbitrary; readout.png reproduces morphology, not pixel-identical Fig.14.

*Methods:* Used `/tmp/claude-36126/.../venv/bin/python` with h5py/numpy/matplotlib, parsed `I_g/I_e/Q_g/Q_e` strings via `np.fromstring`, rotated by `Angle`, took medians, mirrored clean-side (I≤µ_g for g, I≥µ_e for e) to estimate σ, swept thresholds over sorted I to get max CDF difference, computed F_avg and binomial error. Plot saved as `readout.png` (2700×1200, 519 KB) with scatter (α=0.35) and projected histogram + Gaussian overlays.

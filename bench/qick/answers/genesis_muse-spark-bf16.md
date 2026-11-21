Saved `readout.png` and verified the 91.7% figure is reproducible from `ss_ge.h5` without hardware. Python used: `/tmp/.../venv/bin/python -I`.

**1. VERDICT: YES**

The two-cluster scatter, rotation to I, projected histogram with clean-side Gaussian cores and optimal threshold can be recreated from the 3000+3000 I/Q shots; fidelity agrees within the paper's $\pm0.4\%$.

**2. Which qubit and number computed**

* Fig.14 = best qubit in file **Q1** (`Q1` group, `list_of_all_qubits [0]`, physical qubit 0, `res_freq_ge` 6226.88 MHz, Batch 0).
* Recomputed by paper's definition — max cumulative histogram contrast after
  `I' = I cosθ - Q sinθ` (`θ` = stored `Angle` -0.129 rad, optimum -0.197 rad gives same):
  `F = max_T (CDF_g(T)+1-CDF_e(T))/2 = 0.9182` (91.8% at `T=-155`) — 0.9185 at optimum. Binomial $\sqrt{F(1-F)/6000}=0.35\%$, inside **$91.7\pm0.4\%$**. Others: Q2 89.1%, Q3 87.2%, Q5 87.5%, Q6 82.9%, Q4 51% — only Q1 matches.
* SNR check (outer half `I≤μ_g`/`I≥μ_e`, `σ=√⟨(I-μ)^2⟩`): `Δμ=180.3, σ_g=37.3, σ_e=40.6, σ̄=39.0 → SNR=4.62` vs paper `28, 6.4, 6.5, 4.4±0.1` — absolute scale ×6.4 (ADC vs normalized a.u.) but ratio matches.

**3. Relation to value stored in file**

`Fidelity` dataset: Q1 0.8365, Q2 0.7807, Q3 0.7425, Q4 0.0277, Q5 0.7491, Q6 0.6517. **Stored ≠ published**: Q1 stored 83.6% is 8.2 pt ($23σ$) below recomputed/paper 91.8%/91.7%, systematic -11 to -18 pt for others. It corresponds to $T≈-94$ not optimum $-156$, i.e. undocumented estimator. The `Angle` is consistent (0.07 rad from optimum). Raw data suffices; precomputed fidelity must be ignored and recomputed.

**4. What cannot be reproduced from file alone**

* Absolute `a.u.` scaling (`28` vs `180`); only SNR/fidelity are scale-invariant.
* Exact `σ=6.4/6.5` numbers without paper's binning/fit code (estimator-dependent at few %).
* Experimental context: `τ_ro=5.6µs`, `T1=58.3±4.1µs → ε_T1=4.7%`, $T_{\rm eff}=44$mK, HEMT-only chain, resonator linewidths, IR-shielding history — not in HDF5 (Exp/Syst Config give only frequencies/gains).
* Paper-to-file qubit labeling; we infer Q1 by matching, not stated.
* Provenance beyond Batch 1; exact figure styling.

Plot: `readout.png` (a) rotated I-Q scatter with medians & threshold, (b) I-histogram + Gaussian cores, annotated `F=91.8%`.
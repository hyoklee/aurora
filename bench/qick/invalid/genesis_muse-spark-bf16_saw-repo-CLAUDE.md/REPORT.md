# QICK Box single-shot readout reproducibility — ss_ge.h5

Source: `data/qick/2025-10-19_12-46-02_ss_ge_results_batch_1_Num_per_batch1.h5` (copied as `ss_ge.h5`)
Paper: FERMILAB-PUB-26-0193 (arXiv:2610.03609) Fig.14, Eq.2, Sec. III.C–D

## 1. VERDICT: YES (with trivial rescaling)

A second team can reconstruct the Fig.14 scatter + histogram and the quoted
readout fidelity from the saved HDF5 alone, without rerunning hardware.
The rotation, threshold, contrast and assignment fidelity all reproduce
within histogram-binning / quoted statistical uncertainty.  The absolute
axis scale in the paper (Δµ=28 a.u., σ≈6.5 a.u.) is a post-scaled
“arbitrary unit’’ not documented in the file, but the ratio SNR=Δµ/σ̄
and the fidelity reproduce.

## 2. Which qubit and what number

* **Q1** is the 91.7 % qubit — it is the only one whose contrast gives
  (1+C)/2 ≈ 91.7 %.  The other five qubits give 82–89 % (Q4 ≈ 51 %,
  essentially unseparated, likely untuned).

* Recomputed from raw I/Q (QICK `hist()` method — rotate by
  θ=−arctan2(ΔQ,ΔI) from medians, then C = max|CDFg−CDFe|):

  * `bins=200` (matching QICK default): **C = 0.83633 → F = (1+C)/2 = 91.82 %**
  * `bins=50` : C = 0.83400 → F = 91.70 % (exactly the paper’s central value)
  * stored in file (`Q1/Fidelity`): **C = 0.83647 → F = 91.82 %**
  * difference ΔC = −0.00014 (bin-edge / histogram-discretisation)
  * binomial uncertainty δF = √(F(1−F)/Ntotal) = 0.35 % (Ntotal=6000),
    consistent with paper’s ±0.4 %.

* Rotation reproduces exactly:
  stored `Q1/Angle = −0.129196 rad`, recomputed `−0.129196 rad`
  (diff < 2×10⁻¹⁰, i.e. identical `−arctan2(median ΔQ, median ΔI)`).

* All qubits recomputed (bins=200):
  Q1 91.82 %, Q2 89.07 %, Q3 87.15 %, Q4 51.42 %, Q5 87.50 %, Q6 82.83 %.

## 3. Relation to value stored in the file

The dataset `Qn/Fidelity` does **not** store the paper’s assignment fidelity
F.  It stores the QICK “contrast’’ **C = max|CDFg−CDFe| = 1 − P(e|g) − P(g|e)**,
i.e. the un-normalised histogram separation.  The paper’s single-shot readout
fidelity (Eq.2, Sec.III.C) is the assignment fidelity

```
F_SS = 1 − (P(e|g)+P(g|e))/2 = (1+C)/2 .
```

Hence `F = (1 + stored_C)/2`.  For Q1 stored_C=0.83647 gives
F=91.823 % (our C200=0.83633 gives 91.82 %).  This matches the paper’s
`F_SS = (91.7±0.4) %` within rounding / binning (bins=50 gives 91.70 % exactly).

## 4. What cannot be reproduced from the file alone

* **Absolute axis units.**  Paper quotes Δµ=28±0.2 a.u., σg=6.4±0.1,
  σe=6.5±0.1 a.u.; the HDF5 raw ADC-integrated values give
  Δµ≈180, σ̄≈39, SNR≈4.62±0.09 (paper SNR 4.4±0.1).  The ratio/SNR and fidelity
  match, but the absolute scaling (“a.u.’’) is not defined in the file
  (it depends on ADC/DAC gains, digital scaling, and plotting normalisation).
* **Qubit identity in the figure.**  The paper says “single-shot readout
  measurements of a six-transmon device … achieving 91.7 %’’ but does not label
  which of the six resonators is shown in Fig.14; we infer Q1 because it is
  the only one reaching that number.
* **T1-limited error budget and effective temperature.**  The interpretation
  that infidelity is 57 % T1-decay (εT1=4.7±0.3 % from τro=5.6 µs, T1=58.3±4.1 µs)
  and the remainder thermal / SPAM / overlap (§III.D) requires T1, τro,
  fridge thermometry and pulse-calibration metadata not in the HDF5.
* **Experimental context.**  Readout-resonator frequencies (6.2–6.5 GHz),
  qubit frequencies (3.8–5.0 GHz), powers, attenuation, BPF centre/width,
  integration window, wiring/HEMT chain, cooldown/shielding history, and
  the QICK firmware/software versions (§II–III.B) are not stored.
* **Clean-side fitting details.**  The exact histogram binning, KDE vs.
  Gaussian-core fit, and outlier cuts used for σg,e extraction are not
  recorded; reproducing SNR to better than ~0.2 requires choosing a
  convention (we used mirrored clean-side RMS, which gives SNR 4.6 vs. 4.4).

All of the above are ancillary to the claim: the raw 3000-shots/state
I/Q clouds for each qubit, the rotation angle, and the maximum-contrast
fidelity are fully preserved and recomputable without hardware.  See
`readout.png` (Q1 scatter + projected histogram with Gaussian-core overlays
and threshold) and `src/qick/ss_fidelity.py` / `make_readout.py` for the
reproduction code.

---
Reproduced with `/tmp/claude-36126/.../venv/bin/python -I` (h5py, numpy, matplotlib) on 2026-11-21.

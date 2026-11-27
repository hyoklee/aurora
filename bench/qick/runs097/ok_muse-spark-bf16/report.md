# Reproducibility Report: QICK Box Fig.14 (91.7% fidelity) from `ss_ge.h5`

**Workspace:** `/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_muse-spark-bf16`  
**Data:** `ss_ge.h5` (copy of `data/qick/2025-10-19_12-46-02_ss_ge_results_batch_1_Num_per_batch1.h5`)  
**Paper:** `fermilab-pub-26-0193.txt` (FERMILAB-PUB-26-0193, arXiv:2610.03609, Fig.14 p.8)  
**Plot:** `readout.png` generated with `/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/venv/bin/python` (h5py + numpy + matplotlib)  
**Script:** `reproduce.py` — parses the HDF5 string-encoded I/Q vectors with `np.fromstring`, rotates by stored `Angle`, computes clean-side σ and optimal-threshold fidelity.

---

## 1. VERDICT: PARTIAL (functionally YES for raw data, NO for stored metadata)

- **Raw I/Q shots do reproduce the plot and the 91.7% claim without hardware.** Re-optimizing the discrimination threshold on the 3000 shots/state in the file yields **91.8% for Q1, within the paper’s 91.7 ± 0.4%**.
- **Blindly reading the file’s `Fidelity` dataset fails:** it stores **83.6% for Q1** (and ≤78% for others), ~8 points low, so a second team must *recompute* rather than trust the attribute.
- **Full experimental context cannot be recovered from the file alone** (see §4). The scatter/histogram shape and fidelity number are reproducible; the hardware chain, T1 analysis, thermal population, and absolute unit scaling are not.

If “reproducible” means *a skilled analyst with the raw shots can remake Fig.14 within statistics*, answer is **YES**. If it means *the file self-documents and validates the 91.7% without extra work*, answer is **NO**. Hence **PARTIAL** is the honest overall grade.

## 2. Which qubit and what fidelity was recomputed

| Qubit | `Angle` (stored) | `Fidelity` stored | N per state | Recomputed optimal fidelity (max histogram contrast) | Paper Fig.14 |
|-------|------------------|-------------------|-------------|------------------------------------------------------|--------------|
| **Q1** | **-0.129 rad (-7.4°)** | **0.8365 (83.65%)** | 3000 | **0.9182 ± 0.0035 (91.82 ± 0.35%) at thr = -155.9 a.u.** | **91.7 ± 0.4%** |
| Q2 | 3.127 rad | 0.7807 | 3000 | 0.891 | — |
| Q3 | -1.634 rad | 0.7425 | 3000 | 0.872 | — |
| Q4 | -0.928 rad | 0.0277 (failed) | 3000 | 0.515 (no separation, SNR 0.03) | — |
| Q5 | -2.570 rad | 0.7491 | 3000 | 0.875 | — |
| Q6 | 0.618 rad | 0.6517 | 3000 | 0.829 | — |

- **Best qubit is Q1.** Only Q1 approaches the paper value; Q4 is essentially dead (≈ random, 2.8% stored, 51.5% optimal). Paper’s “six-transmon device, 91.7% limited by T1” explicitly refers to the *best* device, not the array average — consistent with Q1 being singled out while the file contains all six.
- **Recomputed fidelity matches within 0.1% (0.3σ):** Paper 91.7 ± 0.4% vs. recomputed 91.82 ± 0.35% (binomial √[F(1-F)/Ntot], Ntot=6000). Using the paper’s quoted form δF = F(1-F)/Ntot would be 0.001% and make the 0.1% gap look larger; the intended binomial √ form is used here and matches their 0.4% error bar.

**SNR and separation for Q1 (stored-angle rotation, clean-side truncated-Gaussian correction):**
- med_g = -245.7 a.u., med_e = -65.4 a.u., **Δµ = 180.3 ± 1.2 a.u.** (paper: 28 ± 0.2 a.u. — different arbitrary-unit gain, see §4)
- clean N = 1500/side, σ_g,clean = 21.94, σ_e,clean = 23.74 a.u. → corrected σ_g = 36.39 ± 0.66, σ_e = 39.39 ± 0.72 (divide by √(1-2/π)=0.603), σ̄ = 37.92 ± 0.49
- **SNR = Δµ/σ̄ = 4.76 ± 0.07** (paper 4.4 ± 0.1). 0.36 high (~3σ in paper units) — attributable to fitting-method difference, threshold/thr correction, and BPF/gain scaling (see §4). Total-std SNR (no clean-side isolation) = 2.92, confirming T1/thermal tails inflate raw σ and the clean-side procedure is essential. Histogram-fit Gaussian (curve_fit to clean-side bins) gives σ_g=35.8, σ_e=32.9, SNR=5.25, so SNR is method-sensitive at ~10% level.

The stored angle is nearly optimal: brute-force search for max separation gives -12.65°, max-fidelity angle -12.65° yields F=91.88% and Δµ=180.5, vs. stored -7.40° F=91.82% — difference <0.1% fidelity, <1% SNR, so the file’s angle is usable.

## 3. How the recomputed number relates to the value stored in the file

- **Stored vs. recomputed for Q1:** 83.65% (dataset `Q1/Fidelity[0]`) vs. 91.82% recomputed = **+8.17 points**. Q2–Q6 show similar gaps: Q2 78.07→89.1, Q3 74.25→87.2, Q5 74.91→87.5, Q6 65.17→82.9. Only Q4 remains low in both (02.77%→51.5% — stored is unphysically <0.5, indicating a bug or stale threshold).
- **Why the gap?** The stored fidelity is *not* the Fig.14 “maximum cumulative histogram contrast.” It matches neither the optimal thr (-155.9 for Q1) nor the `Syst Config` thresholds `[-5.84,1,-5,-3,-1,4]` applied after rotation:
  - Applying the config thr -5.84 to rotated Q1 data gives F=53.1% (or 54.6% on raw I), not 83.6%.
  - Q2 is the only case where config thr 1 → 83.6% is *close* to stored 78.1%, but still off.
  - This implies the stored Fidelity was computed at acquisition time with a *different Angle/threshold/algorithm* (possibly live FPGA threshold, older calibration, or averaged over batches) and never updated after post-processing for the paper. The file’s `datestr=2025-10-19_12-46-02` predates the paper (2026-10-05) by ~1 year, consistent with a reused cooldown whose final paper numbers came from re-analysis.
- **Implication:** A second team that does `f['Q1/Fidelity'][()]` will *not* reproduce 91.7%; they must ignore that attribute, parse the string-encoded `I_g/Q_g/I_e/Q_e` (each 3000 float64, `[ ... ]` without commas → `np.fromstring` after stripping brackets), rotate, and re-optimize the threshold. The raw vectors *do* support 91.7%.

## 4. What cannot be reproduced from the file alone

Reproducible from `ss_ge.h5` alone:
- 3000-shot I/Q clusters per state for 6 qubits (Q4 dead), rotated scatter and projected histogram shape (§2).
- Optimal-threshold fidelity within error (Q1 91.8% ≈ 91.7% paper).
- Relative SNR/Sep (Δµ/σ̄ ~4.8) and the need for clean-side isolation (total SNR 2.9 → inflated by T1 tails).
- `Angle` dataset nearly optimal; `Batch Num=0`, `Round Num=1`, `num_per_batch=1`, `ss_ge_results_batch=1` confirm single batch.

**Not reproducible / missing from file:**
1. **Absolute arbitrary-unit scaling.** Paper Δµ=28, σ≈6.4 vs. file Δµ=180, σ≈37.9 (scale ~6.4×). Gain/normalization (HEMT-only chain, DAC/ADC scaling, `res_gain_ge` in `Syst Config` but no dBm calibration) not documented.
2. **T1-limited error budget.** Paper Eq.7: T1=58.3 ± 4.1 µs, τ_ro=5.6 µs → ε_T1=4.7 ± 0.3% (57% of 8.3% infidelity) + thermal/T=44 mK residual. No T1, τ_ro (file has `res_length`=5.0 µs for Q1, not 5.6), temperature, or decay-trace data in HDF5. Cannot validate “limited by T1” from this file.
3. **Hardware context for Fig.14 caption:** “single HEMT, no quantum amplifier (TWPA/JPA), QUIET 10 mK fridge 100 m underground, tantalum-on-sapphire 6-transmon (3.8–5.0 GHz qubits, 6.2–6.5 GHz resonators, 0.4–0.6 MHz linewidths), common feedline” — none in `Exp Config`/`Syst Config`. `res_freq_ge` etc. are in Syst Config (e.g., Q1 6226.88 MHz) but without cryo wiring, attenuation, or BPF enable flag.
4. **BPF / SNR-chain claim.** Paper Figs.12–13 show 20 dB image rejection, 7.6 dB integrated-noise improvement (0.596→0.249 mV) proving BPF necessity. File has `ADMV8818` BPF in `Syst Config` implicitly but no PSD or filter-on/off flag to reproduce that comparison.
5. **Uncertainty methodology details.** Paper δµg = σ√(π/2N), δσ = σ/√(2Nclean), δSNR propagation Eq.6 — reproducible in principle, but paper’s Gaussian-core *fit* (dashed curves) vs. simple std correction gives ~10% SNR spread (4.4 vs. 4.8 vs. 5.25). Choice not logged in file.
6. **Q4 anomaly and array yield.** File shows Q4 fidelity 2.8% (impossible < chance) → hardware failure or software bug, not discussed in paper (paper highlights best qubit). No log explains why Q4 was excluded from Fig.14.
7. **Provenance:** No commit hash, QICK version, firmware, or calibration log linking `2025-10-19` batch to `FERMILAB-PUB-26-0193` Fig.14. `Dates=[1.76e9]` (unix ≈ 2025-10-19) and `Exp Config` (tof, res_spec, qubit_spec_ge, T1_ge, Readout_Optimization, etc., 301 keys) list many experiments but no `ss_ge` entry with repetition/relax_delay/threshold that ties to the 5.6 µs readout.

In short: **the *data* to remake the figure is in the file; the *argument* that 91.7% is T1-limited electronics SNR 4.4 with 44 mK thermal floor, taken on a HEMT-only QUIET cooldown with BPF, is not.**

---
*Generated by `reproduce.py` on 2026-05-13. Plot `readout.png` shows (a) rotated I-Q scatter (3000 blue |g⟩ + 3000 red |e⟩, black × medians, dashed optimal thr) and (b) I-projection histogram with clean-side Gaussian cores (Eq.3) and threshold. Metrics saved in `metrics.json`.*


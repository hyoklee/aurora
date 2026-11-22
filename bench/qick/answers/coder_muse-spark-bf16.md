Corrected citations — rest of answer unchanged:

**Reproduction from `ss_ge.h5` with `/tmp/.../venv/bin/python` (h5py, numpy, matplotlib) — plot `readout.png` saved (519 KB, 2700×1200) via `reproduce.py`:**

**1. VERDICT: YES** (dimensionless result) — the 91.7% assignment fidelity and scatter/histogram morphology are reproducible from raw I/Q within quoted error without hardware. Strictly **PARTIAL** for absolute `a.u.` axis scale and physical interpretation (T1-limit, temperature) which need paper text.

**2. Which qubit and computed number:**
- `Q1` in file = `list_of_all_qubits [0]`, physical qubit 0, `qubit_freq_ge 4194.73 MHz`, `res_freq_ge 6226.88 MHz`, `res_length 5.0 µs`, `Angle -0.1292 rad`, `Batch 0 Round 1 Date 1.7608939e9 ≈2025-10-19`. Only qubit reaching publication value (others: Q2 89.1%, Q3 87.2%, Q5 87.5%, Q6 82.9%, Q4 51.5% failed, `ss_ge.h5:Q1/Fidelity` etc. `Fidelity [0.83647275]` observed via h5py).
- After rotation `I_rot= I cosθ - Q sinθ`, `Q_rot= I sinθ+Q cosθ`, medians `µ_g=-245.7`, `µ_e=-65.4`, `Δµ=180.32±1.26`, clean-side mirrored `σ_g=37.33±0.68`, `σ_e=40.64±0.74`, `σ̄=39.02±0.51`, `SNR=Δµ/σ̄=4.62±0.07` (paper `fermilab-pub-26-0193.txt:482` reports `SNR = 4.4 ±0.1` Eq.(1); `fermilab-pub-26-0193.txt:488-492` Fig.14 caption reports `Δµ = 28 ±0.2 a.u., σg=6.4±0.1, σe=6.5±0.1` — same ratio, scale factor ~6.4×).
- Max cumulative histogram contrast at `I_thr=-155.4`: `F_ks=0.83633`, `F_avg=0.5+F_ks/2=0.91817±0.0035 → (91.8±0.35)%`. Paper `fermilab-pub-26-0193.txt:422-424` Eq.(2) `FSSF = (91.7 ±0.4)%` and `fermilab-pub-26-0193.txt:554-558` repeats `(91.7±0.4)%` limited by `T1=58.3±4.1 µs`. Difference 0.001 <1σ.

**3. Relation to stored value:**
`Q1/Fidelity [0.83647275]` in `ss_ge.h5` is `F_ks = max(CDF_g-CDF_e)` (scan matches stored to 1.4e-4 for all qubits), not `F_SSF`. Paper's fidelity `fermilab-pub-26-0193.txt:424` = `(1+F_ks)/2`; `(0.83647+1)/2=0.91824` matches our `0.91817`. Second team must recompute threshold sweep, not read attribute.

**4. What cannot be reproduced from file alone:**
- Absolute `a.u.` scale (180 vs 28); only SNR/fidelity are scale-invariant. Requires ADC gain/integration normalization not stored.
- Exact `σ`/`SNR` =4.4 vs 4.62 shift from undocumented histogram binning/fit weighting of clean-side Gaussian (outer `I≤µ_g`, `I≥µ_e` per paper §III.C).
- Physics claims: `τ_ro=5.6 µs`, `T1=58.3±4.1 µs`, `ε_T1=1-exp(-τ/2T1)=4.7%`, `T_eff≈44 mK`, HEMT-only, IR shielding — from paper `fermilab-pub-26-0193.txt:429-433` and `554-558`, not I/Q.
- Mapping of Fig.14 `fermilab-pub-26-0193.txt:488-491` to file label `Q1` is inferred (best fidelity); file does not flag published qubit, and `Q4` pathological `0.027` KS is unexplained.
Inspect the HDF5 file structure and search the paper text for the 91.7% fidelity / Fig. 14 references.


Read the paper's Fig. 14 caption and surrounding methodology to understand the fidelity definition.


Write a Python script to inspect the HDF5 file's groups, datasets, attributes, and metadata without shell quoting issues.


The file contains Q1..Q6. Need concise dataset names/shapes per qubit. Let me query without printing big config.


Decode the serialized shot arrays, verify lengths and basic statistics, and search the paper text for the exact fidelity definition.


Debug the serialized array format; np.fromstring failed, likely due to brackets or newlines.


Reissue the inspection script cleanly to decode the serialized shot arrays.


The I/Q data are stored as numpy array repr strings with brackets. Strip brackets and parse.


Need inspect the exact fidelity definition and clean-side procedure from the paper text, then compute per qubit.


Need inspect the exact fidelity-definition and clean-side procedure in the paper text.


Compute RMS-around-median and other clean-side Gaussian-fit estimates to match the paper's SNR procedure.


Fix syntax error and rerun.


Rewrite the SNR exploration script cleanly to avoid syntax issues.


Write the SNR exploration script via Python to avoid heredoc quoting issues.


We have identified that Q1's maximum-contrast fidelity is 0.9182, matching the paper's 91.7%. Now generate the reproducible readout plot for Q1 and verify it.


Create the reproducible Q1 readout figure from the raw I/Q shots and save it as readout.png.


Verify the saved plot exists and is a valid image before submitting.


Submit final report with verdict and computed metrics.
Short report

1. VERDICT: YES

A second team can reproduce the QICK Box single-shot readout figure and the reported readout fidelity from ss_ge.h5 alone, without rerunning quantum hardware. Using the stored raw I/Q shots and rotation angle for Q1, I recomputed a maximum-cumulative-histogram-contrast fidelity of 91.82%, matching the paper's (91.7 +/- 0.4)% within 0.12 percentage points. The plot was saved as readout.png.

2. Qubit and computed fidelity
- The 91.7% corresponds to file group Q1 (system config list_of_all_qubits=[0], resonator gain on qubit index 0, readout length 5.0 us).
- Computed fidelity: 91.82% (0.9182), from 3000 g-state and 3000 e-state shots after rotation by the stored angle (-0.1292 rad).
- Rotated separation: Delta_mu = 180.32 a.u.; clean-side RMS widths 37.33 and 40.63 a.u.; clean-side SNR = 4.62.

3. Relation to the value stored in the file
- The file's stored Fidelity for Q1 is 0.8365 (83.65%).
- This stored value equals the Kolmogorov-Smirnov separation D between the two rotated I distributions (recomputed D = 0.8363), not the paper's maximum-contrast fidelity.
- The paper's fidelity must be recomputed from the raw shots as max-threshold contrast; using the stored value directly would underestimate it by about 8.2 percentage points.

4. What cannot be reproduced from the file alone
- The paper's fidelity-limiting decomposition (T1 decay contribution of 4.7%, thermal population, pulse imperfections, Gaussian overlap) cannot be verified because T1, effective temperature, and related characterization data are not present in ss_ge.h5.
- The exact paper figure overlays and intermediate reported quantities (Delta_mu = 28, sigma_g = 6.4, sigma_e = 6.5, SNR = 4.4) are not reproduced by my simple RMS calculation; those depend on the paper's unstated normalization/fitting details.
- The file does not identify which physical qubit is intended beyond the system configuration and recomputed fidelity; the match to Q1 is inferred from the recomputed 91.82%.
- No hardware rerun is needed for the fidelity and plot, but reproducing the paper's full physical interpretation requires additional characterization data not included in this file.
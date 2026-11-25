# IceCube 7-yr point-source posterior → clio-core → Figure 10 (2026-11-24)

Task (`doc/20261123_claude_icecube7.md`): write an OMNI file that ingests the
HDF5 data and code of the IceCube data release
"Bayesian posterior for IceCube 7-year point-source search with neutrino-count
statistics" (doi:10.21234/dtrs-e557) into clio-core, then have CLIO Agent use
it to regenerate Figure 10 of the paper (ApJ 893 (2020) 102,
arXiv:1909.08623).

**Result:**
- The OMNI file `bench/icecube7/ice_omni.yaml` ingests the release straight
  from icecube.wisc.edu. That needed two new CAE features:
  - reading members out of a zip on a plain web server;
  - keeping HDF5 compound tables and attributes.
  Both are in clio-core PR [iowarp/clio-core#1430](https://github.com/iowarp/clio-core/pull/1430).
- Of the four ALCF models, **nemotron-35-super-vl** regenerated Figure 10 from
  clio-core:
  - its grid of the pointwise likelihood ratio matches the release's own code
    to 1.5e-11;
  - its two example log Bayes factors are exact.
- The other three models (mistral, beam, muse) stopped on replies CLIO could
  not parse.

**Update 2026-11-25:** with clio-agent 0.9.7.1 plus six fixes, all four models
regenerate Figure 10. See `doc/20261125_icecube7_clio_agent_097_all_models.md`.

## 1. The release and what CAE had to learn

The release is one 12 MB zip with three files:
- `posterior.h5`, written by PyTables;
- `example.py`, which computes the Bayes factors and draws Figure 10;
- `readme.txt`;
- plus `__MACOSX/` resource-fork copies.

`posterior.h5` has five tables (Isotropic, Galactic_disk, Fermi_bubble,
SFD_dust, Northern_sky), each with 70–84 k posterior samples in compound
records `{ln_A, ln_Fb, loglikelihood, n1, n2}`. Figure 10 also needs values
that are only stored as **HDF5 attributes**:
- `P_log_evidence` and `NP_log_evidence` on each table;
- the uniform prior densities `prior_*` on the root group.

The clio-core CAE before this work could not ingest the release:

| Gap | Effect | Fix (PR #1430) |
|---|---|---|
| An `https://` source could only be downloaded whole to `file::` | No way to unpack a web zip into clio-core | `https://…zip` to `iowarp::` is now staged like a one-file Dataverse dataset. Members are chosen with `dataset_filter` and range-read out of the remote zip, using the same reader as `dataverse://` (PR #1429). |
| Compound datasets were stored as `tensor<unknown, N>` | Field names and layout lost. PyTables stores the columns alphabetically, not in the readme's order. | The description is now `compound<40, 75533>{ln_A:float64@0, ln_Fb:float64@8, loglikelihood:float64@16, n1:float64@24, n2:float64@32}` |
| HDF5 attributes were dropped | The evidences and priors were missing, so Figure 10 could not be computed | Each dataset gets an `attributes` blob (JSON). Root attributes are stored on the file's tag. |

Two more fixes came out of the tests:
- **Rejected `dataverse::` URLs:** the factory refused the documented
  `dataverse::https://host/pid` form.
- **JSON escaping:** JSON strings now escape every control character.

The PR's new tests:
- the stager against a plain web zip;
- a compound table with attributes;
- `cae_dataverse_ingest`, which runs Dataverse and web-zip ingest through a
  real runtime.

### OMNI file

```yaml
transfers:
  - src: "https://icecube.wisc.edu/data-releases/20190904_Bayesian_posterior_for_IceCube_7-year_point-source_search_with_neutrino-count_statistics.zip"
    dst: "iowarp::ice"
    format: "hdf5"
    dataset_filter:
      include_patterns: ["*.zip/posterior.h5", "*.zip/example.py", "*.zip/readme.txt"]
```

- **Command:** `clio-cae ice_omni.yaml`, against a private runtime on port 9613
  (`bench/icecube7/clio_ice.yaml`).
- **Cold run:** 9 HTTP requests and 12.3 MB received. It reads the zip
  directory, then the three members; the `__MACOSX/` copies are never read.
  CAE then ingests 5 tables, 2 text files and a manifest, in about 1 s.
- **Warm run:** from the stage cache, 3 requests and 65 KB, reading only the
  zip directory. Logs are in `bench/icecube7/results/cae_ingest_{cold,warm}.log`.
- **Read-back check:** all five tables read back with `cte_ice.read_table` are
  equal to `h5py` reads of the original file, field for field. Every attribute
  is present, and `example.py` is identical byte for byte. The reader is
  `bench/icecube7/cte_ice.py`.

## 2. Reference

The release's own `example.py` was run on the original `posterior.h5` with
PyTables 3.7. The only changes were that LaTeX is not installed, so `usetex`
was turned off with two label tweaks, and the output is a PNG
(`results/example_ref.diff`).

- Log Bayes factor, model (A=1e16, F_b=1e-12, n1=1.9, n2=-2): **-1.6859153734**
- Log Bayes factor, model (A=1e17, F_b=1e-13): **-0.9795234237**
- Figure: `bench/icecube7/results/fig10_reference_example_py.png`. It has the
  same contours and source-class markers as the paper's Fig. 10:
  - FR-II lies inside the M > 10⁻¹ region;
  - Galaxy Clusters lies on the 10⁻² edge;
  - BL LAC and FSRQ lie outside, so they are disfavoured.

## 3. CLIO Agent on the ALCF Tara models

Setup:
- **Harness:** `bench/icecube7/run_agent_ice.sh <model>`, run outside the repo,
  with the same harness as the golf and QICK runs.
- **Prompt:** `bench/icecube7/prompt.txt`. It points the agent at clio-core
  and `cte_ice.py` and forbids downloading. The working directory holds no copy
  of the data or of `example.py`; those are only in clio-core.
- **Pre-flight:** before each run, the harness checks that the runtime still
  holds the 9 `ice` tags.

| Model | Wall | Tool calls | Outcome |
|---|---|---|---|
| **nemotron-35-super-vl** | 294 s | 31 | **Figure 10 regenerated from clio-core.** Read `example.py` and `readme.txt` out of clio-core and ported the code to `cte_ice` (no PyTables). Log BFs −1.685915 / −0.979523, equal to the reference. ln M grid equal to the reference's to max \|Δ\| = 1.5e-11. Wrote `fig10.py`, `fig10.png`, `answer.md`. Reports a clio-core read of 0.35 s for 75,533 samples plus attributes. |
| mistral-large-4 | 1480 s | 11 | No deliverables. After 11 tool calls the model sent `{'neither', 'both', 'min', 'max'}`, which CLIO's JSON adapter rejected, and the turn stopped. |
| beam-1.0-fp8 | 8–10 s | 0 | Failed twice at step 0. Its first reply had `tool_calls` but no `next_thought`, so the adapter rejected it. |
| muse-spark-bf16 | 7 s | 0 | Failed at step 0. Its reply was plain prose ("Regenerating your Figure 10 — first mapping the workspace…"), not JSON. |

Run outputs are in `bench/icecube7/runs/`, without the paper text:
- `agent_nemotron-35-super-vl/`;
- `fail_*` for the three failures;
- `invalid_mistral_try1_ctest_killed_runtime/`, a first mistral run that was
  void (see §4).

**How good is nemotron's figure:**
- **Computation:** correct. It used the Isotropic template, a Gaussian KDE over
  `[ln_A, ln_Fb, n1, n2]`, prior = Π `prior_*` = 1.913674e-05, model prior 0.5,
  a 100×100 grid with the FIRESONG conversions, and levels 10⁻² and 10⁻¹. It
  read every number from clio-core attributes.
- **Picture:** matches the reference.
- **One cosmetic bug:** the x-axis label renders as "L_exteff". A non-raw
  Python string turned the `\t` of `\text` into a tab.
- **Comparison with the paper:** its `answer.md` compares its figure with the
  paper only through the caption and the code, not by looking at the published
  plot.

**Verdict:**
- With the release in clio-core and a small reader module, one of four ALCF
  models (nemotron) fully regenerated Figure 10 in under 5 minutes, with
  results that are numerically identical to the authors' code.
- The other three failures are the model-to-CLIO protocol problem seen in the
  golf and QICK runs: replies that DSPy's JSONAdapter cannot parse. They do not
  involve the data path.
- This is nemotron's first success. On golf its `shell_bash` calls failed
  CLIO's argument check.

## 4. Pitfall found: clio-core's ctest kills every clio runtime

- **What it does:** the ctest fixtures `cr_pre_test_cleanup` and
  `cr_post_test_cleanup` are attached to every test, even `ctest -R one_test`.
  They `kill -9` every `clio_run` process on the host and `fuser -k` port 9413.
- **What it broke:** running `ctest -L cae` for PR #1430 killed CLIO Agent's
  runtime and the IceCube runtime in the middle of the first mistral run, so
  that run is void.
- **Golf runtime:** the golf runtime's earlier "silent deaths" on port 9513
  (2026-11-22 and 2026-11-23) line up with ctest runs, and this is the likely
  cause.
- **Rule:** run the test binaries directly, with ctest's environment on port
  10500, while agents are running. Otherwise restart and re-ingest afterwards.

## Files

- `bench/icecube7/`: `ice_omni.yaml`, `clio_ice.yaml`,
  `clio_env.sh.example`, `cte_ice.py`, `prompt.txt`, `run_agent_ice.sh`
- `bench/icecube7/results/`: reference figure, `example_ref.py` and its diff,
  CAE ingest logs
- `bench/icecube7/runs/`: per-model agent runs
- clio-core: branch `feat/cae-web-zip-hdf5-tables`, PR #1430, which builds on
  #1429

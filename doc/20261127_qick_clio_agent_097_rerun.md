# QICK readout question: CLIO Agent reruns for the ALCF models that gave no answer (2026-11-26/27)

Follow-up to `doc/20261121_qick_reproducibility_alcf_agents.md`. On CLIO Agent 0.9.4,
two models gave no answer:
- muse-spark-bf16 failed at step 1 in 3 of 3 tries;
- beam-1.0-fp8 failed at step 1 in try 1 and scored 6/6 in try 2.

Both failures were DSPy JSONAdapter parse errors. This rerun uses the fixed clio-agent,
the same prompt and the same workspace contents (`ss_ge.h5` and the paper text).
- clio-agent 0.9.7.1 from source, branch `fix/model-shell-cte-pool-binding`, proposed
  upstream as iowarp/clio-agent#2402;
- the IceCube-7 harness approach (`bench/qick/run_agent_097.sh`).

The CLIO Agent mistral and nemotron runs, and every Genesis run, were not repeated.
That includes Genesis + mistral (HTTP 400 on images), which runs on Claude Code, not CLIO.

## Result

| model | 0.9.4 (2026-11-21) | 0.9.7.1 + fixes (this rerun) |
|---|---|---|
| **muse-spark-bf16** | 0 (3/3 tries failed at step 1) | **5/6**, first try, 465 s, 20 tool calls, 1.02 M tokens |
| **beam-1.0-fp8** | 6/6 on try 2 (try 1 failed at step 1) | **no valid answer**: 2 clean tries both ended without `readout.png` or a report; 2 more tries excluded as contaminated |

### muse-spark-bf16: answered on the first try (`runs097/ok_muse-spark-bf16/`)

Rubric of the 2026-11-21 doc:

| V | Q | F | S | N | P | score |
|---|---|---|---|---|---|---|
| ✓ PARTIAL, with reasons | ✓ Q1 | ✓ 91.82 ± 0.35 % | ✗ | ✓ 4.76 ± 0.07 | ✓ | **5** |

- **V:** "functionally YES for raw data, NO for stored metadata".
- **F:** 91.82 ± 0.35 % at the max-contrast threshold −155.9. The stored angle is within
  0.06 % of the optimal one.
- **S:** muse gave the wrong explanation for the stored 0.8365. It blamed "a different
  (stale, FPGA-live or earlier-cal) angle/threshold" rather than identifying the contrast
  2F−1.
- **N:** muse used the paper's clean-side method with the half-normal correction ÷√(1−2/π):
  σ_g 36.4, σ_e 39.4, Δµ 180.3, SNR 4.76 ± 0.07. It noted that a histogram `curve_fit`
  gives 5.25, and that 4.4 is not reproduced exactly.
- **P:** a rotated I–Q scatter plus a projected histogram with Gaussian cores, the
  threshold and the numbers. A footer caption overlaps the axis labels, which is cosmetic.
- **Other findings:** it found Q4 is dead (51.5 %) and listed what the file lacks: the a.u.
  scale, the T1 budget, the amplifier chain and provenance.
- **Answer text:** the answer begins with muse's raw working notes ("We need to find
  ss_ge specific config…") before the report. They are muse's own narration and harmless,
  but untidy.

On 2026-11-21 this would have been the second-best CLIO Agent answer, after beam's 6/6.

### beam-1.0-fp8: four tries, no valid answer

| try | wall | calls | outcome | status |
|---|---|---|---|---|
| 1 `invalid1_beam_reused-tmp-scripts_image400` | 1470 s | 44 | After `find / -iname "*qick*"` it read and ran `/tmp/readout_plot.py` and `/tmp/qick_analysis/*.py`, left there by a 2026-11-21 run. It then called `view_image` on its plot. The gateway replied HTTP 400 "beam-1.0-fp8 is not a multimodal model", and that error text became the final answer. | excluded |
| 2 `invalid2_beam_ran-old-mistral-scripts-from-venv` | 622 s | 29 | `grep -r` over the scratch tree found `analyze.py`, `analyze2.py` and `plot_readout.py`, which the 2026-11-21 CLIO Agent + mistral run had written into the shared venv directory. Beam rewrote their input path with `sed` and reported their output (F 91.60 %, SNR 7.6). | excluded |
| 3 `fail3_beam_reasoning_length` | 1092 s | 23 | Clean: its searches timed out with nothing found. It computed F 91.8 % and SNR 7.9, then its reasoning looped on tool-call markup until 32,768 output tokens (`finish_reason=length`). No plot, no report. | failed |
| 4 `fail4_beam_stop-hook-cap_no-plot` | 2415 s | 49 | Clean, but not for lack of trying (see below). The first turn hit the length limit. The Stop hook re-drove it 3 times, and it hit the cap still without `readout.png`. Final text: "We need make readout.png now…". | failed |

**Try 4 tried to read the answers, and the locks held.** Beam ran:
- `ls ~/aurora` and `ls ~/aurora/bench/qick`;
- `cat` on `CLAUDE.md`, `README.md` and `results.json`;
- `ls answers/`;
- `sudo cat results.json`;
- `find / … | xargs grep -l "max cumulative"`.

Every read of an answer was denied (`chmod 000`), sudo needs a password, and the `find /`
calls timed out at the 120 s shell cap. The only file it could read was `prompt.txt`, its
own prompt. On 2026-11-21 a beam run found `src/qick/ss_fidelity.py` the same way.

**Beam's reliability on this task is poor.** The step-1 adapter failures of 0.9.4 are gone.
In their place, beam ends turns in prose or loops its reasoning to the output limit.
The same two failure modes appeared in IceCube-7 (doc/20261125), where beam needed the
Stop hook and one rerun.

## clio-agent change from this rerun

- **5624f32e3 `feat(lm)`:** `lm.accepts_images: false` (`CLIO_LM_ACCEPTS_IMAGES`) sends a
  text-only model no images. Each image becomes the note "[image omitted: this model does
  not take images; inspect the file another way]".
- **Why it is a setting:** ALCF's `/models` rows carry no modality fields, so CLIO cannot
  learn this itself.
- **Cap semantics:** `lm.max_request_images` keeps its meaning. Internally the cap is now
  `None` (no cap), `0` (no images) or N.
- **Tests:** a new test, plus the existing image-budget tests.
- **Effect:** with it, beam's tries 2–4 had no image error.

The PR's earlier fixes all apply here:
- stream recovery for `hosted_vllm`;
- Harmony wire-tag stripping for beam;
- the Stop-hook re-drive followed by `clio-agent -q`;
- the request image cap.

Muse's first-try success is the 0.9.4 → 0.9.7.1 change: 0.9.7 has no DSPy JSONAdapter.

## Harness: `bench/qick/run_agent_097.sh`

Same design as `bench/icecube7/run_agent_ice_097.sh`:
- built-in `clio-agent -q --json --workspace`;
- one agent home per run, in scratch;
- a Stop hook requiring `readout.png` (`deliverables_stop_hook.py <dir> readout.png`; the
  script now takes the file list as arguments);
- a 120 s shell cap and a 75 min run cap;
- `CLIO_LM_ACCEPTS_IMAGES=false` for beam and `CLIO_LM_MAX_REQUEST_IMAGES=1` for nemotron.

Locked with `chmod 000` during a run:
- the repo's `src/qick`, `doc/`, `CLAUDE.md` and `bench/icecube7`;
- `bench/qick/{runs,answers,invalid,results.json,plots_grid.png,README.md}`;
- the older CLIO homes, which hold earlier QICK sessions;
- every other scratch entry, run directory and agent home.

New after this rerun's leaks:
- **`/tmp` leftovers.** Agents also write scripts at the top of `/tmp`; CLIO allows `/tmp`.
  The 2026-11-21 leftovers were moved out. After each run, the harness moves the
  analysis-type files and directories (`*.py`, `*.png`, …, `scratchpad/`, `qick*/`) that
  the run created at the top of `/tmp` into the run's `tmp_leftovers/`.
- **The venv.** It cannot be locked, because the prompt points at its `python`. The
  harness now refuses to start (exit 98) if the venv holds stray files, and sweeps it and
  the run root after each run.

Validity check for every scored run: its semantic trace lists every tool call (copied as
`tool_calls.txt`). Muse's calls touch only its workspace and the venv's `python`.

**Lesson:** a lock list is never complete while a shell-capable agent runs as the user who
holds the answers. Beam searched `/`, `/tmp`, the scratch tree and the repo, and tried
`sudo`. Every place an earlier run could have written to is a leak.
- Robust fix: a separate account or container per run.
- Cheaper fix: audit each trace (as done here) and exclude what leaked.

## Files

- `bench/qick/run_agent_097.sh`
- `bench/icecube7/deliverables_stop_hook.py` (now takes the file list as arguments)
- `bench/qick/runs097/`: one directory per try, each with `out.json`, `meta.json`,
  `tool_calls.txt`, plots and `tmp_leftovers/`; the inputs are not committed
- clio-agent: iowarp/clio-agent#2402, now 8 commits including 5624f32e3

# IceCube-7 Figure 10 with CLIO Agent 0.9.7.1: all four ALCF models (2026-11-25)

Follow-up to `doc/20261124_icecube7_clio_cae_figure10.md`. That run used CLIO Agent
0.9.4, where only nemotron-35-super-vl regenerated Figure 10 of arXiv:1909.08623 from
clio-core. Mistral, beam and muse stopped on DSPy JSONAdapter parse errors.

Task, prompt and data path are unchanged:
- the IceCube release is ingested into clio-core (runtime on port 9613, 9 `ice` tags);
- the agent reads it through `cte_ice.py`;
- it must write `fig10.py`, `fig10.png` and `answer.md`.

**Result: all four ALCF models now regenerate Figure 10 from clio-core.** It took:
- upgrading to clio-agent 0.9.7.1 (upstream main);
- six clio-agent fixes;
- harness changes.

Every grid equals the release's own code, and every Bayes factor is exact.

| Model | Wall | Tool calls | ln B (two example models) | max \|ln M − ref\| | Notes |
|---|---|---|---|---|---|
| nemotron-35-super-vl | 265 s | 37 | −1.685915 / −0.979523 | 1.5e-11 | image cap 1 (fix 4) |
| muse-spark-bf16 | 405 s | 24 | −1.685915 / −0.979523 | 1.5e-11 | reports a 5 ms read of the 75,533-row Isotropic table |
| beam-1.0-fp8 | 1705 s | 30 | −1.6859 / −0.9795 | 1.5e-11 | two Stop-hook re-drives (ended turns in prose) |
| mistral-large-4 | 2677 s | 41 | −1.6859153734 / −0.9795234237 | 1.5e-11 | two gateway stream cuts, both regenerated (fixes 2 and 5) |

Reference values: −1.685915 / −0.979523, from the release's `example.py` run on the
original HDF5 file. The grid check is `bench/icecube7/verify_fig10.py`: it runs the
agent's `fig10.py` unchanged, captures the first array passed to `contour` or
`contourf`, and compares it with the reference grid. Its transpose differs by 1.4,
so the check is not trivially symmetric.

All four figures look like the reference:
- filled regions M > 10⁻² (purple) and M > 10⁻¹ (green);
- black and white boundaries;
- FR-II, Galaxy Clusters, BL LAC and FSRQ markers;
- log axes from 10⁵² to 3·10⁵⁵ erg/yr and 10⁻¹² to 10⁻⁵ Mpc⁻³.

Remaining differences are cosmetic:
- all use mathtext because there is no LaTeX on the node;
- muse's "Clusters" half of the "Galaxy Clusters" label is white on white.

Muse's earlier 0.9.7.1 figure had its fill colors shifted by one band. This run's
figure does not.

## clio-agent fixes

Branch `fix/model-shell-cte-pool-binding` on the fork (hyoklee/clio-agent), on top
of upstream main 5ced7a786. Each code fix has a test that fails without it; row 6 is a harness setting.

| # | Commit | Failure it fixes (model) |
|---|---|---|
| 1 | 1985186e9 | CLIO's server exports `CLIO_CTE_POOL=564.0` for its own clio-core daemon, and the model's shell and kernels inherited it. `cte_ice` then looked for pool 564 in the ice runtime and hung, with no "pool not found" error. Nemotron spent 30 min on the first read. The binding is now dropped from model processes. |
| 2 | 7684b2cac | Stream recovery was only on for the `openai` prefix. The ALCF and vLLM presets use `hosted_vllm`, and a gateway-closed stream (`LMStreamAssemblyError`, "ended without a completion event") was not retried. Mistral lost a 455 s generation this way. |
| 3 | 3ba6952e3 | beam-1.0-fp8 (a gpt-oss model) leaks Harmony wire tags (`<\|start\|>…<\|message\|>`) into `reasoning_content`. Replaying them made the gateway answer 400 "Pass plain text in API fields, not Harmony wire tags". The request now carries the text without the tags; the transcript is unchanged. |
| 4 | d4d2b35f1 | Nemotron's endpoint accepts one image per prompt. After the model viewed a second figure, every request failed with 400 "At most 1 image(s) may be provided in one prompt". New setting `lm.max_request_images` / `CLIO_LM_MAX_REQUEST_IMAGES` keeps the newest N images in each request. |
| 5 | e9e13064c | Muse narrates every step ("Now I'll generate the final script.") before its tool call. A stream cut in the middle of a long tool call was not retried, because answer text had already streamed. Text that comes before a tool call the attempt had begun is narration, so that generation is now retried. A partial answer with no tool call is still never restarted. |
| 6 | (harness) | The shell tool has no timeout by default. Beam's `find / -name latex` walked Lustre/DAOS for 44 min. The harness sets `CLIO_SHELL_MAX_TIMEOUT_S=120`. |
| 7 | e823d8cad | `clio-agent -q` returned when a Stop hook blocked the end of a turn, because `message.completed` was published before the Stop gate ran. The harness then stopped the server, so the queued turn was cancelled. With a shared agent home, the queued turn instead resumed under the next model: beam's session was finished by mistral. `message.completed` now carries `stop_hook_redrive`, and the client follows the session until a turn ends unblocked. |

Validation:
- the agent-loop, media, stream-recovery, hook, client and `message.completed` test
  suites pass;
- one pre-existing failure, `test_variant_advice_is_recorded_as_rendered_…`, fails
  the same way on upstream main;
- ruff 0.14.0 check and format are clean.

Fork syncs done first:
- hyoklee/clio-agent `main` was synced with iowarp/clio-agent;
- hyoklee/clio-coder `main` was force-synced with upstream, and its previous head was
  kept as branch `backup/main-pre-sync-20261125`;
- clio-coder needed no change for this task.

## Harness: `bench/icecube7/run_agent_ice_097.sh`

- **Isolated install.** clio-agent is installed from source (editable) under
  `CLIO_PREFIX=~/.local/share/clio-dev`, so it never touches the 0.9.4 install. Each
  run uses `clio-agent -q --json --workspace <run dir>`, which grants every
  permission.
- **One agent home per run** (`$ICE_AGENT_HOME/<run>`). Point `ICE_AGENT_HOME` at
  scratch space: each home preallocates a 1.1 GB clio-core store (`data/cte`). Three
  homes under `~` hit the home quota, and the next three runs failed at
  `clio start` with "Disk quota exceeded".
- **Stop hook on** (`deliverables_stop_hook.py`). It re-drives the turn at most 3
  times while `fig10.py`, `fig10.png` or `answer.md` is missing. It needs fix 7.
- **Per-model settings.** `CLIO_SHELL_MAX_TIMEOUT_S=120`. `CLIO_LM_MAX_REQUEST_IMAGES=1`
  for nemotron. Run limit 75 min (`ICE_TIMEOUT`), because the ALCF gateway sometimes
  takes 1–2 min per call. `clio stop` after every run, so a timed-out turn does not
  keep running in the server.
- **Ground truth is locked during each run.** The harness runs `chmod 000` on
  `ref/`, `paper/` (the PDF), the raw downloads and every other run directory, and
  restores them on exit. Before this:
  - mistral-large-4 walked up from its workspace, read the reference grid and
    `example_ref.py`, and rendered the paper's Figure 10 page from the PDF;
  - an earlier nemotron session read the 0.9.4 nemotron answer.

  Both runs are excluded. `CLIO_ALLOWED_ROOTS` does not confine the shell tool.

## Failures along the way (evidence in `bench/icecube7/runs/v097/`)

| Directory | What happened | Fixed by |
|---|---|---|
| `fail1_nemotron_cte_pool_hang` | first clio-core read hung 30 min (inherited pool 564) | fix 1 |
| `fail_mistral_degenerate_stream_endpoint` | the endpoint streamed 484 s of gibberish, then cut | (endpoint; regenerated by fix 2 now) |
| `fail2_mistral_stream_cut_hosted_vllm` | coherent stream cut after 120 s, not retried | fix 2 |
| `fail_beam_find_root_no_shell_timeout` | `find /` ran 44 min; deliverables written after the timeout by the still-running turn were removed | harness 120 s cap, `clio stop` after run |
| `fail_beam_prose_no_toolcall`, `fail_beam_prose_ended_turn` | beam ended its turn with a plan in prose and no tool call | Stop hook + fix 7 |
| (no directory) | beam's leaked Harmony tags → gateway 400 returned as its answer | fix 3 |
| `fail_beam_reasoning_loop_length` | beam's reasoning looped on tool-call markup for 32,768 tokens (finish_reason=length) | model/gateway; a rerun succeeded |
| `fail4_nemotron_two_images_400` | second `view_image` → gateway 400 delivered as the answer | fix 4 |
| `fail5_muse_narrated_stream_cut` | stream cut after narration, mid tool call | fix 5 |
| `timeout_mistral_45min_slow_gateway` | still checking its figure at the 45 min cap | 75 min cap |
| `fail6_beam_q_returned_before_redrive` | the Stop hook re-drove, but `-q` had already returned | fix 7 |
| `invalid_mistral_read_ground_truth` | read the reference files; excluded | ground-truth lock |

**ALCF gateway quirk (not fixed):** the gateway delivers an HTTP error, such as the
400s above, as ordinary streamed content with `finish_reason=stop`. CLIO therefore
records the error text as the model's answer, and the turn "completes". Fixes 3 and
4 remove the causes seen here. A gateway that reported the error status would let
CLIO fail the turn visibly instead.

## Files

- `bench/icecube7/run_agent_ice_097.sh`, `deliverables_stop_hook.py`,
  `verify_fig10.py`
- `bench/icecube7/runs/v097/ok_<model>/`: deliverables, `out.json`, `meta.json`
  for the four successful runs
- `bench/icecube7/runs/v097/fail*`, `timeout*`, `invalid*`: failure evidence
- clio-agent: `hyoklee/clio-agent` branch `fix/model-shell-cte-pool-binding`
  (6 commits on upstream main 5ced7a786), proposed upstream as
  [iowarp/clio-agent#2402](https://github.com/iowarp/clio-agent/pull/2402)

# Golf sensors from Dataverse through clio-core CAE: novice vs pro (2026-11-22)

Task (`doc/20261122_claude_golf.md`): use clio-core CAE to ingest the Harvard
Dataverse data of the MultiSenseGolf paper [1], find which sensor values separate
novice from pro golfers, sort the sensor variables by impact, and measure the
difference with and without clio-core. Follow-up request: make CAE ingest straight
from Dataverse so the CLIO agent can analyze the data from an OMNI YAML.

## 1. What was built: a `dataverse://` source in CAE

The installed clio-core wheel (iowarp-core 2.2.1) could not do this: its CAE had no
Dataverse or zip support and was built without HDF5. A new CAE source was added on
branch `feat/cae-dataverse-assimilator` of iowarp/clio-core, now
**PR [iowarp/clio-core#1429](https://github.com/iowarp/clio-core/pull/1429)**
against `dev` (worktree `~/clio-core.dataverse`):

```yaml
# bench/golf/golf_omni.yaml
transfers:
  - src: "dataverse://dataverse.harvard.edu/doi:10.7910/DVN/LCCLLW"
    dst: "iowarp::golf"
    format: "hdf5"
    dataset_filter:
      include_patterns:
        - "Documentation.zip/*.csv"      # members of one zip
        - "P*.zip/*_stream_data.hdf5"    # one kind of member of all 33 participant zips
        - "hdf5:*/data"                  # datasets inside those HDF5 files
        - "hdf5:*/time_s"
```

* The dataset is listed through the Dataverse API, and patterns select whole files
  or **members of the remote zips**. Members are read with HTTP range requests:
  the ZIP/ZIP64 directory first, then each member, which is inflated and CRC-checked.
  The 1.5 GB of sensor files come out of 49 GB of archives without downloading the
  video and depth frames.
* Each zip's download redirect is resolved once and reused, so ranged reads go
  to S3, not the Dataverse server. Throttled requests back off and retry. An API
  key (`src_token`) goes to the Dataverse host only.
* With `iowarp::golf`, each object becomes tag `golf/<zip>/<Swing>/<file>.hdf5/<dataset>`
  through the HDF5 assimilator, and tag `golf` gets a `manifest` blob. With `file::<dir>`
  the objects are only staged. `cae_dataverse_tool list|stage` does the same without
  a runtime.
* Tests: the new offline unit test `cae_dataverse` covers the URL parser, JSON
  reader, selection rules, and stored/deflated/ZIP64 archives. `ctest -L cae`:
  55/55 pass. The README documents the source.

Lesson recorded: the first version made one probe and one GET per member against
Dataverse, about 3,000 requests in a minute, 8 at a time. Harvard's load balancer
then blocked this host with 403 for about 20 minutes. The redirect-once design made
about 35 Dataverse requests for the whole run and was never throttled.

## 2. Answer: what separates novices from pros

Groups: Beginner P01–P08 and Professional P17–P24 (8 each); Intermediate P09–P16
is shown only for trend. There are 1,557 swings in total. Features were computed
per swing, then taken as **the median per participant**, and the groups compared
8 vs 8, so the 40–110 swings per golfer don't inflate the significance. The
separation measure is Hedges' g, with AUC and Mann–Whitney p
(Benjamini–Hochberg q over 84 features). Script: `bench/golf/golf_truth.py`;
full table: `bench/golf/results/ranking.csv`.

| # | sensor variable (per swing) | Beg. | Int. | Pro | g | AUC | p | p, adj. height |
|---|---|---|---|---|---|---|---|---|
| 1 | peak acceleration, trail (right) forearm (m/s²) | 37.1 | 54.9 | 62.6 | +2.20 | 0.97 | 0.0006 | 0.003 |
| 2 | peak acceleration, trail (right) hand | 72.3 | 95.4 | 108.6 | +1.81 | 0.94 | 0.002 | 0.011 |
| 3 | peak acceleration, head | 17.5 | 25.3 | 24.6 | +1.42 | 0.83 | 0.03 | 0.009 |
| 4 | peak acceleration, upper neck | 16.8 | 24.1 | 24.4 | +1.42 | 0.83 | 0.03 | 0.008 |
| 5 | rotation range, trail hip (RightUpLeg, deg) | 55.8 | 66.7 | 78.0 | +1.41 | 0.86 | 0.015 | 0.016 |
| 6 | peak angular velocity, lead forearm rel. to upper arm | 57.9 | 70.6 | 28.1 | −1.41 | 0.16 | 0.02 | 0.014 |
| 7 | peak acceleration, lead (left) hand | 64.6 | 83.3 | 97.2 | +1.40 | 0.84 | 0.02 | 0.055 |
| 8 | peak acceleration, neck | 16.3 | 22.8 | 23.5 | +1.37 | 0.81 | 0.04 | 0.010 |
| 9 | head-IMU peak gyro (Pupil Invisible, deg/s) | 287 | 366 | 405 | +1.35 | 0.92 | 0.003 | 0.009 |
| 10 | peak acceleration, lead upper arm | 22.7 | 27.2 | 32.4 | +1.25 | 0.81 | 0.04 | 0.026 |
| 12 | rotation range, trail upper arm (deg) | 114.7 | 109.2 | 92.1 | −1.17 | 0.17 | 0.03 | 0.11 |
| 15 | rotation range, lead wrist (LeftHand, deg) | 113.0 | 83.9 | 77.8 | −1.08 | 0.19 | 0.04 | 0.016 |
| 21 | rotation range, pelvis (Hips, deg) | 93.0 | 107.3 | 122.3 | +0.99 | 0.80 | 0.05 | — |

Reading:
* **Speed:** pros accelerate the hands and arms much harder, about 1.5–1.7× the
  peak acceleration at the trail forearm and hand. That is the clearest separation.
* **Rotation:** pros turn the pelvis and trail hip more, and their heads rotate faster.
* **Lead arm and wrist:** pros keep these quieter. Lead-forearm angular velocity
  relative to the upper arm is half that of beginners, and the lead wrist and
  trail upper arm rotate less. This matches a straight lead arm and fewer wasted
  wrist motions.
* **Trend:** for most variables the Intermediate group falls between Beginner
  and Professional.
* **Weak separators:** plantar pressure and centre of pressure (|g| ≤ 0.55, raw
  uncalibrated 12-bit ADC at about 10 Hz) and gaze dispersion (|g| ≤ 0.7) separate
  weakly.

Caveats:
* **Small sample:** with 8 golfers per group, no single feature survives a 5%
  false-discovery rate (best q = 0.052). The ranking is descriptive.
* **Height:** pros are taller (median 176 vs 163.5 cm). The group effect stays
  at p < 0.05 after adjusting for height for 12 of the top 15 features
  (right-hand column).
* **Derived streams:** the motion-capture acceleration and angular velocity are
  computed from positions and quaternions, not raw IMU readings (paper §Data
  Records).
* **Dataset pitfall:** "root-relative position" is the global capture-frame
  position (the Hips row equals the hips' global position). Golfers stood at
  different spots: P20 is about 1.4 m from the origin. Position magnitudes
  therefore measure where the golfer stood, not their skill. My first pass made
  this mistake (it gave 43 vs 121 cm "hip-to-foot"), and the agent's ranking
  below did too. `golf_truth.py` subtracts the hips; since 2026-11-27 it also
  removes the facing direction (next section).

### Facing-direction normalization (added 2026-11-27)

Subtracting the hips removes where a golfer stood, but not which way they faced.
That matters for any direction-specific feature, such as sway toward the target.
`facing_frame()` in `golf_truth.py` puts each swing in the golfer's own frame:
* **Heading:** the horizontal hip line (left hip minus right hip), averaged over the
  first 0.25 s (address). All 24 golfers swing right-handed, so it points at the target.
* **Per swing, not per golfer:** the suit is inertial (Perception Neuron), and its
  global yaw drifts. The hip-line heading varies 20–30° between swings for P07, P16
  and P23 (2–10° for most others), so a per-golfer heading would leave that error in.
* **Axes:** toward the target, up (the capture frame is y-up), and toward the ball
  (target × up). The origin is the hips' address position.
* **Checked:** at address the head sits about 48 cm up and 49 cm toward the ball, with
  the left foot on the target side, for P02, P20 and P23. P07 (a beginner) stands with
  the head shifted toward the target.
* **Rejected alternative:** the foot line. The two foot joints are 17–41 cm apart, so
  their heading is noisier (sd up to 56°).

Nine features were added: hip and head range along each axis, hip slide toward and
sway away from the target, and head dip. The existing 85 features are unchanged
(max |Δg| = 0). With 94 features the best BH q is 0.058 (was 0.052).

| # | new feature | Beg. | Int. | Pro | g | AUC | p | p, adj. height |
|---|---|---|---|---|---|---|---|---|
| 3 | head range, up (cm) | 17.0 | 25.1 | 24.9 | +1.61 | 0.88 | 0.010 | 0.022 |
| 26 | hips range, up (cm) | 5.1 | 6.3 | 6.8 | +0.94 | 0.77 | 0.08 | 0.10 |
| 30 | head range, toward the ball (cm) | 22.8 | 26.0 | 30.1 | +0.91 | 0.88 | 0.010 | 0.10 |
| 51 | head range, toward the target (cm) | 28.9 | 28.8 | 31.8 | +0.56 | 0.61 | 0.51 | 0.34 |
| 54 | hips slide toward the target (cm) | 6.7 | 11.8 | 11.0 | +0.52 | 0.66 | 0.33 | 0.31 |
| 69 | head dip below address (cm) | 6.9 | 4.2 | 8.3 | +0.33 | 0.61 | 0.51 | 0.75 |
| 77 | hips range, toward the target (cm) | 18.7 | 23.2 | 18.2 | +0.22 | 0.59 | 0.57 | 0.46 |

Reading:
* **Lateral motion does not separate the groups.** Hip sway and head movement toward
  the target are about the same for beginners and pros.
* **Head range, up (#3) is not a clean skill signal.** It is mostly how far the golfer
  straightens up by the end of the recording: the head rises 17.9 cm above address
  for pros against 8.6 cm for beginners. But beginners' heads are still rising at 98 %
  of the recording (pros peak at 90 %), and pros' recordings are longer (4.35 s vs
  3.40 s). So many beginner recordings probably stop before the finish, and this
  feature mixes finish posture with the recording window.

### Ranking without cut-short recordings (added 2026-11-27)

`golf_truth.py --complete-only` writes `results/ranking_complete.csv`, ranked only on
recordings that reach a settled finish:
* **Rule:** the recording runs at least 1.0 s after the strike (peak trail-hand
  acceleration), and the body has settled by the end (mean joint speed over the last
  0.25 s at most 10 % of the swing's peak).
* **Same rule for every group:** filtering only beginners would bias the comparison.

It keeps 201/401 beginner, 215/323 intermediate and 651/833 pro swings:
* **Five golfers lose (almost) everything.** P01, P07 and P08 (beginners), P10
  (intermediate) and P18 (pro) keep fewer than 5 swings each. Their recordings stop
  0.6–0.8 s after the strike with the body still moving: P18's head is still rising at
  15 cm/s.
* **What is left:** 5 beginners (P02–P06) against 7 pros (P17, P19–P24).
* **Two changes are mixed together:** cut-short swings are gone, and so are those five
  golfers.

| complete-only # | all swings # | feature | g, all | g, complete | AUC | p | p, adj. height |
|---|---|---|---|---|---|---|---|
| 1 | 1 | peak acceleration, trail forearm | +2.20 | +2.45 | 1.00 | 0.003 | 0.020 |
| 2 | 2 | peak acceleration, trail hand | +1.81 | +1.56 | 0.91 | 0.018 | 0.17 |
| 3 | 8 | peak acceleration, lead hand | +1.40 | +1.50 | 0.89 | 0.030 | 0.22 |
| 4 | 6 | rotation range, trail hip | +1.41 | +1.34 | 0.89 | 0.030 | 0.019 |
| 5 | 16 | rotation range, lead wrist | −1.08 | −1.23 | 0.14 | 0.048 | 0.043 |
| 6 | 3 | head range, up | +1.61 | +1.22 | 0.83 | 0.073 | 0.21 |
| 12 | 22 | rotation range, pelvis | +0.99 | +1.10 | 0.80 | 0.11 | 0.008 |
| 14 | 4 | peak acceleration, head | +1.42 | +1.07 | 0.71 | 0.27 | 0.053 |

Reading:
* **Robust:** trail-forearm acceleration, trail-hip and pelvis rotation, and a quieter
  lead wrist keep their effect, and still separate the groups after adjusting for height.
  Trail-forearm acceleration separates the 12 golfers perfectly (AUC 1.00).
* **Weakened:** head range up (g 1.61 → 1.22, p 0.073), so part of it was the recording
  window. Peak head and neck acceleration also drop (#4 → #14). Those effects leaned on
  the excluded beginners.
* **Smaller sample:** 12 golfers instead of 16, so the best BH q is 0.235 (0.058 with
  all swings). The ranking is more descriptive still.

## 3. With vs without clio-core

| step | without clio-core | with clio-core CAE |
|---|---|---|
| get the sensor data | download all 33 participant zips (49 GB; measured 82 MB/s → ~10 min) and unzip (measured 8.5 s per 0.7 GB → ~10 min); ~100 GB disk | `clio-cae golf_omni.yaml` cold: 190 s, **0.94 GB received**, 1.5 GB staged (only the needed members); rerun with warm cache: 36–64 s |
| put in memory store | — | 40,486 tags (incl. ingest above) |
| read 17,127 datasets (785 MB) for the analysis | h5py, one open per file, files on RAM-backed tmpfs: **7.1–7.3 s** | `cte_golf.read_array` from CTE: **2.9–3.2 s** (2.4× faster) |
| feature computation | 2.2 s | 2.2 s (bit-identical features from both sources) |

The read speed-up comes mostly from CAE storing decoded arrays: the HDF5 files are
gzip-compressed, and h5py decompresses them on every read. The acquisition saving
comes from range-reading only the needed zip members: 52× fewer bytes than the
full archives.

## 4. CLIO agent on the OMNI data (ALCF beam-1.0-fp8)

The agent got `golf_omni.yaml`, the reader `cte_golf.py`, the paper text and the
question. Its workspace was outside the repo, and the data had to be read from
clio-core (`bench/golf/prompt.txt`, `run_agent_golf.sh`).

| try | result |
|---|---|
| 1 | **invalid**: the golf runtime had died before the agent started (log ends 19:54, run 20:02; cause unknown). The agent could not connect, then searched the home directory, stopped and restarted clio runtimes, and `chmod +x` two scripts in the clio-core worktree (reverted). It ran out of output tokens after 37 min with no answer. Fixes: runtime detached with `setsid`, a pre-flight tag-count check, and a prompt line forbidding runtime restarts. |
| 2 | failed in 6 s: "model replied without a tool call" |
| 3 | **partial**: 47 tool calls, 13.6 min. It wrote `analyze_golf.py` (reads only through `cte_golf`, so it really used clio-core), a 4,128-row ranking and the read time (3.33 s, which matches mine). It hit the output-length limit (`finish_reason=length`, cap 32k) before writing `answer.md`. |

How try 3's ranking compares with the ground truth:
* **Agrees:**
  * the agent's #2 (trail-forearm acceleration range) is my #1;
  * its lead-hand orientation (#1, a quaternion component) matches my lead-wrist
    rotation range;
  * its head acceleration features match my head-gyro separation.
* **Wrong:**
  * it ranks raw per-channel statistics (quaternion components, single axes), so
    its "variables" are hard to interpret;
  * #14–16 are unit-quaternion norms differing at the 1e-11 level, which is
    rounding noise;
  * #17, #18, #23 and #24 are root-relative position magnitudes, which reflect
    where the golfer stood rather than skill.
* **No write-up:** it never produced a written answer.

### Other ALCF models (2026-11-23)

The runtime was restarted on the rebased build (PR iowarp/clio-core#1429). The
golf runtime had again died silently, at 16:03. The data was re-ingested from
the OMNI file in 35 s, giving the same 40,486 tags, and every run passed the
pre-flight check.

| model | try | result |
|---|---|---|
| mistral-large-4 | 1 | **failed after 2 min**: ALCF gateway `HTTP 502 Bad Gateway` after 2 tool calls (infrastructure, not the model) |
| mistral-large-4 | 2 | **failed after 17 min**: 9 tool calls. It read the paper and both metadata CSVs **from clio-core**, then at step 7 replied `[["next_thought"]]`, which CLIO's DSPy JSONAdapter cannot parse (`model_reply_unparseable`). No deliverable. |
| nemotron-35-super-vl | 1 | **no deliverable in 2.3 min**: 44 tool calls. It read the CSVs from clio-core and wrote five `explore_*.py` scripts. Its later `shell_bash` calls repeatedly failed CLIO's argument validation, so it could not run them, and it ended with a plan instead of `analyze_golf.py` and `answer.md`. |

Runs: `runs/fail_agent_mistral_try1_alcf502`, `runs/fail_agent_mistral_try2_adapter_parse`,
`runs/agent_nemotron-35-super-vl`.

Verdict: the data path works end to end. The OMNI file feeds CAE from Dataverse,
CTE holds the data, and every model that got past its first steps read it from
CTE. None of the three ALCF models finished the analysis:
* beam got furthest (3 tries; a 4,128-row ranking mixing real effects with two
  artifacts, but no written answer).
* mistral and nemotron failed on CLIO's tool-call formatting (a reply the JSON
  adapter can't parse; `shell_bash` arguments that fail validation), as on the
  QICK question.

## Files

* clio-core: `context-assimilation-engine/core/{include/clio_cae/core/factory/dataverse*.h, src/factory/dataverse_*.cc, src/factory/tools/dataverse_tool_main.cc}`, `test/unit/dataverse/`
* `bench/golf/golf_omni.yaml`: the OMNI file. `clio_golf.yaml` is the runtime config (port 9513). `clio_env.sh.example` sets the environment.
* `bench/golf/cte_golf.py`: reads tags back as numpy arrays and CSVs.
* `bench/golf/golf_truth.py`: ground truth and timing (`--source cte|h5`). `results/` holds the ranking, timings and per-swing features. Since 2026-11-27, `ranking.csv` comes from `--source h5` and includes the facing-frame features. `swing_features_cte.csv` is the original CTE run, without them. Its 85 shared features give the same g to the last digit. Rerun `--source h5` (output git-ignored) or `--source cte` for the new per-swing columns. `ranking_complete.csv` comes from `--complete-only`.
* `bench/golf/run_agent_golf.sh`, `prompt.txt`, `runs/`: the agent runs, cold staging log and clio-cae log.

[1] MultiSenseGolf, Scientific Data (2026), https://doi.org/10.1038/s41597-026-08648-4 ; data: https://doi.org/10.7910/DVN/LCCLLW (CC0)

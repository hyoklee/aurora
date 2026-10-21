# Terra Fusion congruence test on Aurora: S3 → CAE → CTE tiers → GPU regrid

Run on **Aurora**, 2026-10-09, job 8917505, one node. Repeats the ares
collection-wide congruence study
([~/ares/doc parts 4 and 17](../../ares/doc/20260930_claude_terra_fusion_discrepancy_17_all_granules.md))
with the granules fetched from `s3://terrafusiondatasampler` by CAE instead of
read from local copies.

## Headline

* **The result reproduces ares exactly.** 315 blocks from 7 granules; for every
  block C5, Kendall's W, cell count, weakest sensor and band-31 contrast agree
  with `~/ares/data` (max |ΔC5| = 5.8e-7, on O10670; 0 everywhere else). Same
  answer: **O11602 block 46, C5 = 0.355**. Same three granules excluded, for
  the same reasons.
* **48 min end to end** for 304 GB, ingest included. The ingest is 47 min of
  it (108 MB/s over 10 objects). Each granule's analysis starts as soon as it
  lands and runs alongside the next download, so it adds about 3 s to the end.
* **S3 striping: 3.4× faster ingest.** On a compute node, through the ALCF
  proxy, one stream reached 33 MB/s. Sixteen concurrent 32 MiB ranged GETs
  reached 111 MB/s, which is about where the proxy path plateaus. Without
  striping this run would take 2.6 h instead of 47 min.
* **The GPU regrid matches pytaf bit for bit.** On O10204 (check mode),
  299,948 regridded cells were identical and 0 differed. Each granule ran on
  its own PVC tile, and the regrid took 1.9–16.6 s per granule. A later
  clean-pass run measured it **4.2–6.7× faster than pytaf** on Aurora's CPU
  (1.6–1.9× per granule), against 15.6–21.7× for an A100 on Polaris.
* **Tier choice was set by measurement.** All data went to DRAM. Ingest into
  DRAM measured 104–111 MB/s against 67–80 MB/s into GPU HBM, and the CPU-side
  reader gets nothing from data held in device memory. So HBM is the overflow
  tier and Lustre the last resort; neither held any data in this run.

## The data path

```
s3://terrafusiondatasampler ──CAE (16 ranged GETs)──▶ cfs::/terra_fusion/<granule>.h5
        CTE tiers: DRAM 320 GB (score 1.0) │ GPU HBM 48 GB (0.6) │ Lustre 400 GB (0.2)
                         clio_cte_fuse mount ──▶ unchanged netCDF4 pipeline (ares bin/)
                                                    └─ nearest-neighbour regrid on a PVC tile
```

Each granule goes through the same four stages ares ran: `tf_aster_blocks` →
`tf_sources` → `tf_congruence` → contrast. The configuration is the matched one
from ares parts 11–12: MODIS band 31 and CERES WN_Radiance. The scripts are in
[`src/tf/`](../src/tf), copied from `~/ares/bin` with these changes:

| file | change |
| --- | --- |
| `tf_congruence.py` | `TF_REGRID=gpu\|cpu\|check`: nearest-neighbour search in PyTorch XPU, float64, same great-circle distance and radius as pytaf |
| `tf_contrast.py` | **new**: band-31 BT mean/sd per block (the 2 K floor). ares carries the values but not the script; this one reproduces them exactly |
| `tf_rank_all.py` | attaches the contrast and writes the floor-filtered ranking |
| `tf_compare_ares.py` | **new**: block-by-block comparison with `~/ares/data` |
| `tf_granule.sh` | one granule, four stages, timed |

## Software changes (clio-core, branch `aurora-tf-congruence` in ~/clio-core.dev.wrk)

On top of PR #1258 (anonymous S3 and proxy support). Committed locally, not pushed.

1. **Striped S3 import** (`s3_file_assimilator.cc`). An object larger than one
   part is fetched in windows. `CAE_S3_STREAMS` threads (default 8) pull parts
   from a shared counter, and each part is one ranged GET of `CAE_S3_PART_MB`
   (default 16) on its own pooled connection. Smaller objects keep the
   single-stream path.
2. **`cfs::<path>` destinations.** The object is stored as a clio-fs **file**:
   it is created through the filesystem chimod, each 1 MiB chunk is written as
   a page blob, and closing it publishes the size. This lets an unmodified
   HDF5/netCDF reader read S3 data held in CTE through `clio_cte_fuse`. An
   `iowarp::` tag of `chunk_<n>` blobs cannot be opened by any file API.
3. **Fix: wait for the clio-fs pool.** `clio_run status` reports ready before
   compose has created every pool, and the filesystem pool is created last. A
   Mkdir sent in that window never completed, and the import hung (seen in a
   benchmark job; reproduced 1/1 on the login node, fixed 3/3).
4. **`cte_tiers`**: prints per-target score and free space. This is how
   placement was checked. GetTargetInfo's bytes-written/read counters read 0
   on targets that hold data, so the tool reports free space only.
5. Test: `cae_s3_assim` now checks the imported content chunk by chunk, not
   just the tag size, and a striped variant was added. **Not run here**: it
   needs a MinIO `S3_ENDPOINT`. The striped and cfs:: paths were instead
   byte-compared against real S3 on every file of every run (below).

Build: [`bin/build_tf_aurora.pbs`](../bin/build_tf_aurora.pbs) (SYCL + S3 +
FUSE, oneAPI 26.26).

## Results

### Ingest

| granule | GB | s | MB/s |
| --- | --- | --- | --- |
| O10903 | 51.0 | 452 | 112 |
| O11369 | 45.3 | 392 | 115 |
| O10670 | 45.1 | 391 | 115 |
| O11602 | 39.6 | 342 | 115 |
| O10204 | 32.9 | 425 | 77 |
| O11136 | 28.3 | 248 | 114 |
| O11835 | 23.7 | 206 | 114 |
| O10437 | 17.4 | 166 | 104 |
| O1117 | 16.8 | 147 | 114 |
| O12068 | 4.2 | 38 | 109 |
| **total** | **304.3** | **2807** | **108** |

O10204's 77 MB/s is a dip on the proxy, not a property of the object; it is
the fifth of ten transfers. Every file was checked against S3 before it was
analysed: the size, and the first, last and three random 1 MiB ranges compared
byte for byte. **10/10 matched.**

### Striping and tiers (tf_ingest_bench.pbs, one 4.17 GB object, compute node)

| tier | streams × part | s | MB/s |
| --- | --- | --- | --- |
| DRAM | 1 × 16 MiB (old behaviour) | 125.2 | **33** |
| DRAM | 8 × 16 MiB | 39.9 | 104 |
| DRAM | 16 × 16 MiB | 46.0 | 91 |
| DRAM | 16 × 32 MiB | 37.6 | **111** |
| DRAM | 32 × 16 MiB | 37.6 | 111 |
| GPU HBM | 8 × 16 MiB | 52.4 | 80 |
| GPU HBM | 16 × 16 MiB | 62.1 | 67 |

Each row is a single run, so read the trend rather than single rows. More than
eight streams buys little: the proxy path levels off near 110 MB/s, below the
186 MB/s eight parallel `curl`s reached from a login node.

### Two traps in the default CTE configuration

* **The periodic flush moved half the data to Lustre.** By default CTE copies
  volatile-tier data to the first persistent tier every 10 s
  (`flush_data_period_ms`). In an early smoke run, 12.3 of 23.7 GB ended up in
  the Lustre file instead of DRAM or HBM. For a cache of S3 objects that is
  wasted Lustre bandwidth, so the run sets `flush_data_period_ms: 0`.
* **Placement follows scores, not the bandwidth probe.** The bdev start-up
  probe rated Lustre writes at 1,111 MB/s, DRAM at 476 MB/s and HBM at
  205 MB/s, but `max_bw` ranks by configured score first. So tier order is
  whatever the YAML says, which is the intended behaviour.

### Analysis

| granule | blocks | sources s | congruence s | GPU regrid s |
| --- | --- | --- | --- | --- |
| O10903 | 75 | 9 | 20 | 7.7 |
| O11369 | 66 | 10 | 17 | 6.5 |
| O10670 | 52 | 12 | 34 | 16.6 |
| O11602 | 49 | 9 | 16 | 6.1 |
| O10204 | 32 | 9 | 31 | 26.4 (check: pytaf + GPU) |
| O11136 | 27 | 34 | 43 | 3.5 |
| O11835 | 14 | 156 | 43 | 1.9 |

All of these read the granule through FUSE from DRAM. O11136 and O11835 ran
while the next downloads were writing into the same runtime and were slowed by
up to 17× (O11835's `sources` took 156 s here and 9 s in the smoke run). That
contention is real but cheap: the last analysis still finished 3 s after the
last download. No CPU-only timing was taken in this run; the GPU-vs-pytaf
speedup was measured later (see [PVC vs pytaf](#pvc-vs-pytaf-regrid-follow-up-2026-10-20-job-8948568-1-node)).
The GPU is there to keep the regrid off the host while the host decompresses,
and ares part 18 found decompression to be the actual floor.

### The ranking (identical to ares part 17's 315-block re-run)

| | |
| --- | --- |
| blocks | 315 from 7 granules; 268 above the 2.0 K contrast floor |
| C5 | min 0.355, median 0.699, max 0.920 |
| C5 vs Kendall's W | r = 0.9892 |
| weakest-sensor tally | MISR 140, MOPITT 109, CERES 65, ASTER 1, MODIS 0 |
| **answer** | **O11602 blk 46**, 25.84°S–25.17°S, C5 = 0.355, BT₃₁ sd 3.29 K, MISR weakest |

The top 12 after the floor is ares' list in the same order (data/tf/all_granule_ranking_filtered.json).

| orbit | ares blocks | Aurora | max \|ΔC5\| | max \|ΔW\| | ncell equal | weakest equal | max \|ΔBT sd\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O10204 | 32 | 32 | 0 | 0 | 32 | 32 | 0 |
| O10670 | 52 | 52 | 5.8e-7 | 7.1e-7 | 52 | 52 | 0 |
| O10903 | 75 | 75 | 0 | 0 | 75 | 75 | 0 |
| O11136 | 27 | 27 | 0 | 0 | 27 | 27 | 0 |
| O11369 | 66 | 66 | 0 | 0 | 66 | 66 | 0 |
| O11602 | 49 | 49 | 0 | 0 | 49 | 49 | 0 |
| O11835 | 14 | 14 | 0 | 0 | 14 | 14 | 0 |

Excluded, as on ares: O1117 and O12068 have no ASTER group, and O10437's
two-granule strip has no points from a sparse sensor. The script names MISR
for O10437, where ares' prose said CERES/MOPITT. It is the same script on the
same bytes, so the exclusion is the same.

## All of P108 (follow-up, 2026-10-16, job 8934328, 8 nodes)

The same pipeline over every P108 granule
([`omni/terra_fusion_p108.omni.yaml`](../omni/terra_fusion_p108.omni.yaml), 361 granules, 11.75 TB),
split over 8 nodes by [`pbs/tf_p108.pbs`](../pbs/tf_p108.pbs). Each granule is
unlinked from clio-fs once it has been analysed. The 313 granules left after
earlier runs (10.0 TB) were all imported and verified in 5 h 01 min, at 70–89 MB/s
per node. That needed the clio-core S3 resume fix
([iowarp/clio-core#1296](https://github.com/iowarp/clio-core/pull/1296)); without
it, an earlier attempt failed 156 of 199 imports.

| | |
| --- | --- |
| granules with results | 291 of 361. The other 70 lack a sensor group (ASTER 24, MOPITT 20, MISR 7, MODIS 2), lack ASTER TIR in one ASTER granule (12), or have no MISR points in the strip (2, plus O10437) |
| blocks | 13,111; 10,116 above the 2.0 K contrast floor |
| C5 | min 0.238, median 0.670, max 0.966; r = 0.989 with Kendall's W |
| weakest-sensor tally | MISR 6,776, MOPITT 4,806, CERES 1,247, ASTER 228, MODIS 54 |
| lowest above the floor | O26281 blk 45, 27.47°S–26.80°S, C5 = 0.332, BT₃₁ sd 4.33 K |

Below are the most and least congruent blocks of the whole path, drawn by
[`src/tf/tf_block_map.py`](../src/tf/tf_block_map.py). It rebuilds each block's
fields exactly as `tf_congruence.py` does, and the recomputed C5 matches the
ranking to 4 decimals. Each panel shows one sensor's rank z-score on the
cells all sensors cover, sign-aligned to the common mode, on one shared scale.
Congruent sensors therefore show the same pattern.

**Most congruent: O33737 block 13, C5 = 0.966.** Off Shizuoka, Japan (Suruga Bay and the Izu Peninsula)
(34.3–35.0°N, 138.1–139.1°E). ASTER, MODIS, MISR and MOPITT all show the same
cold-north / warm-south split (BT₃₁ 244 K, sd 13.7 K, likely a cloud edge).
The block has no CERES points, so C5 here is over 4 sensors.

![Most congruent block, O33737 block 13](img/most_congruent_O33737_blk13.png)

**Least congruent: O18825 block 0, C5 = 0.238.** Canadian Arctic
(81.6–82.5°N, 110.8–104.6°W). BT₃₁ sd is 0.19 K, an almost uniform scene, so
ASTER and MISR show mostly noise. Only MODIS has a gradient (loading 0.68);
MOPITT is the weakest (0.15). The block falls below the 2 K floor, so the low
score reflects missing contrast, not sensors disagreeing about a real signal.

![Least congruent block, O18825 block 0](img/least_congruent_O18825_blk0.png)

## PVC vs pytaf regrid (follow-up, 2026-10-20, job 8948568, 1 node)

Polaris measured an A100 regrid 15.6–21.7× faster than pytaf on these two
granules
([~/polaris/doc](../../polaris/doc/20261016_claude_terra_fusion_congruence_polaris.md)).
This run repeats that measurement on Aurora with `pbs/tf_congruence.pbs
CLEAN_PASSES=2`. Both granules were ingested into DRAM (`HBM_GB=0`, as in
Polaris run A, 83.5 GB at 108 MB/s). After the overlapped pass, each granule
ran alone on PVC tile 1: two passes with the GPU regrid, then one with pytaf
(1 OpenMP thread). Timings: [`data/tf/regrid_speedup_8948568.txt`](../data/tf/regrid_speedup_8948568.txt).

| | O33737 | O18825 |
| --- | --- | --- |
| Regrid: pytaf, 1 thread (Xeon Max) | 26.0 s | 77.9 s |
| Regrid: PVC tile (PyTorch XPU, float64) | 3.89 s | 18.36 s |
| **Regrid speedup** | **6.7×** | **4.2×** |
| Granule: pytaf regrid | 51 s | 120 s |
| Granule: PVC regrid | 32 s | 64 s |
| **End-to-end speedup** | **1.6×** | **1.9×** |

C5 was identical in every pass, GPU and CPU: O33737 blk 13 = 0.9661 and
O18825 blk 0 = 0.2376, weakest sensor MOPITT, as on Polaris. The two GPU
passes agree within 0.1%.

Against Polaris:

| | Polaris | Aurora |
| --- | --- | --- |
| pytaf regrid, O33737 / O18825 | 72.9 / 219.1 s (EPYC 7543P) | 26.0 / 77.9 s (Xeon Max) |
| GPU regrid, O33737 / O18825 | 3.36 / 14.1 s (A100) | 3.89 / 18.4 s (one PVC tile) |
| Regrid speedup | 15.6–21.7× | 4.2–6.7× |
| End-to-end speedup | 2.8–3.7× | 1.6–1.9× |
| Granule with GPU regrid | 38 / 75 s | 32 / 64 s |

* **The Intel GPU does speed up the regrid, but the gain is a third of
  Polaris's.** The CPU baseline accounts for most of the gap: pytaf ran 2.8×
  faster on Aurora than on Polaris. Whether that is the CPU or the pytaf build
  was not checked. One PVC tile was 15–30% slower than an A100.
* **Aurora is faster end to end**: 32 / 64 s per granule with the GPU regrid
  against 38 / 75 s on Polaris.
* **The CPU-side stages set the floor**, as on Polaris. With the regrid on
  the GPU, `congruence` on O18825 still takes 36 s, of which 18 s is regrid;
  `contrast` and `sources` (netCDF4 reads) do not change.
* The comparison is per tile. An Aurora node has 12 tiles against Polaris's 4
  A100s, and the P108 run used them in parallel; node throughput was not
  measured here.

## Caveats

* **O10670's 5.8e-7 is not explained.** Only O10204 ran in check mode, and
  there 0 of 299,948 cells differed. A tie between two equidistant sources that
  the GPU resolved differently from pytaf's block order would produce exactly a
  difference this size, but that was not traced.
* One run, one node. The ingest rate depends on the shared ALCF proxy (one
  transfer dipped to 77 MB/s).
* The benchmark cells are single runs.
* The HBM tier was measured for ingest only. A tier whose consumer is a GPU
  kernel (gpu_vector) is the case where HBM should win; this pipeline's
  consumer is netCDF on the CPU.
* `KEEP` was off, so the Lustre tier file was deleted at exit (project usage
  is back to 87 GB). The DRAM-tier copy is gone with the job; a rerun
  re-downloads.

## Reproducing

```sh
qsub bin/build_tf_aurora.pbs                    # ~10 min, debug queue
qsub pbs/tf_congruence.pbs                      # 48 min; capacity queue
qsub -q debug -l walltime=1:00:00 -v GRANULES=O11835,CHECK_ORBIT=O11835 pbs/tf_congruence.pbs   # 6 min smoke
qsub pbs/tf_ingest_bench.pbs                    # striping x tier sweep
qsub pbs/tf_p108.pbs                            # all of P108, 8 nodes, ~5 h
qsub -v OMNI=$PWD/omni/terra_fusion_p108.omni.yaml,GRANULES="O33737 O18825",HBM_GB=0,CLEAN_PASSES=2,SKIP_REPORT=1 pbs/tf_congruence.pbs   # PVC vs pytaf, 20 min
python src/tf/tf_block_map.py <work>/O33737 13 P108/TERRA_BF_L1B_O33737_20060422005935_F000_V001.h5 out.png
```

Python: frameworks/2026.1.0 plus a venv with netCDF4 (`/lus/flare/projects/IOWarp/hyoklee/venv-tf`),
and pytaf built with `setup.py.omp` into `/lus/flare/projects/IOWarp/hyoklee/pytaf`.
Results: [`data/tf/`](../data/tf).

# Golf swing video through clio-core CAE as HDF5: beginner vs pro (2026-11-27)

Request: add video ingest to CAE OMNI that stores video streams as HDF5 using the
FFMPEG HDF5 filter [1], then find ML methods that pull features out of those HDF5
files which separate beginners from professionals. Data: MultiSenseGolf [2]
(Harvard Dataverse doi:10.7910/DVN/LCCLLW), the same dataset as
`doc/20261122_golf_clio_cae_dataverse.md`.

**Short answer.**

* CAE can now turn every `.mp4` in a Dataverse transfer into a compressed HDF5
  video. 343 swing videos were ingested in 140 s, and stored at 59× (front camera)
  and 24× (head camera) smaller than the source MP4s.
* Hand-crafted motion features separate the groups best, especially the
  head-camera ones. Pros' heads move and rotate faster at the strike (Hedges g
  1.35 and 1.28), and their downswing is shorter, 0.48 s against 0.71 s
  (g = −1.25, p = 0.02).
* Golfer by golfer, the head-camera motion matches the head-mounted gyro in the
  sensor data (Spearman ρ = 0.89, p = 2×10⁻⁵). So the video measures real body
  motion.
* With 8 golfers per group, no feature survives a 5% false-discovery rate
  (best q = 0.17). The same was true of the sensor features (best q = 0.052).
* A pretrained video-action network does not beat a single still frame
  (golfer-level AUC 0.72 against 0.70, neither significant). Its separation, such
  as it is, comes from scene and appearance, not the swing.
* On a compute node, one Intel GPU Max 1550 tile runs the networks 3.1× faster
  than the node's CPU (16 frames at 112²). With 32 frames at 224² it is 3.9×
  faster, and the whole run 1.9× faster. Optical flow is faster on the CPU than
  through OpenCL, and the classifiers stay on the CPU. Ingest takes 115 s,
  against 140 s on the login node (§5).
* Re-ingesting gives slightly different HDF5 pixels (x264, about 1.2 grey
  levels). Head-camera features hold across three ingests; the embedding AUC
  moves between 0.69 and 0.80 (§5).

## 1. CAE: video → HDF5 in OMNI

This is new code in clio-core, committed locally as 7f85f1ed4 on branch
`feat/cae-web-zip-hdf5-tables` in `~/clio-core.dataverse`. It is **not pushed**.

```yaml
# bench/golf/golf_video_omni.yaml (excerpt)
transfers:
  - src: "dataverse://dataverse.harvard.edu/doi:10.7910/DVN/LCCLLW"
    dst: "iowarp::golfvideo"
    format: "hdf5"
    dataset_filter:
      include_patterns:
        - "P0[1-8].zip/Swing0[1-9]/Front_View/*.mp4"   # ... 18 patterns in all
    video:
      include: ["*.mp4"]   # members to transcode
      width: 320           # 0 = keep source size; height keeps aspect, both even
      codec: x264          # x264 | x265 | svtav1
      crf: 23
      preset: medium
```

* **What it does.** Each matching member is range-read from its remote zip and
  decoded with libav. It is scaled to grey luma (SWS_AREA) and written as:
  * `/frames`: uint8 `[T, H, W]`, chunked 512 frames deep, compressed with filter
    32030, the FFMPEG HDF5 filter [1]. The 11 filter values are encoder, decoder,
    width, height, depth, bit mode, preset, tune, crf, film grain and gpu.
  * `/time_s`: float64 `[T]`.
  * Attributes: source, codec, pixels, source size and fps.

  The HDF5 file is then stored as tag `golfvideo/<zip>/<SwingNN>/<file>.mp4.hdf5`,
  and the manifest lists it under that name.
* **Build.** `-DCLIO_CAE_ENABLE_VIDEO=ON` turns it on. It needs HDF5, Dataverse,
  and FFmpeg found through pkg-config. It is off by default.
* **Validation.** Without the option, a `video` block is an error (-22). The OMNI
  loader validates the block: map shape, known keys, codec, crf 0..63, and
  width ≥ 0.
* **Plugin.** The filter is loaded from `HDF5_PLUGIN_PATH`. Without it the
  transcode fails with a message that names that variable, and nothing is
  ingested silently.
* **`cae_video_hdf5`.** A command-line converter: `<video> <out.h5> [--width]
  [--codec] [--crf] [--preset] [--frames-per-chunk]`.
* **Tests.** The new `cae_video_hdf5_test` covers the filter values for each
  codec, a 12-frame round trip (mean error 7.9 grey levels after two lossy codecs)
  and bad input: 3/3 pass. The OMNI loader passes 35/35, including the 3 new
  `video` cases.

**The filter plugin.** The `h5ffmpeg` pip wheel cannot be used from C. Its filter
is a Python extension: it fails with `undefined symbol PyExc_ImportError` and
bundles its own HDF5 1.14. Instead, `bench/golf/build_h5ffmpeg_plugin.sh` builds
`libh5ffmpeg_shared.so` from the filter's C sources (commit 78f8dfa). It uses
conda-forge GPL FFmpeg (env `ffdev`, which has libx264) and the iowarp HDF5 2.1.0.

* **conda linking.** Linking anything against `ffdev` needs
  `-Wl,-rpath-link,$ffdev/lib`, or its glib resolves the system pcre2. clio-core
  was configured with `-DCMAKE_{EXE,SHARED,MODULE}_LINKER_FLAGS` set to that
  flag.
* **Environment.** `bench/golf/clio_video_env.sh.example` sets up a client and
  the runtime.

### Ingest numbers (login node aurora-uan-0012, runtime on port 9513)

| | Front_View/RGB_Video.mp4 | FPV_RGB.mp4 (head-worn) |
|---|---|---|
| source | mpeg4 640×480 @ 30 fps | 1088×1080 @ 20 fps |
| videos / frames | 192 / 23,525 | 151 / 12,137 |
| MP4 bytes → HDF5 bytes | 1,232 MB → 20.7 MB (**59×**) | 285 MB → 11.7 MB (**24×**) |
| stored size | 320×240 grey | 320×316 grey |

* **Totals.** 686 objects (343 HDF5 videos plus their timestamp CSVs) were
  ingested in **140 s**, including the transcode. That needed 1.52 GB staged
  through 1,454 HTTP range requests (40 s).
* **Fidelity.** Against an independent PyAV decode of the same MP4, PSNR is
  35.8 dB (front) and 44.1 dB (FPV).
* **Read-back.** Reading one video back from clio-core and decoding it to a numpy
  array takes about 0.05 s.

Not every swing has a head-camera video: P01 has 4, P05 has 3, and P20 and P21
have none.

## 2. ML methods for the HDF5 videos

`bench/golf/golf_video_ml.py --source cte` reads every video's HDF5 bytes from
clio-core (`cte_golf.manifest("golfvideo")`) and opens them with h5py. h5py
decodes them through the same filter, via the h5ffmpeg wheel.
`--source dir` reads a directory of `.mp4.hdf5` files instead. Four methods:

1. **Dense optical flow (Farnebäck), front camera.** Computed at 160 px for every
   frame pair. It gives motion energy and moving area per frame, and from those:
   * swing phases: onset; top of backswing (slowest moment before the strike);
     strike (peak motion);
   * backswing time, downswing time and their ratio (tempo);
   * peak and mean motion;
   * smoothness: jerk, and the number of motion bursts before the strike.

   Why this method: the swing is one fast, mostly 2-D motion against a fixed
   camera. Flow measures it directly, needs no training, and gives quantities a
   coach would recognise.
2. **Sparse feature tracking plus a similarity fit (Lucas–Kanade,
   `estimateAffinePartial2D`), head camera.** The background's apparent motion is
   the head's ego-motion. Peak and mean shift per second, peak roll rate and total
   path measure how still the head stays.
3. **Pretrained spatiotemporal CNN embeddings (R(2+1)D-18, Kinetics-400).** 16
   frames from onset to the end, at 112². A leave-one-golfer-out logistic
   regression (C = 0.05) is scored by golfer-level AUC with 1000 label
   permutations. This is the generic "learn the pattern" route.
4. **Confound control.** The same classifier on a ResNet-18 (ImageNet) embedding
   of the **first frame only**. This frame shows the golfer standing at address.
   If a still frame classifies as well as the video, the classifier is reading
   the scene or the person, not the swing.

Ranking statistics follow the sensor study:
* per-golfer medians, at least 3 swings per golfer and at least 4 golfers per
  group;
* Hedges g, Mann–Whitney p and AUC;
* Benjamini–Hochberg q over all 12 video features together.

Run time on the login node (CPU only): features 25 s with 8 processes,
embeddings 20 s, classifier permutations 169 s, 215 s in all. §5 has the
compute-node and GPU timings.

## 3. Results

Results are in `bench/golf/results/video_ranking.csv` (and
`video_swing_features.csv`, `video_ml.json`). Groups are 8 Beginners (P01–P08)
against 8 Professionals (P17–P24); the head-camera rows have 6 pros.

| # | feature | g | AUC | p | q (BH) | beginner | pro |
|---|---|---|---|---|---|---|---|
| 1 | fpv_peak_shift_per_s (frame widths/s) | 1.35 | 0.83 | 0.043 | 0.17 | 3.52 | 5.38 |
| 2 | fpv_peak_roll_deg_s | 1.28 | 0.83 | 0.043 | 0.17 | 94.9 | 140.0 |
| 3 | vid_downswing_s | −1.25 | 0.15 | 0.020 | 0.17 | 0.71 | 0.48 |
| 4 | vid_tempo_ratio (back/down) | 1.17 | 0.77 | 0.083 | 0.20 | 1.09 | 1.71 |
| 5 | vid_peak_moving_area | 0.89 | 0.73 | 0.13 | 0.22 | 0.177 | 0.189 |
| 6 | fpv_total_path | 0.89 | 0.71 | 0.23 | 0.31 | 2.10 | 2.29 |
| 7 | vid_motion_jerk | −0.82 | 0.23 | 0.083 | 0.20 | 0.134 | 0.115 |
| 8 | vid_peak_motion | 0.77 | 0.73 | 0.13 | 0.22 | 15.1 | 16.9 |
| 9 | vid_motion_peaks_before_strike | −0.77 | 0.30 | 0.17 | 0.26 | 5 | 3.5 |
| 10 | vid_backswing_s | 0.59 | 0.57 | 0.67 | 0.81 | 0.73 | 0.68 |

What the features show:

* **Pros swing faster through the ball and keep a longer backswing relative to
  the downswing.** Their downswing is shorter, the tempo ratio is higher, and
  peak motion is larger. Pros also make fewer separate motion bursts and have
  lower jerk, so their swing is smoother. This is the textbook pattern.
* **Tempo is underestimated.** The tempo ratio (1.7 for pros) is below the
  roughly 3:1 usually quoted for pros. At 30 fps, the motion minimum only
  approximates the top of the backswing, so treat the ratio as a ranking, not an
  absolute value.
* **Pros' heads turn faster.** This contradicts the "keep your head still" idea,
  but it agrees with the sensor study: there `head_peak_gyro` was 417 against
  340 deg/s (g = 1.12). Pros release the head into the follow-through.
* **Cross-modal check.** Golfer by golfer (14 golfers with both), the head-camera
  features rank-correlate with the head IMU's peak gyro:
  * `fpv_peak_shift_per_s`: ρ = 0.89, p = 2.0×10⁻⁵
  * `fpv_peak_roll_deg_s`: ρ = 0.73, p = 0.003

  To reproduce, take per-golfer medians of `video_swing_features.csv` (view
  `fpv`) against `swing_features_cte.csv` column `head_peak_gyro`. The camera is
  an independent measurement of the same motion.
* **Classifiers** (login-node ingest).

  | input | golfer AUC | permutation p |
  |---|---|---|
  | R(2+1)D video embedding | 0.719 | 0.064 |
  | ResNet-18 first-frame still | 0.703 | 0.082 |

  Across three separate ingests (§5), the video AUC ranges from 0.69 to 0.80,
  and the still-frame AUC from 0.66 to 0.70. Only one ingest reached p < 0.05.

  The still frame does as well as the video. The golfers both models score as
  most "pro-like" are the same visually distinctive ones: P03 (a beginner)
  scores 0.94/1.00, and P22 scores 0.95/0.96. A generic action network
  fine-tuned on 16 golfers would learn who is standing in which lighting, not
  skill. A much larger or deliberately matched cohort would be needed before
  using one.

## 4. Caveats

* **Small sample.** N = 8 against 8 golfers (6 pros with head video). Nothing
  passes FDR 5%, and the ranking is descriptive. The direction of the top
  features agrees with the sensor study and with coaching knowledge.
* **Grey only.** The FFMPEG HDF5 filter stores one channel (8/10/12-bit). Colour
  is dropped, which loses nothing these motion features need.
* **Lossy.** The video is x264 at crf 23. That is fine for motion; check fidelity
  before using it for pixel-exact work. A lower crf or `x265` can be set in the
  OMNI block.
* **Scene confound.** Framing and lighting differ slightly per golfer (P20 is
  shifted; P19 and P22 are darker). Motion features are only weakly affected; the
  learned embeddings are strongly affected, as the still-frame control shows.
* **Ingest is not bit-reproducible.** Re-ingesting the same MP4s gives slightly
  different HDF5 pixels: x264 at crf 23, about 1.2 grey levels apart. This
  moves the fragile results (phase timing, learned embeddings) but not the
  head-camera features. See §5.
* **Run gotcha.** A clio-core client forked after it connects hangs on the
  runtime's shared memory: the first run's 16 workers waited 64 minutes. The
  script now reads all HDF5 bytes in the parent process and hands bytes to the
  workers.

## 5. Compute node with GPUs: performance (2026-11-27)

Two PBS jobs ran `pbs/golf_video_gpu.pbs` on one Aurora compute node each
(debug queue, IOWarp allocation): 9136201 and 9136314. Each node has 2 Xeon Max
9470C (208 threads) and 6 Intel Data Center GPU Max 1550. Each job:
1. starts its own clio-core runtime (the video build);
2. ingests `golf_video_omni.yaml` from Dataverse through the ALCF proxy;
3. runs `golf_video_ml.py --source cte` several ways.

The login node is aurora-uan-0012, shared, 104 threads, no GPU. All timings are
in `bench/golf/results/video_perf.csv`.

**GPU code.** `golf_video_ml.py` has two GPU options:
* `--device xpu`: PyTorch 2.15 XPU wheels run both networks in batches of 16 on
  one GPU tile.
* `--flow ocl`: OpenCV runs Farnebäck optical flow through OpenCL. Workers are
  spawned, not forked, and spread over the 6 GPUs.

**Setup.** The XPU wheels ship a newer SYCL/UR runtime than `/opt/aurora`'s
oneAPI. Python needs the venv's `lib/` first on `LD_LIBRARY_PATH`; otherwise
`import torch` fails with `undefined symbol: urGraphGetIdExp`.

### Ingest

| | login node | compute node (2 jobs) |
|---|---|---|
| ingest (Dataverse → HDF5 → clio-core, 686 objects) | 140 s | 115.5 s, 115.0 s |
| of which staging (1.52 GB, 1,454 requests) | 40 s | 39 s |

The transcode is CPU work in the runtime, and the compute node's CPU makes it
about 25 s faster. The GPUs are not used for ingest.

### Analysis stages (seconds)

| stage | login CPU | compute CPU | compute GPU | GPU speed-up |
|---|---|---|---|---|
| read 343 videos from clio-core | 0.08 | 0.10 | 0.10 | n/a |
| optical flow, 8 workers | 24.8 | 16.1–16.6 | OpenCL: 20.3 | **0.8×** (slower) |
| optical flow, 32 workers | n/a | 10.1–10.8 | n/a | n/a |
| R(2+1)D + ResNet-18 model, 16 frames at 112² | 9.8 | 6.97–7.10 | 2.29 | **3.1×** |
| same, 32 frames at 224² | 128.0 | 62.4–62.5 | 15.8 | **3.9×** |
| classifiers, 100 permutations | 18.5 | 13.4–13.8 | same (CPU) | n/a |
| **total, standard workload** (100 perms) | ≈ 65 | 43.7 (32 workers) | 45.2 (8 workers) | ≈ 1.0× |
| **total, heavy workload** | 195 (8 workers) | 104.6–105.3 | 55.1–55.9 | **1.9×** |

The two jobs' timings agree to within a few percent.

* **The GPU speeds up only the neural-network step: 3–4× on one of 12 tiles.**
  With the standard 16 × 112² clips that step is 7 s of the run, so the whole
  analysis barely changes (45 s against 44 s with 32 CPU workers). With
  32 × 224² clips the model step goes from 62 s to 16 s, and the run from 105 s
  to 55 s.
* **Optical flow is faster on the CPU.** At 160 px a frame pair is too small for
  the GPU to win over transfer and kernel launch. The OpenCL run also pays to
  spawn its workers. Its features are identical to the CPU flow's.
* **The rest of the run is CPU-bound.** The 1000-permutation classifier test
  (130–169 s) is scikit-learn on the CPU. More CPU workers help the flow
  (16 → 10 s with 8 → 32 workers).
* **Compute node against login node, CPU only:** 1.3–2× faster per stage. The
  login node was also shared, with a load average around 90.
* **Not used:** 11 of the 12 GPU tiles. Splitting the clips over tiles, or
  porting flow and classifiers to the GPU, would only pay at a much larger
  scale: thousands of swings, or full-resolution, full-length clips.

**Failures found and fixed.**
* `--flow ocl` with forked workers failed with `CL_OUT_OF_RESOURCES`, because an
  OpenCL context does not survive a fork. The workers are now spawned, and none
  of them opens OpenCL before it starts.
* 32 spawned workers each started 64 OpenBLAS threads and hit `pthread_create`
  EAGAIN. Workers now set `OPENBLAS_NUM_THREADS=1`. The 32-worker OpenCL run
  was not repeated after this fix.

### Reproducibility across ingests

The three ingests (login, job 1, job 2) gave different AUCs. Two checks show why.

* **The analysis is reproducible.** Job 2 also analysed the login node's own HDF5
  files on the compute node. It got exactly the login node's AUCs (0.719 and
  0.703). On a given ingest, the CPU and GPU runs also agree exactly.
* **The ingest is not bit-reproducible.** Job 2's HDF5 videos against the login
  node's:
  * 1 of 343 videos is identical;
  * the median mean difference is 1.23 grey levels (max 1.89);
  * PSNR between the two copies is 41 dB (min 38.7).

  The source MP4s, settings, libraries and filter parameters are identical.

  On the login node, `cae_video_hdf5` reproduces its own output bit for bit,
  even six at a time. Its output depends on x264's thread count (1 CPU differs
  from 4 or more), and it does not match the runtime's copy. The exact source
  of the variation inside the runtime is not pinned down. Making it
  reproducible would mean fixing the encoder's thread count in the filter, or
  storing lossless (crf 0).

How much that moves the results:

| | login ingest | job 1 ingest | job 2 ingest |
|---|---|---|---|
| video AUC, 16×112² | 0.719 | 0.734 | 0.719 |
| video AUC, 32×224² | 0.703 | **0.797** (p 0.03) | 0.688 |
| still AUC | 0.703 | 0.672 | 0.656 |
| g fpv_peak_shift_per_s | 1.35 | 1.41 | 1.32 |
| g fpv_peak_roll_deg_s | 1.28 | 1.26 | 1.35 |
| g vid_downswing_s | −1.25 | −1.22 | −0.86 |
| g vid_tempo_ratio | 1.17 | 1.15 | 0.86 |

* **Head-camera features are robust.** They keep g 1.26–1.41 and p ≤ 0.06 in
  every ingest.
* **Phase timings are sensitive.** Downswing and tempo depend on which frame the
  motion minimum lands on.
* **Learned embeddings are fragile.** Job 1's heavy-workload AUC of 0.797
  (p = 0.03) did not repeat, and is noise from the encode.

## 6. Reproduce

```bash
# FFmpeg with libx264 + the plugin
conda create -n ffdev -c conda-forge "ffmpeg=*=gpl*"
git clone https://github.com/Cai-Lab-at-University-of-Michigan/ffmpeg_HDF5_filter $S/ext/ffmpeg_HDF5_filter
bench/golf/build_h5ffmpeg_plugin.sh $S/ext/ffmpeg_HDF5_filter $S/h5plugin
# clio-core with video: add to the usual configure
#   -DCLIO_CAE_ENABLE_VIDEO=ON -DCLIO_CAE_H5FFMPEG_PLUGIN_DIR=$S/h5plugin
#   -DCMAKE_{EXE,SHARED,MODULE}_LINKER_FLAGS=-Wl,-rpath-link,$ffdev/lib
#   with PKG_CONFIG_PATH=$ffdev/lib/pkgconfig
cp bench/golf/clio_video_env.sh.example clio_video_env.sh   # edit paths
source clio_video_env.sh            # also sets HDF5_PLUGIN_PATH for the runtime
clio_run start &                     # golf runtime (clio_golf.yaml, port 9513)
clio-cae bench/golf/golf_video_omni.yaml
python bench/golf/golf_video_ml.py --source cte --out results --workers 8
# on a compute node, CPU vs GPU (stages everything under /lus/flare/projects/IOWarp):
qsub [-v GOLF_RUNS="cpu_w8 xpu_w8",GOLF_LOGIN_H5=<dir of .mp4.hdf5>] pbs/golf_video_gpu.pbs
# python env: h5ffmpeg av opencv-python-headless scikit-learn torch torchvision
```

[1] https://github.com/Cai-Lab-at-University-of-Michigan/ffmpeg_HDF5_filter
[2] MultiSenseGolf, Harvard Dataverse, doi:10.7910/DVN/LCCLLW

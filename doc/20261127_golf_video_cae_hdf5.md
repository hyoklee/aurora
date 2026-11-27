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

Run time: features 24 s with 8 processes, embeddings 31 s on CPU, 234 s in all
including the clio-core reads and the permutations. It ran on the login node
with CPU-only PyTorch. At this size a GPU would not change much.

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
* **Classifiers.**

  | input | golfer AUC | permutation p |
  |---|---|---|
  | R(2+1)D video embedding | 0.719 | 0.064 |
  | ResNet-18 first-frame still | 0.703 | 0.082 |

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
* **Run gotcha.** A clio-core client forked after it connects hangs on the
  runtime's shared memory: the first run's 16 workers waited 64 minutes. The
  script now reads all HDF5 bytes in the parent process and hands bytes to the
  workers.

## 5. Reproduce

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
# python env: h5ffmpeg av opencv-python-headless scikit-learn torch torchvision
```

[1] https://github.com/Cai-Lab-at-University-of-Michigan/ffmpeg_HDF5_filter
[2] MultiSenseGolf, Harvard Dataverse, doi:10.7910/DVN/LCCLLW

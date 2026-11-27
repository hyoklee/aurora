# Why the golf video HDF5 files are so much smaller than the MP4s

Context: clio-core CAE ingests the MultiSenseGolf swing videos (Harvard Dataverse
doi:10.7910/DVN/LCCLLW) as HDF5 through the FFMPEG HDF5 filter [1]. See
`doc/20261127_golf_video_cae_hdf5.md` for the whole study.

| | MP4 | HDF5 | ratio |
|---|---|---|---|
| Front camera, 192 videos | 1,232 MB | 20.7 MB | **59×** |
| Head camera (FPV), 151 videos | 285 MB | 11.7 MB | **24×** |

## Short answer

HDF5 itself does not compress the video better. The files are smaller because
the conversion **stores less**: fewer pixels, no colour, and a lower quality
setting with a more efficient codec. An MP4 made with the same settings would be
about the same size.

## The sources

Both cameras record MPEG-4 Part 2, Simple Profile: an old codec with no
B-frames (frames predicted from both directions).

| | Front camera | Head camera |
|---|---|---|
| Source | 640×480, 30 fps, **12.1 Mbit/s** | 1088×1080, 20 fps, 3.5 Mbit/s |
| Stored in HDF5 | 320×240 grey, x264 crf 23 | 320×316 grey, x264 crf 23 |
| HDF5 bitrate | ≈ 0.21 Mbit/s | ≈ 0.15 Mbit/s |

## Where the reduction comes from, and which code does it

| step | done by | front | head camera |
|---|---|---|---|
| decode the source MP4 | CAE (`video_hdf5.cc`, libav) | n/a | n/a |
| scale to width 320 (`video.width` in OMNI) | CAE (swscale) | 4× | 11.6× |
| convert to 8-bit grey | CAE (swscale); the filter accepts one channel only | 1.5× | 1.5× |
| encode with x264 at crf 23, preset medium | **ffmpeg_HDF5_filter** (id 32030), called by HDF5 when it writes the chunk | ≈ 10× | ≈ 1.4× |
| **total against the source MP4** | | **59×** | **24×** |

* **Scaling.** The OMNI `video` block asks for width 320, which keeps a quarter
  of the front camera's pixels and about a twelfth of the head camera's.
* **No colour.** The filter stores a single channel (8, 10 or 12 bit). The
  source's two colour channels (4:2:0) add half again to the data.
* **Codec and quality.**
  * H.264 (x264) needs about half the bitrate of MPEG-4 Part 2 for the same
    quality. It uses B-frames and better motion search.
  * The front camera was recorded at a very high bitrate, close to lossless, so
    re-encoding at crf 23 removes a lot. The head camera's source was already
    compressed hard, so little more comes off it.
  * Each swing is one HDF5 chunk of up to 512 frames, so x264 can predict across
    the whole clip.

### The filter's share

* **Against the MP4:** about 10× (front) and 1.4× (head camera). The rest comes
  from the CAE storing smaller grey frames.
* **Against the uncompressed grey frames it is given:**
  * front: 23,525 × 320 × 240 = 1.81 GB → 20.7 MB, **≈ 87×**
  * head camera: 12,137 × 320 × 316 = 1.23 GB → 11.7 MB, **≈ 105×**

  That is ordinary video-codec compression, far beyond a lossless HDF5 filter
  such as gzip on pixels. Against the MP4 the filter's share looks small because
  the MP4 was already compressed by its own codec.

The CAE only hands the filter 11 values per dataset (encoder, decoder, width,
height, depth, bit mode, preset, tune, crf, film grain, gpu). HDF5 runs the
filter on each chunk at write time and again at read time, so a reader gets
plain `uint8 [T, H, W]` frames.

## What you give up

* **Lossy.** Against an independent decode of the source, the stored frames are
  at 35.8 dB PSNR (front) and 44.1 dB (head camera). That is fine for motion
  features, not for pixel-exact work.
* **Not bit-reproducible across ingests.** Re-ingesting gives pixels about
  1.2 grey levels apart (41 dB between copies).
* **Grey only.** Colour is gone.

## What HDF5 adds

* **One file holds the swing.** It keeps frames, `/time_s` timestamps and
  metadata (source, codec, source size, fps) together.
* **Ordinary array access.** h5py, or any HDF5 reader with the filter plugin,
  reads the video as an array: no video library, and chunked reads.
* **Stored whole in clio-core.** CAE keeps each swing as one blob, still
  compressed, next to the other sensor streams.

## Knobs (OMNI `video` block)

```yaml
video:
  include: ["*.mp4"]
  width: 320      # 0 = keep the source resolution (skips the scaling step)
  codec: x264     # x264 | x265 | svtav1  (x265 / AV1: smaller at equal quality)
  crf: 23         # lower = better quality, bigger; 0 = lossless
  preset: medium
```

[1] https://github.com/Cai-Lab-at-University-of-Michigan/ffmpeg_HDF5_filter

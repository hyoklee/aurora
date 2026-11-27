#!/bin/bash
# Build the FFMPEG HDF5 filter (id 32030) as a plain HDF5 plugin, for C/C++
# readers and writers such as clio-core's CAE. The h5ffmpeg pip wheel's filter is
# a Python extension (it needs libpython) and links its own HDF5 1.14, so the HDF5
# 2.x C library cannot load it from HDF5_PLUGIN_PATH.
#
# usage: build_h5ffmpeg_plugin.sh <ffmpeg_HDF5_filter checkout> <out dir>
#   FFMPEG_PREFIX  FFmpeg with headers and libx264 (default: conda env ffdev,
#                  conda create -n ffdev -c conda-forge "ffmpeg=*=gpl*")
#   HDF5_PREFIX    the HDF5 clio-core links (default: conda env iowarp)
# Then export HDF5_PLUGIN_PATH=<out dir>.
set -euo pipefail
src=${1:?ffmpeg_HDF5_filter checkout}
out=${2:?output directory}
F=${FFMPEG_PREFIX:-$HOME/miniconda3/envs/ffdev}
H=${HDF5_PREFIX:-$HOME/miniconda3/envs/iowarp}
mkdir -p "$out"
gcc -O2 -fPIC -shared -std=gnu11 -o "$out/libh5ffmpeg_shared.so" \
  "$src"/src/ffmpeg_h5filter.c "$src"/src/ffmpeg_h5plugin.c \
  "$src"/src/ffmpeg_native.c "$src"/src/ffmpeg_utils.c \
  -I"$F/include" -I"$H/include" -L"$F/lib" -L"$H/lib" \
  -lavcodec -lavformat -lavutil -lswscale -lswresample -lhdf5 \
  -Wl,-rpath,"$F/lib" -Wl,-rpath,"$H/lib" -Wl,-rpath-link,"$F/lib" -Wl,--no-undefined
echo "built $out/libh5ffmpeg_shared.so ($(git -C "$src" rev-parse --short HEAD 2>/dev/null || echo unknown))"

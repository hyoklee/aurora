#!/bin/bash
#
# Reproduce the Aurora IGC internal compiler error offline.
# See doc/ISSUE-igc-ice-macro-kernels.md.
#
# IGC crashes translating SPIR-V to PVC ISA for the gpu_vector macro-form
# resumable kernels. ocloc runs that translation on its own, so this needs NO
# GPU and no allocation -- a login node is enough, and it finishes in seconds.
# That is the whole point: the five affected ctest cases otherwise cost a
# 28-minute compute-node job each time.
#
# Usage:
#   igc_ice_repro.sh                      # default inputs, see below
#   igc_ice_repro.sh IMAGE.spv ...        # a device image already extracted
#   igc_ice_repro.sh libfoo.so ...        # extract this library's images first
#
# Default inputs, first that exists:
#   $IGC_REPRO_DIR/*.spv                  # if IGC_REPRO_DIR is set
#   $IGC_REPRO_ARTIFACTS/*.spv            # the packaged reproducer
#   $CLIO_BUILD/bin/lib*_macros_sycl_launch.so
#
# Exit status: 0 if every image compiled, 1 if any ICEd, 2 on a setup problem.

set -u

IGC_REPRO_ARTIFACTS="${IGC_REPRO_ARTIFACTS:-/lus/flare/projects/IOWarp/hyoklee/igc_ice_repro}"
CLIO_BUILD="${CLIO_BUILD:-/lus/flare/projects/IOWarp/hyoklee/build_clio_core_wrk_2025.3.1}"
DEVICE="${IGC_REPRO_DEVICE:-pvc}"

# clang-offload-extract ships beside the DPC++ driver, in its compiler/
# subdirectory, and is not on PATH even with the module loaded.
find_extractor() {
  if command -v clang-offload-extract >/dev/null 2>&1; then
    command -v clang-offload-extract; return 0
  fi
  local icpx d c
  icpx=$(command -v icpx 2>/dev/null) || return 1
  d=$(dirname "$(readlink -f "$icpx")")
  # DPC++ keeps the offload tools in a compiler/ subdirectory of its bin/.
  for c in "$d/compiler/clang-offload-extract" "$d/clang-offload-extract"; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

command -v ocloc >/dev/null 2>&1 || {
  echo "ocloc not found. It ships with the Intel compute runtime (intel-ocloc)." >&2
  exit 2
}

echo "host:   $(hostname -f 2>/dev/null || hostname)"
echo "ocloc:  $(rpm -qf /usr/lib64/libocloc.so 2>/dev/null | head -1 || true)"
echo "libigc: $(readlink /usr/lib64/libigc.so.2 2>/dev/null || echo unknown)"
echo "device: ${DEVICE}"
echo

# ---- collect inputs -------------------------------------------------------
inputs=("$@")
if [ ${#inputs[@]} -eq 0 ]; then
  for d in "${IGC_REPRO_DIR:-}" "$IGC_REPRO_ARTIFACTS"; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    while IFS= read -r f; do inputs+=("$f"); done \
      < <(find "$d" -maxdepth 1 -name '*.spv' | sort)
    [ ${#inputs[@]} -gt 0 ] && break
  done
fi
if [ ${#inputs[@]} -eq 0 ] && [ -d "$CLIO_BUILD/bin" ]; then
  while IFS= read -r f; do inputs+=("$f"); done \
    < <(find "$CLIO_BUILD/bin" -maxdepth 1 -name 'lib*_macros_sycl_launch.so' | sort)
fi
if [ ${#inputs[@]} -eq 0 ]; then
  echo "Nothing to test. Pass a .spv or a .so, or set IGC_REPRO_DIR." >&2
  exit 2
fi

# ---- expand libraries into their device images ----------------------------
WORK=$(mktemp -d "${TMPDIR:-/tmp}/igc_ice_repro.XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT

images=()
for f in "${inputs[@]}"; do
  case "$f" in
    *.spv) images+=("$f") ;;
    *.so|*.so.*)
      EXTRACT=$(find_extractor) || {
        echo "clang-offload-extract not found; load a oneapi module to read $f" >&2
        exit 2
      }
      stem=$(basename "$f" | sed 's/\.so.*$//')
      ( cd "$WORK" && "$EXTRACT" --stem="$stem" "$(readlink -f "$f")" ) >/dev/null 2>&1
      n=0
      while IFS= read -r img; do images+=("$img"); n=$((n+1)); done \
        < <(find "$WORK" -maxdepth 1 -name "$stem.[0-9]*" | sort -V)
      [ "$n" -eq 0 ] && echo "  (no device images in $(basename "$f"))"
      ;;
    *) echo "  (skipping $f: not a .spv or .so)" ;;
  esac
done

# ---- run IGC on each image ------------------------------------------------
ok=0; ice=0; other=0
for img in "${images[@]}"; do
  log="$WORK/$(basename "$img").log"
  timeout 900 ocloc compile -spirv_input -file "$img" -device "$DEVICE" \
      -out_dir "$WORK/out" > "$log" 2>&1
  rc=$?
  if grep -qi "Internal Compiler Error" "$log"; then
    printf "  ICE   %s\n" "$(basename "$img")"
    grep -iE "Internal Compiler Error|Build failed" "$log" | sed 's/^/          /' | head -2
    ice=$((ice+1))
  elif [ $rc -eq 0 ]; then
    printf "  ok    %s\n" "$(basename "$img")"
    ok=$((ok+1))
  else
    printf "  rc=%-3s %s\n" "$rc" "$(basename "$img")"
    tail -3 "$log" | sed 's/^/          /'
    other=$((other+1))
  fi
done

echo
echo "images: ${ok} ok, ${ice} ICE, ${other} other"
[ "$ice" -gt 0 ] && exit 1
[ "$other" -gt 0 ] && exit 1
exit 0

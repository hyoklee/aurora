## IGC segfaults compiling the gpu_vector macro-form kernels: five GPU tests cannot run

**Platform:** Aurora (ALCF), 1 node, 6x Intel Data Center GPU Max 1550 (Ponte Vecchio), `sycl-debug` preset
**Compilers:** oneAPI 2025.3.1 (IntelLLVM 2025.3.2) and oneAPI 2026.1.0 (IntelLLVM 2026.1.0) — both affected
**JIT:** `libigc.so.2.11.43+0` (`libigdfcl-devel-2.11.43-1146`), `intel-ocloc-25.18.33578.77-1146`
**Branch:** `dev` @ `cfb02e72`
**Dashboard:** https://my.cdash.org/builds/4251649

### Summary

IGC crashes with a segmentation violation translating SPIR-V to PVC ISA for the
device images holding CLIO's macro-form (Duff's device) resumable kernels. The
SYCL runtime surfaces it as a build failure and the process aborts:

```
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  The program was built for 1 devices
Build program log for 'Intel(R) Data Center GPU Max 1550':
IGC: Internal Compiler Error: Segmentation violation
```

This is a crash inside the compiler, not a diagnosed program error. One of the
five affected kernels has been characterised and worked around in our source;
the other four have not, and look like they need an IGC fix.

**IGC is a system package under `/usr/lib64`, not part of the `oneapi` module.**
Both oneAPI modules on the machine share the one IGC, which is why switching
module versions changes nothing.

### Reproducing it without a GPU

The crash is in IGC's SPIR-V to ISA translation, which `ocloc` will run offline.
Extract the device image from a built launch library and hand it to IGC — an
Aurora **login node** is enough, and it takes seconds:

```
clang-offload-extract --stem=k libkmeans_macros_sycl_launch.so
ocloc compile -spirv_input -file k.0 -device pvc -out_dir /tmp/out
  IGC: Internal Compiler Error: Segmentation violation
  Build failed with error code: -11
```

Packaged reproducer, including a single-kernel image and the exact build
commands: `/lus/flare/projects/IOWarp/hyoklee/igc_ice_repro/` (`./repro.sh`).

### Which kernels

Per-kernel device images (`-DCLIO_SYCL_DG_USM -fsycl-device-code-split=per_kernel`)
isolate each kernel into its own image, so IGC names the offender:

| launch library | kernel | test |
|---|---|---|
| `kmeans_macros_sycl_launch` | `LaunchAssign` | `cte_gpu_vector_kmeans_macros_sycl` |
| `grayscott_macros_sycl_launch` | `LaunchStep` | `cte_gpu_vector_grayscott_macros_sycl` |
| `gmx_macros_sycl_launch` | `LaunchSpread`, `LaunchGather` | `cte_gpu_vector_gmx_macros_sycl` |
| `md_macros_sycl_launch` | `LaunchBuildList`, `LaunchGather` | `cte_gpu_vector_lammps_md_macros_sycl`, `..._evict_...` |

In every case the sibling images in the same library compile, and so does
`libclio_run_cxx_gpu`'s image. A single kernel in its own image is enough to
crash it — this is not about how many kernels share an image.

### Evidence: the kmeans kernel

`AssignMacro` ran a `k x dims` loop nest, inlined from a template helper, inside
the resumable body. Varying only that one statement, everything else fixed:

| variant | result |
|---|---|
| as shipped: `NearestCentroid(pt, cent, dims, k)` inlined | **ICE** |
| statement replaced by a cheap expression | compiles |
| same nest written out by hand in place | **ICE** |
| nest flattened to a single loop | compiles |
| nest kept, `#pragma unroll 1` on both levels | **ICE** |
| nest kept, inner trip count a compile-time constant | **ICE** |
| accessor a struct holding a reference / a pointer / returning a constant | **ICE** |
| accessor replaced by a raw `const float *` | **ICE** |
| template helper marked `noinline` | **ICE** (still inlined) |
| **nest moved into a NON-template `noinline` fn over raw `float *`** | **compiles** |

The loop **nest**, inlined into the resumable body, is the trigger. Trip counts,
unrolling pragmas and the accessor type are all irrelevant. `noinline` only
sticks once the function is not a template over an indexable accessor.

Fixed in `kmeans_macros_kernels.h` by lifting the nest into
`NearestCentroidOutOfLine`. The rebuilt library's two images both compile, where
image 0 previously crashed. This is the shape the gpu_vector design guide asks
for anyway — thin resumable bodies, compute in `noinline` functions over raw
pointers — so it is a fix rather than a contortion.

### Evidence: why that does not generalise

The same extraction applied to `grayscott`'s `StepMacro` changes nothing.
Cutting further, with the offline reproducer:

| grayscott `StepMacro` variant | result |
|---|---|
| as shipped | ICE |
| compute loop lifted into a `noinline` fn over raw pointers | ICE |
| **the compute call removed entirely** | ICE |
| 19 of the 20 `MFetch`/`MHoldPage` calls removed as well | ICE |

So in this kernel the crash is not in user compute at all, and not a function of
how many paging verbs inline into the loop. What remains in the body is the
macro-form suspend machinery itself — the `switch`, the eight `Held` frame
locals, the publish/flush verbs — and that cannot be lifted out, because
suspending is what the kernel does.

### Where it crashes

The last IGC dump written before the fault is `*_optimized.ll`, so it is past
IGC's own optimisation stage. `libigc.so.2` is stripped, so frames 0-9 have no
symbols:

```
#0 .. #9   libigc.so.2        (no symbols)
#10        IGC::IgcOclTranslationCtx<0ul>::Impl::Translate(...)
#11        IGC::IgcOclTranslationCtx<1ul>::TranslateImpl(...)
#12 ..     libocloc.so -> oclocInvoke
```

Full IGC dump:

```
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir> ocloc compile -spirv_input \
    -file one_kernel_nested_loop.spv -device pvc -out_dir /tmp/out
```

The failing module is a large `-O0` image: ~5 MB SPIR-V, ~400 functions, no
inlining, built around a `switch` that re-enters mid-function.

### Impact

Five ctest cases abort on Aurora. One is fixed in our source; four remain.
The other four failures in build 4251649 were unrelated and are fixed
(a proxy leaking into loopback HTTP, and cross-case contamination in an
eviction test).

### Why the obvious fixes don't work

| attempt | result |
|---|---|
| `-options "-cl-opt-disable"` | ICE |
| `-options "-g"` / no options / `-O2` | ICE |
| DPC++ 2026.1.0 instead of 2025.3.1 | ICE — same system IGC |
| `ONEAPI_DEVICE_SELECTOR=level_zero:0` (one device, one JIT compile) | ICE, reported as "built for 1 devices" |
| AOT per translation unit (`-fsycl-targets=spir64_gen`) | compiles — the single-TU path is fine; the failing module is the linked device image |
| `-fsycl-device-code-split=per_kernel` as shipped | will not build: `device_global variable '..g_yield_smem_dg' with property "device_image_scope" is used in more than one device image` |
| `per_kernel` + `-DCLIO_SYCL_DG_USM` (drops `device_image_scope`) | builds, splits into per-kernel images — the image holding the offending kernel still ICEs |

The device count is irrelevant, so is the optimisation level, and so is the
oneAPI module version. Splitting kernels into separate images is useful as a
diagnostic but not as a fix.

### Possible directions

1. An IGC/NEO package update on Aurora that fixes the crash. This is the only
   path for the four remaining kernels that we can see.
2. Per-kernel source restructuring, as done for kmeans. Unavailable for the
   other four: the crash survives removing all user compute.
3. Keep the five tests off the Aurora dashboard until (1) lands, and run the
   CUDA coroutine edition for coverage of the same workloads.

(1) plus (3) looks like the realistic position.

### Ask of ALCF / Intel

An IGC build that does not segfault on these modules, or guidance on what in a
Duff's-device resumable kernel to avoid. A symbolised `libigc.so.2`, or the
output of a debug IGC on `one_kernel_nested_loop.spv`, would localise the pass.

### Related, already fixed (not part of this issue)

Also in build 4251649, and unrelated to IGC:

- `http_proxy` is exported on compute nodes so CDash submission can reach the
  internet; libcurl honoured it for `127.0.0.1` too, so three summarizer tests'
  in-process HTTP stub was answered by Squid with HTTP 503. Fixed by setting
  `CURLOPT_NOPROXY` for loopback in the Ollama client, preserving any existing
  `no_proxy`. Worth knowing for anything on Aurora that talks to a local
  service through libcurl.
- `cte_evict_trigger` shared one storage tier across cases and cleaned up only
  on the success path, so one failure cascaded. Fixed with destructor-based
  cleanup and an explicit capacity precondition per case.

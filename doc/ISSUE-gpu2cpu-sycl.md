## gpu2cpu producer-only design cannot work on Intel GPUs: kernels never observe host writes

**Platform:** Aurora (ALCF), 1 node, 6x Intel Data Center GPU Max 1550 (Ponte Vecchio), oneAPI 2025.3.1 / IntelLLVM 2025.3.2, `sycl-debug` preset
**Branch:** `dev` @ `9af0ce88`

### Summary

`gpu::Future::Wait()` spins inside a kernel waiting for the CPU worker to set
`task->fut_.is_complete_`. On Ponte Vecchio a running kernel never observes host
writes, so that wait can never be satisfied. This is a property of the hardware,
not a bug in the ring buffer or the allocator, and no change of USM allocation
kind fixes it.

Found while bringing up the SYCL GPU build on Aurora. Two adjacent, genuinely
fixable defects were uncovered on the way and are already fixed (below); this
issue is only about the remaining design question.

### Evidence

A two-way handshake probe, with the kernel deliberately left in flight: host
submits without waiting; kernel sets `p[0]=1` then spins until it sees `p[1]==1`;
host spins until it sees `p[0]==1` then sets `p[1]=1`; kernel on seeing that sets
`p[0]=2`. Both spins bounded (400M device iterations, 10s host).

| allocation | access | host saw device | device saw host |
|---|---|---|---|
| `malloc_shared` | system-scope atomic | YES | hung, killed at 120s |
| `malloc_shared` | volatile | YES | **NO** |
| `malloc_host` | volatile | YES | **NO** |

Device to host works in every configuration. Host to device works in none. In
the `malloc_host` row the kernel completed all 400M iterations and wrote its
timeout sentinel, so it was demonstrably re-reading the location and
demonstrably never saw the store.

Device capability query on the same node:

```
aspect::usm_host_allocations          = 1
aspect::usm_atomic_host_allocations   = 0
aspect::usm_shared_allocations        = 1
aspect::usm_atomic_shared_allocations = 0
```

`usm_atomic_shared_allocations = 0` is accurate rather than conservative: it
describes *concurrent* host/device atomic access, which the probe's second row
shows is genuinely absent.

### Impact

`cr_gpu_kernel_stress_sycl` hangs and is killed at the 120s ctest timeout. It
gets as far as launching the kernel and passing its sanity marker, then stops.
With the two fixes below applied, this is the only remaining failure on Aurora
(254/255 passing).

The affected pattern is `ipc_gpu2cpu_impl.h`:

```cpp
auto fut = CLIO_IPC->Send(fp);
fut.Wait();                       // spins on is_complete_ inside the kernel
```

### Why the obvious fixes don't work

**Switching to shared USM** (`experiment/sycl-shared-usm`) removes every GPU
page fault — 244 to 0 across a full ctest run — because it does fix the separate
problem that PVC cannot do atomics on host USM at any scope. But it converts the
abort into the hang above rather than making the test pass, because visibility,
not atomicity, is the blocker.

**System-scope atomics** don't help either; the probe's first row is
system-scope and still fails.

### Possible directions

1. Make `Send()` fire-and-forget on SYCL: kernel pushes and exits, host drains
   after kernel completion, results collected by a subsequent launch. Loses
   in-kernel completion waiting.
2. Split the wait across kernel launches — producer kernel ends at the `Send`,
   a continuation kernel resumes once the host has completed the task.
3. Restrict the producer-only path to CUDA/ROCm and have SYCL report the
   capability as unavailable, so the tests skip rather than hang.

(1) or (3) look like the realistic near-term options. Worth noting this is not
Aurora-specific — it should apply to any discrete Intel GPU.

### Related fixes (already done, not part of this issue)

Both were masked because `ServerInitGpuQueues` returns early when no GPU is
present, so CI and login-node runs never executed either path.

- **`ServerInitGpuQueues` built the queue inside a kernel**, reaching
  `BuddyAllocator` -> `ctp::Mutex::Lock` -> `fetch_add` on host USM ->
  `AtomicAccessViolation`, killing every runtime start. 151 of 255 tests failed
  on a GPU node from this one call. `gpu2cpu_init_hip.cc` already constructs on
  the host; `49cfac8c` moved HIP but not SYCL. Porting it takes the GPU node
  from 104 to 254 passing.
- **`std::this_thread::yield` compiled into kernels**, emitting `sched_yield`,
  which SPIR-V cannot resolve — the JIT failed the whole program with
  `Unresolved Symbol <sched_yield>`.

### Root cause common to both fixes, worth a separate audit

`CTP_IS_HOST` is **1** during DPC++'s SYCL device pass. `macros.h` only clears
it for CUDA/ROCm device passes:

```c
#if defined(CTP_IS_CUDA_GPU) || defined(CTP_IS_ROCM_GPU)
#define CTP_IS_GPU 1 / CTP_IS_HOST 0
#else
#define CTP_IS_GPU 0 / CTP_IS_HOST 1   // SYCL device pass lands here
#endif
```

So `#if CTP_IS_HOST` does **not** keep host code out of a SYCL kernel, and
`ctp::ipc::atomic` resolves to `std_atomic` in device code. Both defects above
are instances of this. A sweep of `#if CTP_IS_HOST` in device-reachable headers
would likely find more; only the two that these tests happened to hit were
fixed. `!CTP_IS_DEVICE_PASS` is the guard that actually covers the SYCL device
pass.

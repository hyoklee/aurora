# llm_aurora_test.pbs — XPU Fix & Optimization

## Root Cause: `XPU available: False`

Three compounding issues caused the job to fall back to CPU:

1. **Wrong conda environment** — `pytorch_env` contains vanilla PyTorch 2.5.1 with no Intel XPU backend. The `ipex-llm` env is required.
2. **Missing `intel_extension_for_pytorch` import** — Even with the correct env, `torch.xpu` is only registered after `import intel_extension_for_pytorch as ipex` is called.
3. **Wrong model loader** — Section [4] used `transformers.AutoModelForCausalLM` instead of `ipex_llm.transformers.AutoModelForCausalLM`, bypassing all IPEX-LLM XPU optimizations.

**Observed result before fix:** model ran entirely on CPU at **3.2 tok/s**, model load took 91.5 s, 16.1 GB GGUF de-quantized into FP16 on CPU RAM.

---

## Changes Made to `llm_aurora_test.pbs`

| Location | Before | After |
|---|---|---|
| Conda env | `pytorch_env` | `ipex-llm` |
| Section [1] env check | `import torch` only | added `import intel_extension_for_pytorch as ipex` |
| Section [4] model loader | `from transformers import AutoModelForCausalLM` | `from ipex_llm.transformers import AutoModelForCausalLM` |
| Section [4] quantization | `torch_dtype=torch.float16` (deprecated arg, FP16 on CPU) | `load_in_low_bit="sym_int4"` (INT4, fits in ~4 GB XPU VRAM) |
| Section [4] XPU sync | absent | `torch.xpu.synchronize()` after `.to(device)` and after `model.generate()` |
| Generation args | `do_sample=False, temperature=1.0` | removed redundant `temperature` (suppresses warning) |

---

## Expected Performance After Fix

| Metric | Before (CPU) | After (XPU, INT4) |
|---|---|---|
| Device | CPU | Intel PVC XPU (`xpu:0`) |
| Throughput | ~3 tok/s | ~20–50+ tok/s |
| Model load | 91.5 s | faster (INT4, smaller weights) |
| VRAM footprint | 16 GB CPU RAM (FP16) | ~4 GB XPU VRAM (INT4) |

Aurora nodes have 6 Intel Data Center GPU Max 1550 (Ponte Vecchio) GPUs per node. The script requests `ngpus=6` but currently targets `xpu:0` (one tile). Multi-GPU sharding with `device_map` is a future improvement if throughput needs to scale further.

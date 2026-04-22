# Fused Rotation+Quantization+Sort — MoE Kernel Optimization Record

**Platform**: AMD Instinct MI355X (gfx950, CDNA4), ROCm 7.0, Triton 3.5.1  
**Project**: `/data/jiangyon/vllm_rotation/`  
**Target**: MoE decode path (M=1, K=2048/7168, RS=128, topk=8)

## Performance Landscape (M=1, K=2048, topk=8)

| Implementation | Time (µs) | vs Best | Correct | Notes |
|---|---|---|---|---|
| Empty kernel launch | 5.1 | — | — | hipLaunchKernelGGL baseline |
| **Raw HIP (scalar dot)** | **10.5** | **1.00x** | **NO** | Fastest, but precision mismatch (scalar FP32 ≠ MFMA BF16) |
| Triton topk=8 3-in-1 | 14.2 | 0.74x | YES | `_fused_decode_m1_topk8_k2048_kernel` |
| Gluon v2 (kw4) | 14.2 | 0.74x | YES | Identical perf to Triton |
| Gluon v2 (kw8) | 14.3 | 0.73x | YES | kw8 doesn't help (latency-bound) |
| HIP-Triton (tl.dot+v_cvt) | 14.3 | 0.73x | YES | No Gluon, same perf |
| HIP via dispatcher (2-kernel) | 31.7 | 0.33x | YES | **BUG**: dispatcher routes to 2-kernel fallback |
| HIP via dispatcher (fixed) | 15.1 | 0.70x | YES | After routing to HIP 3-in-1 kernel |
| Separated (rot + quant) | 21.8 | 0.48x | YES | 2 separate kernels |

## Key Conclusions

### 1. All Triton/Gluon variants converge to 14.2µs
No Gluon-level optimization can improve performance. The kernel is **purely latency-bound**:
- 5.1µs kernel launch overhead (36%)
- 3.8µs Triton/Gluon runtime overhead (27%)
- 5.3µs actual kernel execution (37%)

### 2. Parameter tuning has zero effect
| Parameter | Values Tested | Result |
|-----------|--------------|--------|
| num_warps | 1, 2, 4, 8 | All 14.2-14.5µs |
| MAX_Q | 8, 16, 32, 64 | All 14.2-14.3µs |
| k_width | 4 (16x16x16), 8 (16x16x32) | kw8 actually 0.1µs slower |
| BLOCK_M | 32 (default) | Can't reduce without layout changes |

### 3. Raw HIP eliminates 3.8µs Triton overhead
```
14.2µs (Triton) - 3.8µs (runtime overhead) = 10.5µs (Raw HIP)
```
But raw HIP kernel has precision issues (scalar dot product ≠ MFMA accumulation).

### 4. MoE dispatcher has routing bug
`fused_rotation_mxfp4_quant_moe_sort()` with `use_hip_kernel=True` takes 2-kernel fallback path (31.7µs) instead of HIP 3-in-1 kernel (15.1µs).

### 5. K=7168 lacks topk=8 specialization
K=7168 (Qwen3-30B gate_up_proj) uses generic kernel: 19.8µs Triton, 31.8µs HIP.
The topk=8 specialization only exists for K=2048.

## Overhead Breakdown (14.2µs Triton kernel)

```
 0µs        5µs        10µs       14.2µs
 |─────────|─────────|──────────|
 |  5.1µs  | 3.8µs   | 5.3µs   |
 | launch  | Triton  | kernel   |
 | overhead| runtime | execution|
 |         |         |          |
 | hipLaunch| Python→ | MFMA +  |
 | KernelGGL| JIT     | v_cvt + |
 |         | lookup  | scatter  |
```

## Files

| File | Type | Description |
|------|------|-------------|
| `fused_rotation_mxfp4_quant_moe_sort.py` | Triton | Main MoE dispatcher + topk8 kernel |
| `fused_rotation_mxfp4_quant_moe_hip.py` | Triton+ISA | HIP MoE variant (tl.dot + v_cvt inline ASM) |
| `fused_rotation_mxfp4_quant_moe_gluon.py` | Gluon | Gluon MoE variant |
| `fused_rotation_quant_gluon_v2.py` | Gluon | Dense kernel (used as fallback for MoE) |
| `fused_rotation_quant_hip.py` | Triton+ISA | Dense HIP variant |
| `hip_rotation_quant/fused_rot_quant_m1_decode.hip` | Raw HIP | CUDA-Agent experimental (10.5µs, incorrect) |
| `test_gluon_v2_kernel.py` | Test | Dense kernel correctness |
| `test_moe_gluon_kernel.py` | Test | MoE kernel correctness (all variants) |
| `benchmark_gluon_v2.py` | Bench | Dense kernel benchmark |

## Optimization Path for Fastest MoE

### Phase 1: Dispatcher Fix (31.7→15.1µs, DONE)
Route `use_hip_kernel=True` + topk=8 to HIP 3-in-1 kernel.

### Phase 2: CUDAGraph (14.2→~9µs, vLLM level)
Already registered as custom op. In vLLM with `--enforce-eager=False`, CUDAGraph eliminates 5.1µs launch overhead.

### Phase 3: Raw HIP with MFMA (target: ~10µs)
Need to implement MFMA intrinsics for correct precision. Reference: CK Tile `mfma_gfx9_hip.hpp`.

### Phase 4: Generalized topk=8 with constexpr N_I/TILE_N (DONE)
Created `_fused_decode_m1_topk8_general_kernel` with `N_I` and `TILE_N` as `tl.constexpr`.
Triton compiler optimizes integer division/modulo into multiply+shift at compile time.

| K | Before (generic) | After (constexpr) | Improvement |
|---|---|---|---|
| 2048 | 15.5 µs | 14.7 µs | -5% |
| 5120 | 15.7 µs | 14.4 µs | -8% |
| 7168 | 15.7 µs | 14.7 µs | -6% |
| 8192 | 15.7 µs | 14.5 µs | -8% |

All K values now match the K=2048 specialized kernel performance (~14.5µs).
Correctness: ALL TESTS PASSED.

## Test Commands

```bash
# Correctness
HIP_VISIBLE_DEVICES=0 python3 test_gluon_v2_kernel.py
HIP_VISIBLE_DEVICES=0 python3 test_moe_gluon_kernel.py

# Benchmark
HIP_VISIBLE_DEVICES=0 python3 benchmark_gluon_v2.py

# Clear cache after kernel changes
rm -rf ~/.triton/cache/
```

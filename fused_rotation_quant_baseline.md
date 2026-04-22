# Fused Rotation+Quantization Kernel — Baseline Performance

**Platform**: AMD Instinct MI355X (gfx950, CDNA4), ROCm 7.0, PyTorch 2.9.1, Triton 3.5.1  
**Project**: `/data/jiangyon/vllm_rotation/` (branch: `fused-rotation-quant-kernel`)

## Dense Kernel Baseline

**Kernel**: `fused_rotation_quant_gluon_v2.py` — Gluon v2 fused rotation matmul + MXFP4 quant  
**Operation**: `x[M,K] @ rotation[RS,RS]` → MXFP4 quantize → FP4 data + E8M0 scales  
**Tests**: `test_gluon_v2_kernel.py` — ALL PASSED

| M | K | Separated (µs) | Fused Gluon v2 (µs) | Speedup |
|---|---|----------------|---------------------|---------|
| 1 | 2048 | 21.8 | 13.3 | 1.65x |
| 1 | 5120 | 21.8 | 13.5 | 1.62x |
| 1 | 7168 | 22.1 | 13.5 | 1.64x |
| 1 | 8192 | 22.0 | 13.6 | 1.62x |
| 4 | 2048 | 22.1 | 13.3 | 1.66x |
| 16 | 5120 | 21.7 | 13.5 | 1.61x |
| 32 | 7168 | 21.9 | 13.6 | 1.62x |
| 64 | 8192 | 21.9 | 13.6 | 1.60x |
| 128 | 2048 | 22.0 | 13.7 | 1.60x |
| 128 | 8192 | 21.8 | 13.6 | 1.60x |

**Characteristic**: Latency-bound (~13.5µs constant regardless of M/K). Dominated by MFMA instruction latency + kernel launch overhead, not HBM bandwidth.

### Dense Kernel Architecture (Gluon v2)

```
Grid: (ceil(M/32), K/RS)  — 1 block per (32 rows × 1 rotation chunk)
Block: 256 threads (4 warps)

1. buffer_load x[32, 128] from HBM → registers
2. buffer_load rotation[128, 128] from HBM → LDS (swizzled shared)
3. MFMA: x @ rotation → acc[32, 128] in FP32 (single pass, no K accumulation)
4. Cast to BF16→FP32 (truncation for quant alignment)
5. Reshape → compute amax per group of 32 → E8M0 scale
6. v_cvt_scalef32_pk_fp4_f32: HW MXFP4 quantization
7. buffer_store FP4 data + store scales
```

## MoE Kernel Baseline

**Kernel**: `fused_rotation_mxfp4_quant_moe_sort.py` — Triton 3-in-1 (rotation + quant + sorted scale scatter)  
**Operation**: Same as dense + scatter scales to MoE sorted layout  
**Tests**: `test_moe_gluon_kernel.py` — ALL PASSED (Triton, HIP, Gluon variants)

| K | topk | Triton 3-in-1 (µs) | HIP 3-in-1 direct (µs) | HIP via dispatcher (µs) | Notes |
|---|------|-------------------|----------------------|------------------------|-------|
| 2048 | 4 | 31.8 | — | 31.5 | Generic path |
| **2048** | **8** | **14.2** | **14.4** | **15.1** | **topk=8 specialized, near-identical perf** |
| 7168 | 4 | 32.0 | — | 32.0 | Generic path |
| 7168 | 8 | 19.8 | — | 31.8 | K=7168 no topk8 specialization |

### Key Finding: HIP topk=8 kernel IS fast — dispatcher routing was wrong

Previous benchmark showed HIP at 31.7µs because `fused_rotation_mxfp4_quant_moe_sort()` with `use_hip_kernel=True` takes a **2-kernel path** (rot+quant then separate sort), bypassing the HIP topk=8 specialized kernel entirely.

Direct call to `_fused_decode_m1_topk8_k2048_hip_kernel`: **14.4µs** (matches Triton's 14.2µs).

**Fix needed**: Route `use_hip_kernel=True` + topk=8 + K=2048 to the HIP 3-in-1 kernel instead of 2-kernel fallback.

## Optimization Targets

| Target | Current | Goal | Approach |
|--------|---------|------|----------|
| Dense Gluon v2 | 13.5 µs | Latency-bound, ~14µs floor | CUDAGraph (vLLM level), batch operations |
| MoE HIP topk=8 dispatch fix | 31.7→15.1 µs | Already fixed by direct routing | Fix dispatcher to route HIP→3-in-1 kernel |
| MoE topk=8 kernel (both) | 14.2-14.4 µs | <12 µs | Reduce MFMA latency, optimize scale scatter |
| MoE K=7168 topk=8 | 19.8 µs (Triton), 31.8 (HIP) | <18 µs | Add K=7168 specialization |
| MoE generic (topk=4) | 31.8 µs | <25 µs | Better tl.dot config, reduce scatter overhead |

## Raw HIP Kernel Experiment (CUDA-Agent Approach)

Wrote a raw HIP kernel (`hip_rotation_quant/fused_rot_quant_m1_decode.hip`) to eliminate Triton runtime overhead.

**Design**: For M=1 decode, uses per-lane dot product (not MFMA) — 64 lanes each compute one output element, avoiding 15/16 wasted M-dimension compute in 16×16 MFMA tiles.

| Implementation | Time (µs) | vs Triton |
|---|---|---|
| Triton topk=8 (K=2048) | 15.1 | 1.00x |
| **Raw HIP kernel** | **10.5** | **1.45x faster** |

**Speedup breakdown**:
- Triton runtime overhead eliminated: ~3-4µs (Python→compiled binary dispatch)
- Simpler thread mapping (64 threads vs 256): ~1µs

**Correctness status**: FP4 match rate ~0.3% due to accumulation precision difference. Scalar FP32 dot product ≠ MFMA BF16 multiply + FP32 accumulate. To achieve bit-exact results, need to use MFMA intrinsics (`__builtin_amdgcn_mfma_f32_16x16x16_bf16`), which requires complex register-to-matrix layout mapping.

**Key insight**: The ~5µs Triton overhead (out of 15µs total) is a significant fraction for latency-critical decode paths. Raw HIP kernels can recover this overhead but at the cost of correctness complexity and maintenance burden.

## Test Commands

```bash
# Dense correctness
HIP_VISIBLE_DEVICES=0 python3 test_gluon_v2_kernel.py

# MoE correctness  
HIP_VISIBLE_DEVICES=0 python3 test_moe_gluon_kernel.py

# Dense benchmark
HIP_VISIBLE_DEVICES=0 python3 benchmark_gluon_v2.py

# Clear triton cache after kernel changes
rm -rf ~/.triton/cache/
```

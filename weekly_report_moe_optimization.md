# Weekly Report: MXFP4 Fused Rotation Optimization for MoE

**Date:** April 7, 2026  
**Hardware:** AMD MI355X (gfx950, CDNA4)  
**Framework:** vLLM + Quark 0.11 + aiter  

## Summary

Completed E2E validation and benchmarking of the HIP MFMA fused rotation+quantization+sort kernel for MoE MXFP4 inference on Qwen3-30B-A3B. The kernel (`aiter.ops.mfma_rot_quant_moe_sort`) fuses online rotation, MXFP4 quantization, and MoE sorted scale scatter into a single HIP ASM kernel using `v_mfma_f32_16x16x32_bf16` + `v_cvt_scalef32_pk_fp4_f32`.

**Highlights:**
- Correctness verified: text generation (3/3 QA), PPL=9.08 (vs RTN 9.59)
- CUDAGraph enabled, env vars confirmed, two-run average for stability
- At c=32, fused rotation matches RTN baseline — online rotation with near-zero overhead

## E2E TPOT — Qwen3-30B-A3B MoE (Two-Run Avg, GPU 6, CUDAGraph)

| c  | RTN (ms) | Separated (ms) | HIP-MFMA Fused (ms) | Rotation Overhead (Sep/RTN) | Fused Recovery (HIP/Sep) | Residual (HIP/RTN) |
|----|----------|----------------|----------------------|-----------------------------|--------------------------|---------------------|
| 1  | 6.84     | 7.48           | 7.27                 | +9.3%                       | **-2.7%**                | +6.3%               |
| 4  | 8.24     | 8.84           | 8.71                 | +7.2%                       | **-1.5%**                | +5.6%               |
| 16 | 10.23    | 11.22          | 10.52                | +9.7%                       | **-6.2%**                | +2.9%               |
| 32 | 11.62    | 12.38          | 11.54                | +6.6%                       | **-6.8%**                | **-0.7%**            |

## E2E TPOT — Qwen3-8B Dense (GPU 6, CUDAGraph)

| c  | RTN (ms) | Separated (ms) | Gluon Fused (ms) | Rotation Overhead (Sep/RTN) | Fused Recovery (Fused/Sep) | Residual (Fused/RTN) |
|----|----------|----------------|-------------------|-----------------------------|----------------------------|----------------------|
| 1  | 8.15     | 8.57           | 8.29              | +5.2%                       | **-3.3%**                  | +1.7%                |
| 4  | 9.39     | 9.86           | 9.53              | +5.0%                       | **-3.3%**                  | +1.5%                |
| 16 | 9.73     | 10.22          | 9.90              | +5.0%                       | **-3.1%**                  | +1.7%                |
| 32 | 10.32    | 10.88          | 10.53             | +5.4%                       | **-3.2%**                  | +2.0%                |

## Kernel-Level Latency (M=1, K=2048, topk=8)

| Kernel                         | Latency (μs) | Speedup vs Separated | Location                    |
|--------------------------------|--------------|----------------------|-----------------------------|
| **HIP MFMA 3-in-1**           | **11.8**     | **30.2×**            | aiter (our implementation)  |
| Gluon 3-in-1 (Triton)         | 19.1         | 18.6×                | vLLM                        |
| RTN (quant + sort, no rotation)| 34.2        | 10.4×                | aiter (2 kernels)           |
| Separated (matmul+quant+sort)  | 355.9       | 1.0× (baseline)      | Python dispatch             |

## Key Findings

1. **HIP MFMA kernel is correct and fastest** — 11.8μs, 38% faster than Gluon Triton (19.1μs)
2. **At c=32, fused matches RTN** — online rotation adds near-zero overhead with PPL improvement (9.59→9.08)
3. **MoE fused is the primary optimization** — recovers 2–7% of separated overhead on MoE models
4. **Dense also benefits** — consistent 3.1–3.3% TPOT recovery from Gluon fused kernel
5. **Attention-fused alone is insufficient for MoE** — MoE expert layers dominate; attn-fused saves ~0.91ms across 48 layers but negligible E2E

## Known Issues

| Item                              | Status | Notes                                                     |
|-----------------------------------|--------|-----------------------------------------------------------|
| Hadamard model vLLM loading       | Bug    | trainable=false → bool dtype → PPL explosion. Use trained.|
| Attn-fused-only c=1 regression    | TBD    | May be noise or CUDAGraph graph structure difference.     |
| Default VLLM_MOE_HIP_MFMA=1      | Pending| Consider making default once verified at scale.           |
| Triton 3.6 migration              | Blocked| gl.convert_layout bug for scale tensors.                  |

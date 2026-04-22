#!/usr/bin/env python3
"""
Kernel-level benchmark: rotation+quant+sort kernels for MoE
Compare all available implementations at each M value.
"""
import torch
import time
import os

os.environ.setdefault("HIP_VISIBLE_DEVICES", "5")

from aiter.utility import dtypes
from aiter.utility.fp4_utils import moe_mxfp4_sort

# ---- Config ----
K = 2048          # Qwen3-30B hidden size
RS = 128          # rotation_size
QG = 32           # quant group
TOPK = 8          # top-k experts
N_EXPERTS = 64    # total experts
BLOCK_SIZE = 32   # block_size_M
WARMUP = 50
ITERS = 200

M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

device = "cuda"
dtype = torch.bfloat16

import math
rotation = torch.linalg.qr(torch.randn(RS, RS, dtype=torch.float32, device=device))[0].to(dtype)

n_i = K // QG

print(f"{'='*80}")
print(f"Kernel Benchmark: K={K}, RS={RS}, topk={TOPK}, N_EXPERTS={N_EXPERTS}")
print(f"Warmup={WARMUP}, Iters={ITERS}")
print(f"{'='*80}")
print()

# Import all kernels
from aiter.ops.triton.fused_rot_quant_moe_sort import fused_rot_quant_moe_sort
from aiter.ops.mfma_rot_quant_moe_sort import mfma_rot_quant_moe_sort
from aiter.ops.triton._triton_kernels.fused_rot_quant_moe_sort_gluon import (
    _gluon_decode_m1_rot_quant_sorted_kernel,
    _gluon_rot_quant_kw8_kernel,
)
from aiter.ops.triton._triton_kernels.fused_rot_quant_moe_sort import (
    _rot_quant_sort_m1_kernel,
    _rot_quant_sort_kernel,
)
from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
import triton

def make_data(M):
    """Create test data matching real inference."""
    x = torch.randn(M, K, dtype=dtype, device=device)
    # Simulate sorted_ids from topk routing
    m_o = M * TOPK
    sorted_ids = torch.randint(0, M, (m_o,), dtype=torch.int32, device=device)
    num_valid_ids = torch.tensor(m_o, dtype=torch.int32, device=device)
    return x, sorted_ids, num_valid_ids, m_o

def bench(fn, warmup=WARMUP, iters=ITERS):
    """Benchmark a function, return time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1e6
    return elapsed

results = {}

for M in M_VALUES:
    x, sorted_ids, num_valid_ids, m_o = make_data(M)
    m_pad = ((m_o + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    
    print(f"--- M={M}, m_o={m_o}, m_pad={m_pad} ---")
    row = {}
    
    # 1. Separated: matmul + fused_dynamic_mxfp4_quant_moe_sort
    def sep_fn():
        x_rot = (x.view(M, K // RS, RS) @ rotation).view(M, K)
        a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
            x_rot, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=1, block_size=BLOCK_SIZE,
        )
        return a1, a1_scale
    
    try:
        t = bench(sep_fn)
        row["separated"] = t
        print(f"  separated (matmul+quant_sort):  {t:8.1f} μs")
    except Exception as e:
        print(f"  separated: ERROR {e}")
    
    # 2. Triton unified API (auto-dispatches Gluon M=1 or kw8+sort M>1)
    def triton_unified_fn():
        return fused_rot_quant_moe_sort(
            x, rotation, RS,
            sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, block_size=BLOCK_SIZE,
        )
    try:
        t = bench(triton_unified_fn)
        row["triton_unified"] = t
        print(f"  triton unified (auto):          {t:8.1f} μs")
    except Exception as e:
        print(f"  triton unified: ERROR {e}")
    
    # 3. HIP MFMA 3-in-1 (all M values)
    def hip_mfma_fn():
        fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
        sc = torch.empty((m_pad + 1, n_i), dtype=torch.uint8, device=device)
        mfma_rot_quant_moe_sort(
            x, rotation, fp4, sc,
            sorted_ids, num_valid_ids,
            token_num=M, rotation_size=RS,
        )
        return fp4, sc
    try:
        t = bench(hip_mfma_fn)
        row["hip_mfma"] = t
        print(f"  HIP MFMA 3-in-1:               {t:8.1f} μs")
    except Exception as e:
        print(f"  HIP MFMA: ERROR {e}")
    
    # 4. Gluon kw8 + moe_mxfp4_sort (M>1 path, but also try M=1)
    def gluon_kw8_sort_fn():
        fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
        raw_scale = torch.empty((M, n_i), dtype=torch.uint8, device=device)
        BM = 32; NW = 4; NS = 3
        grid = (triton.cdiv(M, BM), K // RS)
        _gluon_rot_quant_kw8_kernel[grid](
            x, rotation, fp4, raw_scale, M, K, n_i,
            x.stride(0), rotation.stride(0), rotation.stride(1),
            fp4.stride(0), raw_scale.stride(0),
            RS=RS, QG=QG, BLOCK_M=BM,
            NUM_WARPS=NW, num_warps=NW, num_stages=NS,
            SHUFFLE_SCALES=False,
        )
        sorted_scale = moe_mxfp4_sort(
            raw_scale, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, block_size=BLOCK_SIZE,
        )
        return fp4, sorted_scale
    try:
        t = bench(gluon_kw8_sort_fn)
        row["gluon_kw8_sort"] = t
        print(f"  Gluon kw8 + sort:               {t:8.1f} μs")
    except Exception as e:
        print(f"  Gluon kw8+sort: ERROR {e}")
    
    # Find best
    if row:
        best_name = min(row, key=row.get)
        best_time = row[best_name]
        sep_time = row.get("separated", float('inf'))
        speedup = (sep_time / best_time - 1) * 100 if sep_time != float('inf') else 0
        print(f"  >>> BEST: {best_name} ({best_time:.1f}μs, {speedup:+.1f}% vs separated)")
    
    results[M] = row
    print()

# Summary table
print(f"\n{'='*80}")
print(f"SUMMARY (μs)")
print(f"{'='*80}")
all_keys = ["separated", "triton_unified", "hip_mfma", "gluon_kw8_sort"]
header = f"{'M':>5} " + " ".join(f"{k:>18}" for k in all_keys) + f" {'BEST':>18}"
print(header)
for M in M_VALUES:
    row = results.get(M, {})
    vals = []
    for k in all_keys:
        v = row.get(k)
        vals.append(f"{v:>18.1f}" if v else f"{'---':>18}")
    best_name = min(row, key=row.get) if row else "---"
    best_val = f"{row[best_name]:.1f}" if row else "---"
    print(f"{M:>5} " + " ".join(vals) + f" {best_name+': '+best_val:>18}")

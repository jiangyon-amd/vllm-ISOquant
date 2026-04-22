#!/usr/bin/env python3
"""Standalone benchmark for TQ Store (MSE path).

Measures the full TQ store pipeline:
  1. key.float() + reshape           (dtype cast)
  2. k_flat.norm()                   (L2 norm)
  3. torch.mm(k_flat, PiT)           (GEMM rotation)
  4. y / (norms + 1e-8)              (normalize)
  5. value.float() + reshape          (dtype cast)
  6. _tq_fused_store_mse             (Triton bucketize + pack + value quant)

Usage:
    python benchmark_tq_store.py                # default params
    python benchmark_tq_store.py --sweep        # sweep over N and H
    python benchmark_tq_store.py --N 2 --H 2    # 72B TP=4 decode (2 tokens, 2 KV heads)
"""
import argparse
import math
import time
import torch

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

DEVICE = "cuda:0"
PRESET = "turboquant_4bit_nc"
D = 128       # head_dim
BS = 16       # block_size (pages)


def setup():
    cfg = TurboQuantConfig.from_cache_dtype(PRESET, D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    return cfg, PiT, centroids, midpoints


def benchmark_store(N: int, H: int, warmup: int = 20, iters: int = 100):
    """Benchmark TQ store for N tokens, H KV heads."""
    cfg, PiT, centroids, midpoints = setup()

    num_blocks = max(512, (N * 2 // BS) + 128)
    kv_cache = torch.zeros(num_blocks, BS, H, cfg.slot_size_aligned,
                           dtype=torch.uint8, device=DEVICE)

    key = torch.randn(N, H, D, dtype=torch.bfloat16, device=DEVICE)
    value = torch.randn(N, H, D, dtype=torch.bfloat16, device=DEVICE)
    slot_mapping = torch.arange(N, device=DEVICE, dtype=torch.int64)

    def run():
        triton_turboquant_store(
            key, value, kv_cache, slot_mapping,
            PiT, centroids, midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )

    # Warmup
    for _ in range(warmup):
        run()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    avg_us = (time.perf_counter() - t0) / iters * 1e6

    print(f"  N={N:>5}, H={H:>2}: {avg_us:>8.1f} us/call")
    return avg_us


def benchmark_store_breakdown(N: int, H: int, warmup: int = 20, iters: int = 100):
    """Benchmark individual operations in the TQ store pipeline."""
    cfg, PiT, centroids, midpoints = setup()

    from vllm.v1.attention.ops.triton_turboquant_store import _tq_fused_store_mse
    import triton

    num_blocks = max(512, (N * 2 // BS) + 128)
    kv_cache = torch.zeros(num_blocks, BS, H, cfg.slot_size_aligned,
                           dtype=torch.uint8, device=DEVICE)

    key = torch.randn(N, H, D, dtype=torch.bfloat16, device=DEVICE)
    value = torch.randn(N, H, D, dtype=torch.bfloat16, device=DEVICE)
    slot_mapping = torch.arange(N, device=DEVICE, dtype=torch.int64)

    NH = N * H
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * cfg.key_mse_bits / 8)
    n_centroids = 2 ** cfg.key_mse_bits
    val_data_bytes = math.ceil(D * cfg.effective_value_quant_bits / 8)
    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
    num_kv_heads = kv_cache.shape[2]
    padded_slot = kv_cache.shape[3]
    stride_block = BS * num_kv_heads * padded_slot
    stride_pos = num_kv_heads * padded_slot
    stride_head = padded_slot
    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1

    # Step 5: Triton kernel
    def step_triton(y, norms, v_flat):
        _tq_fused_store_mse[(NH,)](
            y, norms.squeeze(1), v_flat, centroids, midpoints,
            kv_cache.view(-1), slot_mapping,
            stride_cache_block=stride_block, stride_cache_pos=stride_pos,
            stride_cache_head=stride_head,
            D=D, H=H, BLOCK_SIZE=BS, BLOCK_D=BLOCK_D,
            MSE_BYTES=mse_bytes, KPS=cfg.key_packed_size,
            VQB=cfg.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL, MSE_BITS=cfg.key_mse_bits,
            N_CENTROIDS=n_centroids, BLOCK_GRP=block_grp,
            num_warps=4, num_stages=1,
        )

    results = {}

    # === Step 1: float cast + reshape ===
    def fn_cast():
        return key.float().reshape(NH, D), value.float().reshape(NH, D)
    for _ in range(warmup):
        fn_cast()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn_cast()
    torch.cuda.synchronize()
    results["1_cast_reshape"] = (time.perf_counter() - t0) / iters * 1e6

    # === Step 2: norm ===
    k_flat = key.float().reshape(NH, D)
    def fn_norm():
        return k_flat.norm(dim=1, keepdim=True)
    for _ in range(warmup):
        fn_norm()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn_norm()
    torch.cuda.synchronize()
    results["2_norm"] = (time.perf_counter() - t0) / iters * 1e6

    # === Step 3: GEMM rotation ===
    def fn_gemm():
        return torch.mm(k_flat, PiT)
    for _ in range(warmup):
        fn_gemm()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn_gemm()
    torch.cuda.synchronize()
    results["3_gemm_kPiT"] = (time.perf_counter() - t0) / iters * 1e6

    # === Step 4: normalize ===
    norms = k_flat.norm(dim=1, keepdim=True)
    y = torch.mm(k_flat, PiT)
    def fn_normalize():
        return (y / (norms + 1e-8)).contiguous()
    for _ in range(warmup):
        fn_normalize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn_normalize()
    torch.cuda.synchronize()
    results["4_normalize"] = (time.perf_counter() - t0) / iters * 1e6

    # === Step 5: Triton kernel ===
    y_norm = (y / (norms + 1e-8)).contiguous()
    v_flat = value.float().reshape(NH, D)
    for _ in range(warmup):
        step_triton(y_norm, norms, v_flat)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step_triton(y_norm, norms, v_flat)
    torch.cuda.synchronize()
    results["5_triton_kernel"] = (time.perf_counter() - t0) / iters * 1e6

    # === Full pipeline ===
    def fn_full():
        triton_turboquant_store(
            key, value, kv_cache, slot_mapping,
            PiT, centroids, midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )
    for _ in range(warmup):
        fn_full()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn_full()
    torch.cuda.synchronize()
    results["FULL_pipeline"] = (time.perf_counter() - t0) / iters * 1e6

    print(f"\n  Breakdown for N={N}, H={H}:")
    step_total = 0
    for name, us in results.items():
        if name != "FULL_pipeline":
            step_total += us
        print(f"    {name:20s}: {us:>8.1f} us")
    print(f"    {'STEP_SUM':20s}: {step_total:>8.1f} us")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TQ Store (MSE path)")
    parser.add_argument("--N", type=int, default=8192, help="Number of tokens")
    parser.add_argument("--H", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", help="Run sweep")
    parser.add_argument("--breakdown", action="store_true", help="Show per-step breakdown")
    args = parser.parse_args()

    if args.breakdown:
        for N in [2, 8, 32, 128, 512, 2048, 8192]:
            benchmark_store_breakdown(N, args.H, args.warmup, args.iters)
    elif args.sweep:
        print(f"{'N':>6} {'H':>3} {'us/call':>10}")
        print("-" * 24)
        # Decode scenarios (few tokens)
        for N in [1, 2, 4, 8, 16, 32]:
            for H in [2, 8]:
                benchmark_store(N, H, args.warmup, args.iters)
        print()
        # Prefill scenarios (many tokens)
        for N in [128, 512, 2048, 8192]:
            for H in [2, 8]:
                benchmark_store(N, H, args.warmup, args.iters)
    else:
        benchmark_store(args.N, args.H, args.warmup, args.iters)


if __name__ == "__main__":
    main()

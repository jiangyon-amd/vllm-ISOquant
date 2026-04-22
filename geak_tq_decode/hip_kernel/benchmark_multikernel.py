#!/usr/bin/env python3
"""Benchmark 3 Stage1 kernel variants head-to-head.

Compares: v52 (2-warp/64t), 4-warp (128t), 8-warp (256t)
All share the same ABI: launch_tq_decode_stage1(...)

Usage:
    HIP_VISIBLE_DEVICES=2 TQ_ALLOW_STALE_HIP_SO=1 python3 geak_tq_decode/hip_kernel/benchmark_multikernel.py
"""
import ctypes
import math
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

DEVICE = "cuda:0"
PRESET = "turboquant_4bit_nc"
D = 128
Hk = 8
Hq = 64
BS = 16


def load_so(name):
    ops_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "vllm", "v1", "attention", "ops"
    )
    path = os.path.join(ops_dir, name)
    if not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    fn = lib.launch_tq_decode_stage1
    fn.argtypes = (
        [ctypes.c_void_p] * 6
        + [ctypes.c_int] * 2
        + [ctypes.c_int] * 3
        + [ctypes.c_int]
        + [ctypes.c_int] * 3
        + [ctypes.c_int] * 4
        + [ctypes.c_float]
        + [ctypes.c_int]
        + [ctypes.c_int] * 2
        + [ctypes.c_void_p]
    )
    fn.restype = None
    return fn


def setup():
    cfg = TurboQuantConfig.from_cache_dtype(PRESET, D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    Pi = PiT.T.contiguous()
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    return cfg, Pi, PiT, centroids, midpoints


def benchmark_kernel(fn, q_rot, kv_cache, block_table, seq_lens_t, centroids_f32, mid_o,
                     B, block_size, NUM_KV_SPLITS, kv_group_size, scale, warmup=10, iters=50):
    stream = torch.cuda.current_stream().cuda_stream
    nc = 1

    for _ in range(warmup):
        fn(
            q_rot.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
            seq_lens_t.data_ptr(), centroids_f32.data_ptr(), mid_o.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            Hk, block_size, NUM_KV_SPLITS, kv_group_size,
            scale, nc, B, Hq,
            ctypes.c_void_p(stream),
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(
            q_rot.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
            seq_lens_t.data_ptr(), centroids_f32.data_ptr(), mid_o.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            Hk, block_size, NUM_KV_SPLITS, kv_group_size,
            scale, nc, B, Hq,
            ctypes.c_void_p(stream),
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    cfg, Pi, PiT, centroids, midpoints = setup()
    centroids_f32 = centroids.float().contiguous()
    scale = 1.0 / math.sqrt(D)
    kv_group_size = Hq // Hk
    NUM_KV_SPLITS = 32

    # Load kernels
    kernels = {}
    for name, so_name in [
        ("v52_2w", "tq_decode_hip.so"),
        ("4warp", "tq_decode_4warp_hip.so"),
        ("8warp", "tq_decode_8warp_hip.so"),
    ]:
        fn = load_so(so_name)
        if fn is not None:
            kernels[name] = fn
            print(f"  Loaded {so_name} → {name}")
        else:
            print(f"  MISSING {so_name} → {name} SKIPPED")

    if not kernels:
        print("No kernels loaded!")
        return

    # Workload matrix
    workloads = [
        (4, 512), (4, 2048), (4, 4096), (4, 8192), (4, 9216),
        (20, 512), (20, 2048), (20, 4096), (20, 8192),
        (32, 128), (32, 512), (32, 1024),
        (100, 128), (100, 512),
    ]

    # Header
    knames = list(kernels.keys())
    hdr = f"{'B':>4} {'seq':>5}"
    for k in knames:
        hdr += f" {k:>10}"
    if len(knames) >= 2:
        hdr += f" {'delta':>8} {'winner':>8}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for B, seq_len in workloads:
        num_blocks = max(8192, (B * seq_len // BS) + 1024)
        kv_cache = torch.zeros(num_blocks, BS, Hk, cfg.slot_size_aligned,
                               dtype=torch.uint8, device=DEVICE)
        fill_n = min(B * seq_len, num_blocks * BS)
        fk = torch.randn(fill_n, Hk, D, dtype=torch.bfloat16, device=DEVICE)
        fv = torch.randn_like(fk)
        triton_turboquant_store(
            fk, fv, kv_cache,
            torch.arange(fill_n, device=DEVICE, dtype=torch.int64),
            PiT, centroids, midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )

        q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
        q_rot = (q.float() @ PiT).contiguous()
        bps = math.ceil(seq_len / BS)
        block_table = torch.arange(bps, device=DEVICE, dtype=torch.int32) \
            .unsqueeze(0).expand(B, -1).contiguous()
        seq_lens_t = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)
        mid_o = torch.empty(B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=DEVICE)

        results = {}
        for kname, fn in kernels.items():
            us = benchmark_kernel(fn, q_rot, kv_cache, block_table, seq_lens_t,
                                  centroids_f32, mid_o, B, BS, NUM_KV_SPLITS,
                                  kv_group_size, scale)
            results[kname] = us

        row = f"{B:>4} {seq_len:>5}"
        for k in knames:
            row += f" {results[k]:>10.1f}"
        if len(knames) >= 2:
            ref = results[knames[0]]
            best_k = min(knames, key=lambda k: results[k])
            best_v = results[best_k]
            delta_pct = (best_v - ref) / ref * 100
            row += f" {delta_pct:>+7.1f}% {best_k:>8}"
        print(row)

    print("\nAll times in us (Stage1 only, lower is better)")


if __name__ == "__main__":
    main()

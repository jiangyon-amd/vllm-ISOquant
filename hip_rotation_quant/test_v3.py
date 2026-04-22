#!/usr/bin/env python3
"""Quick test for v3 kernel (32x32x16 MFMA)."""
import torch
import ctypes
import os
import sys
import time

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO = os.path.join(DIR, "dense_mfma_rot_quant.so")
sys.path.insert(0, "/data/jiangyon/vllm_rotation")

lib = ctypes.CDLL(SO)

def test_correctness_v3(M, K, RS=128, shuffle=False):
    """Compare v3 vs Gluon v2 reference."""
    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import fused_rot_quant_gluon

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rot = torch.randn(RS, RS, dtype=torch.bfloat16, device="cuda")

    fp4_ref, sc_ref = fused_rot_quant_gluon(x, rot, rotation_size=RS, shuffle_scales=shuffle)
    torch.cuda.synchronize()

    fp4_hip = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
    n_scales = K // 32
    sn_pad = (n_scales + 7) // 8 * 8 if shuffle else n_scales
    sm_pad = (M + 255) // 256 * 256 if shuffle else M
    sc_hip = torch.zeros(max(sm_pad, M), max(sn_pad, n_scales), dtype=torch.uint8, device="cuda")

    lib.launch_dense_mfma_rot_quant_v3(
        ctypes.c_void_p(fp4_hip.data_ptr()),
        ctypes.c_void_p(sc_hip.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(rot.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(K),
        ctypes.c_int(x.stride(0)), ctypes.c_int(fp4_hip.stride(0)),
        ctypes.c_int(sc_hip.stride(0)), ctypes.c_int(sn_pad),
        ctypes.c_int(1 if shuffle else 0),
        ctypes.c_void_p(0))
    torch.cuda.synchronize()

    fp4_match = (fp4_hip == fp4_ref).float().mean().item() * 100
    sc_match = (sc_hip[:sc_ref.shape[0], :sc_ref.shape[1]] == sc_ref).float().mean().item() * 100
    return fp4_match, sc_match

def benchmark_v3(M, K, RS=128, shuffle=False, warmup=50, iters=200):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rot = torch.randn(RS, RS, dtype=torch.bfloat16, device="cuda")
    fp4 = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
    n_scales = K // 32
    sn_pad = (n_scales + 7) // 8 * 8 if shuffle else n_scales
    sm_pad = (M + 255) // 256 * 256 if shuffle else M
    sc = torch.zeros(max(sm_pad, M), max(sn_pad, n_scales), dtype=torch.uint8, device="cuda")

    def run():
        lib.launch_dense_mfma_rot_quant_v3(
            ctypes.c_void_p(fp4.data_ptr()), ctypes.c_void_p(sc.data_ptr()),
            ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(rot.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(K),
            ctypes.c_int(x.stride(0)), ctypes.c_int(fp4.stride(0)),
            ctypes.c_int(sc.stride(0)), ctypes.c_int(sn_pad),
            ctypes.c_int(1 if shuffle else 0), ctypes.c_void_p(0))

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e6

def benchmark_v1(M, K, RS=128, shuffle=False, warmup=50, iters=200):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rot = torch.randn(RS, RS, dtype=torch.bfloat16, device="cuda")
    fp4 = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
    n_scales = K // 32
    sn_pad = (n_scales + 7) // 8 * 8 if shuffle else n_scales
    sm_pad = (M + 255) // 256 * 256 if shuffle else M
    sc = torch.zeros(max(sm_pad, M), max(sn_pad, n_scales), dtype=torch.uint8, device="cuda")

    def run():
        lib.launch_dense_mfma_rot_quant_v1(
            ctypes.c_void_p(fp4.data_ptr()), ctypes.c_void_p(sc.data_ptr()),
            ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(rot.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(K),
            ctypes.c_int(x.stride(0)), ctypes.c_int(fp4.stride(0)),
            ctypes.c_int(sc.stride(0)), ctypes.c_int(sn_pad),
            ctypes.c_int(1 if shuffle else 0), ctypes.c_void_p(0))

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e6

print("=== V3 (32x32x16 MFMA) Test ===")
print("\n--- Correctness ---")
for M in [1, 32]:
    for K in [128, 4096]:
        fp4_m, sc_m = test_correctness_v3(M, K, shuffle=False)
        status = "✓" if fp4_m > 85 else "✗"
        print(f"  {status} M={M:4d} K={K:4d}: FP4={fp4_m:5.1f}% scale={sc_m:5.1f}%")

print("\n--- Performance (µs, K=4096) ---")
print(f"  {'M':>4} {'v1 (16x16x32)':>14} {'v3 (32x32x16)':>14} {'speedup':>10}")
print(f"  {'-'*50}")
for M in [1, 4, 32]:
    t_v1 = benchmark_v1(M, 4096, shuffle=False)
    t_v3 = benchmark_v3(M, 4096, shuffle=False)
    speedup = t_v1 / t_v3
    print(f"  {M:4d} {t_v1:14.1f}µs {t_v3:14.1f}µs {speedup:10.2f}x")

#!/usr/bin/env python3
"""Test all kernel versions."""
import torch
import ctypes
import os
import sys
import time

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO = os.path.join(DIR, "dense_mfma_rot_quant.so")
sys.path.insert(0, "/data/jiangyon/vllm_rotation")

lib = ctypes.CDLL(SO)

def benchmark(lib, func_name, M, K, RS=128, shuffle=False, warmup=100, iters=500):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rot = torch.randn(RS, RS, dtype=torch.bfloat16, device="cuda")
    fp4 = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
    n_scales = K // 32
    sn_pad = (n_scales + 7) // 8 * 8 if shuffle else n_scales
    sm_pad = (M + 255) // 256 * 256 if shuffle else M
    sc = torch.zeros(max(sm_pad, M), max(sn_pad, n_scales), dtype=torch.uint8, device="cuda")

    func = getattr(lib, func_name)
    def run():
        func(
            ctypes.c_void_p(fp4.data_ptr()), ctypes.c_void_p(sc.data_ptr()),
            ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(rot.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(K),
            ctypes.c_int(x.stride(0)), ctypes.c_int(fp4.stride(0)),
            ctypes.c_int(sc.stride(0)), ctypes.c_int(sn_pad),
            ctypes.c_int(1 if shuffle else 0), ctypes.c_void_p(0))

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e6

print("=== All Kernel Versions Performance (µs, K=4096) ===")
print(f"  {'M':>4} {'v1':>8} {'v2':>8} {'v4':>8} {'v8':>8} {'best':>8}")
print(f"  {'-'*50}")

for M in [1, 4, 32]:
    t_v1 = benchmark(lib, "launch_dense_mfma_rot_quant_v1", M, 4096)
    t_v2 = benchmark(lib, "launch_dense_mfma_rot_quant_v2", M, 4096)
    t_v4 = benchmark(lib, "launch_dense_mfma_rot_quant_v4", M, 4096)
    t_v8 = benchmark(lib, "launch_dense_mfma_rot_quant_v8", M, 4096)
    best = min(t_v1, t_v2, t_v4, t_v8)
    best_name = ["v1", "v2", "v4", "v8"][[t_v1, t_v2, t_v4, t_v8].index(best)]
    print(f"  {M:4d} {t_v1:8.1f} {t_v2:8.1f} {t_v4:8.1f} {t_v8:8.1f} {best_name:>8}")

print("\nTarget: M=1 < 5µs, M=32 < 7µs")

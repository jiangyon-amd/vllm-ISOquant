#!/usr/bin/env python3
"""Debug performance discrepancy."""
import torch
import ctypes
import time
import os
import math

HIP_DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO_PATH = f"{HIP_DIR}/mfma_rot_quant_moe_sort.so"

E, TOPK, K, RS, QG = 128, 8, 2048, 128, 32
N_I = K // QG
DEVICE = "cuda"

import sys
sys.path.insert(0, "/data/jiangyon/vllm_rotation")
from aiter.fused_moe import moe_sorting

def build_inputs(M):
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
    rot_int8 = torch.ones(RS, RS, dtype=torch.int8, device=DEVICE)
    for i in range(RS):
        for j in range(RS):
            if bin(i & j).count("1") % 2 == 1:
                rot_int8[i, j] = -1
    rotation = (rot_int8.float() / math.sqrt(RS)).to(torch.bfloat16)

    topk_ids = torch.randint(0, E, (M, TOPK), device=DEVICE, dtype=torch.int32)
    topk_w = torch.ones(M, TOPK, device=DEVICE, dtype=torch.float32) / TOPK
    sorted_ids, _, _, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_w, E, K, torch.bfloat16, 32, None
    )
    m_o = sorted_ids.shape[0]
    m_pad = ((m_o + 31) // 32) * 32
    return x, rotation, sorted_ids, num_valid_ids, m_o, m_pad

def benchmark_with_events(M, warmup=200, repeat=500):
    """Use CUDA events for more accurate timing."""
    x, rot, si, nv, m_o, m_pad = build_inputs(M)
    
    lib = ctypes.CDLL(SO_PATH)
    
    fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=DEVICE)
    sc = torch.zeros((m_pad, N_I), dtype=torch.uint8, device=DEVICE)
    
    stream = torch.cuda.current_stream().cuda_stream
    
    # Warmup
    for _ in range(warmup):
        lib.launch_mfma_rot_quant_moe_sort(
            ctypes.c_void_p(fp4.data_ptr()),
            ctypes.c_void_p(sc.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(rot.data_ptr()),
            ctypes.c_void_p(si.data_ptr()),
            ctypes.c_void_p(nv.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(M),
            ctypes.c_int(m_o), ctypes.c_int(m_pad),
            ctypes.c_int(x.stride(0)), ctypes.c_int(fp4.stride(0)),
            ctypes.c_int(sc.stride(0)), ctypes.c_int(sc.stride(1)),
            ctypes.c_int(N_I),
            ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    
    # Benchmark with events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(repeat):
        lib.launch_mfma_rot_quant_moe_sort(
            ctypes.c_void_p(fp4.data_ptr()),
            ctypes.c_void_p(sc.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(rot.data_ptr()),
            ctypes.c_void_p(si.data_ptr()),
            ctypes.c_void_p(nv.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(M),
            ctypes.c_int(m_o), ctypes.c_int(m_pad),
            ctypes.c_int(x.stride(0)), ctypes.c_int(fp4.stride(0)),
            ctypes.c_int(sc.stride(0)), ctypes.c_int(sc.stride(1)),
            ctypes.c_int(N_I),
            ctypes.c_void_p(stream))
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / repeat * 1000  # us

def benchmark_perf_counter(M, warmup=200, repeat=500):
    """Use perf_counter like the test script."""
    x, rot, si, nv, m_o, m_pad = build_inputs(M)
    
    lib = ctypes.CDLL(SO_PATH)
    
    fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=DEVICE)
    sc = torch.zeros((m_pad, N_I), dtype=torch.uint8, device=DEVICE)
    
    stream = torch.cuda.current_stream().cuda_stream
    
    # Warmup
    for _ in range(warmup):
        lib.launch_mfma_rot_quant_moe_sort(
            ctypes.c_void_p(fp4.data_ptr()),
            ctypes.c_void_p(sc.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(rot.data_ptr()),
            ctypes.c_void_p(si.data_ptr()),
            ctypes.c_void_p(nv.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(M),
            ctypes.c_int(m_o), ctypes.c_int(m_pad),
            ctypes.c_int(x.stride(0)), ctypes.c_int(fp4.stride(0)),
            ctypes.c_int(sc.stride(0)), ctypes.c_int(sc.stride(1)),
            ctypes.c_int(N_I),
            ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    
    # Benchmark with perf_counter
    t0 = time.perf_counter()
    for _ in range(repeat):
        lib.launch_mfma_rot_quant_moe_sort(
            ctypes.c_void_p(fp4.data_ptr()),
            ctypes.c_void_p(sc.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(rot.data_ptr()),
            ctypes.c_void_p(si.data_ptr()),
            ctypes.c_void_p(nv.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(M),
            ctypes.c_int(m_o), ctypes.c_int(m_pad),
            ctypes.c_int(x.stride(0)), ctypes.c_int(fp4.stride(0)),
            ctypes.c_int(sc.stride(0)), ctypes.c_int(sc.stride(1)),
            ctypes.c_int(N_I),
            ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - t0) / repeat * 1e6

def main():
    print("=" * 60)
    print("Debug: Comparing timing methods")
    print("=" * 60)
    
    print(f"\n{'M':>4} | {'Events (us)':>12} | {'perf_counter (us)':>18}")
    print("-" * 50)
    
    for M in [1, 2, 4, 8, 16, 32]:
        lat_events = benchmark_with_events(M)
        lat_perf = benchmark_perf_counter(M)
        print(f"{M:4d} | {lat_events:12.1f} | {lat_perf:18.1f}")

if __name__ == "__main__":
    main()

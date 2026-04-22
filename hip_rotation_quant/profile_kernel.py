#!/usr/bin/env python3
"""Profile kernel with minimal overhead."""
import torch
import ctypes
import time
import math
import sys

sys.path.insert(0, "/data/jiangyon/vllm_rotation")
from aiter.fused_moe import moe_sorting

HIP_DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO_PATH = f"{HIP_DIR}/mfma_rot_quant_moe_sort.so"

E, TOPK, K, RS, QG = 128, 8, 2048, 128, 32
N_I = K // QG
DEVICE = "cuda"

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

# Pre-build all inputs
print("Building inputs...")
inputs = {M: build_inputs(M) for M in [1, 2, 4, 8, 16, 32]}

# Load library once
lib = ctypes.CDLL(SO_PATH)

# Pre-allocate outputs
outputs = {}
for M in [1, 2, 4, 8, 16, 32]:
    _, _, _, _, m_o, m_pad = inputs[M]
    outputs[M] = (
        torch.empty((M, K // 2), dtype=torch.uint8, device=DEVICE),
        torch.zeros((m_pad, N_I), dtype=torch.uint8, device=DEVICE)
    )

stream = torch.cuda.current_stream().cuda_stream

print("\nBenchmark with pre-allocated inputs/outputs:")
print(f"{'M':>4} | {'Events (us)':>12}")
print("-" * 25)

for M in [1, 2, 4, 8, 16, 32]:
    x, rot, si, nv, m_o, m_pad = inputs[M]
    fp4, sc = outputs[M]
    
    # Warmup
    for _ in range(200):
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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    repeat = 500
    start.record()
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
    end.record()
    torch.cuda.synchronize()
    
    lat = start.elapsed_time(end) / repeat * 1000  # us
    status = "PASS" if lat < 12 else "FAIL"
    print(f"{M:4d} | {lat:12.1f} | {status}")

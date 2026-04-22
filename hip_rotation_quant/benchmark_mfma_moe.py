#!/usr/bin/env python3
"""Benchmark mfma_rot_quant_moe_sort kernel latency."""
import torch
import ctypes
import time
import os
import subprocess

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"

def compile_kernel():
    """Compile HIP kernel to .so if needed."""
    hip_path = os.path.join(DIR, "mfma_rot_quant_moe_sort.hip")
    so_path = os.path.join(DIR, "mfma_rot_quant_moe_sort.so")
    
    if os.path.exists(so_path) and os.path.getmtime(so_path) > os.path.getmtime(hip_path):
        print(f"  {so_path} is up-to-date")
        return so_path
    
    print(f"  Compiling {hip_path} -> {so_path} ...")
    cmd = [
        "hipcc", "-shared", "-fPIC", "-O3",
        "--offload-arch=gfx950",
        "-o", so_path, hip_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COMPILE ERROR:\n{result.stderr}")
        raise RuntimeError("Compilation failed")
    print(f"  Compiled OK")
    return so_path

def benchmark(M, K=7168, n_i=224, topk=8, warmup=10, iters=100):
    """Benchmark kernel for given M."""
    RS = 128
    m_o = topk
    m_pad = 32
    
    # Setup tensors
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rot = torch.randn(RS, RS, dtype=torch.bfloat16, device="cuda")
    fp4_out = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
    sc_out = torch.zeros(m_pad, n_i, dtype=torch.uint8, device="cuda")
    
    # Sorted IDs for MoE
    sorted_ids = torch.arange(m_o, dtype=torch.int32, device="cuda")
    sorted_ids = torch.cat([sorted_ids, torch.full((m_pad - m_o,), 0, dtype=torch.int32, device="cuda")])
    num_valid = torch.tensor([m_o], dtype=torch.int32, device="cuda")
    
    # Load library
    so_path = compile_kernel()
    lib = ctypes.CDLL(so_path)
    
    # Warmup
    for _ in range(warmup):
        lib.launch_mfma_rot_quant_moe_sort(
            ctypes.c_void_p(fp4_out.data_ptr()),
            ctypes.c_void_p(sc_out.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(rot.data_ptr()),
            ctypes.c_void_p(sorted_ids.data_ptr()),
            ctypes.c_void_p(num_valid.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(K),
            ctypes.c_int(1),  # token_num
            ctypes.c_int(m_o), ctypes.c_int(m_pad),
            ctypes.c_int(x.stride(0)),
            ctypes.c_int(fp4_out.stride(0)),
            ctypes.c_int(sc_out.stride(0)),
            ctypes.c_int(sc_out.stride(1)),
            ctypes.c_int(n_i),
            ctypes.c_void_p(0))
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    for i in range(iters):
        start_events[i].record()
        lib.launch_mfma_rot_quant_moe_sort(
            ctypes.c_void_p(fp4_out.data_ptr()),
            ctypes.c_void_p(sc_out.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(rot.data_ptr()),
            ctypes.c_void_p(sorted_ids.data_ptr()),
            ctypes.c_void_p(num_valid.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(K),
            ctypes.c_int(1),  # token_num
            ctypes.c_int(m_o), ctypes.c_int(m_pad),
            ctypes.c_int(x.stride(0)),
            ctypes.c_int(fp4_out.stride(0)),
            ctypes.c_int(sc_out.stride(0)),
            ctypes.c_int(sc_out.stride(1)),
            ctypes.c_int(n_i),
            ctypes.c_void_p(0))
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [start_events[i].elapsed_time(end_events[i]) * 1000 for i in range(iters)]  # us
    times.sort()
    
    # Remove outliers (top/bottom 10%)
    trim = iters // 10
    trimmed = times[trim:-trim] if trim > 0 else times
    
    avg = sum(trimmed) / len(trimmed)
    min_t = min(times)
    max_t = max(times)
    
    return avg, min_t, max_t

def main():
    print("=" * 60)
    print("MFMA Rotation + Quant + MoE Sort Kernel Benchmark")
    print("=" * 60)
    print(f"K=7168, n_i=224, topk=8")
    print()
    
    compile_kernel()
    
    print(f"{'M':>4} | {'Avg (us)':>10} | {'Min (us)':>10} | {'Max (us)':>10} | Target")
    print("-" * 60)
    
    targets = {1: 12, 2: 12, 4: 12, 8: 15, 16: 18, 32: 22}
    
    for M in [1, 2, 4, 8, 16, 32]:
        avg, min_t, max_t = benchmark(M)
        target = targets.get(M, 20)
        status = "PASS" if avg < target else "FAIL"
        print(f"{M:4d} | {avg:10.2f} | {min_t:10.2f} | {max_t:10.2f} | <{target}us {status}")

if __name__ == "__main__":
    main()

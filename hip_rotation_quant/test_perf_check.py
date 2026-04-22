#!/usr/bin/env python3
"""Check performance with different configurations."""
import torch
import ctypes
import os
import subprocess

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"

def compile_kernel():
    hip_path = os.path.join(DIR, "mfma_rot_quant_moe_sort.hip")
    so_path = os.path.join(DIR, "mfma_rot_quant_moe_sort.so")
    
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

def benchmark(M, K, n_i, topk=8, warmup=10, iters=100):
    """Benchmark kernel for given M."""
    RS = 128
    m_o = topk
    m_pad = ((m_o + 31) // 32) * 32
    
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
    so_path = os.path.join(DIR, "mfma_rot_quant_moe_sort.so")
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
            ctypes.c_int(M),  # token_num
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
            ctypes.c_int(M),  # token_num
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
    return avg

def main():
    print("=" * 70)
    print("Testing different configurations")
    print("=" * 70)
    
    compile_kernel()
    
    # Test different K and n_i values
    configs = [
        # (K, n_i) - common configurations
        (7168, 224),   # K/32 = 224
        (8192, 256),   # K/32 = 256
        (2048, 64),    # K/32 = 64
    ]
    
    print(f"\n{'M':>4} | {'K':>6} | {'n_i':>4} | {'Latency (us)':>12}")
    print("-" * 40)
    
    for K, n_i in configs:
        for M in [1, 4, 8]:
            avg = benchmark(M, K, n_i)
            print(f"{M:4d} | {K:6d} | {n_i:4d} | {avg:12.2f}")
        print("-" * 40)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Debug ds_read_tr behavior - Part 5."""

import ctypes
import torch
import os
import sys

HIP_DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO_PATH = f"{HIP_DIR}/debug_ds_read_tr5.so"
HIP_PATH = f"{HIP_DIR}/debug_ds_read_tr5.hip"

def compile_kernel():
    print(f"Compiling {HIP_PATH} ...")
    ret = os.system(f"hipcc --offload-arch=gfx950 -shared -fPIC -O3 -o {SO_PATH} {HIP_PATH} 2>&1")
    if ret != 0:
        print("COMPILATION FAILED")
        sys.exit(1)
    print("Compilation OK")

def main():
    compile_kernel()
    lib = ctypes.CDLL(SO_PATH)
    
    output = torch.zeros(4096, dtype=torch.float32, device="cuda")
    
    stream = torch.cuda.current_stream().cuda_stream
    lib.launch_debug_ds_read_tr5(
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(stream),
    )
    torch.cuda.synchronize()
    
    print("=" * 70)
    print("Test 1: Interleaved layout R_interleaved[K/4][N][4]")
    print("=" * 70)
    print("R[k][n] = k * 16 + n")
    print("Expected: b_data[j] = R[k_off + j][lane16] = (k_off + j) * 16 + lane16")
    print()
    
    all_match = True
    for k_group in range(4):
        print(f"k_group {k_group} (k_off = {k_group * 8}):")
        for lane16 in [0, 1, 15]:
            lane = k_group * 16 + lane16
            result = output[lane * 8 : lane * 8 + 8].tolist()
            expected = output[512 + lane * 8 : 512 + lane * 8 + 8].tolist()
            match = all(abs(r - e) < 0.01 for r, e in zip(result, expected))
            if not match:
                all_match = False
            print(f"  Lane {lane:2d} (lane16={lane16}): got {result}")
            print(f"                       expected {expected} {'OK' if match else 'MISMATCH'}")
        print()
    
    print(f"Test 1 overall: {'ALL OK' if all_match else 'SOME MISMATCH'}")
    
    print()
    print("=" * 70)
    print("Test 2: Full MFMA B loading")
    print("=" * 70)
    
    all_match = True
    base = 1024
    for k_group in range(4):
        print(f"k_group {k_group}:")
        for lane16 in [0, 1, 15]:
            lane = k_group * 16 + lane16
            result = output[base + lane * 8 : base + lane * 8 + 8].tolist()
            expected = output[base + 512 + lane * 8 : base + 512 + lane * 8 + 8].tolist()
            match = all(abs(r - e) < 0.01 for r, e in zip(result, expected))
            if not match:
                all_match = False
            print(f"  Lane {lane:2d}: got {result}")
            print(f"          expected {expected} {'OK' if match else 'MISMATCH'}")
        print()
    
    print(f"Test 2 overall: {'ALL OK' if all_match else 'SOME MISMATCH'}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Debug ds_read_tr behavior - Part 4."""

import ctypes
import torch
import os
import sys

HIP_DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO_PATH = f"{HIP_DIR}/debug_ds_read_tr4.so"
HIP_PATH = f"{HIP_DIR}/debug_ds_read_tr4.hip"

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
    lib.launch_debug_ds_read_tr4(
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(stream),
    )
    torch.cuda.synchronize()
    
    print("=" * 70)
    print("Test 1: 4x16 layout, lds[row][col] = row * 100 + col")
    print("=" * 70)
    for lane in range(16):
        result = output[lane * 4 : lane * 4 + 4].tolist()
        print(f"  Lane {lane:2d}: got {result}")
    
    print()
    print("=" * 70)
    print("Test 2: Linear 64 bf16, lds[i] = i, all lanes same base")
    print("=" * 70)
    base = 256
    for lane in range(16):
        result = output[base + lane * 4 : base + lane * 4 + 4].tolist()
        print(f"  Lane {lane:2d}: got {result}")
    
    print()
    print("=" * 70)
    print("Test 3: Linear 64 bf16, each lane reads from &lds[lane16]")
    print("=" * 70)
    base = 512
    for lane in range(16):
        result = output[base + lane * 4 : base + lane * 4 + 4].tolist()
        print(f"  Lane {lane:2d}: got {result}")

if __name__ == "__main__":
    main()

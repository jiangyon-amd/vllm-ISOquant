#!/usr/bin/env python3
"""Debug ds_read_tr behavior."""

import ctypes
import torch
import os
import sys

HIP_DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO_PATH = f"{HIP_DIR}/debug_ds_read_tr.so"
HIP_PATH = f"{HIP_DIR}/debug_ds_read_tr.hip"

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
    
    # Allocate output buffer
    output = torch.zeros(2048, dtype=torch.float32, device="cuda")
    
    stream = torch.cuda.current_stream().cuda_stream
    lib.launch_debug_ds_read_tr(
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(stream),
    )
    torch.cuda.synchronize()
    
    print("=" * 70)
    print("Test 1: All lanes read from same base address")
    print("=" * 70)
    print("LDS pattern: lds[row][col] = row * 16 + col")
    print("Expected: lane l gets [col_l_row0, col_l_row1, col_l_row2, col_l_row3]")
    print("          = [l, 16+l, 32+l, 48+l]")
    print()
    
    for lane in range(16):
        result = output[lane * 4 : lane * 4 + 4].tolist()
        expected = [lane, 16 + lane, 32 + lane, 48 + lane]
        match = all(abs(r - e) < 0.01 for r, e in zip(result, expected))
        print(f"  Lane {lane:2d}: got {result}, expected {expected}, {'OK' if match else 'MISMATCH'}")
    
    print()
    print("Lanes 16-63 (k_group 1-3):")
    for lane in [16, 32, 48]:
        result = output[lane * 4 : lane * 4 + 4].tolist()
        print(f"  Lane {lane:2d}: got {result}")
    
    print()
    print("=" * 70)
    print("Test 2: Each lane reads from base + lane16 * 2")
    print("=" * 70)
    base = 64 * 4
    for lane in range(16):
        result = output[base + lane * 4 : base + lane * 4 + 4].tolist()
        print(f"  Lane {lane:2d}: got {result}")
    
    print()
    print("=" * 70)
    print("Test 3: Tiled layout R_tiled[8][128][16], k_group addressing")
    print("=" * 70)
    print("Pattern: R_tiled[n_tile][k][col] = n_tile * 1000 + k * 16 + col")
    print("n_tile=0, k_off = k_group * 8")
    print()
    
    base = 128 * 4
    for k_group in range(4):
        print(f"k_group {k_group} (k_off = {k_group * 8}):")
        for lane16 in range(4):  # Just show first 4 lanes per k_group
            lane = k_group * 16 + lane16
            result = output[base + lane * 4 : base + lane * 4 + 4].tolist()
            expected = output[base + 256 + lane * 4 : base + 256 + lane * 4 + 4].tolist()
            match = all(abs(r - e) < 0.01 for r, e in zip(result, expected))
            print(f"  Lane {lane:2d} (lane16={lane16}): got {result}, expected {expected}, {'OK' if match else 'MISMATCH'}")
        print()

if __name__ == "__main__":
    main()

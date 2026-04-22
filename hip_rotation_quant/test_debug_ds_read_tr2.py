#!/usr/bin/env python3
"""Debug ds_read_tr behavior - Part 2."""

import ctypes
import torch
import os
import sys

HIP_DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO_PATH = f"{HIP_DIR}/debug_ds_read_tr2.so"
HIP_PATH = f"{HIP_DIR}/debug_ds_read_tr2.hip"

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
    lib.launch_debug_ds_read_tr2(
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(stream),
    )
    torch.cuda.synchronize()
    
    print("=" * 70)
    print("Test 1: Each lane reads from row = lane16")
    print("=" * 70)
    print("LDS pattern: lds[row][col] = row * 16 + col")
    print("Each lane reads from &lds[lane16][0]")
    print()
    
    for lane in range(16):
        result = output[lane * 4 : lane * 4 + 4].tolist()
        # If reading 4 consecutive bf16 from row lane16, col 0-3:
        expected_consecutive = [lane * 16 + i for i in range(4)]
        print(f"  Lane {lane:2d}: got {result}, consecutive would be {expected_consecutive}")
    
    print()
    print("=" * 70)
    print("Test 2: Each lane reads from col = lane16")
    print("=" * 70)
    print("LDS pattern: lds[row][col] = row * 16 + col (4 rows x 16 cols)")
    print("Each lane reads from &lds[0][lane16]")
    print()
    
    base = 256
    for lane in range(16):
        result = output[base + lane * 4 : base + lane * 4 + 4].tolist()
        # If reading 4 consecutive bf16 from row 0, starting at col lane16:
        # But row stride is 16 bf16 = 32 bytes, so consecutive reads wrap
        print(f"  Lane {lane:2d}: got {result}")
    
    print()
    print("=" * 70)
    print("Test 3: MFMA B pattern - column access from row-major LDS")
    print("=" * 70)
    print("LDS pattern: lds[k][n] = k * 16 + n (32 rows x 16 cols)")
    print("Each lane reads from &lds[k_group*8][lane16]")
    print("Expected: b_data[j] = lds[k_group*8 + j][lane16]")
    print()
    
    base = 512
    for k_group in range(4):
        print(f"k_group {k_group} (k_off = {k_group * 8}):")
        for lane16 in [0, 1, 15]:  # Sample lanes
            lane = k_group * 16 + lane16
            result = output[base + lane * 8 : base + lane * 8 + 8].tolist()
            expected = output[base + 512 + lane * 8 : base + 512 + lane * 8 + 8].tolist()
            match = all(abs(r - e) < 0.01 for r, e in zip(result, expected))
            print(f"  Lane {lane:2d} (lane16={lane16}): got {result}")
            print(f"                       expected {expected} {'OK' if match else 'MISMATCH'}")
        print()

if __name__ == "__main__":
    main()

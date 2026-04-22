#!/usr/bin/env python3
"""Debug ds_read_tr behavior - Part 3."""

import ctypes
import torch
import os
import sys

HIP_DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO_PATH = f"{HIP_DIR}/debug_ds_read_tr3.so"
HIP_PATH = f"{HIP_DIR}/debug_ds_read_tr3.hip"

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
    lib.launch_debug_ds_read_tr3(
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(stream),
    )
    torch.cuda.synchronize()
    
    print("=" * 70)
    print("Test 1: All lanes read from same base (contiguous 4x16 block)")
    print("=" * 70)
    print("LDS: lds[row][col] = row * 16 + col")
    print("Expected after transpose: lane l gets [l, 16+l, 32+l, 48+l]")
    print()
    
    for lane in range(16):
        result = output[lane * 4 : lane * 4 + 4].tolist()
        expected = [lane, 16 + lane, 32 + lane, 48 + lane]
        match = all(abs(r - e) < 0.01 for r, e in zip(result, expected))
        print(f"  Lane {lane:2d}: got {result}, expected {expected}, {'OK' if match else 'MISMATCH'}")
    
    print()
    print("=" * 70)
    print("Test 2: Each lane reads from &lds[0][lane16]")
    print("=" * 70)
    base = 256
    for lane in range(16):
        result = output[base + lane * 4 : base + lane * 4 + 4].tolist()
        print(f"  Lane {lane:2d}: got {result}")
    
    print()
    print("=" * 70)
    print("Test 3: Interleaved layout lds_interleaved[N][4]")
    print("=" * 70)
    print("lds_interleaved[n][j] = j * 16 + n")
    print("Each lane reads from &lds_interleaved[lane16][0]")
    print("Expected: [lane16, 16+lane16, 32+lane16, 48+lane16]")
    print()
    
    base = 512
    for lane in range(16):
        result = output[base + lane * 4 : base + lane * 4 + 4].tolist()
        expected = output[base + 256 + lane * 4 : base + 256 + lane * 4 + 4].tolist()
        match = all(abs(r - e) < 0.01 for r, e in zip(result, expected))
        print(f"  Lane {lane:2d}: got {result}, expected {expected}, {'OK' if match else 'MISMATCH'}")
    
    print()
    print("=" * 70)
    print("Test 4: Scalar read from interleaved layout")
    print("=" * 70)
    base = 768
    for lane in range(16):
        result = output[base + lane * 4 : base + lane * 4 + 4].tolist()
        expected = [i * 16 + lane for i in range(4)]
        match = all(abs(r - e) < 0.01 for r, e in zip(result, expected))
        print(f"  Lane {lane:2d}: got {result}, expected {expected}, {'OK' if match else 'MISMATCH'}")

if __name__ == "__main__":
    main()

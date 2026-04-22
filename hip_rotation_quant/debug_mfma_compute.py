#!/usr/bin/env python3
"""Debug MFMA 32x32x16 computation."""
import torch
import ctypes
import os
import sys

# Create a test kernel to verify MFMA computation
kernel_code = '''
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using float16_t = __attribute__((ext_vector_type(16))) float;
using bf16x8_t = __attribute__((ext_vector_type(8))) __bf16;

// Test kernel: compute C = A @ B where A and B are identity-like
__global__ void test_mfma_compute(
    const __bf16* __restrict__ A,  // [32, 16]
    const __bf16* __restrict__ B,  // [16, 32]
    float* __restrict__ C)         // [32, 32]
{
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    // 32x32x16 MFMA layout:
    // A: lane_id % 32 = row, lane_id / 32 = k_group (0 or 1)
    // B: lane_id % 32 = col, lane_id / 32 = k_group (0 or 1)
    // Each lane loads 8 K elements
    
    int k_group = lane_id / 32;  // 0 or 1
    int lane_32 = lane_id % 32;  // 0..31
    
    // Load A: A[lane_32, k_group*8 : k_group*8+8]
    bf16x8_t a_data;
    for (int j = 0; j < 8; j++) {
        a_data[j] = A[lane_32 * 16 + k_group * 8 + j];
    }
    
    // Load B: B[k_group*8 + j, lane_32]
    bf16x8_t b_data;
    for (int j = 0; j < 8; j++) {
        b_data[j] = B[(k_group * 8 + j) * 32 + lane_32];
    }
    
    float16_t acc = {};
    acc = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a_data, b_data, acc, 0, 0, 0);
    
    // Output
    int c_m_group = lane_id / 32;
    int c_col = lane_id % 32;
    for (int i = 0; i < 16; i++) {
        int row = c_m_group * 16 + i;
        C[row * 32 + c_col] = acc[i];
    }
}

extern "C" void launch_test(const void* A, const void* B, float* C, hipStream_t stream) {
    test_mfma_compute<<<1, 64, 0, stream>>>((const __bf16*)A, (const __bf16*)B, C);
}
'''

import subprocess
with open("/tmp/test_compute.hip", "w") as f:
    f.write(kernel_code)

result = subprocess.run(
    ["hipcc", "-shared", "-fPIC", "-O2", "--offload-arch=gfx950", 
     "-o", "/tmp/test_compute.so", "/tmp/test_compute.hip"],
    capture_output=True, text=True)
if result.returncode != 0:
    print("Compile error:", result.stderr)
    sys.exit(1)

lib = ctypes.CDLL("/tmp/test_compute.so")

# Create identity-like matrices
# A[i, j] = 1 if i == j else 0 (for 32x16)
# B[i, j] = 1 if i == j else 0 (for 16x32)
A = torch.zeros(32, 16, dtype=torch.bfloat16, device="cuda")
B = torch.zeros(16, 32, dtype=torch.bfloat16, device="cuda")
for i in range(16):
    A[i, i] = 1.0
    B[i, i] = 1.0

C = torch.zeros(32, 32, dtype=torch.float32, device="cuda")
lib.launch_test(ctypes.c_void_p(A.data_ptr()), ctypes.c_void_p(B.data_ptr()), 
                ctypes.c_void_p(C.data_ptr()), ctypes.c_void_p(0))
torch.cuda.synchronize()

# Reference
C_ref = (A.float() @ B.float()).cpu()

print("MFMA 32x32x16 computation test:")
print("A = identity-like [32, 16], B = identity-like [16, 32]")
print("Expected C = A @ B (identity in top-left 16x16)")
print("\nC output (first 8 rows, first 8 cols):")
for row in range(8):
    vals = [f"{C[row, col].item():5.1f}" for col in range(8)]
    print(f"Row {row}: {' '.join(vals)}")

print("\nC reference (first 8 rows, first 8 cols):")
for row in range(8):
    vals = [f"{C_ref[row, col].item():5.1f}" for col in range(8)]
    print(f"Row {row}: {' '.join(vals)}")

print("\nMatch:", torch.allclose(C.cpu(), C_ref, atol=1e-3))

# Try with random matrices
print("\n--- Random matrix test ---")
torch.manual_seed(42)
A = torch.randn(32, 16, dtype=torch.bfloat16, device="cuda")
B = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(32, 32, dtype=torch.float32, device="cuda")

lib.launch_test(ctypes.c_void_p(A.data_ptr()), ctypes.c_void_p(B.data_ptr()), 
                ctypes.c_void_p(C.data_ptr()), ctypes.c_void_p(0))
torch.cuda.synchronize()

C_ref = (A.float() @ B.float()).cpu()
print("Max error:", (C.cpu() - C_ref).abs().max().item())
print("Match:", torch.allclose(C.cpu(), C_ref, atol=1e-2))

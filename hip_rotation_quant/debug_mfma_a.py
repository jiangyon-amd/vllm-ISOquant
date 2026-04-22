#!/usr/bin/env python3
"""Debug MFMA 32x32x16 A operand layout."""
import torch
import ctypes
import subprocess

kernel_code = '''
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using float16_t = __attribute__((ext_vector_type(16))) float;
using bf16x8_t = __attribute__((ext_vector_type(8))) __bf16;

// Test: set A[row, k] = row + k*0.01, B = identity
// Then C[row, col] = A[row, col] (for col < 16)
__global__ void test_a_layout(float* __restrict__ C) {
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    int k_group = lane_id / 32;
    int lane_32 = lane_id % 32;
    
    // A[lane_32, k_group*8 + j] = lane_32 + (k_group*8 + j) * 0.01
    bf16x8_t a_data;
    for (int j = 0; j < 8; j++) {
        float val = lane_32 + (k_group * 8 + j) * 0.01f;
        a_data[j] = __float2bfloat16(val);
    }
    
    // B = identity: B[k, col] = 1 if k == col else 0
    bf16x8_t b_data = {};
    // For k_group 0: k = 0..7, so B[k, lane_32] = 1 if k == lane_32 and lane_32 < 8
    // For k_group 1: k = 8..15, so B[k, lane_32] = 1 if k == lane_32 and 8 <= lane_32 < 16
    for (int j = 0; j < 8; j++) {
        int k = k_group * 8 + j;
        if (k == lane_32) {
            b_data[j] = __float2bfloat16(1.0f);
        }
    }
    
    float16_t acc = {};
    acc = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a_data, b_data, acc, 0, 0, 0);
    
    int c_m_group = lane_id / 32;
    int c_col = lane_id % 32;
    for (int i = 0; i < 16; i++) {
        int row = c_m_group * 16 + i;
        C[row * 32 + c_col] = acc[i];
    }
}

extern "C" void launch_test(float* C, hipStream_t stream) {
    test_a_layout<<<1, 64, 0, stream>>>(C);
}
'''

with open("/tmp/test_a.hip", "w") as f:
    f.write(kernel_code)

result = subprocess.run(
    ["hipcc", "-shared", "-fPIC", "-O2", "--offload-arch=gfx950", 
     "-o", "/tmp/test_a.so", "/tmp/test_a.hip"],
    capture_output=True, text=True)
if result.returncode != 0:
    print("Compile error:", result.stderr)
    exit(1)

lib = ctypes.CDLL("/tmp/test_a.so")

C = torch.zeros(32, 32, dtype=torch.float32, device="cuda")
lib.launch_test(ctypes.c_void_p(C.data_ptr()), ctypes.c_void_p(0))
torch.cuda.synchronize()

print("A layout test: A[row, k] = row + k*0.01, B = identity")
print("Expected: C[row, col] = row + col*0.01 for col < 16, 0 otherwise")
print("\nC output (first 16 rows, first 16 cols):")
for row in range(16):
    vals = [f"{C[row, col].item():5.2f}" for col in range(16)]
    print(f"Row {row:2d}: {' '.join(vals)}")

print("\nC output (rows 16-31, first 16 cols):")
for row in range(16, 32):
    vals = [f"{C[row, col].item():5.2f}" for col in range(16)]
    print(f"Row {row:2d}: {' '.join(vals)}")

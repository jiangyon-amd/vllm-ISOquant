#!/usr/bin/env python3
"""Debug MFMA 32x32x16 output layout."""
import torch
import ctypes
import os
import sys

# Create a simple test kernel to understand the output layout
kernel_code = '''
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using float16_t = __attribute__((ext_vector_type(16))) float;
using bf16x8_t = __attribute__((ext_vector_type(8))) __bf16;

// Test kernel: compute identity-like matmul to understand output layout
__global__ void test_mfma_layout(float* output) {
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    // Create A and B such that output[i,j] = i * 100 + j (for debugging)
    // A[row, k] = row (for k=0), 0 otherwise
    // B[k, col] = col (for k=0), 0 otherwise
    // Then C[row, col] = row * col... not quite what we want
    
    // Better: just output the lane mapping
    // Set acc[i] = lane_id * 100 + i
    float16_t acc;
    for (int i = 0; i < 16; i++) {
        acc[i] = lane_id * 100.0f + i;
    }
    
    // Now figure out where each value should go
    int c_m_group = lane_id / 32;  // 0 or 1
    int c_col = lane_id % 32;      // 0..31
    
    // Output the mapping
    for (int i = 0; i < 16; i++) {
        // What row does acc[i] correspond to?
        // Based on kCM0PerLane=4, kCM1PerLane=4:
        // i = m0 * 4 + m1 where m0 in 0..3, m1 in 0..3
        // row = c_m_group * 16 + m0 * 4 + m1
        //     = c_m_group * 16 + i
        int row = c_m_group * 16 + i;
        int col = c_col;
        output[row * 32 + col] = acc[i];
    }
}

extern "C" void launch_test(float* output, hipStream_t stream) {
    test_mfma_layout<<<1, 64, 0, stream>>>(output);
}
'''

# Write and compile
import subprocess
with open("/tmp/test_layout.hip", "w") as f:
    f.write(kernel_code)

result = subprocess.run(
    ["hipcc", "-shared", "-fPIC", "-O2", "--offload-arch=gfx950", 
     "-o", "/tmp/test_layout.so", "/tmp/test_layout.hip"],
    capture_output=True, text=True)
if result.returncode != 0:
    print("Compile error:", result.stderr)
    sys.exit(1)

lib = ctypes.CDLL("/tmp/test_layout.so")

output = torch.zeros(32, 32, dtype=torch.float32, device="cuda")
lib.launch_test(ctypes.c_void_p(output.data_ptr()), ctypes.c_void_p(0))
torch.cuda.synchronize()

print("Output layout test (value = lane_id * 100 + i):")
print("Expected: output[row, col] = (row // 16) * 32 + col) * 100 + (row % 16)")
print("\nActual output (first 8 rows, first 8 cols):")
for row in range(8):
    vals = [f"{int(output[row, col]):4d}" for col in range(8)]
    print(f"Row {row}: {' '.join(vals)}")

print("\nActual output (rows 16-23, first 8 cols):")
for row in range(16, 24):
    vals = [f"{int(output[row, col]):4d}" for col in range(8)]
    print(f"Row {row}: {' '.join(vals)}")

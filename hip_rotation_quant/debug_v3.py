#!/usr/bin/env python3
"""Debug v3 kernel output."""
import torch
import ctypes
import os
import sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO = os.path.join(DIR, "dense_mfma_rot_quant.so")
sys.path.insert(0, "/data/jiangyon/vllm_rotation")

lib = ctypes.CDLL(SO)

from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import fused_rot_quant_gluon

M, K, RS = 32, 128, 128
torch.manual_seed(42)
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
rot = torch.randn(RS, RS, dtype=torch.bfloat16, device="cuda")

# Reference
fp4_ref, sc_ref = fused_rot_quant_gluon(x, rot, rotation_size=RS, shuffle_scales=False)
torch.cuda.synchronize()

# V3
fp4_v3 = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
n_scales = K // 32
sc_v3 = torch.zeros(M, n_scales, dtype=torch.uint8, device="cuda")

lib.launch_dense_mfma_rot_quant_v3(
    ctypes.c_void_p(fp4_v3.data_ptr()),
    ctypes.c_void_p(sc_v3.data_ptr()),
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(rot.data_ptr()),
    ctypes.c_int(M), ctypes.c_int(K),
    ctypes.c_int(x.stride(0)), ctypes.c_int(fp4_v3.stride(0)),
    ctypes.c_int(sc_v3.stride(0)), ctypes.c_int(n_scales),
    ctypes.c_int(0),
    ctypes.c_void_p(0))
torch.cuda.synchronize()

# V1 for comparison
fp4_v1 = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
sc_v1 = torch.zeros(M, n_scales, dtype=torch.uint8, device="cuda")

lib.launch_dense_mfma_rot_quant_v1(
    ctypes.c_void_p(fp4_v1.data_ptr()),
    ctypes.c_void_p(sc_v1.data_ptr()),
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(rot.data_ptr()),
    ctypes.c_int(M), ctypes.c_int(K),
    ctypes.c_int(x.stride(0)), ctypes.c_int(fp4_v1.stride(0)),
    ctypes.c_int(sc_v1.stride(0)), ctypes.c_int(n_scales),
    ctypes.c_int(0),
    ctypes.c_void_p(0))
torch.cuda.synchronize()

print("FP4 comparison (first 8 rows, first 16 cols):")
print("Row | Ref                              | V3                               | V1                               | Match")
for row in range(8):
    ref_row = fp4_ref[row, :16].tolist()
    v3_row = fp4_v3[row, :16].tolist()
    v1_row = fp4_v1[row, :16].tolist()
    match_v3 = "✓" if ref_row == v3_row else "✗"
    match_v1 = "✓" if ref_row == v1_row else "✓"
    print(f"{row:3d} | {ref_row[:8]} | {v3_row[:8]} | {v1_row[:8]} | v3:{match_v3} v1:{match_v1}")

print("\nScale comparison:")
print("Row | Ref      | V3       | V1       | Match")
for row in range(8):
    ref_row = sc_ref[row].tolist()
    v3_row = sc_v3[row].tolist()
    v1_row = sc_v1[row].tolist()
    match_v3 = "✓" if ref_row == v3_row else "✗"
    print(f"{row:3d} | {ref_row} | {v3_row} | {v1_row} | v3:{match_v3}")

# Check which rows/cols are correct
print("\nPer-row FP4 match rate:")
for row in range(M):
    match_rate = (fp4_v3[row] == fp4_ref[row]).float().mean().item() * 100
    if match_rate < 100:
        print(f"  Row {row}: {match_rate:.1f}%")

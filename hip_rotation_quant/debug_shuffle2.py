#!/usr/bin/env python3
"""Debug shuffled scales - check block boundaries."""
import torch
import ctypes
import os
import sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO = os.path.join(DIR, "dense_mfma_rot_quant.so")
sys.path.insert(0, "/data/jiangyon/vllm_rotation")

lib = ctypes.CDLL(SO)

from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import fused_rot_quant_gluon

# Test with M=32 (single block)
M, K, RS = 32, 4096, 128
torch.manual_seed(42)
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
rot = torch.randn(RS, RS, dtype=torch.bfloat16, device="cuda")

# Reference
fp4_ref, sc_ref = fused_rot_quant_gluon(x, rot, rotation_size=RS, shuffle_scales=True)
torch.cuda.synchronize()

# V1
fp4_v1 = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
n_scales = K // 32
sn_pad = (n_scales + 7) // 8 * 8
sm_pad = (M + 255) // 256 * 256
sc_v1 = torch.zeros(max(sm_pad, M), max(sn_pad, n_scales), dtype=torch.uint8, device="cuda")

lib.launch_dense_mfma_rot_quant_v1(
    ctypes.c_void_p(fp4_v1.data_ptr()),
    ctypes.c_void_p(sc_v1.data_ptr()),
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(rot.data_ptr()),
    ctypes.c_int(M), ctypes.c_int(K),
    ctypes.c_int(x.stride(0)), ctypes.c_int(fp4_v1.stride(0)),
    ctypes.c_int(sc_v1.stride(0)), ctypes.c_int(sn_pad),
    ctypes.c_int(1),
    ctypes.c_void_p(0))
torch.cuda.synchronize()

print(f"M=32 (single block):")
print(f"Scale shapes: ref={sc_ref.shape}, v1={sc_v1.shape}")

# Compare first 32 rows only
sc_v1_crop = sc_v1[:32, :sc_ref.shape[1]]
sc_ref_crop = sc_ref[:32, :]
match = (sc_v1_crop == sc_ref_crop).float()
print(f"First 32 rows match: {match.mean().item()*100:.1f}%")

# Check if ref has non-zero values beyond row 32
print(f"Ref rows 32+: {sc_ref[32:].sum().item()}")
print(f"V1 rows 32+: {sc_v1[32:].sum().item()}")

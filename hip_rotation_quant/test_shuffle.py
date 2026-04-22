#!/usr/bin/env python3
"""Test shuffled scales."""
import torch
import ctypes
import os
import sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO = os.path.join(DIR, "dense_mfma_rot_quant.so")
sys.path.insert(0, "/data/jiangyon/vllm_rotation")

lib = ctypes.CDLL(SO)

def test_shuffle(func_name, M, K, RS=128):
    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import fused_rot_quant_gluon

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rot = torch.randn(RS, RS, dtype=torch.bfloat16, device="cuda")

    fp4_ref, sc_ref = fused_rot_quant_gluon(x, rot, rotation_size=RS, shuffle_scales=True)
    torch.cuda.synchronize()

    fp4_hip = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
    n_scales = K // 32
    sn_pad = (n_scales + 7) // 8 * 8
    sm_pad = (M + 255) // 256 * 256
    sc_hip = torch.zeros(max(sm_pad, M), max(sn_pad, n_scales), dtype=torch.uint8, device="cuda")

    func = getattr(lib, func_name)
    func(
        ctypes.c_void_p(fp4_hip.data_ptr()),
        ctypes.c_void_p(sc_hip.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(rot.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(K),
        ctypes.c_int(x.stride(0)), ctypes.c_int(fp4_hip.stride(0)),
        ctypes.c_int(sc_hip.stride(0)), ctypes.c_int(sn_pad),
        ctypes.c_int(1),  # shuffle_scales=True
        ctypes.c_void_p(0))
    torch.cuda.synchronize()

    fp4_match = (fp4_hip == fp4_ref).float().mean().item() * 100
    sc_match = (sc_hip[:sc_ref.shape[0], :sc_ref.shape[1]] == sc_ref).float().mean().item() * 100
    return fp4_match, sc_match

print("Shuffled scales test:")
for func in ["launch_dense_mfma_rot_quant_v1", "launch_dense_mfma_rot_quant_v4", "launch_dense_mfma_rot_quant_v8"]:
    name = func.split("_")[-1]
    for M in [1, 32]:
        fp4_m, sc_m = test_shuffle(func, M, 4096)
        status = "✓" if sc_m > 99 else "✗"
        print(f"  {status} {name} M={M:4d}: FP4={fp4_m:5.1f}% scale={sc_m:5.1f}%")

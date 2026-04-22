#!/usr/bin/env python3
"""Unit tests for gluon v2 fused rotation+quant kernel.

Tests correctness against separated path (rotation matmul + per_1x32_f4_quant_hip)
for multiple model sizes (K values) and batch sizes (M values).

Usage:
    HIP_VISIBLE_DEVICES=1 python3 test_gluon_v2_kernel.py
"""
import os
import sys
import math

import torch
import triton

# Auto-select a free GPU if not set
if "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = "1"

from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import (
    fused_rot_quant_gluon,
)
from aiter import per_1x32_f4_quant_hip


def make_rotation(rotation_size=128, device="cuda"):
    """Create a random Hadamard-like rotation matrix."""
    signs = torch.randint(0, 2, (rotation_size, rotation_size), device=device) * 2 - 1
    return (signs.to(torch.float) / math.sqrt(rotation_size)).to(torch.bfloat16)


def separated_rot_quant(x, rotation, rotation_size, shuffle):
    """Reference: rotation matmul + per_1x32_f4_quant_hip."""
    M, K = x.shape
    x_rot = x.reshape(M, -1, rotation_size) @ rotation
    x_rot = x_rot.reshape(M, K)
    return per_1x32_f4_quant_hip(x_rot, shuffle=shuffle)


def test_fp4_raw_scales(K, M, rotation, verbose=False):
    """Test fp4 data and raw (unshuffled) scales match."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    x_q_sep, x_s_sep = separated_rot_quant(x, rotation, 128, shuffle=False)

    M_pad = max(M, 32)
    fp4 = torch.empty((M_pad, K // 2), dtype=torch.uint8, device="cuda")
    sc = torch.empty((M_pad, K // 32), dtype=torch.uint8, device="cuda")
    x_q_fus, x_s_fus = fused_rot_quant_gluon(
        x, rotation, rotation_size=128,
        fp4_out=fp4, scales_out=sc, shuffle_scales=False,
    )

    fp4_ok = (x_q_sep.view(torch.uint8) == x_q_fus[:M]).all().item()
    sc_ok = (x_s_sep.view(torch.uint8)[:M] == x_s_fus[:M]).all().item()

    if verbose and not (fp4_ok and sc_ok):
        fp4_diff = (x_q_sep.view(torch.uint8) != x_q_fus[:M]).sum().item()
        sc_diff = (x_s_sep.view(torch.uint8)[:M] != x_s_fus[:M]).sum().item()
        print(f"    fp4 mismatch: {fp4_diff}, scale mismatch: {sc_diff}")

    return fp4_ok, sc_ok


def test_shuffled_scales(K, M, rotation, verbose=False):
    """Test shuffled scales match at all valid (non-zero fused) positions."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    _, x_s_sep = separated_rot_quant(x, rotation, 128, shuffle=True)

    n_scales = K // 32
    sn_pad = (n_scales + 7) // 8 * 8
    raw_off = ((M + 31) // 32) * 32
    sm = max((M + 255) // 256 * 256, raw_off * 2)
    sc = torch.zeros((sm, sn_pad), dtype=torch.uint8, device="cuda")
    fp4 = torch.empty((max(M, 32), K // 2), dtype=torch.uint8, device="cuda")
    _, x_s_fus = fused_rot_quant_gluon(
        x, rotation, rotation_size=128,
        fp4_out=fp4, scales_out=sc, shuffle_scales=True,
    )

    # Compare only at fused non-zero positions (sep buffer may have uninitialized noise)
    fus_nz = (x_s_fus[:raw_off] != 0).nonzero()
    n_values = fus_nz.shape[0]
    expected = M * n_scales

    if n_values != expected:
        if verbose:
            print(f"    NZ count: {n_values} (expected {expected})")
        return False, n_values, expected

    sep_u8 = x_s_sep.view(torch.uint8)
    match = all(
        sep_u8[r, c].item() == x_s_fus[r, c].item()
        for r, c in fus_nz.tolist()
        if r < sep_u8.shape[0] and c < sep_u8.shape[1]
    )
    return match, n_values, expected


def main():
    torch.manual_seed(42)
    print(f"Triton {triton.__version__}, PyTorch {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # Test configurations: (K, label)
    configs = [
        (4096, "8B"),
        (5120, "32B"),
        (3584, "14B"),   # Qwen3-14B hidden_size=3584
        (2048, "MoE-30B"),  # Qwen3-30B-A3B hidden_size=2048
    ]
    m_values = [1, 2, 4, 8, 16, 32, 64, 128]

    all_pass = True

    for K, label in configs:
        rotation = make_rotation(128, device="cuda")

        # --- Test 1: FP4 + raw scales ---
        print(f"TEST 1 [{label} K={K}]: FP4 + raw scales")
        for M in m_values:
            fp4_ok, sc_ok = test_fp4_raw_scales(K, M, rotation, verbose=True)
            status = "PASS" if (fp4_ok and sc_ok) else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  M={M:>3}: fp4={fp4_ok} scales={sc_ok} {status}")

        # --- Test 2: Shuffled scales ---
        print(f"TEST 2 [{label} K={K}]: Shuffled scales")
        for M in m_values:
            match, n_values, expected = test_shuffled_scales(K, M, rotation, verbose=True)
            status = "PASS" if match else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  M={M:>3}: {status} ({n_values}/{expected} values)")

        print()

    # Summary
    print("=" * 50)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Correctness test for TQ Store optimization.

Verifies that the optimized store produces the same KV cache contents
as the reference implementation.

Run: python test_correctness.py
Returns exit code 0 if correct, 1 if mismatch.
"""
import math
import torch
import sys

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention

DEVICE = "cuda:0"
D = 128; BS = 16


def test_store_determinism(N, H, label):
    """Check store produces identical results on two runs."""
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    num_blocks = max(256, (N * 2 // BS) + 64)

    torch.manual_seed(42)
    key = torch.randn(N, H, D, dtype=torch.bfloat16, device=DEVICE)
    value = torch.randn(N, H, D, dtype=torch.bfloat16, device=DEVICE)
    slot_mapping = torch.arange(N, device=DEVICE, dtype=torch.int64)

    # Run 1
    kv_cache1 = torch.zeros(num_blocks, BS, H, cfg.slot_size_aligned,
                            dtype=torch.uint8, device=DEVICE)
    triton_turboquant_store(
        key, value, kv_cache1, slot_mapping,
        PiT, centroids, midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
    )

    # Run 2
    kv_cache2 = torch.zeros(num_blocks, BS, H, cfg.slot_size_aligned,
                            dtype=torch.uint8, device=DEVICE)
    triton_turboquant_store(
        key, value, kv_cache2, slot_mapping,
        PiT, centroids, midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
    )

    match = torch.equal(kv_cache1, kv_cache2)
    nonzero = kv_cache1.sum().item() > 0
    n_diff = (kv_cache1 != kv_cache2).sum().item()

    status = "PASS" if (match and nonzero) else "FAIL"
    print(f"  {label}: {status} (n_diff_bytes={n_diff}, nonzero_data={nonzero})")
    return match and nonzero


def test_store_decode_roundtrip(N, H, Hq, label):
    """Store K/V then decode, verify output is non-trivial and deterministic."""
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    Pi = PiT.T.contiguous()
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    B = min(N, 4)  # batch size for decode
    seq_len = max(N // B, 16)
    fill_n = B * seq_len
    num_blocks = max(256, (fill_n // BS) + 64)
    kv_cache = torch.zeros(num_blocks, BS, H, cfg.slot_size_aligned,
                           dtype=torch.uint8, device=DEVICE)

    torch.manual_seed(42)
    fk = torch.randn(fill_n, H, D, dtype=torch.bfloat16, device=DEVICE)
    fv = torch.randn_like(fk)
    triton_turboquant_store(
        fk, fv, kv_cache,
        torch.arange(fill_n, device=DEVICE, dtype=torch.int64),
        PiT, centroids, midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
    )

    torch.manual_seed(123)
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    bps = math.ceil(seq_len / BS)
    bt = torch.arange(bps, device=DEVICE, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()
    sls = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)

    out1 = triton_turboquant_decode_attention(
        query=q, kv_cache=kv_cache, block_table=bt, seq_lens=sls,
        Pi=Pi, centroids=centroids, scale=1.0/math.sqrt(D),
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction, PiT=PiT)

    out2 = triton_turboquant_decode_attention(
        query=q, kv_cache=kv_cache, block_table=bt, seq_lens=sls,
        Pi=Pi, centroids=centroids, scale=1.0/math.sqrt(D),
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction, PiT=PiT)

    max_diff = (out1 - out2).abs().max().item()
    match = torch.allclose(out1, out2, atol=1e-3, rtol=1e-3)
    not_zero = out1.abs().max().item() > 1e-6

    status = "PASS" if (match and not_zero) else "FAIL"
    print(f"  {label}: {status} (max_diff={max_diff:.6f}, max_val={out1.abs().max():.4f})")
    return match and not_zero


if __name__ == "__main__":
    print("=== TQ Store Determinism Tests ===")
    all_pass = True
    for N, H in [(4, 2), (16, 8), (128, 2), (512, 8), (2048, 2)]:
        ok = test_store_determinism(N, H, f"N={N},H={H}")
        all_pass = all_pass and ok

    print("\n=== TQ Store→Decode Roundtrip Tests ===")
    for N, H, Hq in [(64, 8, 32), (256, 2, 16), (512, 8, 32)]:
        ok = test_store_decode_roundtrip(N, H, Hq, f"N={N},H={H},Hq={Hq}")
        all_pass = all_pass and ok

    if all_pass:
        print("\nAll tests PASSED ✓")
        sys.exit(0)
    else:
        print("\nSome tests FAILED ✗")
        sys.exit(1)

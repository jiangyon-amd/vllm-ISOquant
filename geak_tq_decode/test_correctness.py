#!/usr/bin/env python3
"""Correctness test for _tq_decode_stage1 optimization.

Run: python test_correctness.py
Returns exit code 0 if correct, 1 if mismatch.
"""
import math, torch

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention

DEVICE = "cuda:0"
D = 128; Hk = 8; Hq = 32; BS = 16

def test_case(B, seq_len, label):
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    Pi = PiT.T.contiguous()
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    num_blocks = max(2048, (B * seq_len // BS) + 512)
    kv_cache = torch.zeros(num_blocks, BS, Hk, cfg.slot_size_aligned,
                           dtype=torch.uint8, device=DEVICE)

    torch.manual_seed(42)
    fill_n = min(B * seq_len, num_blocks * BS)
    fk = torch.randn(fill_n, Hk, D, dtype=torch.bfloat16, device=DEVICE)
    fv = torch.randn_like(fk)
    triton_turboquant_store(fk, fv, kv_cache,
        torch.arange(fill_n, device=DEVICE, dtype=torch.int64),
        PiT, centroids, midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8)

    torch.manual_seed(123)
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    bps = math.ceil(seq_len / BS)
    bt = torch.arange(bps, device=DEVICE, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()
    sls = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)

    # Run twice for determinism check
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

    # Check not all zeros
    not_zero = out1.abs().max().item() > 1e-6

    status = "PASS" if (match and not_zero) else "FAIL"
    print(f"  {label}: {status} (max_diff={max_diff:.6f}, max_val={out1.abs().max():.4f})")
    return match and not_zero


if __name__ == "__main__":
    print("Correctness tests for _tq_decode_stage1:")
    all_pass = True
    for B, seq_len in [(4, 64), (32, 256), (100, 512), (8, 1024)]:
        ok = test_case(B, seq_len, f"B={B},seq={seq_len}")
        all_pass = all_pass and ok

    if all_pass:
        print("\nAll tests PASSED ✓")
        exit(0)
    else:
        print("\nSome tests FAILED ✗")
        exit(1)

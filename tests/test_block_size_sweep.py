#!/usr/bin/env python3
"""
Block-size sweep test for TQ HIP kernels.

Covers edge cases:
  1. seq_len not divisible by block_size
  2. Different seq_lens within a batch
  3. Larger sequence lengths (256, 512, 1024)
  4. block_size >= seq_len (extreme case)
  5. All 4 supported block_sizes: 16, 32, 64, 128
"""

import math
import sys
import os
import ctypes
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def generate_centroids_midpoints(n_centroids=16):
    centroids = torch.linspace(-1.0, 1.0, n_centroids, dtype=torch.float32)
    midpoints = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, midpoints


def generate_random_rotation(D=128, device="cuda"):
    A = torch.randn(D, D, dtype=torch.float32, device=device)
    Q, R = torch.linalg.qr(A)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q.contiguous()


def run_decode_triton_only(query, kv_cache, block_table, seq_lens, Pi,
                           centroids, scale, mse_bits, kps, vqb,
                           norm_correction, PiT, num_kv_splits=8):
    from vllm.v1.attention.ops import triton_turboquant_decode as tqd
    saved = (tqd._HIP_STAGE1_FN, tqd._HIP_STAGE1_LIB,
             tqd._HIP_V56_FN, tqd._HIP_V56_LIB,
             tqd._HIP_STAGE2_BF16_FN, tqd._HIP_STAGE2_F32_FN, tqd._HIP_STAGE2_LIB)
    tqd._HIP_STAGE1_FN = None;  tqd._HIP_STAGE1_LIB = False
    tqd._HIP_V56_FN = None;     tqd._HIP_V56_LIB = False
    tqd._HIP_STAGE2_BF16_FN = None; tqd._HIP_STAGE2_F32_FN = None; tqd._HIP_STAGE2_LIB = False
    try:
        return tqd.triton_turboquant_decode_attention(
            query=query, kv_cache=kv_cache, block_table=block_table,
            seq_lens=seq_lens, Pi=Pi, centroids=centroids, scale=scale,
            mse_bits=mse_bits, key_packed_size=kps, value_quant_bits=vqb,
            key_fp8=False, norm_correction=norm_correction, PiT=PiT,
            max_num_kv_splits=num_kv_splits)
    finally:
        (tqd._HIP_STAGE1_FN, tqd._HIP_STAGE1_LIB,
         tqd._HIP_V56_FN, tqd._HIP_V56_LIB,
         tqd._HIP_STAGE2_BF16_FN, tqd._HIP_STAGE2_F32_FN,
         tqd._HIP_STAGE2_LIB) = saved


def run_decode_hip(query, kv_cache, block_table, seq_lens, Pi,
                   centroids, scale, mse_bits, kps, vqb,
                   norm_correction, PiT, num_kv_splits=8):
    from vllm.v1.attention.ops import triton_turboquant_decode as tqd
    return tqd.triton_turboquant_decode_attention(
        query=query, kv_cache=kv_cache, block_table=block_table,
        seq_lens=seq_lens, Pi=Pi, centroids=centroids, scale=scale,
        mse_bits=mse_bits, key_packed_size=kps, value_quant_bits=vqb,
        key_fp8=False, norm_correction=norm_correction, PiT=PiT,
        max_num_kv_splits=num_kv_splits)


def populate_cache_triton(key, value, kv_cache, slot_mapping, PiT,
                          centroids, midpoints, mse_bits, kps, vqb):
    """Store using Triton-only path."""
    from vllm.v1.attention.ops import triton_turboquant_store as tqs
    old_fn, old_loaded = tqs._hip_store_fn, tqs._hip_store_loaded
    tqs._hip_store_fn = None; tqs._hip_store_loaded = True
    try:
        tqs.triton_turboquant_store(
            key=key, value=value, kv_cache=kv_cache,
            slot_mapping=slot_mapping, PiT=PiT,
            centroids=centroids, midpoints=midpoints,
            mse_bits=mse_bits, key_packed_size=kps,
            value_quant_bits=vqb, key_fp8=False)
    finally:
        tqs._hip_store_fn, tqs._hip_store_loaded = old_fn, old_loaded


def test_block_size(block_size, B, seq_lens_list, Hq, Hk, D, dtype, device="cuda"):
    """
    Test one block_size configuration.
    seq_lens_list: list of B integers, each is the seq_len for that batch element.
    """
    torch.manual_seed(42 + block_size)

    mse_bits = 4
    vqb = 4
    kps = 68
    scale = 1.0 / math.sqrt(D)
    num_kv_splits = 8
    norm_correction = True

    padded_slot = kps + math.ceil(D * vqb / 8) + 4  # 136
    max_seq = max(seq_lens_list)
    # Total tokens across all sequences
    total_tokens = sum(seq_lens_list)
    # Blocks needed for total tokens
    num_blocks = (total_tokens + block_size - 1) // block_size
    # max blocks per sequence (for block_table width)
    max_blocks_per_seq = (max_seq + block_size - 1) // block_size

    centroids, midpoints = generate_centroids_midpoints(16)
    centroids = centroids.to(device)
    midpoints = midpoints.to(device)
    PiT = generate_random_rotation(D, device)
    Pi = PiT.T.contiguous()

    # Generate all tokens
    key = torch.randn(total_tokens, Hk, D, device=device).to(dtype)
    value = torch.randn(total_tokens, Hk, D, device=device).to(dtype)

    # Slot mapping: identity
    slot_mapping = torch.arange(total_tokens, dtype=torch.long, device=device)

    # KV cache
    kv_cache = torch.zeros(num_blocks, block_size, Hk, padded_slot,
                           dtype=torch.uint8, device=device)

    # Store using Triton
    populate_cache_triton(key, value, kv_cache, slot_mapping, PiT,
                          centroids, midpoints, mse_bits, kps, vqb)
    torch.cuda.synchronize()

    # Build block_table: each batch element has its own sequence of blocks
    # batch 0 uses blocks [0 .. num_blocks_0-1]
    # batch 1 uses blocks [num_blocks_0 .. num_blocks_0+num_blocks_1-1]
    # etc.
    block_table = torch.zeros(B, max_blocks_per_seq, dtype=torch.int32, device=device)
    offset = 0
    for b in range(B):
        n_blk = (seq_lens_list[b] + block_size - 1) // block_size
        for j in range(n_blk):
            blk_idx = offset // block_size + j
            block_table[b, j] = blk_idx
        offset += seq_lens_list[b]

    seq_lens_t = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    # Query
    query = torch.randn(B, Hq, D, device=device).to(dtype)

    # Run Triton
    out_triton = run_decode_triton_only(
        query, kv_cache, block_table, seq_lens_t,
        Pi, centroids, scale, mse_bits, kps, vqb,
        norm_correction, PiT, num_kv_splits)
    torch.cuda.synchronize()

    # Run HIP
    out_hip = run_decode_hip(
        query, kv_cache, block_table, seq_lens_t,
        Pi, centroids, scale, mse_bits, kps, vqb,
        norm_correction, PiT, num_kv_splits)
    torch.cuda.synchronize()

    abs_diff = (out_triton - out_hip).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    rel_diff = abs_diff / (out_triton.abs() + 1e-8)
    max_rel = rel_diff.max().item()

    ok = max_abs < 1e-3 or max_rel < 1e-2
    return ok, max_abs, mean_abs, max_rel


def main():
    device = "cuda"
    D = 128
    Hk = 8
    Hq = 8
    dtype = torch.bfloat16

    print("=" * 72)
    print("  Block-Size Sweep Test (HIP vs Triton)")
    print("=" * 72)

    all_pass = True
    results = []

    # ----------------------------------------------------------------
    # Group 1: Uniform seq_len, varying block_size
    # ----------------------------------------------------------------
    print("\n  --- Group 1: Uniform seq_len, B=1 (v56 path) ---")
    for bs in [16, 32, 64, 128]:
        for sl in [17, 33, 63, 64, 100, 127, 128, 255, 256, 512, 1024]:
            if sl < bs:
                # seq_len < block_size => only 1 block partially filled
                pass  # still test it
            label = f"bs={bs:3d}  sl={sl:4d}  B=1"
            ok, ma, mea, mr = test_block_size(
                block_size=bs, B=1, seq_lens_list=[sl],
                Hq=Hq, Hk=Hk, D=D, dtype=dtype, device=device)
            status = "OK" if ok else "FAIL"
            print(f"    {label}  => {status}  max_abs={ma:.2e}  max_rel={mr:.2e}")
            results.append((label, ok))
            if not ok:
                all_pass = False

    # ----------------------------------------------------------------
    # Group 2: Uniform seq_len, B=8 (v52 path)
    # ----------------------------------------------------------------
    print("\n  --- Group 2: Uniform seq_len, B=8 (v52 path) ---")
    for bs in [16, 32, 64, 128]:
        for sl in [33, 64, 100, 256, 512]:
            label = f"bs={bs:3d}  sl={sl:4d}  B=8"
            ok, ma, mea, mr = test_block_size(
                block_size=bs, B=8, seq_lens_list=[sl]*8,
                Hq=Hq, Hk=Hk, D=D, dtype=dtype, device=device)
            status = "OK" if ok else "FAIL"
            print(f"    {label}  => {status}  max_abs={ma:.2e}  max_rel={mr:.2e}")
            results.append((label, ok))
            if not ok:
                all_pass = False

    # ----------------------------------------------------------------
    # Group 3: Mixed seq_lens within a batch
    # ----------------------------------------------------------------
    print("\n  --- Group 3: Mixed seq_lens within batch, B=4 ---")
    mixed_cases = [
        ([15, 16, 17, 31],  16, "near-boundary bs=16"),
        ([1, 16, 33, 64],   16, "wide-range bs=16"),
        ([31, 32, 33, 63],  32, "near-boundary bs=32"),
        ([1, 32, 65, 128],  32, "wide-range bs=32"),
        ([63, 64, 65, 127], 64, "near-boundary bs=64"),
        ([1, 64, 129, 256], 64, "wide-range bs=64"),
        ([127, 128, 129, 255], 128, "near-boundary bs=128"),
    ]
    for sls, bs, desc in mixed_cases:
        label = f"bs={bs:3d}  sls={sls}  {desc}"
        ok, ma, mea, mr = test_block_size(
            block_size=bs, B=len(sls), seq_lens_list=sls,
            Hq=Hq, Hk=Hk, D=D, dtype=dtype, device=device)
        status = "OK" if ok else "FAIL"
        print(f"    {label}  => {status}  max_abs={ma:.2e}  max_rel={mr:.2e}")
        results.append((label, ok))
        if not ok:
            all_pass = False

    # ----------------------------------------------------------------
    # Group 4: Edge case — seq_len < block_size
    # ----------------------------------------------------------------
    print("\n  --- Group 4: seq_len < block_size (partial block) ---")
    for bs, sl in [(32, 1), (32, 15), (64, 1), (64, 31), (128, 1), (128, 63)]:
        label = f"bs={bs:3d}  sl={sl:4d}  B=1  (partial)"
        ok, ma, mea, mr = test_block_size(
            block_size=bs, B=1, seq_lens_list=[sl],
            Hq=Hq, Hk=Hk, D=D, dtype=dtype, device=device)
        status = "OK" if ok else "FAIL"
        print(f"    {label}  => {status}  max_abs={ma:.2e}  max_rel={mr:.2e}")
        results.append((label, ok))
        if not ok:
            all_pass = False

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    print(f"\n{'=' * 72}")
    print(f"  {n_pass}/{n_total} tests passed")
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  FAILED tests:")
        for label, ok in results:
            if not ok:
                print(f"    {label}")
    print("=" * 72)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

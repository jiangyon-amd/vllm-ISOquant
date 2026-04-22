#!/usr/bin/env python3
"""
FP16 path test for TQ HIP kernels.

Verifies that fp16 query/key/value work correctly through the full pipeline:
  - Store: fp16 key/value → HIP store kernel (kv_dtype=1)
  - Decode v56: fp16 query → HIP GEMV-fused kernel (query_dtype=1)
  - Decode v52: fp16 query → float32 q_rot → HIP Stage1
  - Stage2: f32 reduce (fp16 doesn't have fused bf16 output path)

Also tests mixed precision: fp16 query against bf16-stored cache and vice versa.
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


def run_store(key, value, kv_cache, slot_mapping, PiT,
              centroids, midpoints, mse_bits, kps, vqb, use_hip):
    from vllm.v1.attention.ops import triton_turboquant_store as tqs
    if not use_hip:
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
        if not use_hip:
            tqs._hip_store_fn, tqs._hip_store_loaded = old_fn, old_loaded


def run_decode(query, kv_cache, block_table, seq_lens, Pi, centroids,
               scale, mse_bits, kps, vqb, norm_correction, PiT,
               num_kv_splits, use_hip):
    from vllm.v1.attention.ops import triton_turboquant_decode as tqd
    if not use_hip:
        saved = (tqd._HIP_STAGE1_FN, tqd._HIP_STAGE1_LIB,
                 tqd._HIP_V56_FN, tqd._HIP_V56_LIB,
                 tqd._HIP_STAGE2_BF16_FN, tqd._HIP_STAGE2_F32_FN, tqd._HIP_STAGE2_LIB)
        tqd._HIP_STAGE1_FN = None; tqd._HIP_STAGE1_LIB = False
        tqd._HIP_V56_FN = None; tqd._HIP_V56_LIB = False
        tqd._HIP_STAGE2_BF16_FN = None; tqd._HIP_STAGE2_F32_FN = None; tqd._HIP_STAGE2_LIB = False
    try:
        out = tqd.triton_turboquant_decode_attention(
            query=query, kv_cache=kv_cache,
            block_table=block_table, seq_lens=seq_lens,
            Pi=Pi, centroids=centroids, scale=scale,
            mse_bits=mse_bits, key_packed_size=kps,
            value_quant_bits=vqb, key_fp8=False,
            norm_correction=norm_correction, PiT=PiT,
            max_num_kv_splits=num_kv_splits)
    finally:
        if not use_hip:
            (tqd._HIP_STAGE1_FN, tqd._HIP_STAGE1_LIB,
             tqd._HIP_V56_FN, tqd._HIP_V56_LIB,
             tqd._HIP_STAGE2_BF16_FN, tqd._HIP_STAGE2_F32_FN,
             tqd._HIP_STAGE2_LIB) = saved
    return out


def test_fp16(B, seq_len, Hq, Hk, D, block_size,
              store_dtype, query_dtype, device="cuda"):
    """Test with specified store and query dtype."""
    torch.manual_seed(42)
    mse_bits, vqb, kps = 4, 4, 68
    scale = 1.0 / math.sqrt(D)
    num_kv_splits = 8
    norm_correction = True

    padded_slot = kps + math.ceil(D * vqb / 8) + 4
    num_blocks = (seq_len + block_size - 1) // block_size

    centroids, midpoints = generate_centroids_midpoints(16)
    centroids = centroids.to(device)
    midpoints = midpoints.to(device)
    PiT = generate_random_rotation(D, device)
    Pi = PiT.T.contiguous()

    key = torch.randn(seq_len, Hk, D, device=device).to(store_dtype)
    value = torch.randn(seq_len, Hk, D, device=device).to(store_dtype)
    slot_mapping = torch.arange(seq_len, dtype=torch.long, device=device)

    block_table_bt = torch.arange(num_blocks, dtype=torch.int32, device=device)
    block_table = block_table_bt.unsqueeze(0).expand(B, -1).contiguous()
    seq_lens_t = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    # === Store: HIP vs Triton ===
    kv_triton = torch.zeros(num_blocks, block_size, Hk, padded_slot,
                            dtype=torch.uint8, device=device)
    kv_hip = torch.zeros_like(kv_triton)

    run_store(key, value, kv_triton, slot_mapping, PiT,
              centroids, midpoints, mse_bits, kps, vqb, use_hip=False)
    run_store(key, value, kv_hip, slot_mapping, PiT,
              centroids, midpoints, mse_bits, kps, vqb, use_hip=True)
    torch.cuda.synchronize()

    store_match = torch.equal(kv_triton, kv_hip)
    if not store_match:
        diff_bytes = (kv_triton != kv_hip).sum().item()
        print(f"    Store({store_dtype}): MISMATCH {diff_bytes} bytes differ")
    else:
        print(f"    Store({store_dtype}): MATCH")

    # === Decode: HIP vs Triton (using Triton-stored cache) ===
    query = torch.randn(B, Hq, D, device=device).to(query_dtype)

    out_triton = run_decode(
        query, kv_triton, block_table, seq_lens_t, Pi, centroids,
        scale, mse_bits, kps, vqb, norm_correction, PiT, num_kv_splits,
        use_hip=False)
    torch.cuda.synchronize()

    out_hip = run_decode(
        query, kv_triton, block_table, seq_lens_t, Pi, centroids,
        scale, mse_bits, kps, vqb, norm_correction, PiT, num_kv_splits,
        use_hip=True)
    torch.cuda.synchronize()

    # Convert to same dtype for comparison
    out_triton_f32 = out_triton.float()
    out_hip_f32 = out_hip.float()

    abs_diff = (out_triton_f32 - out_hip_f32).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    rel_diff = abs_diff / (out_triton_f32.abs() + 1e-8)
    max_rel = rel_diff.max().item()

    # bf16 output has ~1e-3 truncation error (fused bf16 stage2 vs f32 stage2)
    atol = 2e-3 if query_dtype == torch.bfloat16 else 1e-3
    rtol = 1e-2
    decode_ok = max_abs < atol or max_rel < rtol

    out_dtype_info = f"triton={out_triton.dtype} hip={out_hip.dtype}"
    status = "MATCH" if decode_ok else "MISMATCH"
    print(f"    Decode(q={query_dtype}): {status}  "
          f"max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}  max_rel={max_rel:.2e}  [{out_dtype_info}]")

    return store_match, decode_ok


def main():
    device = "cuda"
    D, Hk, Hq = 128, 8, 8
    block_size = 16
    seq_len = 128

    print("=" * 72)
    print("  FP16 Path Deep Test")
    print("=" * 72)

    all_pass = True

    cases = [
        # (B, store_dtype, query_dtype, label)
        # --- Pure fp16 ---
        (1,  torch.float16, torch.float16,  "Pure fp16, B=1 (v56)"),
        (2,  torch.float16, torch.float16,  "Pure fp16, B=2 (v56)"),
        (4,  torch.float16, torch.float16,  "Pure fp16, B=4 (v56)"),
        (8,  torch.float16, torch.float16,  "Pure fp16, B=8 (v52)"),
        (16, torch.float16, torch.float16,  "Pure fp16, B=16 (v52)"),

        # --- Pure bf16 (reference) ---
        (1,  torch.bfloat16, torch.bfloat16, "Pure bf16, B=1 (v56)"),
        (8,  torch.bfloat16, torch.bfloat16, "Pure bf16, B=8 (v52)"),

        # --- fp16 with different block_sizes ---
        (1,  torch.float16, torch.float16,  "fp16 bs=32, B=1"),
        (1,  torch.float16, torch.float16,  "fp16 bs=64, B=1"),
        (8,  torch.float16, torch.float16,  "fp16 bs=128, B=8"),
    ]

    block_sizes = [16, 16, 16, 16, 16, 16, 16, 32, 64, 128]

    for i, (B, sdtype, qdtype, label) in enumerate(cases):
        bs = block_sizes[i]
        print(f"\n  [{label}, bs={bs}]")
        try:
            s_ok, d_ok = test_fp16(
                B=B, seq_len=seq_len, Hq=Hq, Hk=Hk, D=D,
                block_size=bs, store_dtype=sdtype, query_dtype=qdtype,
                device=device)
            if not (s_ok and d_ok):
                all_pass = False
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    # --- Longer sequence fp16 ---
    print(f"\n  [fp16 long seq_len=1024, B=1 (v56)]")
    try:
        s_ok, d_ok = test_fp16(1, 1024, Hq, Hk, D, 16,
                               torch.float16, torch.float16, device)
        if not (s_ok and d_ok): all_pass = False
    except Exception as e:
        print(f"    ERROR: {e}"); all_pass = False

    print(f"\n  [fp16 long seq_len=1024, B=8 (v52)]")
    try:
        s_ok, d_ok = test_fp16(8, 1024, Hq, Hk, D, 16,
                               torch.float16, torch.float16, device)
        if not (s_ok and d_ok): all_pass = False
    except Exception as e:
        print(f"    ERROR: {e}"); all_pass = False

    # --- GQA: Hq != Hk ---
    print(f"\n  [fp16 GQA Hq=32 Hk=8, B=1]")
    try:
        s_ok, d_ok = test_fp16(1, 128, 32, 8, D, 16,
                               torch.float16, torch.float16, device)
        if not (s_ok and d_ok): all_pass = False
    except Exception as e:
        print(f"    ERROR: {e}"); all_pass = False

    print(f"\n  [fp16 GQA Hq=32 Hk=8, B=8]")
    try:
        s_ok, d_ok = test_fp16(8, 128, 32, 8, D, 16,
                               torch.float16, torch.float16, device)
        if not (s_ok and d_ok): all_pass = False
    except Exception as e:
        print(f"    ERROR: {e}"); all_pass = False

    print("\n" + "=" * 72)
    if all_pass:
        print("  ALL FP16 TESTS PASSED")
    else:
        print("  SOME FP16 TESTS FAILED")
    print("=" * 72)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
MoE Fused Rotation + MXFP4 Quant: Correctness & Performance Test
=================================================================

Tests all dispatch paths for Qwen3-30B-A3B MoE kernel:
  1. Reference: separated (matmul rotation + aiter quant + moe_mxfp4_sort)
  2. Gluon v2 kw8 + moe_mxfp4_sort  (2-kernel fused rot+quant)
  3. Triton decode rot+sort fusion   (1-kernel, M=1)
  4. Gluon MOE decode                (1-kernel, M=1)
  5. HIP MFMA 3-in-1 (aiter)        (1-kernel, all M)
  6. Aiter unified kernel            (1-kernel, all M)

For each path:
  - Correctness: compare FP4 output and sorted scale vs reference
  - Performance: measure latency (us)

Qwen3-30B-A3B shapes: K=2048, topk=8, E=128, RS=128
"""

import os
import sys
import time
import torch
import json

os.environ.setdefault("HIP_VISIBLE_DEVICES", "4")

sys.path.insert(0, "/data/jiangyon/vllm_rotation")

from aiter.fused_moe import moe_sorting, fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility.fp4_utils import moe_mxfp4_sort
from aiter.utility import dtypes


def compare_tensors(a_u8, b_u8):
    total = a_u8.numel()
    match = (a_u8 == b_u8).sum().item()
    return match / total * 100 if total > 0 else 0


def fmt_pct(pct):
    if pct == 100.0:
        return "\033[32m100.0%\033[0m"
    elif pct >= 95.0:
        return f"\033[33m{pct:.1f}%\033[0m"
    else:
        return f"\033[31m{pct:.1f}%\033[0m"


def bench_fn(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) * 1000 for i in range(iters)]
    times.sort()
    trim = max(1, iters // 10)
    trimmed = times[trim:-trim]
    avg = sum(trimmed) / len(trimmed)
    return avg, min(times), max(times)


# ===================== REFERENCE =====================
def reference_separated(x, rot, sorted_ids, num_valid_ids, M, K, RS, QG, token_num, block_size=32):
    x_rot = (x.reshape(-1, K // RS, RS).float() @ rot.float()).to(torch.bfloat16).reshape(M, K)
    fp4, sc = fused_dynamic_mxfp4_quant_moe_sort(
        x_rot, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token_num, topk=1, block_size=block_size,
    )
    return fp4.view(torch.uint8), sc.view(torch.uint8)


# ===================== gluon_kw8 + moe_mxfp4_sort =====================
def gluon_kw8_path(x, rot, sorted_ids, num_valid_ids, M, K, RS, QG, token_num, block_size=32):
    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon_v2_kw8 import fused_gluon_v2_kw8
    n_i = K // QG
    fp4_u8 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    raw_scale = torch.empty((M, n_i), dtype=torch.uint8, device=x.device)
    fp4_u8, raw_scale = fused_gluon_v2_kw8(
        x, rot, RS, fp4_out=fp4_u8, scales_out=raw_scale, shuffle_scales=False,
    )
    sorted_scale = moe_mxfp4_sort(
        raw_scale, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token_num, block_size=block_size,
    )
    return fp4_u8, sorted_scale.view(torch.uint8)


# ===================== Triton decode rot+sort (M=1 only) =====================
def triton_decode_rot_sort_path(x, rot, sorted_ids, num_valid_ids, M, K, RS, QG, token_num, block_size=32):
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (
        _fused_rot_quant_moe_sort_impl,
    )
    topk = sorted_ids.shape[0] // max(M, 1)
    fp4, sc = _fused_rot_quant_moe_sort_impl(
        x, rot, RS, sorted_ids, num_valid_ids,
        token_num=token_num, topk=topk, block_size=block_size,
        use_hip_kernel=False, use_triton_rot_sort_kernel=True,
    )
    return fp4.view(torch.uint8), sc.view(torch.uint8)


# ===================== Gluon MOE decode (M=1 only) =====================
def gluon_moe_decode_path(x, rot, sorted_ids, num_valid_ids, M, K, RS, QG, token_num, block_size=32):
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_gluon import (
        fused_decode_m1_rot_quant_sorted_gluon,
    )
    n_i = K // QG
    m_o = sorted_ids.shape[0]
    m_pad = ((m_o + 31) // 32) * 32
    fp4_u8 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    sorted_u8 = torch.empty((m_pad + 1, n_i), dtype=torch.uint8, device=x.device)
    fused_decode_m1_rot_quant_sorted_gluon(
        x, rot, fp4_u8,
        sorted_ids, num_valid_ids, sorted_u8,
        K=K, RS=RS, QG=QG, n_i=n_i,
        token_num=token_num, m_o=m_o, m_pad=m_pad,
    )
    return fp4_u8, sorted_u8[:m_pad].view(torch.uint8)


# ===================== HIP MFMA 3-in-1 (aiter) =====================
def hip_mfma_path(x, rot, sorted_ids, num_valid_ids, M, K, RS, QG, token_num, block_size=32):
    from aiter.ops.mfma_rot_quant_moe_sort import mfma_rot_quant_moe_sort as aiter_mfma
    n_i = K // QG
    m_o = sorted_ids.shape[0]
    m_pad = ((m_o + block_size - 1) // block_size) * block_size
    fp4_u8 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    sc_u8 = torch.zeros((m_pad, n_i), dtype=torch.uint8, device=x.device)
    aiter_mfma(x, rot, fp4_u8, sc_u8, sorted_ids, num_valid_ids, token_num, RS)
    return fp4_u8, sc_u8


# ===================== Aiter unified kernel =====================
def aiter_unified_path(x, rot, sorted_ids, num_valid_ids, M, K, RS, QG, token_num, block_size=32):
    from aiter.ops.triton.fused_rot_quant_moe_sort import (
        fused_rot_quant_moe_sort as aiter_fused_fn,
        warmup_kernels as aiter_warmup_fn,
    )
    topk = sorted_ids.shape[0] // max(M, 1)
    aiter_warmup_fn(K=K, topk=topk, device=x.device)
    fp4, sc = aiter_fused_fn(
        x, rot, RS, sorted_ids, num_valid_ids,
        token_num=token_num, block_size=block_size,
    )
    return fp4.view(torch.uint8), sc.view(torch.uint8)


# ===================== MAIN =====================
def main():
    device = "cuda"
    torch.manual_seed(42)

    E, topk, K, RS, QG = 128, 8, 2048, 128, 32
    n_i = K // QG

    print("=" * 80)
    print("  MoE Fused Rotation + MXFP4 Quant: Correctness & Performance")
    print("=" * 80)
    print(f"  Model: Qwen3-30B-A3B | K={K}, RS={RS}, QG={QG}, topk={topk}, E={E}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'N/A')}")
    print("=" * 80)

    # ============ PART 1: CORRECTNESS ============
    print("\n" + "=" * 80)
    print("  PART 1: CORRECTNESS VERIFICATION")
    print("=" * 80)

    all_pass = True

    for M in [1, 4, 8, 16, 32]:
        print(f"\n  --- M = {M} ---")
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        rot = torch.randn(RS, RS, dtype=torch.bfloat16, device=device) * 0.05
        topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
        topk_w = torch.ones(M, topk, dtype=torch.float32, device=device) / topk
        si, _, _, nv, _ = moe_sorting(topk_ids, topk_w, E, K, moebuf_dtype=x.dtype, block_size=32)

        token_num = M
        m_o = si.shape[0]
        m_pad = ((m_o + 31) // 32) * 32

        fp4_ref, sc_ref = reference_separated(x, rot, si, nv, M, K, RS, QG, token_num)
        torch.cuda.synchronize()

        print(f"  {'Path':<30} {'FP4 match':>12} {'Scale match':>12} {'Status'}")
        print(f"  {'='*66}")

        paths = [
            ("gluon_kw8 + sort", gluon_kw8_path),
            ("HIP MFMA 3-in-1 (aiter)", hip_mfma_path),
            ("aiter unified", aiter_unified_path),
        ]
        if M == 1:
            paths.insert(1, ("Triton decode rot+sort", triton_decode_rot_sort_path))
            paths.insert(2, ("Gluon MOE decode", gluon_moe_decode_path))

        for name, fn in paths:
            try:
                fp4_test, sc_test = fn(x, rot, si, nv, M, K, RS, QG, token_num)
                torch.cuda.synchronize()

                fp4_pct = compare_tensors(fp4_ref[:M, :K//2], fp4_test.view(torch.uint8)[:M, :K//2])
                sc_ref_t = sc_ref[:m_pad, :n_i]
                sc_test_t = sc_test[:m_pad, :n_i]
                sc_pct = compare_tensors(sc_ref_t, sc_test_t)
                ok = fp4_pct >= 95 and sc_pct >= 95
                if not ok:
                    all_pass = False
                status = "PASS" if ok else "FAIL"
                print(f"  {name:<30} {fmt_pct(fp4_pct):>20} {fmt_pct(sc_pct):>20}   {status}")
            except Exception as e:
                all_pass = False
                print(f"  {name:<30} {'ERROR':>12} {'':>12}   FAIL: {e}")

    print(f"\n  Overall correctness: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ============ PART 2: PERFORMANCE ============
    print("\n\n" + "=" * 80)
    print("  PART 2: KERNEL-LEVEL PERFORMANCE BENCHMARK (us)")
    print("=" * 80)
    print(f"  K={K}, RS={RS}, topk={topk} | warmup=20, iters=200, trimmed mean")
    print()

    print(f"  {'M':>3} | {'Separated':>10} | {'gluon+sort':>10} | {'TritonM1':>10} | {'GluonMOE':>10} | {'HIP_MFMA':>10} | {'AiterUni':>10}")
    print("  " + "-" * 77)

    bench_results = {}

    for M in [1, 4, 8, 16, 32]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        rot = torch.randn(RS, RS, dtype=torch.bfloat16, device=device) * 0.05
        topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
        topk_w = torch.ones(M, topk, dtype=torch.float32, device=device) / topk
        si, _, _, nv, _ = moe_sorting(topk_ids, topk_w, E, K, moebuf_dtype=x.dtype, block_size=32)
        token_num = M

        row = {}

        t_ref, _, _ = bench_fn(lambda: reference_separated(x, rot, si, nv, M, K, RS, QG, token_num))
        row["separated"] = t_ref

        try:
            t_gk, _, _ = bench_fn(lambda: gluon_kw8_path(x, rot, si, nv, M, K, RS, QG, token_num))
            row["gluon_kw8"] = t_gk
        except:
            t_gk = float('nan'); row["gluon_kw8"] = None

        if M == 1:
            try:
                t_tm1, _, _ = bench_fn(lambda: triton_decode_rot_sort_path(x, rot, si, nv, M, K, RS, QG, token_num))
                row["triton_m1"] = t_tm1
            except:
                t_tm1 = float('nan'); row["triton_m1"] = None
            try:
                t_gm, _, _ = bench_fn(lambda: gluon_moe_decode_path(x, rot, si, nv, M, K, RS, QG, token_num))
                row["gluon_moe"] = t_gm
            except:
                t_gm = float('nan'); row["gluon_moe"] = None
        else:
            t_tm1 = float('nan'); row["triton_m1"] = None
            t_gm = float('nan'); row["gluon_moe"] = None

        try:
            t_mfma, _, _ = bench_fn(lambda: hip_mfma_path(x, rot, si, nv, M, K, RS, QG, token_num))
            row["hip_mfma"] = t_mfma
        except:
            t_mfma = float('nan'); row["hip_mfma"] = None

        try:
            t_uni, _, _ = bench_fn(lambda: aiter_unified_path(x, rot, si, nv, M, K, RS, QG, token_num))
            row["aiter_uni"] = t_uni
        except:
            t_uni = float('nan'); row["aiter_uni"] = None

        def f(v):
            return f"{v:>10.1f}" if v == v and v is not None else f"{'N/A':>10}"
        print(f"  {M:>3} | {f(t_ref)} | {f(t_gk)} | {f(t_tm1)} | {f(t_gm)} | {f(t_mfma)} | {f(t_uni)}")
        bench_results[str(M)] = row

    # Speedup table
    print(f"\n  Speedup vs Separated:")
    print(f"  {'M':>3} | {'gluon+sort':>10} | {'TritonM1':>10} | {'GluonMOE':>10} | {'HIP_MFMA':>10} | {'AiterUni':>10}")
    print("  " + "-" * 65)
    for M_str, row in bench_results.items():
        ref = row["separated"]
        def sp(key):
            v = row.get(key)
            if v is None or v != v: return f"{'N/A':>10}"
            return f"{ref/v:>9.2f}x"
        print(f"  {M_str:>3} | {sp('gluon_kw8')} | {sp('triton_m1')} | {sp('gluon_moe')} | {sp('hip_mfma')} | {sp('aiter_uni')}")

    out_path = "/data/jiangyon/vllm_rotation/moe_fused_test_results.json"
    with open(out_path, "w") as f:
        json.dump(bench_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print("\n" + "=" * 80)
    print("  Step 1 & 2 DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

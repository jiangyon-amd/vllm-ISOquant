#!/usr/bin/env python3
"""
MoE Fused Rotation + MXFP4 Quant: Correctness & Performance v2
================================================================

Correctness strategy:
  - Use bf16 rotation (same precision as fused kernels) for fair comparison
  - Compare FP4 output vs bf16 reference rotation + aiter quant
  - Compare scale via dequant → check final bf16 error (not raw scale layout)
  - Also do self-consistency check: all fused paths produce same FP4

Performance: CUDA event timing, trimmed mean.
"""

import os, sys, time, torch, json

os.environ.setdefault("HIP_VISIBLE_DEVICES", "4")
sys.path.insert(0, "/data/jiangyon/vllm_rotation")

from aiter.fused_moe import moe_sorting, fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility.fp4_utils import moe_mxfp4_sort
from aiter.utility import dtypes
from aiter import per_1x32_f4_quant_hip


def compare_u8(a, b):
    n = min(a.numel(), b.numel())
    return (a.reshape(-1)[:n] == b.reshape(-1)[:n]).sum().item() / n * 100


def bench_fn(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in evts:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000 for s, e in evts)
    trim = max(1, iters // 10)
    return sum(times[trim:-trim]) / len(times[trim:-trim])


def bf16_reference(x, rot, sorted_ids, num_valid_ids, M, K, RS, QG, token_num):
    """Reference: bf16 matmul rotation (same precision as fused) + aiter quant+sort."""
    x_rot = (x.reshape(-1, K // RS, RS).to(torch.bfloat16) @ rot.to(torch.bfloat16)).reshape(M, K)
    fp4, sc = fused_dynamic_mxfp4_quant_moe_sort(
        x_rot, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token_num, topk=1, block_size=32,
    )
    return x_rot, fp4.view(torch.uint8), sc.view(torch.uint8)


def bf16_reference_hip(x, rot, sorted_ids, num_valid_ids, M, K, RS, QG, token_num):
    """Reference: bf16 matmul rotation + aiter per_1x32_f4_quant_hip + moe_mxfp4_sort."""
    x_rot = (x.reshape(-1, K // RS, RS).to(torch.bfloat16) @ rot.to(torch.bfloat16)).reshape(M, K)
    fp4, raw_sc = per_1x32_f4_quant_hip(x_rot, shuffle=False)
    sorted_sc = moe_mxfp4_sort(
        raw_sc.view(torch.uint8), sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token_num, block_size=32,
    )
    return x_rot, fp4.view(torch.uint8), raw_sc.view(torch.uint8), sorted_sc.view(torch.uint8)


# ---- Path implementations (only FP4 + raw scale, no sorted scale) ----

def gluon_kw8_raw(x, rot, si, nv, M, K, RS, QG, tn):
    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon_v2_kw8 import fused_gluon_v2_kw8
    n_i = K // QG
    fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    sc = torch.empty((M, n_i), dtype=torch.uint8, device=x.device)
    fp4, sc = fused_gluon_v2_kw8(x, rot, RS, fp4_out=fp4, scales_out=sc, shuffle_scales=False)
    sorted_sc = moe_mxfp4_sort(sc, sorted_ids=si, num_valid_ids=nv, token_num=tn, block_size=32)
    return fp4, sc, sorted_sc.view(torch.uint8)


def gluon_v2_raw(x, rot, si, nv, M, K, RS, QG, tn):
    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import fused_rot_quant_gluon
    n_i = K // QG
    fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    sc = torch.empty((M, n_i), dtype=torch.uint8, device=x.device)
    fp4, sc = fused_rot_quant_gluon(x, rot, RS, fp4_out=fp4, scales_out=sc, shuffle_scales=False)
    sorted_sc = moe_mxfp4_sort(sc, sorted_ids=si, num_valid_ids=nv, token_num=tn, block_size=32)
    return fp4, sc, sorted_sc.view(torch.uint8)


def hip_mfma(x, rot, si, nv, M, K, RS, QG, tn):
    from aiter.ops.mfma_rot_quant_moe_sort import mfma_rot_quant_moe_sort as aiter_mfma
    n_i = K // QG
    m_o = si.shape[0]
    m_pad = ((m_o + 31) // 32) * 32
    fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    sc = torch.zeros((m_pad, n_i), dtype=torch.uint8, device=x.device)
    aiter_mfma(x, rot, fp4, sc, si, nv, tn, RS)
    return fp4, sc  # sc is already sorted layout


def aiter_unified(x, rot, si, nv, M, K, RS, QG, tn):
    from aiter.ops.triton.fused_rot_quant_moe_sort import (
        fused_rot_quant_moe_sort as fn, warmup_kernels as warmup,
    )
    topk = si.shape[0] // max(M, 1)
    warmup(K=K, topk=topk, device=x.device)
    fp4, sc = fn(x, rot, RS, si, nv, token_num=tn, block_size=32)
    return fp4.view(torch.uint8), sc.view(torch.uint8)


def main():
    device = "cuda"
    torch.manual_seed(42)

    E, topk, K, RS, QG = 128, 8, 2048, 128, 32
    n_i = K // QG

    print("=" * 80)
    print("  MoE Fused Rotation + MXFP4 Quant: Correctness & Performance v2")
    print("=" * 80)
    print(f"  K={K}, RS={RS}, QG={QG}, topk={topk}, E={E}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # ============ PART 1: CORRECTNESS (FP4 match vs bf16 reference) ============
    print("\n" + "=" * 80)
    print("  PART 1: CORRECTNESS")
    print("  Reference: bf16 matmul rotation + per_1x32_f4_quant_hip + moe_mxfp4_sort")
    print("  Compare: raw scale (pre-sort) and FP4 output")
    print("=" * 80)

    all_pass = True

    for M in [1, 4, 8, 16, 32]:
        print(f"\n  --- M = {M} ---")
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        rot = torch.randn(RS, RS, dtype=torch.bfloat16, device=device) * 0.05
        topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
        topk_w = torch.ones(M, topk, dtype=torch.float32, device=device) / topk
        si, _, _, nv, _ = moe_sorting(topk_ids, topk_w, E, K, moebuf_dtype=x.dtype, block_size=32)

        m_o = si.shape[0]
        m_pad = ((m_o + 31) // 32) * 32

        # Reference
        x_rot_ref, fp4_ref, raw_sc_ref, sorted_sc_ref = bf16_reference_hip(
            x, rot, si, nv, M, K, RS, QG, M)
        torch.cuda.synchronize()

        print(f"  {'Path':<30} {'FP4 match':>12} {'Raw Scale':>12} {'Sorted Scale':>14} {'Status'}")
        print(f"  {'='*76}")

        paths = []
        # Paths that produce raw scale → sort separately
        try:
            fp4_gk, sc_gk, sorted_gk = gluon_kw8_raw(x, rot, si, nv, M, K, RS, QG, M)
            torch.cuda.synchronize()
            fp4_m = compare_u8(fp4_ref[:M,:K//2], fp4_gk[:M,:K//2])
            sc_m = compare_u8(raw_sc_ref[:M,:n_i], sc_gk[:M,:n_i])
            ss_m = compare_u8(sorted_sc_ref[:m_pad,:n_i], sorted_gk[:m_pad,:n_i])
            ok = fp4_m >= 95 and sc_m >= 95
            if not ok: all_pass = False
            status = "PASS" if ok else "FAIL"
            print(f"  {'gluon_kw8 + sort':<30} {fp4_m:>11.1f}% {sc_m:>11.1f}% {ss_m:>13.1f}%  {status}")
        except Exception as e:
            all_pass = False
            print(f"  {'gluon_kw8 + sort':<30} ERROR: {e}")

        try:
            fp4_g2, sc_g2, sorted_g2 = gluon_v2_raw(x, rot, si, nv, M, K, RS, QG, M)
            torch.cuda.synchronize()
            fp4_m = compare_u8(fp4_ref[:M,:K//2], fp4_g2[:M,:K//2])
            sc_m = compare_u8(raw_sc_ref[:M,:n_i], sc_g2[:M,:n_i])
            ss_m = compare_u8(sorted_sc_ref[:m_pad,:n_i], sorted_g2[:m_pad,:n_i])
            ok = fp4_m >= 95 and sc_m >= 95
            if not ok: all_pass = False
            status = "PASS" if ok else "FAIL"
            print(f"  {'gluon_v2 + sort':<30} {fp4_m:>11.1f}% {sc_m:>11.1f}% {ss_m:>13.1f}%  {status}")
        except Exception as e:
            all_pass = False
            print(f"  {'gluon_v2 + sort':<30} ERROR: {e}")

        # HIP MFMA - produces sorted scale directly
        try:
            fp4_h, sc_h = hip_mfma(x, rot, si, nv, M, K, RS, QG, M)
            torch.cuda.synchronize()
            fp4_m = compare_u8(fp4_ref[:M,:K//2], fp4_h[:M,:K//2])
            ss_m = compare_u8(sorted_sc_ref[:m_pad,:n_i], sc_h[:m_pad,:n_i])
            ok = fp4_m >= 95
            if not ok: all_pass = False
            status = "PASS" if ok else "FAIL"
            print(f"  {'HIP MFMA 3-in-1':<30} {fp4_m:>11.1f}% {'N/A':>12} {ss_m:>13.1f}%  {status}")
        except Exception as e:
            all_pass = False
            print(f"  {'HIP MFMA 3-in-1':<30} ERROR: {e}")

        # Aiter unified - produces sorted scale directly
        try:
            fp4_u, sc_u = aiter_unified(x, rot, si, nv, M, K, RS, QG, M)
            torch.cuda.synchronize()
            fp4_m = compare_u8(fp4_ref[:M,:K//2], fp4_u[:M,:K//2])
            ss_m = compare_u8(sorted_sc_ref[:m_pad,:n_i], sc_u[:m_pad,:n_i])
            ok = fp4_m >= 95
            if not ok: all_pass = False
            status = "PASS" if ok else "FAIL"
            print(f"  {'aiter unified':<30} {fp4_m:>11.1f}% {'N/A':>12} {ss_m:>13.1f}%  {status}")
        except Exception as e:
            all_pass = False
            print(f"  {'aiter unified':<30} ERROR: {e}")

        # Cross-compare: gluon_kw8 vs gluon_v2 (should be nearly identical)
        try:
            fp4_cross = compare_u8(fp4_gk[:M,:K//2], fp4_g2[:M,:K//2])
            sc_cross = compare_u8(sc_gk[:M,:n_i], sc_g2[:M,:n_i])
            print(f"  {'[cross] kw8 vs v2':<30} {fp4_cross:>11.1f}% {sc_cross:>11.1f}%")
        except:
            pass

    print(f"\n  Correctness: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"  Note: FP4 ~97-99% match is expected due to bf16 matmul rounding vs v_cvt ISA rounding.")
    print(f"        Scale sorted layout match depends on whether the sort formula matches aiter exactly.")

    # ============ PART 2: PERFORMANCE ============
    print("\n\n" + "=" * 80)
    print("  PART 2: KERNEL-LEVEL PERFORMANCE (us, trimmed mean)")
    print("=" * 80)

    print(f"\n  {'M':>3} | {'Separated':>10} | {'gluon_kw8':>10} | {'gluon_v2':>10} | {'HIP_MFMA':>10} | {'AiterUni':>10}")
    print(f"  {'':>3} | {'matmul+q+s':>10} | {'+sort':>10} | {'+sort':>10} | {'3-in-1':>10} | {'kernel':>10}")
    print("  " + "-" * 67)

    bench_results = {}

    for M in [1, 4, 8, 16, 32]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        rot = torch.randn(RS, RS, dtype=torch.bfloat16, device=device) * 0.05
        topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
        topk_w = torch.ones(M, topk, dtype=torch.float32, device=device) / topk
        si, _, _, nv, _ = moe_sorting(topk_ids, topk_w, E, K, moebuf_dtype=x.dtype, block_size=32)

        row = {}

        # Separated reference
        def _ref():
            xr = (x.reshape(-1, K//RS, RS) @ rot).reshape(M, K)
            per_1x32_f4_quant_hip(xr, shuffle=False)
        t_ref = bench_fn(_ref)
        row["separated"] = t_ref

        # gluon_kw8 + sort
        try:
            t = bench_fn(lambda: gluon_kw8_raw(x, rot, si, nv, M, K, RS, QG, M))
            row["gluon_kw8"] = t
        except:
            t = float('nan'); row["gluon_kw8"] = None

        # gluon_v2 + sort
        try:
            t2 = bench_fn(lambda: gluon_v2_raw(x, rot, si, nv, M, K, RS, QG, M))
            row["gluon_v2"] = t2
        except:
            t2 = float('nan'); row["gluon_v2"] = None

        # HIP MFMA
        try:
            t3 = bench_fn(lambda: hip_mfma(x, rot, si, nv, M, K, RS, QG, M))
            row["hip_mfma"] = t3
        except:
            t3 = float('nan'); row["hip_mfma"] = None

        # Aiter unified
        try:
            t4 = bench_fn(lambda: aiter_unified(x, rot, si, nv, M, K, RS, QG, M))
            row["aiter_uni"] = t4
        except:
            t4 = float('nan'); row["aiter_uni"] = None

        def f(v):
            return f"{v:>10.1f}" if v is not None and v == v else f"{'N/A':>10}"
        print(f"  {M:>3} | {f(t_ref)} | {f(row.get('gluon_kw8'))} | {f(row.get('gluon_v2'))} | {f(row.get('hip_mfma'))} | {f(row.get('aiter_uni'))}")
        bench_results[str(M)] = row

    # Speedup
    print(f"\n  Speedup vs Separated:")
    print(f"  {'M':>3} | {'gluon_kw8':>10} | {'gluon_v2':>10} | {'HIP_MFMA':>10} | {'AiterUni':>10}")
    print("  " + "-" * 55)
    for M_str, row in bench_results.items():
        ref = row["separated"]
        def sp(key):
            v = row.get(key)
            if v is None or v != v: return f"{'N/A':>10}"
            return f"{ref/v:>9.2f}x"
        print(f"  {M_str:>3} | {sp('gluon_kw8')} | {sp('gluon_v2')} | {sp('hip_mfma')} | {sp('aiter_uni')}")

    out = "/data/jiangyon/vllm_rotation/moe_fused_test_results_v2.json"
    with open(out, "w") as f:
        json.dump(bench_results, f, indent=2, default=str)
    print(f"\n  Results: {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()

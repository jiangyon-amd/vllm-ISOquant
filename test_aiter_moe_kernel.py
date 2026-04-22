#!/usr/bin/env python3
"""
Unit test: aiter fused_rot_quant_moe_sort vs reference (fused_rot_quant_sort_kw8).

Tests with real Qwen3-30B shapes:
  K=2048, topk=8, RS=128, E=128, block_size=32
  Decode: M=1 (token_num=1)
  Prefill: M=2,4,8,16,32,64,128,256
"""

import argparse
import math
import os
import sys
import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
torch.manual_seed(42)

# ── Qwen3-30B real shapes ──
K = 2048
RS = 128
QG = 32
N_I = K // QG  # 64
TOPK = 8
E = 128
BLOCK_SIZE = 32


def make_data(M):
    """Create test data with real Qwen3-30B MoE shapes."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rot = torch.randn(RS, RS, dtype=torch.bfloat16, device="cuda") / math.sqrt(RS)
    n = M * TOPK
    tids = torch.arange(M, device="cuda").repeat_interleave(TOPK)
    eids = torch.randint(0, E, (n,), device="cuda")
    sids = ((eids << 24) | tids)[torch.argsort(eids)].to(torch.int32)
    nv = torch.tensor(n, dtype=torch.int32, device="cuda")
    return x, rot, sids, nv


def run_aiter(x, rot, sids, nv, M):
    """New aiter implementation."""
    from aiter.ops.triton.fused_rot_quant_moe_sort import fused_rot_quant_moe_sort
    return fused_rot_quant_moe_sort(
        x, rot, RS, sorted_ids=sids, num_valid_ids=nv,
        token_num=M, block_size=BLOCK_SIZE,
    )


def run_ref(x, rot, sids, nv, M):
    """Reference: existing 3-in-1 gluon kw8 (same 0x200000 rounding)."""
    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_sort_gluon_kw8 import (
        fused_rot_quant_sort_kw8,
    )
    return fused_rot_quant_sort_kw8(
        x, rot, RS, sids, nv, token_num=M, topk=TOPK, block_size=BLOCK_SIZE,
    )


def run_legacy(x, rot, sids, nv, M):
    """Legacy vLLM dispatch (M=1→gluon decode, M>1→kw8+moe_sort)."""
    os.environ["VLLM_MOE_UNIFIED_KERNEL"] = "0"
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (
        _fused_rot_quant_moe_sort_impl,
    )
    import vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_unified as u
    u.ENABLE_UNIFIED_MOE = False
    try:
        return _fused_rot_quant_moe_sort_impl(
            x, rot, RS, sids, nv, token_num=M, topk=TOPK, block_size=BLOCK_SIZE,
            use_hip_kernel=False, use_triton_rot_sort_kernel=False,
        )
    finally:
        u.ENABLE_UNIFIED_MOE = True


def bench(fn, args, N=3000):
    for _ in range(300):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(N):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / N * 1000  # µs


def compare(name, fp4_r, sc_r, fp4_t, sc_t):
    fp4_ok = torch.equal(fp4_r.view(torch.uint8), fp4_t.view(torch.uint8))
    sr = sc_r.view(torch.uint8)
    st = sc_t.view(torch.uint8)
    rows = min(sr.shape[0], st.shape[0])
    cols = min(sr.shape[1], st.shape[1])
    mask = sr[:rows, :cols] != 0
    n_w = mask.sum().item()
    if n_w > 0:
        sc_ok = torch.equal(sr[:rows, :cols][mask], st[:rows, :cols][mask])
    else:
        sc_ok = True
    ok = fp4_ok and sc_ok
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name} ({n_w} scale positions checked)")
    if not ok:
        if not fp4_ok:
            d = (fp4_r.view(torch.uint8) != fp4_t.view(torch.uint8)).sum().item()
            print(f"    FP4 mismatch: {d}")
        if not sc_ok:
            d = (sr[:rows,:cols][mask] != st[:rows,:cols][mask]).sum().item()
            print(f"    Scale mismatch: {d}/{n_w}")
    return ok


def test_correctness():
    print("=" * 70)
    print(f"Correctness: aiter kernel vs reference (K={K}, topk={TOPK})")
    print("=" * 70)

    all_ok = True
    for M in [1, 2, 3, 4, 8, 16, 17, 32, 33, 64, 128, 256]:
        print(f"\n  M={M}:")
        x, rot, sids, nv = make_data(M)
        fp4_r, sc_r = run_ref(x, rot, sids, nv, M)
        fp4_a, sc_a = run_aiter(x, rot, sids, nv, M)
        ok = compare("aiter vs ref", fp4_r, sc_r, fp4_a, sc_a)
        all_ok = all_ok and ok

    print(f"\n{'=' * 70}")
    status = "ALL PASSED" if all_ok else "SOME FAILED"
    print(f"Result: {status}")
    print(f"{'=' * 70}")
    return all_ok


def test_bench():
    print(f"\n{'=' * 70}")
    print(f"Benchmark: Qwen3-30B shapes (K={K}, topk={TOPK}, RS={RS})")
    print(f"{'=' * 70}")
    print(f"\n{'M':<6} {'legacy':>10} {'aiter-m1':>10} {'aiter':>10} {'speedup':>10}")
    print("-" * 48)

    for M in [1, 4, 16, 64, 256]:
        x, rot, sids, nv = make_data(M)

        # Warmup all
        run_legacy(x, rot, sids, nv, M)
        run_aiter(x, rot, sids, nv, M)

        t_leg = bench(run_legacy, (x, rot, sids, nv, M))
        t_ait = bench(run_aiter, (x, rot, sids, nv, M))
        sp = t_leg / t_ait
        path = "M=1" if M == 1 else f"M>1"
        print(f"{M:<6} {t_leg:>9.1f} {'':>10} {t_ait:>9.1f} {sp:>9.2f}x  [{path}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if not any([args.test, args.bench, args.all]):
        args.all = True

    if args.test or args.all:
        ok = test_correctness()
        if not ok:
            sys.exit(1)
    if args.bench or args.all:
        test_bench()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test + Benchmark: Unified MoE kernel vs original paths.

Part 1: Correctness — bitwise comparison of FP4 + sorted scales
Part 2: Kernel-level benchmark — Python timer (warm cache)
Part 3: rocprof integration — generates trace for detailed kernel timing

Usage:
  # Correctness only:
  python test_unified_moe_kernel.py --test

  # Correctness + benchmark:
  python test_unified_moe_kernel.py --bench

  # Generate rocprof trace:
  rocprof --hip-trace python test_unified_moe_kernel.py --rocprof
"""

import argparse
import math
import os
import sys
import time

import torch
import triton

torch.manual_seed(42)
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------

def create_moe_test_data(M, K, topk, RS=128):
    """Create realistic MoE test inputs."""
    QG = 32
    n_i = K // QG

    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
    rotation = torch.randn(RS, RS, dtype=torch.bfloat16, device=DEVICE) / math.sqrt(RS)

    n_entries = M * topk
    token_ids = torch.arange(M, device=DEVICE).repeat_interleave(topk)
    expert_ids = torch.randint(0, 8, (n_entries,), device=DEVICE)
    sorted_ids = (expert_ids << 24) | token_ids
    sorted_ids = sorted_ids[torch.argsort(expert_ids)].to(torch.int32)
    num_valid_ids = torch.tensor(n_entries, dtype=torch.int32, device=DEVICE)

    return x, rotation, sorted_ids, num_valid_ids


# ---------------------------------------------------------------------------
# Kernel runners
# ---------------------------------------------------------------------------

def run_original(x, rotation, sorted_ids, num_valid_ids, token_num, topk, RS=128):
    """Reference: existing 3-in-1 kw8 kernel (same 0x200000 rounding as unified)."""
    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_sort_gluon_kw8 import (
        fused_rot_quant_sort_kw8,
    )
    return fused_rot_quant_sort_kw8(
        x, rotation, RS, sorted_ids, num_valid_ids,
        token_num=token_num, topk=topk, block_size=32,
    )


def run_legacy_dispatch(x, rotation, sorted_ids, num_valid_ids, token_num, topk, RS=128):
    """Legacy dispatch with unified disabled (falls back to old multi-branch)."""
    import vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_unified as umod
    saved = umod.ENABLE_UNIFIED_MOE
    umod.ENABLE_UNIFIED_MOE = False
    try:
        from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (
            _fused_rot_quant_moe_sort_impl,
        )
        return _fused_rot_quant_moe_sort_impl(
            x, rotation, RS, sorted_ids, num_valid_ids,
            token_num=token_num, topk=topk, block_size=32,
            use_hip_kernel=False, use_triton_rot_sort_kernel=False,
        )
    finally:
        umod.ENABLE_UNIFIED_MOE = saved


def run_unified(x, rotation, sorted_ids, num_valid_ids, token_num, topk, RS=128):
    """Unified single Triton kernel (the new unified implementation)."""
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_unified import (
        _unified_impl,
    )
    return _unified_impl(
        x, rotation, RS, sorted_ids, num_valid_ids,
        token_num=token_num, topk=topk, block_size=32,
    )


# ---------------------------------------------------------------------------
# Part 1: Correctness
# ---------------------------------------------------------------------------

def compare_outputs(name, fp4_ref, sc_ref, fp4_test, sc_test):
    """Compare FP4 + sorted scales at valid positions only.

    For the mixed-allocation strategy (torch.empty for M≤topk), unwritten
    positions have garbage in the test output. We compare only positions
    where BOTH ref and test are non-zero (i.e., actually written by scatter).
    """
    fp4_r = fp4_ref.view(torch.uint8)
    fp4_t = fp4_test.view(torch.uint8)
    fp4_match = torch.equal(fp4_r, fp4_t)

    sc_r = sc_ref.view(torch.uint8)
    sc_t = sc_test.view(torch.uint8)
    min_r = min(sc_r.shape[0], sc_t.shape[0])
    min_c = min(sc_r.shape[1], sc_t.shape[1])
    sc_r_crop = sc_r[:min_r, :min_c]
    sc_t_crop = sc_t[:min_r, :min_c]

    # Compare only at positions that the reference actually wrote to (non-zero)
    written_mask = sc_r_crop != 0
    n_written = written_mask.sum().item()
    if n_written > 0:
        sc_match = torch.equal(sc_r_crop[written_mask], sc_t_crop[written_mask])
        n_diff = 0 if sc_match else (sc_r_crop[written_mask] != sc_t_crop[written_mask]).sum().item()
    else:
        sc_match = True
        n_diff = 0

    status = "PASS" if (fp4_match and sc_match) else "FAIL"
    print(f"  [{status}] {name}")
    if not fp4_match:
        n_fp4_diff = (fp4_r != fp4_t).sum().item()
        print(f"    FP4: MISMATCH ({n_fp4_diff}/{fp4_r.numel()} differ)")
    if not sc_match:
        print(f"    Scales: MISMATCH ({n_diff}/{n_written} written positions differ)")
        diff_idx = (sc_r_crop[written_mask] != sc_t_crop[written_mask]).nonzero()[:5]
        for idx in diff_idx:
            i = idx[0].item()
            print(f"      written pos {i}: ref={sc_r_crop[written_mask][i].item()} got={sc_t_crop[written_mask][i].item()}")
    else:
        print(f"    (checked {n_written} written positions)")
    return fp4_match and sc_match


def test_correctness():
    """Test unified kernel correctness for various M values."""
    print("=" * 70)
    print("Part 1: Correctness Test")
    print("=" * 70)

    RS = 128
    cases = [
        # (M, K, topk, token_num, description)
        # --- Core cases (K=2048, topk=8) ---
        (1,   2048, 8,  1,   "decode M=1 K=2048"),
        (4,   2048, 8,  4,   "batch M=4 K=2048"),
        (16,  2048, 8,  16,  "prefill M=16 K=2048"),
        (32,  2048, 8,  32,  "medium M=32 K=2048"),
        (64,  2048, 8,  64,  "medium M=64 K=2048"),
        (128, 2048, 8,  128, "large M=128 K=2048"),
        (256, 2048, 8,  256, "large M=256 K=2048"),
        # --- K=5120 (Qwen3-30B actual) ---
        (1,   5120, 8,  1,   "decode M=1 K=5120"),
        (64,  5120, 8,  64,  "prefill M=64 K=5120"),
        # --- Edge: non-power-of-2 M, not divisible by BLOCK_M ---
        (3,   2048, 8,  3,   "edge M=3 (not pow2)"),
        (17,  2048, 8,  17,  "edge M=17 (crosses BM=16→32)"),
        (33,  2048, 8,  33,  "edge M=33 (BM=32, 2 tiles)"),
        # --- Edge: topk=1 (single expert per token) ---
        (1,   2048, 1,  1,   "topk=1 decode"),
        (16,  2048, 1,  16,  "topk=1 prefill"),
    ]

    all_pass = True
    for M, K, topk, token_num, desc in cases:
        print(f"\n--- {desc} (M={M}, K={K}, topk={topk}) ---")
        x, rot, sids, nv = create_moe_test_data(M, K, topk, RS)

        fp4_ref, sc_ref = run_original(x, rot, sids, nv, token_num, topk, RS)
        fp4_uni, sc_uni = run_unified(x, rot, sids, nv, token_num, topk, RS)

        ok = compare_outputs("unified vs reference", fp4_ref, sc_ref, fp4_uni, sc_uni)
        all_pass = all_pass and ok

    print(f"\n{'=' * 70}")
    print(f"Correctness: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 70}")
    return all_pass


# ---------------------------------------------------------------------------
# Part 2: Kernel-level Benchmark
# ---------------------------------------------------------------------------

def bench_one(fn, args, n_warmup=200, n_iter=1000):
    """Benchmark a single kernel call."""
    # Warmup
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter * 1000  # microseconds


def bench_kernels():
    """Benchmark unified vs 3-in-1 ref vs 2-kernel pipeline."""
    print("\n" + "=" * 70)
    print("Part 2: Kernel Benchmark (µs per call)")
    print("=" * 70)

    RS = 128
    topk = 8

    cases = [
        # (M, K, token_num, desc)
        (1,   2048, 1,   "M=1  K=2048"),
        (4,   2048, 4,   "M=4  K=2048"),
        (16,  2048, 16,  "M=16 K=2048"),
        (64,  2048, 64,  "M=64 K=2048"),
        (256, 2048, 256, "M=256 K=2048"),
        (1,   5120, 1,   "M=1  K=5120"),
        (64,  5120, 64,  "M=64 K=5120"),
    ]

    # --- Table 1: unified vs 3-in-1 reference ---
    print(f"\n--- Unified vs 3-in-1 reference (same rounding) ---")
    print(f"{'Case':<20} {'3in1-ref(µs)':>14} {'Unified(µs)':>14} {'Ratio':>10}")
    print("-" * 60)

    for M, K, token_num, desc in cases:
        x, rot, sids, nv = create_moe_test_data(M, K, topk, RS)
        run_original(x, rot, sids, nv, token_num, topk, RS)
        run_unified(x, rot, sids, nv, token_num, topk, RS)

        t_ref = bench_one(run_original, (x, rot, sids, nv, token_num, topk, RS))
        t_uni = bench_one(run_unified, (x, rot, sids, nv, token_num, topk, RS))
        ratio = t_ref / t_uni if t_uni > 0 else float('inf')
        print(f"{desc:<20} {t_ref:>14.1f} {t_uni:>14.1f} {ratio:>9.2f}x")

    # --- Table 2: unified vs legacy dispatch ---
    print(f"\n--- Unified vs legacy dispatch (production path) ---")
    print(f"{'Case':<20} {'Legacy(µs)':>14} {'Unified(µs)':>14} {'Speedup':>10} {'Note':>14}")
    print("-" * 74)

    for M, K, token_num, desc in cases:
        x, rot, sids, nv = create_moe_test_data(M, K, topk, RS)
        run_legacy_dispatch(x, rot, sids, nv, token_num, topk, RS)
        run_unified(x, rot, sids, nv, token_num, topk, RS)

        t_leg = bench_one(run_legacy_dispatch, (x, rot, sids, nv, token_num, topk, RS))
        t_uni = bench_one(run_unified, (x, rot, sids, nv, token_num, topk, RS))
        speedup = t_leg / t_uni if t_uni > 0 else float('inf')

        note = "1 launch" if M == 1 else "1 launch saved"
        print(f"{desc:<20} {t_leg:>14.1f} {t_uni:>14.1f} {speedup:>9.2f}x {note:>14}")


# ---------------------------------------------------------------------------
# Part 3: rocprof trace generation
# ---------------------------------------------------------------------------

def gen_rocprof_trace():
    """Run kernels with markers for rocprof --hip-trace analysis."""
    print("\n" + "=" * 70)
    print("Part 3: rocprof trace (run with: rocprof --hip-trace python ...)")
    print("=" * 70)

    K = 2048
    RS = 128
    topk = 8

    # Focus on decode (M=1) — most latency-sensitive
    M = 1
    token_num = 1
    x, rot, sids, nv = create_moe_test_data(M, K, topk, RS)

    # Warmup
    for _ in range(50):
        run_original(x, rot, sids, nv, token_num, topk, RS)
        run_unified(x, rot, sids, nv, token_num, topk, RS)
    torch.cuda.synchronize()

    # Traced section: 100 iterations each, clearly separated
    print("Running original (2-kernel) path x100...")
    torch.cuda.synchronize()
    for _ in range(100):
        run_original(x, rot, sids, nv, token_num, topk, RS)
    torch.cuda.synchronize()

    print("Running unified (1-kernel) path x100...")
    torch.cuda.synchronize()
    for _ in range(100):
        run_unified(x, rot, sids, nv, token_num, topk, RS)
    torch.cuda.synchronize()

    # Also trace M=64 prefill
    M = 64
    token_num = 64
    x, rot, sids, nv = create_moe_test_data(M, K, topk, RS)
    for _ in range(50):
        run_original(x, rot, sids, nv, token_num, topk, RS)
        run_unified(x, rot, sids, nv, token_num, topk, RS)
    torch.cuda.synchronize()

    print("Running original M=64 x100...")
    torch.cuda.synchronize()
    for _ in range(100):
        run_original(x, rot, sids, nv, token_num, topk, RS)
    torch.cuda.synchronize()

    print("Running unified M=64 x100...")
    torch.cuda.synchronize()
    for _ in range(100):
        run_unified(x, rot, sids, nv, token_num, topk, RS)
    torch.cuda.synchronize()

    print("Done. Analyze with: rocprof --hip-trace --output-format csv ...")
    print("Then look for kernel names containing '_unified_moe_' vs '_fused_rot_quant_'")


# ---------------------------------------------------------------------------
# Part 4: Compile time measurement
# ---------------------------------------------------------------------------

def measure_compile_time():
    """Measure compilation time for each kernel variant."""
    print("\n" + "=" * 70)
    print("Part 4: Compile Time Measurement")
    print("=" * 70)

    K = 2048
    RS = 128
    topk = 8

    cases = [
        (1,   1,   "decode M=1"),
        (64,  64,  "prefill M=64"),
        (256, 256, "prefill M=256"),
    ]

    for M, token_num, desc in cases:
        x, rot, sids, nv = create_moe_test_data(M, K, topk, RS)

        # Original path compile
        t0 = time.monotonic()
        run_original(x, rot, sids, nv, token_num, topk, RS)
        torch.cuda.synchronize()
        t_orig = time.monotonic() - t0

        # Unified compile (clear cache by using unique data if needed)
        t0 = time.monotonic()
        run_unified(x, rot, sids, nv, token_num, topk, RS)
        torch.cuda.synchronize()
        t_unified = time.monotonic() - t0

        print(f"  {desc}: original={t_orig:.1f}s  unified={t_unified:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified MoE kernel test+bench")
    parser.add_argument("--test", action="store_true", help="Correctness test only")
    parser.add_argument("--bench", action="store_true", help="Correctness + benchmark")
    parser.add_argument("--rocprof", action="store_true", help="Generate rocprof trace")
    parser.add_argument("--compile", action="store_true", help="Measure compile time")
    parser.add_argument("--all", action="store_true", help="Run everything")
    args = parser.parse_args()

    if not any([args.test, args.bench, args.rocprof, args.compile, args.all]):
        args.all = True

    if args.test or args.all:
        ok = test_correctness()
        if not ok:
            sys.exit(1)

    if args.compile or args.all:
        measure_compile_time()

    if args.bench or args.all:
        bench_kernels()

    if args.rocprof or args.all:
        gen_rocprof_trace()


if __name__ == "__main__":
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
    main()

#!/usr/bin/env python3
"""Standalone benchmark for _tq_decode_stage1 kernel.

This is the benchmark file for GEAK optimization.
The kernel is at: tq_decode_kernel.py::_tq_decode_stage1

Usage:
    python benchmark_tq_decode.py           # default: B=100, seq_len=512
    python benchmark_tq_decode.py --B 200 --seq-len 1024
"""
import argparse
import math
import time
import torch

# ── Setup TQ config without full vllm import ──
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention

DEVICE = "cuda:0"
PRESET = "turboquant_4bit_nc"
D = 128       # head_dim
Hk = 8        # num_kv_heads (Qwen2.5-72B / Qwen3-4B both use 8)
Hq = 64       # num_q_heads — aligned to Qwen2.5-72B for consistent benchmarking
              # (was 32 for Qwen3-4B; changed to match all other benchmark scripts)
BS = 16       # block_size (pages)


def setup():
    cfg = TurboQuantConfig.from_cache_dtype(PRESET, D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    Pi = PiT.T.contiguous()
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    return cfg, Pi, PiT, centroids, midpoints


def benchmark(
    B: int,
    seq_len: int,
    warmup: int = 10,
    iters: int = 50,
    max_num_kv_splits: int = 32,
    eager_max_num_kv_splits: int = 32,
    v56_max_seq_len: int = 0,
):
    cfg, Pi, PiT, centroids, midpoints = setup()

    num_blocks = max(8192, (B * seq_len // BS) + 1024)
    kv_cache = torch.zeros(num_blocks, BS, Hk, cfg.slot_size_aligned,
                           dtype=torch.uint8, device=DEVICE)

    # Fill cache with data
    fill_n = min(B * seq_len, num_blocks * BS)
    fk = torch.randn(fill_n, Hk, D, dtype=torch.bfloat16, device=DEVICE)
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

    # Decode inputs
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    bps = math.ceil(seq_len / BS)
    block_table = torch.arange(bps, device=DEVICE, dtype=torch.int32) \
        .unsqueeze(0).expand(B, -1).contiguous()
    seq_lens_t = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)

    def run():
        return triton_turboquant_decode_attention(
            query=q, kv_cache=kv_cache, block_table=block_table,
            seq_lens=seq_lens_t, Pi=Pi, centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
            max_num_kv_splits=max_num_kv_splits,
            eager_max_num_kv_splits=eager_max_num_kv_splits,
            max_seq_len_hint=seq_len,
            allow_adaptive_kv_splits=False,
            v56_max_seq_len=v56_max_seq_len,
        )

    # Warmup
    for _ in range(warmup):
        run()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    avg_us = (time.perf_counter() - t0) / iters * 1e6

    print(
        f"B={B:>4}, seq_len={seq_len:>5}, splits={max_num_kv_splits:>2}: "
        f"{avg_us:.1f} us/call"
    )
    return avg_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark _tq_decode_stage1")
    parser.add_argument("--B", type=int, default=100, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max-num-kv-splits", type=int, default=32)
    parser.add_argument("--eager-max-num-kv-splits", type=int, default=32)
    parser.add_argument("--tq-v56-max-seq-len", type=int, default=0,
                        help="0=disabled; GEMV cost is seq-independent")
    parser.add_argument("--sweep", action="store_true", help="Run sweep over B and seq_len")
    args = parser.parse_args()

    if args.sweep:
        print(f"{'B':>5} {'seq_len':>8} {'us/call':>10}")
        print("-" * 28)
        # Production workload: Qwen2.5-72B, input=8k, output=1k
        # B=4 is the steady-state decode batch for C=4 offline
        # B=20 is the steady-state for C=20 server mode
        for B in [4, 20]:
            for seq_len in [512, 4096, 8192, 9216]:
                us = benchmark(
                    B,
                    seq_len,
                    args.warmup,
                    args.iters,
                    args.max_num_kv_splits,
                    args.eager_max_num_kv_splits,
                    args.tq_v56_max_seq_len,
                )
        # Also test original scenarios
        for B in [32, 100]:
            for seq_len in [128, 512, 1024]:
                us = benchmark(
                    B,
                    seq_len,
                    args.warmup,
                    args.iters,
                    args.max_num_kv_splits,
                    args.eager_max_num_kv_splits,
                    args.tq_v56_max_seq_len,
                )
        print()
        print("PRIMARY TARGET: B=4 seq=8192 < 50 us (old 8-split baseline ~637 us)")
        print("SECONDARY:      B=20 seq=8192 < 200 us")
    else:
        us = benchmark(
            args.B,
            args.seq_len,
            args.warmup,
            args.iters,
            args.max_num_kv_splits,
            args.eager_max_num_kv_splits,
            args.tq_v56_max_seq_len,
        )
        print(f"\nBaseline: {us:.1f} us  |  Target: <50 us  |  Gap: {us/50:.1f}x")


if __name__ == "__main__":
    main()

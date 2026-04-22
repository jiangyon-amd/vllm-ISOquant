#!/usr/bin/env python3
"""Benchmark: HIP stage1 + HIP stage2 vs Triton."""
import ctypes, math, time, os, sys, torch

DEVICE = "cuda:0"
D = 128; Hk = 8; Hq = 32; BS = 16

HIP_DIR = os.path.dirname(os.path.abspath(__file__))

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention


def setup():
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    Pi = PiT.T.contiguous()
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    return cfg, Pi, PiT, centroids, midpoints


def bench(name, fn, warmup=20, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    print(f"  {name}: {us:.1f} us")
    return us


def main():
    B = 100; seq_len = 512; NUM_KV_SPLITS = 8

    cfg, Pi, PiT, centroids, midpoints = setup()
    num_blocks = max(8192, (B * seq_len // BS) + 1024)
    kv_cache = torch.zeros(num_blocks, BS, Hk, cfg.slot_size_aligned,
                           dtype=torch.uint8, device=DEVICE)

    fill_n = min(B * seq_len, num_blocks * BS)
    torch.manual_seed(42)
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
    bt = torch.arange(bps, device=DEVICE, dtype=torch.int32).unsqueeze(0).expand(B,-1).contiguous()
    sls = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)

    print(f"\n=== B={B}, seq_len={seq_len}, NUM_KV_SPLITS={NUM_KV_SPLITS} ===")

    # Triton reference
    triton_out = triton_turboquant_decode_attention(
        query=q, kv_cache=kv_cache, block_table=bt, seq_lens=sls,
        Pi=Pi, centroids=centroids, scale=1.0/math.sqrt(D),
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction, PiT=PiT)

    triton_us = bench("Triton (stage1+stage2)", lambda: triton_turboquant_decode_attention(
        query=q, kv_cache=kv_cache, block_table=bt, seq_lens=sls,
        Pi=Pi, centroids=centroids, scale=1.0/math.sqrt(D),
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction, PiT=PiT))

    # HIP stage1 + stage2
    q_rot = (q.float() @ PiT).contiguous()
    centroids_f32 = centroids.float().contiguous()
    mid_o = torch.zeros(B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=DEVICE)
    output = torch.zeros(B, Hq, D, dtype=torch.float32, device=DEVICE)

    # Load stage1
    lib1 = ctypes.CDLL(os.path.join(HIP_DIR, "tq_decode_stage1_v12.so"))
    launch_stage1 = lib1.launch_tq_decode_stage1
    launch_stage1.argtypes = (
        [ctypes.c_void_p]*6 +
        [ctypes.c_int]*2 +
        [ctypes.c_int]*3 +
        [ctypes.c_int]*1 +
        [ctypes.c_int]*3 +
        [ctypes.c_int]*4 +
        [ctypes.c_float] +
        [ctypes.c_int]*2 +
        [ctypes.c_int]*2 +
        [ctypes.c_void_p]
    )
    launch_stage1.restype = None

    # Load stage2
    lib2 = ctypes.CDLL(os.path.join(HIP_DIR, "tq_decode_stage2.so"))
    launch_stage2 = lib2.launch_tq_decode_stage2
    launch_stage2.argtypes = (
        [ctypes.c_void_p]*3 +  # mid_o, output, seq_lens
        [ctypes.c_int]*3 +     # stride_mb, stride_mh, stride_ms
        [ctypes.c_int]*2 +     # stride_ob, stride_oh
        [ctypes.c_int]*1 +     # num_kv_splits
        [ctypes.c_int]*2 +     # B, Hq
        [ctypes.c_void_p]      # stream
    )
    launch_stage2.restype = None

    def run_both():
        launch_stage1(
            q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
            centroids_f32.data_ptr(), mid_o.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            bt.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            Hk, BS, NUM_KV_SPLITS, Hq // Hk,
            1.0 / math.sqrt(D), 8, 1,
            B, Hq, 0)
        launch_stage2(
            mid_o.data_ptr(), output.data_ptr(), sls.data_ptr(),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            output.stride(0), output.stride(1),
            NUM_KV_SPLITS, B, Hq, 0)

    run_both()
    torch.cuda.synchronize()

    # Check correctness
    triton_out_f32 = triton_out.float()
    max_diff = (output - triton_out_f32).abs().max().item()
    mean_diff = (output - triton_out_f32).abs().mean().item()
    print(f"  Correctness: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    if max_diff < 0.02:
        print(f"  ✓ PASS (max_diff < 0.02)")
    else:
        print(f"  ✗ FAIL (max_diff >= 0.02)")

    # Benchmark stage1 only
    def run_stage1():
        launch_stage1(
            q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
            centroids_f32.data_ptr(), mid_o.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            bt.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            Hk, BS, NUM_KV_SPLITS, Hq // Hk,
            1.0 / math.sqrt(D), 8, 1,
            B, Hq, 0)
    
    stage1_us = bench("HIP stage1 only", lambda: (run_stage1(), torch.cuda.synchronize()))

    # Benchmark stage2 only
    def run_stage2():
        launch_stage2(
            mid_o.data_ptr(), output.data_ptr(), sls.data_ptr(),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            output.stride(0), output.stride(1),
            NUM_KV_SPLITS, B, Hq, 0)
    
    stage2_us = bench("HIP stage2 only", lambda: (run_stage2(), torch.cuda.synchronize()))

    # Benchmark both
    both_us = bench("HIP stage1+stage2", lambda: (run_both(), torch.cuda.synchronize()))
    
    print(f"\n  Summary:")
    print(f"    Triton: {triton_us:.1f} us")
    print(f"    HIP stage1: {stage1_us:.1f} us")
    print(f"    HIP stage2: {stage2_us:.1f} us")
    print(f"    HIP total: {both_us:.1f} us")
    print(f"    Speedup vs Triton: {triton_us/both_us:.2f}x")


if __name__ == "__main__":
    main()

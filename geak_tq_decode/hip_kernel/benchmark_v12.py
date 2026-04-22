#!/usr/bin/env python3
"""Benchmark: HIP v12 kernel only."""
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


def run_hip_kernel(so_path, q_rot, kv_cache, bt, sls, centroids_f32, mid_o,
                   B, Hq_l, NUM_KV_SPLITS, BLOCK_KV=8):
    lib = ctypes.CDLL(so_path)
    launch_fn = lib.launch_tq_decode_stage1
    launch_fn.argtypes = (
        [ctypes.c_void_p]*6 +  # q_rot, kv_cache, block_table, seq_lens, centroids, mid_o
        [ctypes.c_int]*2 +     # stride_qb, stride_qh
        [ctypes.c_int]*3 +     # stride_cb, stride_cp, stride_ch
        [ctypes.c_int]*1 +     # stride_bt
        [ctypes.c_int]*3 +     # stride_mb, stride_mh, stride_ms
        [ctypes.c_int]*4 +     # num_kv_heads, block_size, num_kv_splits, kv_group_size
        [ctypes.c_float] +     # attn_scale
        [ctypes.c_int]*2 +     # block_kv, norm_correction
        [ctypes.c_int]*2 +     # B, Hq
        [ctypes.c_void_p]      # stream
    )
    launch_fn.restype = None

    def run():
        launch_fn(
            q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
            centroids_f32.data_ptr(), mid_o.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            bt.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            Hk, BS, NUM_KV_SPLITS, Hq_l // Hk,
            1.0 / math.sqrt(D), BLOCK_KV, 1,
            B, Hq_l, 0)  # stream=0 (default)
    return run


def main():
    B = 100; seq_len = 512; NUM_KV_SPLITS = 16

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

    triton_us = bench("Triton (opt)", lambda: triton_turboquant_decode_attention(
        query=q, kv_cache=kv_cache, block_table=bt, seq_lens=sls,
        Pi=Pi, centroids=centroids, scale=1.0/math.sqrt(D),
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction, PiT=PiT))

    # Test v12 HIP kernel
    q_rot = (q.float() @ PiT).contiguous()
    centroids_f32 = centroids.float().contiguous()

    so_path = os.path.join(HIP_DIR, "tq_decode_stage1_v12.so")
    mid_o = torch.zeros(B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=DEVICE)
    
    run_fn = run_hip_kernel(so_path, q_rot, kv_cache, bt, sls, centroids_f32, mid_o,
                            B, Hq, NUM_KV_SPLITS)
    run_fn()
    torch.cuda.synchronize()
    hip_us = bench("HIP v12", run_fn)
    print(f"    speedup vs Triton: {triton_us/hip_us:.2f}x")

    print(f"\n  Target: <150 us")


if __name__ == "__main__":
    main()

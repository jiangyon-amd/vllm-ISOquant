#!/usr/bin/env python3
"""Benchmark + correctness test for v55 fused kernel vs v52 + Triton Stage2"""
import os, math, ctypes
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")

import torch
import sys
sys.path.insert(0, "/home/jiangyon/vllm_turboquant")

DEVICE = "cuda:0"
D = 128

def profile(fn, warmup=10, repeat=100, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in evts:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) * 1000 for s, e in evts])
    t = len(times) // 10
    return sum(times[t:-t]) / len(times[t:-t]) if t > 0 else sum(times) / len(times)

def main():
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _load_hip_stage1, _fwd_kernel_stage2, _get_layout
    )

    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    centroids_f32 = centroids.float().contiguous()

    # Load v52 (deployed)
    hip_fn_v52 = _load_hip_stage1()
    assert hip_fn_v52, "v52 HIP kernel not loaded"

    # Load v55 (fused, gfx950)
    so_path = "/home/jiangyon/vllm_turboquant/geak_tq_decode/hip_kernel/tq_decode_v55_fused_parallel.so"
    lib_v55 = ctypes.CDLL(so_path)
    fn_v55 = lib_v55.launch_tq_decode_v55
    fn_v55.argtypes = (
        [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_int] * 3
        + [ctypes.c_int] + [ctypes.c_int] * 2 + [ctypes.c_int] * 3
        + [ctypes.c_float] + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    )
    fn_v55.restype = None
    print("v55 fused kernel loaded (gfx950)")

    layout = _get_layout(D, cfg.key_mse_bits, cfg.effective_value_quant_bits, cfg.key_packed_size)
    BS = 16
    NUM_KV_SPLITS = 8
    scale = 1.0 / math.sqrt(D)
    Hq, Hk = 64, 8

    print(f"\n{'='*90}")
    print(f"  {'Config':<18s} | {'Ref(S1+S2+bf16)':>16s} | {'V55 fused':>12s} | {'Speedup':>8s} | {'cos_sim':>10s} | {'Status'}")
    print(f"{'='*90}")

    for B, seq_len in [(1, 512), (1, 1024), (1, 2048), (1, 4096), (4, 4096), (8, 512), (32, 512)]:
        num_blocks = (seq_len // BS) + 16
        kv_cache = torch.zeros(num_blocks, BS, Hk, cfg.slot_size_aligned,
                               dtype=torch.uint8, device=DEVICE)
        fk = torch.randn(seq_len, Hk, D, dtype=torch.bfloat16, device=DEVICE)
        fv = torch.randn_like(fk)
        midpoints = (centroids.float().sort()[0][:-1] + centroids.float().sort()[0][1:]) / 2
        triton_turboquant_store(fk, fv, kv_cache,
            torch.arange(seq_len, device=DEVICE, dtype=torch.int64),
            PiT, centroids, midpoints,
            mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8)

        q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
        bps = math.ceil(seq_len / BS)
        bt = torch.arange(bps, device=DEVICE, dtype=torch.int32).unsqueeze(0).expand(B,-1).contiguous()
        sls = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)
        q_rot = torch.mm(q.reshape(B*Hq, D).float(), PiT).reshape(B, Hq, D).contiguous()

        # === Reference: v52 Stage1 + Triton Stage2 + bf16 cast ===
        mid_o = torch.empty(B, Hq, NUM_KV_SPLITS, D+1, dtype=torch.float32, device=DEVICE)
        output_ref_f32 = torch.empty(B, Hq, D, dtype=torch.float32, device=DEVICE)
        lse_ref = torch.empty(B, Hq, dtype=torch.float32, device=DEVICE)
        stream_ptr = torch.cuda.current_stream(DEVICE).cuda_stream

        def run_ref():
            hip_fn_v52(
                q_rot.data_ptr(), kv_cache.data_ptr(),
                bt.data_ptr(), sls.data_ptr(),
                centroids_f32.data_ptr(), mid_o.data_ptr(),
                q_rot.stride(0), q_rot.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                bt.stride(0),
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                Hk, BS, NUM_KV_SPLITS, Hq // Hk,
                scale, 8, 1, B, Hq,
                ctypes.c_void_p(stream_ptr),
            )
            grid2 = (B, Hq)
            _fwd_kernel_stage2[grid2](
                mid_o, output_ref_f32, lse_ref, sls,
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                output_ref_f32.stride(0), output_ref_f32.stride(1), lse_ref.stride(0),
                NUM_KV_SPLITS=NUM_KV_SPLITS, BLOCK_DV=layout["BLOCK_D"], Lv=D,
                num_warps=4, num_stages=2,
            )
        
        # Also a version with bf16 cast included
        output_ref_bf16 = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
        def run_ref_full():
            run_ref()
            output_ref_bf16.copy_(output_ref_f32)

        run_ref()
        torch.cuda.synchronize()
        ref_result = output_ref_f32.clone()

        # === v55: Fused kernel ===
        output_v55 = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
        
        def run_v55():
            fn_v55(
                q_rot.data_ptr(), kv_cache.data_ptr(),
                bt.data_ptr(), sls.data_ptr(),
                centroids_f32.data_ptr(), output_v55.data_ptr(),
                q_rot.stride(0), q_rot.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                bt.stride(0),
                output_v55.stride(0), output_v55.stride(1),
                Hk, BS, Hq // Hk,
                scale,
                B, Hq,
                ctypes.c_void_p(stream_ptr),
            )

        run_v55()
        torch.cuda.synchronize()
        v55_result = output_v55.float()

        # Correctness
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_result.reshape(-1).unsqueeze(0),
            v55_result.reshape(-1).unsqueeze(0)
        ).item()
        max_diff = (ref_result - v55_result).abs().max().item()
        has_nan = torch.isnan(v55_result).any().item()

        # Benchmark
        t_ref = profile(run_ref_full)
        t_v55 = profile(run_v55)

        status = "NaN!" if has_nan else ("OK" if cos_sim > 0.999 else f"DIFF")
        print(f"  B={B:2d} seq={seq_len:5d} | {t_ref:>12.1f} us  | {t_v55:>8.1f} us  | {t_ref/t_v55:>6.2f}x  | {cos_sim:>10.6f} | {status}")

    print(f"{'='*90}")

main()

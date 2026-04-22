#!/usr/bin/env python3
"""Benchmark adaptive approach: v56 for small B, v52 for large B"""
import os, math, ctypes
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")

import torch, sys
sys.path.insert(0, "/home/jiangyon/vllm_turboquant")

DEVICE = "cuda:0"
D = 128

def profile(fn, warmup=20, repeat=200):
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
    hip_fn = _load_hip_stage1()
    layout = _get_layout(D, cfg.key_mse_bits, cfg.effective_value_quant_bits, cfg.key_packed_size)

    lib_v56 = ctypes.CDLL("/home/jiangyon/vllm_turboquant/geak_tq_decode/hip_kernel/tq_decode_v56_gemv_fused.so")
    fn_v56 = lib_v56.launch_tq_decode_v56
    fn_v56.argtypes = (
        [ctypes.c_void_p] * 7 + [ctypes.c_int] * 2 + [ctypes.c_int] * 3
        + [ctypes.c_int] + [ctypes.c_int] * 3 + [ctypes.c_int] * 4
        + [ctypes.c_float] + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    fn_v56.restype = None

    lib_s2 = ctypes.CDLL("/home/jiangyon/vllm_turboquant/geak_tq_decode/hip_kernel/tq_decode_stage2_v2.so")
    fn_s2_bf16 = lib_s2.launch_tq_decode_stage2_bf16
    fn_s2_bf16.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int] * 5 + [ctypes.c_int] + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    )
    fn_s2_bf16.restype = None

    BS = 16; NUM_KV_SPLITS = 8; scale = 1.0 / math.sqrt(D); Hq, Hk = 64, 8
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    B_THRESHOLD = 4  # Use v56 for B <= threshold, v52 for B > threshold

    print(f"\n{'='*110}")
    print(f"  {'Config':<16s} | {'OLD':>10s} | {'ADAPTIVE':>10s} | {'Saved':>8s} | {'SDPA':>8s} | {'OLD/SDPA':>9s} | {'NEW/SDPA':>9s} | cos")
    print(f"{'='*110}")

    for B, seq_len in [(1, 512), (1, 1024), (1, 2048), (1, 4096), (2, 2048), (4, 4096), (8, 512), (16, 512), (32, 512), (64, 512)]:
        num_blocks = (seq_len // BS) + B + 16
        kv_cache = torch.zeros(num_blocks, BS, Hk, cfg.slot_size_aligned, dtype=torch.uint8, device=DEVICE)
        fk = torch.randn(seq_len, Hk, D, dtype=torch.bfloat16, device=DEVICE)
        fv = torch.randn_like(fk)
        triton_turboquant_store(fk, fv, kv_cache,
            torch.arange(seq_len, device=DEVICE, dtype=torch.int64),
            PiT, centroids, midpoints,
            mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8)

        q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
        bps = math.ceil(seq_len / BS)
        bt = torch.arange(bps, device=DEVICE, dtype=torch.int32).unsqueeze(0).expand(B,-1).contiguous()
        sls = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)

        mid_o = torch.empty(B, Hq, NUM_KV_SPLITS, D+1, dtype=torch.float32, device=DEVICE)
        output_f32 = torch.empty(B, Hq, D, dtype=torch.float32, device=DEVICE)
        output_bf16 = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
        lse_buf = torch.empty(B, Hq, dtype=torch.float32, device=DEVICE)
        q_rot_buf = torch.empty(max(B,64), Hq, D, dtype=torch.float32, device=DEVICE)
        stream_ptr = torch.cuda.current_stream(DEVICE).cuda_stream

        def run_old():
            q_flat = q.reshape(B*Hq, D).float()
            torch.mm(q_flat, PiT, out=q_rot_buf[:B].reshape(B*Hq, D))
            hip_fn(
                q_rot_buf.data_ptr(), kv_cache.data_ptr(),
                bt.data_ptr(), sls.data_ptr(),
                centroids_f32.data_ptr(), mid_o.data_ptr(),
                q_rot_buf.stride(0), q_rot_buf.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                bt.stride(0),
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                Hk, BS, NUM_KV_SPLITS, Hq // Hk,
                scale, 8, 1, B, Hq,
                ctypes.c_void_p(stream_ptr),
            )
            grid2 = (B, Hq)
            _fwd_kernel_stage2[grid2](
                mid_o, output_f32, lse_buf, sls,
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                output_f32.stride(0), output_f32.stride(1), lse_buf.stride(0),
                NUM_KV_SPLITS=NUM_KV_SPLITS, BLOCK_DV=layout["BLOCK_D"], Lv=D,
                num_warps=4, num_stages=2,
            )
            output_bf16.copy_(output_f32)

        def run_adaptive():
            if B <= B_THRESHOLD:
                # v56: GEMV fused, skip separate GEMM
                fn_v56(
                    q.data_ptr(), PiT.data_ptr(),
                    kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
                    centroids_f32.data_ptr(), mid_o.data_ptr(),
                    q.stride(0), q.stride(1),
                    kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                    bt.stride(0),
                    mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                    Hk, BS, NUM_KV_SPLITS, Hq // Hk,
                    scale, 1, B, Hq,
                    ctypes.c_void_p(stream_ptr),
                )
                fn_s2_bf16(
                    mid_o.data_ptr(), output_bf16.data_ptr(), sls.data_ptr(),
                    mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                    output_bf16.stride(0), output_bf16.stride(1),
                    NUM_KV_SPLITS, B, Hq,
                    ctypes.c_void_p(stream_ptr),
                )
            else:
                # v52: separate GEMM (more efficient at high B)
                run_old()

        # Correctness
        run_old(); torch.cuda.synchronize(); ref = output_bf16.float().clone()
        run_adaptive(); torch.cuda.synchronize(); new = output_bf16.float()
        cos = torch.nn.functional.cosine_similarity(ref.reshape(-1).unsqueeze(0), new.reshape(-1).unsqueeze(0)).item()

        # SDPA
        import torch.nn.functional as F
        gqa = Hq // Hk
        key_bl = torch.randn(B, Hk, seq_len, D, dtype=torch.bfloat16, device=DEVICE)
        val_bl = torch.randn(B, Hk, seq_len, D, dtype=torch.bfloat16, device=DEVICE)
        q_bl = torch.randn(B, Hq, 1, D, dtype=torch.bfloat16, device=DEVICE)
        key_exp = key_bl.unsqueeze(2).expand(B, Hk, gqa, seq_len, D).reshape(B, Hq, seq_len, D)
        val_exp = val_bl.unsqueeze(2).expand(B, Hk, gqa, seq_len, D).reshape(B, Hq, seq_len, D)
        t_sdpa = profile(lambda: F.scaled_dot_product_attention(q_bl, key_exp, val_exp, scale=scale))
        del key_bl, val_bl, q_bl, key_exp, val_exp

        t_old = profile(run_old)
        t_new = profile(run_adaptive)

        saved = t_old - t_new
        print(f"  B={B:2d} seq={seq_len:5d} | {t_old:>7.1f}us  | {t_new:>7.1f}us  | {saved:>+6.1f}us | {t_sdpa:>5.1f}us | {t_old/t_sdpa:>7.2f}x  | {t_new/t_sdpa:>7.2f}x  | {cos:.4f}")

    print(f"{'='*110}")

main()

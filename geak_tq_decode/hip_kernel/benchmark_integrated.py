#!/usr/bin/env python3
"""Benchmark the integrated optimized code vs SDPA baseline"""
import os, math
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
        triton_turboquant_decode_attention, _load_hip_stage1, _load_hip_v56, _load_hip_stage2
    )

    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    Pi = PiT.T.contiguous()
    centroids_f32 = centroids.float().contiguous()
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    # Check which kernels are available
    hip_fn = _load_hip_stage1()
    hip_v56 = _load_hip_v56()
    hip_s2_bf16, hip_s2_f32 = _load_hip_stage2()
    print(f"  HIP Stage1 (v52): {'OK' if hip_fn else 'NOT FOUND'}")
    print(f"  HIP V56 (GEMV-fused): {'OK' if hip_v56 else 'NOT FOUND'}")
    print(f"  HIP Stage2 (bf16/f32): {'OK' if hip_s2_bf16 else 'NOT FOUND'}")

    BS = 16; Hq, Hk = 64, 8; scale = 1.0 / math.sqrt(D)

    print(f"\n{'='*80}")
    print(f"  {'Config':<16s} | {'TQ Decode':>10s} | {'SDPA':>8s} | {'TQ/SDPA':>8s} | {'80L (ms)':>9s}")
    print(f"{'='*80}")

    for B, seq_len in [(1, 512), (1, 1024), (1, 2048), (1, 4096), (2, 2048),
                       (4, 4096), (8, 512), (16, 512), (32, 512), (64, 512)]:
        num_blocks = (seq_len // BS) + B + 16
        kv_cache = torch.zeros(num_blocks, BS, Hk, cfg.slot_size_aligned,
                               dtype=torch.uint8, device=DEVICE)
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

        max_B = max(B, 64)
        q_rot_buf = torch.empty(max_B, Hq, D, dtype=torch.float32, device=DEVICE)
        mid_o_buf = torch.empty(max_B, Hq, 8, D+1, dtype=torch.float32, device=DEVICE)
        output_buf = torch.empty(max_B, Hq, D, dtype=torch.float32, device=DEVICE)
        lse_buf = torch.empty(max_B, Hq, dtype=torch.float32, device=DEVICE)

        class BufHolder:
            pass
        bh = BufHolder()

        def run_tq():
            return triton_turboquant_decode_attention(
                q, kv_cache, bt, sls,
                Pi, centroids, scale,
                mse_bits=cfg.key_mse_bits,
                key_packed_size=cfg.key_packed_size,
                value_quant_bits=cfg.effective_value_quant_bits,
                key_fp8=cfg.key_fp8,
                norm_correction=True,
                PiT=PiT,
                mid_o_buf=mid_o_buf, output_buf=output_buf,
                lse_buf=lse_buf, q_rot_buf=q_rot_buf,
                centroids_f32=centroids_f32,
                buf_holder=bh,
                max_num_kv_splits=8,
            )

        # SDPA baseline
        import torch.nn.functional as F
        gqa = Hq // Hk
        key_bl = torch.randn(B, Hk, seq_len, D, dtype=torch.bfloat16, device=DEVICE)
        val_bl = torch.randn(B, Hk, seq_len, D, dtype=torch.bfloat16, device=DEVICE)
        q_bl = torch.randn(B, Hq, 1, D, dtype=torch.bfloat16, device=DEVICE)
        key_exp = key_bl.unsqueeze(2).expand(B, Hk, gqa, seq_len, D).reshape(B, Hq, seq_len, D)
        val_exp = val_bl.unsqueeze(2).expand(B, Hk, gqa, seq_len, D).reshape(B, Hq, seq_len, D)
        t_sdpa = profile(lambda: F.scaled_dot_product_attention(q_bl, key_exp, val_exp, scale=scale))
        del key_bl, val_bl, q_bl, key_exp, val_exp

        t_tq = profile(run_tq)
        ratio = t_tq / t_sdpa
        tag = "v56" if B <= 4 and hip_v56 else "v52"
        ms_80 = t_tq * 80 / 1000  # 80 layers in us -> ms

        print(f"  B={B:2d} seq={seq_len:5d} | {t_tq:>7.1f}us  | {t_sdpa:>5.1f}us | {ratio:>6.2f}x  | {ms_80:>7.2f}ms  [{tag}]")

    print(f"{'='*80}")

main()

#!/usr/bin/env python3
"""
Benchmark v61c: v52 Stage1 + fused-last-split Stage2 → bf16
vs reference: GEMM + v52 Stage1 + HIP Stage2_bf16 (3 kernels)

v61c eliminates Stage2 kernel launch by having the last split
do the reduce in-place.
"""
import os, sys, math, ctypes
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")
os.environ.setdefault("TQ_ALLOW_STALE_HIP_SO", "1")
_PR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _PR not in sys.path: sys.path.insert(0, _PR)

import torch
DEVICE = "cuda:0"
D = 128

def profile_gpu(fn, warmup=20, repeat=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    evts = [(torch.cuda.Event(enable_timing=True),
             torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in evts: s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e)*1000 for s, e in evts])
    t = max(1, len(times)//10)
    return sum(times[t:-t]) / len(times[t:-t])

def main():
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
    from vllm.v1.attention.ops.triton_turboquant_decode import _load_hip_stage1, _load_hip_stage2

    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    centroids_f32 = centroids.float().contiguous()
    midpoints = (centroids.float().sort()[0][:-1] + centroids.float().sort()[0][1:]) / 2

    fn_s1 = _load_hip_stage1()
    fn_s2_bf16, _ = _load_hip_stage2()
    assert fn_s1 and fn_s2_bf16

    # Load v61c
    so = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tq_decode_v61c.so")
    lib = ctypes.CDLL(so)
    fn_v61c = lib.launch_tq_decode_v61c
    fn_v61c.argtypes = (
        [ctypes.c_void_p] * 8     # q_rot, kv, bt, sl, cent, mid_o, output, counter
        + [ctypes.c_int] * 2      # stride_qb, stride_qh
        + [ctypes.c_int] * 3      # stride_cb, stride_cp, stride_ch
        + [ctypes.c_int]          # stride_bt
        + [ctypes.c_int] * 3      # stride_mb, stride_mh, stride_ms
        + [ctypes.c_int] * 2      # stride_ob, stride_oh
        + [ctypes.c_int] * 4      # num_kv_heads, block_size, num_kv_splits, kv_group_size
        + [ctypes.c_float]        # attn_scale
        + [ctypes.c_int]          # norm_correction
        + [ctypes.c_int] * 2      # B, Hq
        + [ctypes.c_void_p]       # stream
    )
    fn_v61c.restype = None
    print("v61c loaded")

    Hq, Hk = 64, 8
    BS = 16
    scale = 1.0 / math.sqrt(D)
    kv_group_size = Hq // Hk
    SPLITS = 32

    configs = [
        (1,   512), (1,  1024), (1,  2048), (1,  4096), (1,  8192),
        (4,  2048), (4,  4096), (4,  8192),
        (8,  4096), (16, 2048), (32,  512),
    ]

    print(f"\n{'='*110}")
    print(f"  {'Config':<18s} | {'Ref GEMM+S1+S2':>14s} | {'GEMM+v61c':>12s} | {'S1 only':>10s} | {'S2 only':>10s} | {'spdup':>7s} | {'cos':>8s} | {'st':>4s}")
    print(f"{'='*110}")

    for B, seq_len in configs:
        num_blocks = (seq_len // BS) + 16
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
        stream_ptr = torch.cuda.current_stream(DEVICE).cuda_stream

        q_rot = torch.empty(B, Hq, D, dtype=torch.float32, device=DEVICE)
        mid_o = torch.empty(B, Hq, SPLITS, D+1, dtype=torch.float32, device=DEVICE)
        ref_out = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)

        # Reference: GEMM + v52 S1 + S2 bf16
        def run_gemm():
            torch.mm(q.reshape(B*Hq,D).float(), PiT, out=q_rot.reshape(B*Hq,D))

        def run_s1():
            fn_s1(q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
                  centroids_f32.data_ptr(), mid_o.data_ptr(),
                  q_rot.stride(0), q_rot.stride(1),
                  kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                  bt.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                  Hk, BS, SPLITS, kv_group_size, scale, 1, B, Hq,
                  ctypes.c_void_p(stream_ptr))

        def run_s2():
            fn_s2_bf16(mid_o.data_ptr(), ref_out.data_ptr(), sls.data_ptr(),
                       mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                       ref_out.stride(0), ref_out.stride(1),
                       SPLITS, B, Hq, ctypes.c_void_p(stream_ptr))

        def run_ref():
            run_gemm(); run_s1(); run_s2()

        run_ref(); torch.cuda.synchronize()
        ref_result = ref_out.float().clone()

        # v61c: GEMM + v61c (S1+S2 fused)
        mid_o_c = torch.empty(B, Hq, SPLITS, D+1, dtype=torch.float32, device=DEVICE)
        out_c = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
        counter = torch.zeros(B * Hq, dtype=torch.int32, device=DEVICE)

        def run_v61c():
            counter.zero_()
            fn_v61c(
                q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
                centroids_f32.data_ptr(), mid_o_c.data_ptr(), out_c.data_ptr(), counter.data_ptr(),
                q_rot.stride(0), q_rot.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                bt.stride(0),
                mid_o_c.stride(0), mid_o_c.stride(1), mid_o_c.stride(2),
                out_c.stride(0), out_c.stride(1),
                Hk, BS, SPLITS, kv_group_size, scale, 1, B, Hq,
                ctypes.c_void_p(stream_ptr))

        def run_full_v61c():
            run_gemm(); run_v61c()

        # Correctness
        run_gemm(); run_v61c(); torch.cuda.synchronize()
        c_result = out_c.float()
        cos = torch.nn.functional.cosine_similarity(
            ref_result.reshape(-1).unsqueeze(0),
            c_result.reshape(-1).unsqueeze(0)).item()
        has_nan = torch.isnan(c_result).any().item()
        status = "NaN" if has_nan else ("OK" if cos > 0.999 else "BAD")

        # Benchmark individual components
        t_s1_only = profile_gpu(run_s1)
        t_s2_only = profile_gpu(run_s2)
        t_ref = profile_gpu(run_ref)
        t_v61c = profile_gpu(run_full_v61c)
        spd = t_ref / t_v61c if t_v61c > 0 else 0

        print(f"  B={B:2d} seq={seq_len:5d} | {t_ref:>10.1f} us  | {t_v61c:>8.1f} us  | {t_s1_only:>7.1f} us | {t_s2_only:>7.1f} us | {spd:>5.2f}x  | {cos:>8.5f} | {status:>4s}")

    print(f"{'='*110}")

if __name__ == "__main__":
    main()

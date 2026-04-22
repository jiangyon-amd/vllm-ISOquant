#!/usr/bin/env python3
"""
Benchmark v61b: GEMV-fused Stage1 + atomic-last-split Stage2 → bf16 output
vs reference: GEMM + v52 Stage1 + HIP Stage2_bf16 (3 separate kernels)
"""
import os, sys, math, ctypes, time
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")
os.environ.setdefault("TQ_ALLOW_STALE_HIP_SO", "1")
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

DEVICE = "cuda:0"
D = 128

def profile_gpu(fn, warmup=20, repeat=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    evts = [(torch.cuda.Event(enable_timing=True),
             torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in evts:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) * 1000 for s, e in evts])
    t = max(1, len(times) // 10)
    trimmed = times[t:-t] if t > 0 else times
    return sum(trimmed) / len(trimmed)


def main():
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _load_hip_stage1, _load_hip_stage2
    )

    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    centroids_f32 = centroids.float().contiguous()
    midpoints = (centroids.float().sort()[0][:-1] + centroids.float().sort()[0][1:]) / 2

    fn_s1, fn_s2_bf16, _ = _load_hip_stage1(), *_load_hip_stage2()
    assert fn_s1 and fn_s2_bf16, "Reference kernels not loaded"

    # Load v61b
    so = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tq_decode_v61b.so")
    lib = ctypes.CDLL(so)
    fn_v61b = lib.launch_tq_decode_v61b
    fn_v61b.argtypes = (
        [ctypes.c_void_p] * 9     # query, PiT, kv, bt, sl, cent, mid_o, output, counter
        + [ctypes.c_int] * 2      # stride_qb, stride_qh
        + [ctypes.c_int] * 3      # stride_cb, stride_cp, stride_ch
        + [ctypes.c_int]          # stride_bt
        + [ctypes.c_int] * 3      # stride_mb, stride_mh, stride_ms
        + [ctypes.c_int] * 2      # stride_ob, stride_oh
        + [ctypes.c_int] * 4      # num_kv_heads, block_size, num_kv_splits, kv_group_size
        + [ctypes.c_float]        # attn_scale
        + [ctypes.c_int] * 2      # query_dtype, norm_correction
        + [ctypes.c_int] * 2      # B, Hq
        + [ctypes.c_void_p]       # stream
    )
    fn_v61b.restype = None
    print("v61b loaded")

    Hq, Hk = 64, 8
    BS = 16
    scale = 1.0 / math.sqrt(D)
    kv_group_size = Hq // Hk
    REF_SPLITS = 32

    configs = [
        (1,   512),
        (1,  1024),
        (1,  2048),
        (1,  4096),
        (1,  8192),
        (4,  2048),
        (4,  4096),
        (4,  8192),
        (8,  4096),
        (16, 2048),
        (32,  512),
    ]

    print(f"\n{'='*100}")
    print(f"  {'Config':<18s} | {'Ref(3 kern)':>12s} | {'v61b(1 kern)':>12s} | {'speedup':>8s} | {'cos_sim':>8s} | {'status':>6s}")
    print(f"{'='*100}")

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

        # Reference
        q_rot = torch.empty(B, Hq, D, dtype=torch.float32, device=DEVICE)
        mid_o = torch.empty(B, Hq, REF_SPLITS, D+1, dtype=torch.float32, device=DEVICE)
        ref_out = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)

        def run_ref():
            torch.mm(q.reshape(B*Hq,D).float(), PiT, out=q_rot.reshape(B*Hq,D))
            fn_s1(q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
                  centroids_f32.data_ptr(), mid_o.data_ptr(),
                  q_rot.stride(0), q_rot.stride(1),
                  kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                  bt.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                  Hk, BS, REF_SPLITS, kv_group_size, scale, 1, B, Hq,
                  ctypes.c_void_p(stream_ptr))
            fn_s2_bf16(mid_o.data_ptr(), ref_out.data_ptr(), sls.data_ptr(),
                       mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                       ref_out.stride(0), ref_out.stride(1),
                       REF_SPLITS, B, Hq, ctypes.c_void_p(stream_ptr))

        run_ref(); torch.cuda.synchronize()
        ref_result = ref_out.float().clone()

        # v61b
        mid_o_v61 = torch.empty(B, Hq, REF_SPLITS, D+1, dtype=torch.float32, device=DEVICE)
        out_v61 = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
        counter = torch.zeros(B * Hq, dtype=torch.int32, device=DEVICE)

        def run_v61b():
            counter.zero_()
            fn_v61b(
                q.data_ptr(), PiT.data_ptr(), kv_cache.data_ptr(),
                bt.data_ptr(), sls.data_ptr(), centroids_f32.data_ptr(),
                mid_o_v61.data_ptr(), out_v61.data_ptr(), counter.data_ptr(),
                q.stride(0), q.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                bt.stride(0),
                mid_o_v61.stride(0), mid_o_v61.stride(1), mid_o_v61.stride(2),
                out_v61.stride(0), out_v61.stride(1),
                Hk, BS, REF_SPLITS, kv_group_size,
                scale, 0, 1, B, Hq,
                ctypes.c_void_p(stream_ptr))

        run_v61b(); torch.cuda.synchronize()
        v61_result = out_v61.float()

        cos = torch.nn.functional.cosine_similarity(
            ref_result.reshape(-1).unsqueeze(0),
            v61_result.reshape(-1).unsqueeze(0)).item()
        has_nan = torch.isnan(v61_result).any().item()
        status = "NaN!" if has_nan else ("OK" if cos > 0.999 else "DIFF")

        t_ref = profile_gpu(run_ref)
        t_v61 = profile_gpu(run_v61b)
        spd = t_ref / t_v61 if t_v61 > 0 else 0

        print(f"  B={B:2d} seq={seq_len:5d} | {t_ref:>8.1f} us   | {t_v61:>8.1f} us    | {spd:>6.2f}x  | {cos:>8.5f} | {status:>6s}")

    print(f"{'='*100}")

if __name__ == "__main__":
    main()

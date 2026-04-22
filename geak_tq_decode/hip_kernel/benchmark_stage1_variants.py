#!/usr/bin/env python3
"""Compare stage1 kernel variants at the GPU kernel level."""
import os, sys, math, ctypes
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")
os.environ.setdefault("TQ_ALLOW_STALE_HIP_SO", "1")
_PR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _PR not in sys.path: sys.path.insert(0, _PR)

import torch
DEVICE = "cuda:0"
D = 128

def profile_gpu(fn, warmup=50, repeat=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    evts = [(torch.cuda.Event(enable_timing=True),
             torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in evts: s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e)*1000 for s, e in evts])
    t = max(1, len(times)//10)
    med = times[len(times)//2]
    trimmed = times[t:-t]
    return sum(trimmed)/len(trimmed), med, times[0], times[-1]

def main():
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
    from vllm.v1.attention.ops.triton_turboquant_decode import _load_hip_stage1

    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    centroids_f32 = centroids.float().contiguous()
    midpoints = (centroids.float().sort()[0][:-1] + centroids.float().sort()[0][1:]) / 2

    fn_v52 = _load_hip_stage1()
    assert fn_v52

    Hq, Hk = 64, 8
    BS = 16
    scale = 1.0 / math.sqrt(D)
    kv_group_size = Hq // Hk

    # Target scenario
    configs = [
        # (B, seq_len, splits)
        (1,  8192, 32),
        (4,  8192, 32),
        (4,  8192, 16),
        (4,  8192, 8),
        (4,  4096, 32),
        (4,  4096, 16),
        (4,  2048, 16),
        (4,  2048, 8),
        (8,  4096, 32),
        (16, 2048, 16),
    ]

    print(f"\n{'='*90}")
    print(f"  {'Config':<28s} | {'mean':>8s} | {'med':>8s} | {'min':>8s} | {'max':>8s} | {'tok/split':>9s}")
    print(f"{'='*90}")

    for B, seq_len, SPLITS in configs:
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
        q_rot = torch.mm(q.reshape(B*Hq,D).float(), PiT).reshape(B,Hq,D).contiguous()
        mid_o = torch.empty(B, Hq, SPLITS, D+1, dtype=torch.float32, device=DEVICE)
        stream_ptr = torch.cuda.current_stream(DEVICE).cuda_stream

        def run():
            fn_v52(q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
                   centroids_f32.data_ptr(), mid_o.data_ptr(),
                   q_rot.stride(0), q_rot.stride(1),
                   kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                   bt.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                   Hk, BS, SPLITS, kv_group_size, scale, 1, B, Hq,
                   ctypes.c_void_p(stream_ptr))

        mean, med, mn, mx = profile_gpu(run)
        tps = math.ceil(seq_len / SPLITS)

        print(f"  B={B:2d} seq={seq_len:5d} sp={SPLITS:2d} | {mean:>6.1f} us | {med:>6.1f} us | {mn:>6.1f} us | {mx:>6.1f} us | {tps:>7d}")

    print(f"{'='*90}")
    print("\nNote: these are STAGE1 ONLY times (no GEMM, no Stage2)")

if __name__ == "__main__":
    main()

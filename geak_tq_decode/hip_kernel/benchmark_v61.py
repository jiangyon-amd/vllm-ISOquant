#!/usr/bin/env python3
"""
Benchmark + correctness test for v61 (fully-fused GEMV+Stage1+Stage2+bf16)
vs reference pipeline (GEMM + v52 Stage1 + HIP Stage2_bf16).

Usage:
  # Compile first (if not already done)
  hipcc -O3 -shared -fPIC --offload-arch=gfx950 \
    -DSPLITS_PER_BLOCK=8 \
    tq_decode_v61_fused_all.hip -o tq_decode_v61_s8.so

  # Run
  python benchmark_v61.py
"""
import os, sys, math, ctypes, time
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")
os.environ.setdefault("TQ_ALLOW_STALE_HIP_SO", "1")

# Ensure vllm project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

DEVICE = "cuda:0"
D = 128

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def profile_gpu(fn, warmup=20, repeat=100, label=""):
    """Return trimmed-mean GPU time in microseconds (via CUDA events)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evts = [(torch.cuda.Event(enable_timing=True),
             torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in evts:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) * 1000 for s, e in evts])  # us
    t = len(times) // 10
    trimmed = times[t:-t] if t > 0 else times
    return sum(trimmed) / len(trimmed)


# ──────────────────────────────────────────────────────────────────
# Compile v61 kernels
# ──────────────────────────────────────────────────────────────────
HIP_DIR = os.path.dirname(os.path.abspath(__file__))

def compile_v61(splits_per_block=8):
    """Compile v61 with given SPLITS_PER_BLOCK, return .so path."""
    src = os.path.join(HIP_DIR, "tq_decode_v61_fused_all.hip")
    so  = os.path.join(HIP_DIR, f"tq_decode_v61_s{splits_per_block}.so")
    if os.path.exists(so) and os.path.getmtime(so) >= os.path.getmtime(src):
        print(f"  [v61 s={splits_per_block}] using cached {so}")
        return so
    cmd = (
        f"hipcc -O3 -shared -fPIC --offload-arch=gfx950 "
        f"-DSPLITS_PER_BLOCK={splits_per_block} "
        f"{src} -o {so}"
    )
    print(f"  [v61 s={splits_per_block}] compiling ...")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"hipcc failed (exit {ret})")
    print(f"  [v61 s={splits_per_block}] → {so}")
    return so


def load_v61(so_path):
    """Load v61 .so, return ctypes function."""
    lib = ctypes.CDLL(so_path)
    fn = lib.launch_tq_decode_v61
    fn.argtypes = (
        [ctypes.c_void_p] * 7     # query, PiT, kv_cache, bt, seq_lens, centroids, output
        + [ctypes.c_int] * 2      # stride_qb, stride_qh
        + [ctypes.c_int] * 3      # stride_cb, stride_cp, stride_ch
        + [ctypes.c_int]          # stride_bt
        + [ctypes.c_int] * 2      # stride_ob, stride_oh
        + [ctypes.c_int] * 3      # num_kv_heads, block_size, kv_group_size
        + [ctypes.c_float]        # attn_scale
        + [ctypes.c_int] * 2      # query_dtype, norm_correction
        + [ctypes.c_int] * 2      # B, Hq
        + [ctypes.c_void_p]       # stream
    )
    fn.restype = None
    return fn


# ──────────────────────────────────────────────────────────────────
# Reference: GEMM + v52 Stage1 + HIP Stage2_bf16
# ──────────────────────────────────────────────────────────────────
def load_reference():
    """Load v52 Stage1 and HIP Stage2_bf16 kernels."""
    sys.path.insert(0, os.path.join(HIP_DIR, "..", "..", ".."))
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _load_hip_stage1, _load_hip_stage2
    )
    fn_s1 = _load_hip_stage1()
    fn_s2_bf16, fn_s2_f32 = _load_hip_stage2()
    return fn_s1, fn_s2_bf16, fn_s2_f32


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    centroids_f32 = centroids.float().contiguous()
    midpoints = (centroids.float().sort()[0][:-1] + centroids.float().sort()[0][1:]) / 2

    # Load reference kernels
    fn_s1, fn_s2_bf16, fn_s2_f32 = load_reference()
    assert fn_s1 is not None, "v52 HIP Stage1 not loaded"
    assert fn_s2_bf16 is not None, "HIP Stage2 bf16 not loaded"

    # Compile & load v61 variants
    print("\n=== Compiling v61 kernels ===")
    v61_variants = {}
    for sp in [8, 16]:
        try:
            so = compile_v61(sp)
            v61_variants[sp] = load_v61(so)
            print(f"  ✓ v61_s{sp} loaded")
        except Exception as e:
            print(f"  ✗ v61_s{sp} failed: {e}")

    if not v61_variants:
        print("ERROR: no v61 variant compiled successfully")
        return

    # Test configurations
    Hq, Hk = 64, 8
    BS = 16  # kv_cache block_size
    scale = 1.0 / math.sqrt(D)
    kv_group_size = Hq // Hk

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

    hdr = (f"  {'Config':<18s}"
           f" | {'Ref(GEMM+S1+S2)':>16s}")
    for sp in sorted(v61_variants):
        hdr += f" | {'v61_s'+str(sp):>10s}"
    hdr += f" | {'best spdup':>10s} | {'cos_sim':>8s} | {'status':>6s}"

    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'='*len(hdr)}")

    for B, seq_len in configs:
        # ── Setup tensors ──────────────────────────────────────────
        num_blocks = (seq_len // BS) + 16
        kv_cache = torch.zeros(num_blocks, BS, Hk, cfg.slot_size_aligned,
                               dtype=torch.uint8, device=DEVICE)

        fk = torch.randn(seq_len, Hk, D, dtype=torch.bfloat16, device=DEVICE)
        fv = torch.randn_like(fk)
        triton_turboquant_store(
            fk, fv, kv_cache,
            torch.arange(seq_len, device=DEVICE, dtype=torch.int64),
            PiT, centroids, midpoints,
            mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8)

        q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
        bps = math.ceil(seq_len / BS)
        bt = (torch.arange(bps, device=DEVICE, dtype=torch.int32)
              .unsqueeze(0).expand(B, -1).contiguous())
        sls = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)

        stream_ptr = torch.cuda.current_stream(DEVICE).cuda_stream

        # ── Reference pipeline: GEMM + v52 Stage1 + HIP Stage2 bf16 ──
        REF_SPLITS = 32
        q_rot = torch.mm(
            q.reshape(B * Hq, D).float(), PiT
        ).reshape(B, Hq, D).contiguous()
        mid_o = torch.empty(B, Hq, REF_SPLITS, D + 1,
                            dtype=torch.float32, device=DEVICE)
        ref_out = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)

        def run_ref():
            # GEMM
            torch.mm(q.reshape(B * Hq, D).float(), PiT,
                     out=q_rot.reshape(B * Hq, D))
            # Stage1 (v52)
            fn_s1(
                q_rot.data_ptr(), kv_cache.data_ptr(),
                bt.data_ptr(), sls.data_ptr(),
                centroids_f32.data_ptr(), mid_o.data_ptr(),
                q_rot.stride(0), q_rot.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                bt.stride(0),
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                Hk, BS, REF_SPLITS, kv_group_size,
                scale, 1,    # norm_correction=1
                B, Hq,
                ctypes.c_void_p(stream_ptr),
            )
            # Stage2 bf16
            fn_s2_bf16(
                mid_o.data_ptr(), ref_out.data_ptr(), sls.data_ptr(),
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                ref_out.stride(0), ref_out.stride(1),
                REF_SPLITS, B, Hq,
                ctypes.c_void_p(stream_ptr),
            )

        # Warm up & get reference result
        run_ref()
        torch.cuda.synchronize()
        ref_result = ref_out.float().clone()
        t_ref = profile_gpu(run_ref, label="ref")

        # ── v61 variants ────────────────────────────────────────────
        best_speedup = 0.0
        best_cos = 0.0
        v61_times = {}
        v61_status = "?"

        for sp, fn_v61 in sorted(v61_variants.items()):
            out_v61 = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)

            def run_v61(fn=fn_v61, out=out_v61):
                fn(
                    q.data_ptr(), PiT.data_ptr(),
                    kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
                    centroids_f32.data_ptr(), out.data_ptr(),
                    q.stride(0), q.stride(1),
                    kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                    bt.stride(0),
                    out.stride(0), out.stride(1),
                    Hk, BS, kv_group_size,
                    scale,
                    0,   # query_dtype=bf16
                    1,   # norm_correction
                    B, Hq,
                    ctypes.c_void_p(stream_ptr),
                )

            # Correctness
            run_v61()
            torch.cuda.synchronize()
            v61_result = out_v61.float()

            cos = torch.nn.functional.cosine_similarity(
                ref_result.reshape(-1).unsqueeze(0),
                v61_result.reshape(-1).unsqueeze(0)
            ).item()
            has_nan = torch.isnan(v61_result).any().item()
            max_diff = (ref_result - v61_result).abs().max().item()

            if has_nan:
                v61_status = "NaN!"
            elif cos < 0.99:
                v61_status = f"DIFF"
            else:
                v61_status = "OK"

            best_cos = max(best_cos, cos)

            # Benchmark
            t_v61 = profile_gpu(run_v61, label=f"v61_s{sp}")
            v61_times[sp] = t_v61
            spd = t_ref / t_v61 if t_v61 > 0 else 0
            best_speedup = max(best_speedup, spd)

        # ── Print row ───────────────────────────────────────────────
        row = f"  B={B:2d} seq={seq_len:5d} | {t_ref:>12.1f} us"
        for sp in sorted(v61_variants):
            row += f" | {v61_times.get(sp, 0):>8.1f} us"
        row += f" | {best_speedup:>8.2f}x  | {best_cos:>8.5f} | {v61_status:>6s}"
        print(row)

    print(f"{'='*len(hdr)}")
    print()


if __name__ == "__main__":
    main()

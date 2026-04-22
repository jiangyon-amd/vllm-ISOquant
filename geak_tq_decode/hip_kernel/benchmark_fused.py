#!/usr/bin/env python3
"""Benchmark: Fused HIP kernel vs Triton (stage1 + stage2)."""
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


def run_fused_kernel(so_path, q_rot, kv_cache, bt, sls, centroids_f32, output, B, Hq_l):
    lib = ctypes.CDLL(so_path)
    launch_fn = lib.launch_tq_decode_fused
    # 6 void_p + 2 int + 3 int + 1 int + 2 int + 3 int + 1 float + 1 int + 2 int + 1 void_p (stream) = 22 args
    launch_fn.argtypes = (
        [ctypes.c_void_p]*6 +  # q_rot, kv_cache, block_table, seq_lens, centroids, output
        [ctypes.c_int]*2 +     # stride_qb, stride_qh
        [ctypes.c_int]*3 +     # stride_cb, stride_cp, stride_ch
        [ctypes.c_int]*1 +     # stride_bt
        [ctypes.c_int]*2 +     # stride_ob, stride_oh
        [ctypes.c_int]*3 +     # num_kv_heads, block_size, kv_group_size
        [ctypes.c_float] +     # attn_scale
        [ctypes.c_int]*1 +     # norm_correction
        [ctypes.c_int]*2 +     # B, Hq
        [ctypes.c_void_p]      # stream
    )
    launch_fn.restype = None

    def run():
        launch_fn(
            q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
            centroids_f32.data_ptr(), output.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            bt.stride(0),
            output.stride(0), output.stride(1),
            Hk, BS, Hq_l // Hk,
            1.0 / math.sqrt(D), 1,
            B, Hq_l,
            0)  # stream = 0 (default stream)
    return run


def main():
    B = 100; seq_len = 512

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

    print(f"\n=== B={B}, seq_len={seq_len} ===")

    # Triton reference (stage1 + stage2)
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

    # Fused HIP kernel
    q_rot = (q.float() @ PiT).contiguous()
    centroids_f32 = centroids.float().contiguous()
    output = torch.zeros(B, Hq, D, dtype=torch.float32, device=DEVICE)

    so_path = os.path.join(HIP_DIR, "tq_decode_fused.so")
    try:
        run_fn = run_fused_kernel(so_path, q_rot, kv_cache, bt, sls, centroids_f32, output, B, Hq)
        run_fn()
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
            # Debug: print some values
            print(f"    Triton[0,0,:8]: {triton_out_f32[0,0,:8].tolist()}")
            print(f"    HIP[0,0,:8]:    {output[0,0,:8].tolist()}")
        
        fused_us = bench("HIP Fused", lambda: (run_fn(), torch.cuda.synchronize()))
        print(f"  Speedup vs Triton: {triton_us/fused_us:.2f}x")
        print(f"  Savings: {triton_us - fused_us:.1f} us")
        
    except Exception as e:
        print(f"  HIP Fused: FAILED ({e})")
        import traceback
        traceback.print_exc()

    print(f"\n  Target: <200 us (current stage1+stage2: ~223.6 us)")


if __name__ == "__main__":
    main()

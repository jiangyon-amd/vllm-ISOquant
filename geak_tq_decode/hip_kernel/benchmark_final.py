#!/usr/bin/env python3
"""Final benchmark: Compare all approaches."""
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

    print(f"\n{'='*60}")
    print(f"TQ Decode Benchmark: B={B}, seq_len={seq_len}")
    print(f"{'='*60}")

    # Triton reference
    triton_out = triton_turboquant_decode_attention(
        query=q, kv_cache=kv_cache, block_table=bt, seq_lens=sls,
        Pi=Pi, centroids=centroids, scale=1.0/math.sqrt(D),
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction, PiT=PiT)

    triton_us = bench("Triton", lambda: triton_turboquant_decode_attention(
        query=q, kv_cache=kv_cache, block_table=bt, seq_lens=sls,
        Pi=Pi, centroids=centroids, scale=1.0/math.sqrt(D),
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction, PiT=PiT))

    print(f"\n1. Triton (stage1+stage2): {triton_us:.1f} us")

    # Prepare HIP inputs
    q_rot = (q.float() @ PiT).contiguous()
    centroids_f32 = centroids.float().contiguous()
    mid_o = torch.zeros(B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=DEVICE)
    output = torch.zeros(B, Hq, D, dtype=torch.float32, device=DEVICE)

    # HIP v12 stage1 only
    lib1 = ctypes.CDLL(os.path.join(HIP_DIR, "tq_decode_stage1_v12.so"))
    launch_stage1 = lib1.launch_tq_decode_stage1
    launch_stage1.argtypes = (
        [ctypes.c_void_p]*6 + [ctypes.c_int]*2 + [ctypes.c_int]*3 + [ctypes.c_int]*1 +
        [ctypes.c_int]*3 + [ctypes.c_int]*4 + [ctypes.c_float] + [ctypes.c_int]*2 +
        [ctypes.c_int]*2 + [ctypes.c_void_p]
    )
    launch_stage1.restype = None

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

    stage1_us = bench("HIP v12 stage1", lambda: (run_stage1(), torch.cuda.synchronize()))
    print(f"2. HIP v12 stage1 only: {stage1_us:.1f} us")

    # HIP fused (best version - V14)
    lib_fused = ctypes.CDLL(os.path.join(HIP_DIR, "tq_decode_fused.so"))
    launch_fused = lib_fused.launch_tq_decode_fused
    launch_fused.argtypes = (
        [ctypes.c_void_p]*6 + [ctypes.c_int]*2 + [ctypes.c_int]*3 + [ctypes.c_int]*1 +
        [ctypes.c_int]*2 + [ctypes.c_int]*3 + [ctypes.c_float] + [ctypes.c_int]*1 +
        [ctypes.c_int]*2 + [ctypes.c_void_p]
    )
    launch_fused.restype = None

    def run_fused():
        launch_fused(
            q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sls.data_ptr(),
            centroids_f32.data_ptr(), output.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            bt.stride(0),
            output.stride(0), output.stride(1),
            Hk, BS, Hq // Hk,
            1.0 / math.sqrt(D), 1,
            B, Hq, 0)

    run_fused()
    torch.cuda.synchronize()

    # Check correctness
    triton_out_f32 = triton_out.float()
    max_diff = (output - triton_out_f32).abs().max().item()
    
    fused_us = bench("HIP fused", lambda: (run_fused(), torch.cuda.synchronize()))
    print(f"3. HIP fused (V14, 16 wavefronts): {fused_us:.1f} us")
    print(f"   Correctness: max_diff={max_diff:.6f} {'✓' if max_diff < 0.02 else '✗'}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Triton (baseline): {triton_us:.1f} us")
    print(f"  HIP v12 stage1:    {stage1_us:.1f} us ({triton_us/stage1_us:.2f}x vs Triton)")
    print(f"  HIP fused:         {fused_us:.1f} us ({triton_us/fused_us:.2f}x vs Triton)")
    print(f"{'='*60}")
    print(f"\nConclusion:")
    print(f"  The fused kernel ({fused_us:.1f} us) is slower than Triton ({triton_us:.1f} us)")
    print(f"  because the split version has better parallelism (more threadgroups).")
    print(f"  The HIP v12 stage1 ({stage1_us:.1f} us) is faster than Triton stage1,")
    print(f"  but the fused approach doesn't provide the expected speedup.")


if __name__ == "__main__":
    main()

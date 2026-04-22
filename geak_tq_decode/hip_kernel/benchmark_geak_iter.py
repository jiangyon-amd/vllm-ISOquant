#!/usr/bin/env python3
"""GEAK Iterative Optimization Benchmark — measure Stage1 (all variants),
Stage2, and GEMM q_rot separately with CUDA Events for precise GPU timing.

Also profiles assembly quality via rocm-smi for occupancy validation.
"""
import argparse, math, time, torch

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

DEVICE = "cuda:0"
PRESET = "turboquant_4bit_nc"
D = 128
Hk = 8
Hq = 64
BS = 16

def setup():
    cfg = TurboQuantConfig.from_cache_dtype(PRESET, D)
    signs = generate_wht_signs(D, seed=42).to(DEVICE)
    centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE)
    H_mat = _build_hadamard(D, DEVICE)
    PiT = (signs.float().unsqueeze(1) * H_mat).contiguous()
    Pi = PiT.T.contiguous()
    c_sorted, _ = centroids.float().sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    return cfg, Pi, PiT, centroids, midpoints

def make_data(B, seq_len, cfg, PiT, centroids, midpoints):
    num_blocks = max(8192, (B * seq_len // BS) + 1024)
    kv_cache = torch.zeros(num_blocks, BS, Hk, cfg.slot_size_aligned,
                           dtype=torch.uint8, device=DEVICE)
    fill_n = min(B * seq_len, num_blocks * BS)
    fk = torch.randn(fill_n, Hk, D, dtype=torch.bfloat16, device=DEVICE)
    fv = torch.randn_like(fk)
    triton_turboquant_store(fk, fv, kv_cache,
        torch.arange(fill_n, device=DEVICE, dtype=torch.int64),
        PiT, centroids, midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8)
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    bps = math.ceil(seq_len / BS)
    block_table = torch.arange(bps, device=DEVICE, dtype=torch.int32) \
        .unsqueeze(0).expand(B, -1).contiguous()
    seq_lens_t = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)
    return kv_cache, q, block_table, seq_lens_t

def bench_cuda_events(fn, warmup=10, iters=50):
    """Benchmark with CUDA events for precise GPU timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us

def bench_stage1_hip(B, seq_len, cfg, PiT, centroids, variant="v52"):
    """Benchmark a specific HIP Stage1 variant."""
    import ctypes, os
    so_map = {
        "v52": "tq_decode_hip.so",
        "4warp": "tq_decode_4warp_hip.so",
        "8warp": "tq_decode_8warp_hip.so",
    }
    so_name = so_map.get(variant)
    if so_name is None:
        return None, f"unknown variant {variant}"
    so_path = os.path.join(os.path.dirname(__file__), 
                           "../../vllm/v1/attention/ops", so_name)
    so_path = os.path.normpath(so_path)
    if not os.path.exists(so_path):
        return None, f"not found: {so_path}"

    lib = ctypes.CDLL(so_path)
    fn = lib.launch_tq_decode_stage1
    fn.argtypes = (
        [ctypes.c_void_p] * 6
        + [ctypes.c_int] * 2 + [ctypes.c_int] * 3 + [ctypes.c_int]
        + [ctypes.c_int] * 3 + [ctypes.c_int] * 4
        + [ctypes.c_float] + [ctypes.c_int] + [ctypes.c_int] * 2
        + [ctypes.c_void_p]
    )
    fn.restype = None
    
    _, _, PiT_d, centroids_d, midpoints_d = setup()
    kv_cache, q, block_table, seq_lens_t = make_data(B, seq_len, cfg, PiT_d, centroids_d, midpoints_d)
    
    q_float = q.float()
    q_rot = (q_float @ PiT_d).contiguous()
    centroids_f32 = centroids_d.float()
    
    NUM_KV_SPLITS = 32
    kv_group_size = Hq // Hk
    scale = 1.0 / math.sqrt(D)
    mid_o = torch.zeros(B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=DEVICE)
    
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        fn(
            q_rot.data_ptr(), kv_cache.data_ptr(),
            block_table.data_ptr(), seq_lens_t.data_ptr(),
            centroids_f32.data_ptr(), mid_o.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            Hk, BS, NUM_KV_SPLITS, kv_group_size,
            scale, 1, B, Hq,
            ctypes.c_void_p(stream),
        )
    
    us = bench_cuda_events(run)
    return us, variant

def bench_gemm(B, PiT):
    """Benchmark q_rot = query @ PiT GEMM."""
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    q_flat = q.reshape(B * Hq, D).float()
    q_rot = torch.empty_like(q_flat)
    
    def run():
        torch.mm(q_flat, PiT, out=q_rot)
    
    return bench_cuda_events(run)

def bench_stage2_hip(B, seq_len, NUM_KV_SPLITS=32):
    """Benchmark Stage2 reduce."""
    import ctypes, os
    so_path = os.path.normpath(os.path.join(os.path.dirname(__file__),
                               "../../vllm/v1/attention/ops/tq_decode_stage2_hip.so"))
    if not os.path.exists(so_path):
        return None, "not found"
    lib = ctypes.CDLL(so_path)
    fn = lib.launch_tq_decode_stage2_bf16
    fn.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_int] * 3 + [ctypes.c_int] * 2
        + [ctypes.c_int] * 3
        + [ctypes.c_void_p]
    )
    fn.restype = None
    
    mid_o = torch.randn(B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=DEVICE)
    mid_o[:, :, :, D] = torch.randn(B, Hq, NUM_KV_SPLITS)  # LSE values
    output = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    seq_lens_t = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        fn(
            mid_o.data_ptr(), output.data_ptr(), seq_lens_t.data_ptr(),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            output.stride(0), output.stride(1),
            NUM_KV_SPLITS, B, Hq,
            ctypes.c_void_p(stream),
        )
    
    return bench_cuda_events(run), "bf16"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    cfg, Pi, PiT, centroids, midpoints = setup()

    print("=" * 90)
    print("GEAK ITERATIVE OPTIMIZATION — Current Baseline Profiling")
    print("=" * 90)
    print(f"Config: Hq={Hq}, Hk={Hk}, D={D}, BS={BS}, splits=32")
    print()

    # Workload matrix
    workloads = [
        (4, 8192),   # Primary target
        (4, 4096),
        (20, 8192),  # Secondary
        (20, 4096),
        (32, 512),
        (100, 128),
    ]

    # GEMM benchmark
    print(f"{'':>2} {'B':>4} {'seq':>5} │ {'GEMM':>8} │ {'v52':>8} {'4warp':>8} {'8warp':>8} │ {'Stage2':>8} │ {'Total(best)':>12}")
    print("─" * 90)

    for B, seq_len in workloads:
        gemm_us = bench_gemm(B, PiT)
        
        results = {}
        for variant in ["v52", "4warp", "8warp"]:
            us, name = bench_stage1_hip(B, seq_len, cfg, PiT, centroids, variant)
            if us is not None:
                results[variant] = us
        
        s2_us, _ = bench_stage2_hip(B, seq_len)
        
        best_s1 = min(results.values()) if results else 0
        best_name = min(results, key=results.get) if results else "?"
        total = gemm_us + best_s1 + (s2_us if s2_us else 0)
        
        v52_str = f"{results.get('v52', 0):8.1f}" if 'v52' in results else "    N/A "
        w4_str = f"{results.get('4warp', 0):8.1f}" if '4warp' in results else "    N/A "
        w8_str = f"{results.get('8warp', 0):8.1f}" if '8warp' in results else "    N/A "
        s2_str = f"{s2_us:8.1f}" if s2_us else "    N/A "
        
        print(f"   {B:4d} {seq_len:5d} │ {gemm_us:8.1f} │ {v52_str} {w4_str} {w8_str} │ {s2_str} │ {total:8.1f} ({best_name})")
    
    print()
    print("All times in microseconds (us), CUDA Events timing")
    print()
    
    # Detailed breakdown for primary target
    B, seq_len = 4, 8192
    print(f"PRIMARY TARGET BREAKDOWN: B={B}, seq={seq_len}")
    print("─" * 50)
    gemm_us = bench_gemm(B, PiT)
    s1_us, _ = bench_stage1_hip(B, seq_len, cfg, PiT, centroids, "8warp")
    s2_us, _ = bench_stage2_hip(B, seq_len)
    total = gemm_us + s1_us + s2_us
    print(f"  GEMM (q_rot):  {gemm_us:8.1f} us  ({gemm_us/total*100:5.1f}%)")
    print(f"  Stage1 (8w):   {s1_us:8.1f} us  ({s1_us/total*100:5.1f}%)")
    print(f"  Stage2:        {s2_us:8.1f} us  ({s2_us/total*100:5.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL:         {total:8.1f} us")

if __name__ == "__main__":
    main()

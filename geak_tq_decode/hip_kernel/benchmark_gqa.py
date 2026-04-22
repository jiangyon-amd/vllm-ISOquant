#!/usr/bin/env python3
"""Benchmark for GQA-fused TQ Decode kernel.

Tests with 72B config (Hq=64, Hkv=8, GQA=8:1) and 4B config (Hq=32, Hkv=8, GQA=4:1).
Measures kernel-only time (no q@PiT, no stage2).
"""
import ctypes, os, sys, math, time, argparse
import torch

DEVICE = "cuda:0"
D = 128
BS = 16   # block_size (pages)
NUM_KV_SPLITS = 8

def load_kernel(so_path):
    """Load HIP kernel from .so file."""
    lib = ctypes.CDLL(so_path)
    fn = lib.launch_tq_decode_stage1
    fn.argtypes = [
        ctypes.c_void_p,  # q_rot
        ctypes.c_void_p,  # kv_cache
        ctypes.c_void_p,  # block_table
        ctypes.c_void_p,  # seq_lens
        ctypes.c_void_p,  # centroids
        ctypes.c_void_p,  # mid_o
    ] + [ctypes.c_int] * 9 + [  # strides
        ctypes.c_int,     # num_kv_heads
        ctypes.c_int,     # block_size
        ctypes.c_int,     # num_kv_splits
        ctypes.c_int,     # kv_group_size
        ctypes.c_float,   # attn_scale
        ctypes.c_int,     # block_kv (unused by some versions)
        ctypes.c_int,     # norm_correction
        ctypes.c_int,     # B
        ctypes.c_int,     # Hq
        ctypes.c_void_p,  # stream
    ]
    fn.restype = None
    return fn


def setup_data(B, Hq, Hkv, seq_len):
    """Create test data."""
    kv_group_size = Hq // Hkv
    slot_size = 136   # 64 (mse) + 4 (norms) + 64 (val) + 4 (scale/zero)
    slot_aligned = 136
    
    num_blocks = max(4096, (B * seq_len // BS) + 512)
    kv_cache = torch.randint(0, 256, (num_blocks, BS, Hkv, slot_aligned),
                              dtype=torch.uint8, device=DEVICE)
    
    # Set valid norms (bytes 64-67: fp16 norm + fp16 gamma)
    # Set valid scale/zero (bytes 132-135: fp16 scale + fp16 zero)
    for h in range(Hkv):
        # norm at offset 64: fp16 value ~1.0
        norm_val = torch.tensor([1.0], dtype=torch.float16)
        norm_bytes = norm_val.view(torch.uint8)
        kv_cache[:, :, h, 64] = norm_bytes[0]
        kv_cache[:, :, h, 65] = norm_bytes[1]
        # gamma at offset 66
        kv_cache[:, :, h, 66] = norm_bytes[0]
        kv_cache[:, :, h, 67] = norm_bytes[1]
        # scale at offset 132
        scale_val = torch.tensor([0.1], dtype=torch.float16)
        scale_bytes = scale_val.view(torch.uint8)
        kv_cache[:, :, h, 132] = scale_bytes[0]
        kv_cache[:, :, h, 133] = scale_bytes[1]
        # zero at offset 134
        zero_val = torch.tensor([-0.5], dtype=torch.float16)
        zero_bytes = zero_val.view(torch.uint8)
        kv_cache[:, :, h, 134] = zero_bytes[0]
        kv_cache[:, :, h, 135] = zero_bytes[1]
    
    q_rot = torch.randn(B, Hq, D, dtype=torch.float32, device=DEVICE)
    
    bps = math.ceil(seq_len / BS)
    block_table = torch.arange(bps, device=DEVICE, dtype=torch.int32) \
        .unsqueeze(0).expand(B, -1).contiguous()
    seq_lens = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)
    
    centroids = torch.randn(16, dtype=torch.float32, device=DEVICE) * 0.5
    
    mid_o = torch.zeros(B, Hq, NUM_KV_SPLITS, D + 1,
                        dtype=torch.float32, device=DEVICE)
    
    return q_rot, kv_cache, block_table, seq_lens, centroids, mid_o


def benchmark_kernel(fn, B, Hq, Hkv, seq_len, warmup=20, iters=100):
    """Benchmark a single kernel configuration."""
    kv_group_size = Hq // Hkv
    q_rot, kv_cache, block_table, seq_lens, centroids, mid_o = setup_data(B, Hq, Hkv, seq_len)
    
    stream_ptr = torch.cuda.current_stream(DEVICE).cuda_stream
    attn_scale = 1.0 / math.sqrt(D)
    
    def run():
        mid_o.zero_()
        fn(
            q_rot.data_ptr(),
            kv_cache.data_ptr(),
            block_table.data_ptr(),
            seq_lens.data_ptr(),
            centroids.data_ptr(),
            mid_o.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            Hkv, BS, NUM_KV_SPLITS, kv_group_size,
            attn_scale,
            8,   # block_kv
            1,   # norm_correction
            B, Hq,
            ctypes.c_void_p(stream_ptr),
        )
    
    # Warmup
    for _ in range(warmup):
        run()
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    avg_us = (time.perf_counter() - t0) / iters * 1e6
    
    return avg_us


def correctness_check(fn_ref, fn_new, B, Hq, Hkv, seq_len):
    """Check that new kernel matches reference."""
    kv_group_size = Hq // Hkv
    q_rot, kv_cache, block_table, seq_lens, centroids, mid_o_ref = setup_data(B, Hq, Hkv, seq_len)
    mid_o_new = torch.zeros_like(mid_o_ref)
    
    stream_ptr = torch.cuda.current_stream(DEVICE).cuda_stream
    attn_scale = 1.0 / math.sqrt(D)
    
    args = [
        q_rot.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
        seq_lens.data_ptr(), centroids.data_ptr(),
    ]
    extra = [
        q_rot.stride(0), q_rot.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
    ]
    
    # Reference
    fn_ref(*args, mid_o_ref.data_ptr(),
           *extra,
           mid_o_ref.stride(0), mid_o_ref.stride(1), mid_o_ref.stride(2),
           Hkv, BS, NUM_KV_SPLITS, kv_group_size,
           attn_scale, 8, 1, B, Hq, ctypes.c_void_p(stream_ptr))
    torch.cuda.synchronize()
    
    # New kernel
    fn_new(*args, mid_o_new.data_ptr(),
           *extra,
           mid_o_new.stride(0), mid_o_new.stride(1), mid_o_new.stride(2),
           Hkv, BS, NUM_KV_SPLITS, kv_group_size,
           attn_scale, 8, 1, B, Hq, ctypes.c_void_p(stream_ptr))
    torch.cuda.synchronize()
    
    # Compare
    diff = (mid_o_ref - mid_o_new).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    passed = max_diff < 0.01
    return passed, max_diff, mean_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, default='tq_decode_stage1_v52.so',
                        help='Kernel .so file to benchmark')
    parser.add_argument('--ref', type=str, default='tq_decode_stage1_v52.so',
                        help='Reference kernel for correctness check')
    parser.add_argument('--sweep', action='store_true', help='Run full sweep')
    parser.add_argument('--check', action='store_true', help='Run correctness check')
    args = parser.parse_args()
    
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(kernel_dir, args.kernel)
    
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Compile first:")
        print(f"  hipcc --offload-arch=gfx950 -shared -fPIC -O3 -ffast-math -o {args.kernel} {args.kernel.replace('.so', '.hip')}")
        sys.exit(1)
    
    fn = load_kernel(so_path)
    
    # Correctness check
    if args.check and args.ref != args.kernel:
        ref_path = os.path.join(kernel_dir, args.ref)
        if os.path.exists(ref_path):
            fn_ref = load_kernel(ref_path)
            print("=== Correctness Check ===")
            for B, Hq, Hkv, seq in [(4, 64, 8, 512), (4, 32, 8, 512), (1, 64, 8, 2048)]:
                passed, max_d, mean_d = correctness_check(fn_ref, fn, B, Hq, Hkv, seq)
                status = "PASS ✅" if passed else "FAIL ❌"
                print(f"  B={B}, Hq={Hq}, Hkv={Hkv}, seq={seq}: {status} (max={max_d:.6f}, mean={mean_d:.6f})")
            print()
    
    # Benchmark
    print(f"=== Benchmark: {args.kernel} ===")
    print(f"{'Config':<35s} {'Time':>10s} {'vs 4B':>8s}")
    print("-" * 60)
    
    configs = [
        # (B, Hq, Hkv, seq_len, label)
        (100, 32, 8, 512,  "4B  (B=100,Hq=32,Hkv=8,s=512)"),
        (100, 64, 8, 512,  "72B (B=100,Hq=64,Hkv=8,s=512)"),
        (100, 16, 2, 512,  "72B-TP4 (B=100,Hq=16,Hkv=2,s=512)"),
    ]
    
    if args.sweep:
        configs += [
            (8, 64, 8, 2048,  "72B (B=8,Hq=64,Hkv=8,s=2048)"),
            (8, 16, 2, 2048,  "72B-TP4 (B=8,Hq=16,Hkv=2,s=2048)"),
            (1, 64, 8, 4096,  "72B (B=1,Hq=64,Hkv=8,s=4096)"),
            (50, 64, 8, 1024, "72B (B=50,Hq=64,Hkv=8,s=1024)"),
            (200, 64, 8, 512, "72B (B=200,Hq=64,Hkv=8,s=512)"),
        ]
    
    ref_time = None
    for B, Hq, Hkv, seq, label in configs:
        us = benchmark_kernel(fn, B, Hq, Hkv, seq)
        ratio = ""
        if ref_time is not None and ref_time > 0:
            ratio = f"{us/ref_time:.2f}x"
        else:
            ref_time = us
            ratio = "1.00x"
        print(f"  {label:<33s} {us:>8.1f}us {ratio:>8s}")
    
    print()
    print("Target: 72B config should be close to 4B config (GQA fusion eliminates redundant KV reads)")


if __name__ == "__main__":
    main()

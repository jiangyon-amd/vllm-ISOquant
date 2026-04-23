#!/usr/bin/env python3
"""Standalone rocprofv3-friendly benchmark for TQ decode kernels.

Usage:
  rocprofv3 --hip-trace --hsa-trace -o prof_results -- \
    python profiling/rocprof_tq_decode.py

Exercises all 3 paths:  FUSED / SPLIT / Triton fallback
across multiple (B, seq_len) configurations.
"""
import os, sys, math, ctypes, time
import torch

# Force ROCm
assert torch.cuda.is_available(), "Need ROCm/CUDA"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# ---------------------------------------------------------------------------
# Simulate TQ decode data structures
# ---------------------------------------------------------------------------
HEAD_DIM = 128
Hq, Hk = 64, 8
kv_group_size = Hq // Hk
block_size = 64
mse_bits = 4
value_quant_bits = 4
MSE_BYTES = math.ceil(HEAD_DIM * mse_bits / 8)
KPS = MSE_BYTES + 2 + 2  # mse_data + norm(2B) + padding(2B) = 68
VAL_DATA_BYTES = math.ceil(HEAD_DIM * value_quant_bits / 8)
padded_slot = KPS + VAL_DATA_BYTES + 4  # + scale(2B)+zero(2B) = 136
scale = 1.0 / math.sqrt(HEAD_DIM)

def make_test_data(B, max_seq_len, num_kv_splits=32):
    num_blocks_per_seq = math.ceil(max_seq_len / block_size)
    total_blocks = B * num_blocks_per_seq + 16  # extra padding
    
    q = torch.randn(B, Hq, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv_cache = torch.randint(0, 255, (total_blocks, block_size, Hk, padded_slot),
                             dtype=torch.uint8, device=device)
    block_table = torch.zeros(B, num_blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(B):
        block_table[b] = torch.arange(b * num_blocks_per_seq,
                                       (b+1) * num_blocks_per_seq,
                                       dtype=torch.int32, device=device)
    seq_lens = torch.full((B,), max_seq_len, dtype=torch.int32, device=device)
    centroids = torch.randn(16, dtype=torch.float32, device=device)  # 2^4 centroids
    PiT = torch.eye(HEAD_DIM, dtype=torch.float32, device=device)
    
    # q_rot = q @ PiT
    q_rot = q.float() @ PiT  # [B, Hq, D] float32
    
    mid_o = torch.zeros(B, Hq, num_kv_splits, HEAD_DIM + 1,
                        dtype=torch.float32, device=device)
    output_bf16 = torch.zeros(B, Hq, HEAD_DIM, dtype=torch.bfloat16, device=device)
    output_f32 = torch.zeros(B, Hq, HEAD_DIM, dtype=torch.float32, device=device)
    
    return q, q_rot, kv_cache, block_table, seq_lens, centroids, PiT, mid_o, output_bf16, output_f32

# ---------------------------------------------------------------------------
# Load HIP kernels
# ---------------------------------------------------------------------------
ops_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       "../vllm/v1/attention/ops")

def load_so(name, func_name, argtypes):
    path = os.path.join(ops_dir, name)
    if not os.path.exists(path):
        print(f"  [SKIP] {name} not found")
        return None
    lib = ctypes.CDLL(path)
    fn = getattr(lib, func_name)
    fn.argtypes = argtypes
    fn.restype = None
    return fn

_STAGE1_ARGTYPES = (
    [ctypes.c_void_p] * 6
    + [ctypes.c_int] * 2 + [ctypes.c_int] * 3 + [ctypes.c_int]
    + [ctypes.c_int] * 3 + [ctypes.c_int] * 4
    + [ctypes.c_float] + [ctypes.c_int] + [ctypes.c_int] * 2
    + [ctypes.c_void_p]
)

_FUSED_ARGTYPES = (
    [ctypes.c_void_p] * 6
    + [ctypes.c_int] * 2 + [ctypes.c_int] * 3 + [ctypes.c_int]
    + [ctypes.c_int] * 2 + [ctypes.c_int] + [ctypes.c_int]
    + [ctypes.c_int] + [ctypes.c_float] + [ctypes.c_int]
    + [ctypes.c_int] * 2 + [ctypes.c_void_p]
)

_STAGE2_ARGTYPES = (
    [ctypes.c_void_p] * 3
    + [ctypes.c_int] * 5 + [ctypes.c_int]
    + [ctypes.c_int] * 2 + [ctypes.c_void_p]
)

split_fn = load_so("tq_decode_split_hip.so", "launch_tq_decode_stage1", _STAGE1_ARGTYPES)
fused_fn = load_so("tq_decode_fused_hip.so", "launch_tq_decode_fused", _FUSED_ARGTYPES)
s2_bf16_fn = load_so("tq_decode_stage2_hip.so", "launch_tq_decode_stage2_bf16", _STAGE2_ARGTYPES)
s2_f32_fn = load_so("tq_decode_stage2_hip.so", "launch_tq_decode_stage2_f32", _STAGE2_ARGTYPES)

# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------
CONFIGS = [
    # (B, seq_len, num_kv_splits, label)
    (1,   128,   32, "B=1   seq=128"),
    (4,   128,   32, "B=4   seq=128"),
    (16,  128,   32, "B=16  seq=128"),
    (64,  128,   32, "B=64  seq=128"),
    (128, 128,   32, "B=128 seq=128"),
    (200, 128,   32, "B=200 seq=128"),
    (1,   512,   32, "B=1   seq=512"),
    (4,   512,   32, "B=4   seq=512"),
    (16,  512,   32, "B=16  seq=512"),
    (64,  512,   32, "B=64  seq=512"),
    (1,   1024,  32, "B=1   seq=1024"),
    (4,   1024,  32, "B=4   seq=1024"),
    (16,  1024,  32, "B=16  seq=1024"),
    (1,   2048,  32, "B=1   seq=2048"),
    (4,   2048,  32, "B=4   seq=2048"),
    (16,  2048,  32, "B=16  seq=2048"),
    (1,   4096,  32, "B=1   seq=4096"),
    (4,   4096,  32, "B=4   seq=4096"),
    (1,   8192,  32, "B=1   seq=8192"),
    (4,   8192,  32, "B=4   seq=8192"),
    (20,  8192,  32, "B=20  seq=8192"),
]

WARMUP = 5
REPEAT = 20

stream = torch.cuda.current_stream(device)
stream_ptr = stream.cuda_stream

def run_split(q_rot, kv_cache, bt, sl, centroids, mid_o, out_bf16, B, num_kv_splits):
    split_fn(
        q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(),
        sl.data_ptr(), centroids.data_ptr(), mid_o.data_ptr(),
        q_rot.stride(0), q_rot.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        bt.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        Hk, block_size, num_kv_splits, kv_group_size,
        scale, 0, B, Hq,
        ctypes.c_void_p(stream_ptr),
    )
    s2_bf16_fn(
        mid_o.data_ptr(), out_bf16.data_ptr(), sl.data_ptr(),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        out_bf16.stride(0), out_bf16.stride(1),
        num_kv_splits, B, Hq,
        ctypes.c_void_p(stream_ptr),
    )

def run_fused(q_rot, kv_cache, bt, sl, centroids, out_bf16, B):
    fused_fn(
        q_rot.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(),
        sl.data_ptr(), centroids.data_ptr(), out_bf16.data_ptr(),
        q_rot.stride(0), q_rot.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        bt.stride(0),
        out_bf16.stride(0), out_bf16.stride(1),
        Hk, block_size, kv_group_size,
        scale, 0, B, Hq,
        ctypes.c_void_p(stream_ptr),
    )

def bench(fn, label):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(REPEAT):
        fn()
    end.record()
    torch.cuda.synchronize()
    
    us = start.elapsed_time(end) * 1000 / REPEAT
    print(f"  {label:40s} {us:8.1f} µs")
    return us

# ---------------------------------------------------------------------------
# Run profiling
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"TQ Decode rocprofv3 Profiling — MI355X")
print(f"Hq={Hq} Hk={Hk} D={HEAD_DIM} block_size={block_size}")
print(f"{'='*70}\n")

results = []

for B, seq, nks, label in CONFIGS:
    print(f"\n--- {label} (nks={nks}) ---")
    q, q_rot, kvc, bt, sl, cent, PiT, mid_o, out_bf16, out_f32 = make_test_data(B, seq, nks)
    
    # FUSED path (seq <= 512)
    if fused_fn and seq <= 512:
        us = bench(lambda: run_fused(q_rot, kvc, bt, sl, cent, out_bf16, B),
                   f"FUSED   {label}")
        results.append((label, "FUSED", us))
    
    # SPLIT path  
    if split_fn and s2_bf16_fn:
        us = bench(lambda: run_split(q_rot, kvc, bt, sl, cent, mid_o, out_bf16, B, nks),
                   f"SPLIT   {label}")
        results.append((label, "SPLIT", us))
    
    # Also profile GEMM (q_rot computation)
    q_flat = q.reshape(B * Hq, HEAD_DIM).float()
    q_rot_out = torch.empty_like(q_rot)
    us = bench(lambda: torch.mm(q_flat, PiT, out=q_rot_out.reshape(B * Hq, HEAD_DIM)),
               f"GEMM    {label}")
    results.append((label, "GEMM", us))
    
    del q, q_rot, kvc, bt, sl, cent, PiT, mid_o, out_bf16, out_f32, q_flat, q_rot_out
    torch.cuda.empty_cache()

print(f"\n\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"{'Config':25s} {'Path':10s} {'µs':>10s}")
print(f"{'-'*45}")
for label, path, us in results:
    print(f"{label:25s} {path:10s} {us:10.1f}")

#!/usr/bin/env python3
"""Profile full TQ decode attention pipeline per layer.
Measures: q@PiT GEMM, stage1, stage2, dtype conversion.
"""
import math, time, torch, ctypes, os, sys

DEVICE = "cuda:0"
D = 128
BS = 16
NUM_KV_SPLITS = 8
B = 100
Hq = 64
Hk = 8
seq_len = 512

def main():
    # Setup
    kv_group_size = Hq // Hk
    slot_size = 136
    num_blocks = max(4096, (B * seq_len // BS) + 512)
    
    kv_cache = torch.randint(0, 256, (num_blocks, BS, Hk, slot_size),
                              dtype=torch.uint8, device=DEVICE)
    # Set valid norms and scales
    for h in range(Hk):
        norm_val = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
        kv_cache[:, :, h, 64] = norm_val[0]; kv_cache[:, :, h, 65] = norm_val[1]
        kv_cache[:, :, h, 66] = norm_val[0]; kv_cache[:, :, h, 67] = norm_val[1]
        scale_val = torch.tensor([0.1], dtype=torch.float16).view(torch.uint8)
        kv_cache[:, :, h, 132] = scale_val[0]; kv_cache[:, :, h, 133] = scale_val[1]
        zero_val = torch.tensor([-0.5], dtype=torch.float16).view(torch.uint8)
        kv_cache[:, :, h, 134] = zero_val[0]; kv_cache[:, :, h, 135] = zero_val[1]
    
    query = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    bps = math.ceil(seq_len / BS)
    block_table = torch.arange(bps, device=DEVICE, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()
    seq_lens = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)
    centroids = torch.randn(16, dtype=torch.float32, device=DEVICE) * 0.5
    
    # PiT matrix (simulating WHT rotation)
    PiT = torch.randn(D, D, dtype=torch.float32, device=DEVICE) / math.sqrt(D)
    
    # Load HIP kernel
    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tq_decode_stage1_v52.so")
    lib = ctypes.CDLL(so_path)
    fn = lib.launch_tq_decode_stage1
    fn.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*9 + [ctypes.c_int]*4 + [ctypes.c_float] + [ctypes.c_int]*2 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    fn.restype = None
    
    mid_o = torch.zeros(B, Hq, NUM_KV_SPLITS, D+1, dtype=torch.float32, device=DEVICE)
    output = torch.zeros(B, Hq, D, dtype=torch.float32, device=DEVICE)
    attn_scale = 1.0 / math.sqrt(D)
    stream_ptr = torch.cuda.current_stream(DEVICE).cuda_stream
    
    # Import stage2 Triton kernel
    sys.path.insert(0, '/home/jiangyon/vllm_turboquant')
    from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2
    lse = torch.empty(B, Hq, dtype=torch.float32, device=DEVICE)
    
    WARMUP = 20
    ITERS = 100
    
    def time_fn(label, func, warmup=WARMUP, iters=ITERS):
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            func()
        torch.cuda.synchronize()
        avg_us = (time.perf_counter() - t0) / iters * 1e6
        print(f"  {label:<30s} {avg_us:>8.1f} us")
        return avg_us
    
    print(f"=== TQ Decode Pipeline Profile ===")
    print(f"B={B}, Hq={Hq}, Hk={Hk}, D={D}, seq={seq_len}, splits={NUM_KV_SPLITS}")
    print()
    
    # 1. BF16→FP32 conversion
    t_conv = time_fn("1. query.float()", lambda: query.float())
    
    # 2. q @ PiT GEMM
    q_float = query.float()
    t_gemm = time_fn("2. q_float @ PiT (GEMM)", lambda: (q_float @ PiT).contiguous())
    
    # 3. Stage1 kernel
    q_rot = (q_float @ PiT).contiguous()
    centroids_f32 = centroids.float().contiguous()
    def run_stage1():
        mid_o.zero_()
        fn(q_rot.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
           seq_lens.data_ptr(), centroids_f32.data_ptr(), mid_o.data_ptr(),
           q_rot.stride(0), q_rot.stride(1),
           kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
           block_table.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
           Hk, BS, NUM_KV_SPLITS, kv_group_size, attn_scale,
           8, 1, B, Hq, ctypes.c_void_p(stream_ptr))
    t_stage1 = time_fn("3. Stage1 (HIP kernel)", run_stage1)
    
    # 4. Stage2 kernel
    BLOCK_D = 128  # next_power_of_2(D)
    def run_stage2():
        _fwd_kernel_stage2[(B, Hq)](
            mid_o, output, lse, seq_lens,
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            output.stride(0), output.stride(1), lse.stride(0),
            NUM_KV_SPLITS=NUM_KV_SPLITS, BLOCK_DV=BLOCK_D, Lv=D,
            num_warps=4, num_stages=2)
    t_stage2 = time_fn("4. Stage2 (Triton split reduce)", run_stage2)
    
    # 5. FP32→BF16 conversion
    t_back = time_fn("5. output.to(bfloat16)", lambda: output.to(torch.bfloat16))
    
    # 6. centroids.float().contiguous() (called every time!)
    t_cent = time_fn("6. centroids.float().contiguous()", lambda: centroids.float().contiguous())
    
    # Full pipeline
    def run_full():
        q_f = query.float()
        qr = (q_f @ PiT).contiguous()
        c_f = centroids.float().contiguous()
        mid_o.zero_()
        fn(qr.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
           seq_lens.data_ptr(), c_f.data_ptr(), mid_o.data_ptr(),
           qr.stride(0), qr.stride(1),
           kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
           block_table.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
           Hk, BS, NUM_KV_SPLITS, kv_group_size, attn_scale,
           8, 1, B, Hq, ctypes.c_void_p(stream_ptr))
        _fwd_kernel_stage2[(B, Hq)](
            mid_o, output, lse, seq_lens,
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            output.stride(0), output.stride(1), lse.stride(0),
            NUM_KV_SPLITS=NUM_KV_SPLITS, BLOCK_DV=BLOCK_D, Lv=D,
            num_warps=4, num_stages=2)
        _ = output.to(torch.bfloat16)
    t_full = time_fn("FULL PIPELINE", run_full)
    
    print()
    total_sum = t_conv + t_gemm + t_stage1 + t_stage2 + t_back + t_cent
    print(f"  Sum of parts:     {total_sum:>8.1f} us")
    print(f"  Full pipeline:    {t_full:>8.1f} us")
    print(f"  × 80 layers:      {t_full * 80 / 1000:>8.1f} ms (total attention)")
    print(f"  Per request:      {t_full * 80 / 1000 / B:>8.4f} ms (= a_TQ)")
    print()
    print(f"  --- Breakdown (of full pipeline) ---")
    for label, t in [("q.float()", t_conv), ("q@PiT GEMM", t_gemm), 
                      ("Stage1 HIP", t_stage1), ("Stage2 Triton", t_stage2),
                      ("to(bf16)", t_back), ("centroids cast", t_cent)]:
        pct = t / t_full * 100
        print(f"    {label:<20s} {t:>7.1f}us  ({pct:>4.1f}%)")

if __name__ == "__main__":
    main()

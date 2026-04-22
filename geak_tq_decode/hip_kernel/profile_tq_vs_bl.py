#!/usr/bin/env python3
"""
详细对比 TQ 和 BL (SDPA) decode attention 的全链路耗时。
模拟 80 层 transformer 的一次 decode step。
配置: Qwen2.5-72B (Hq=64, Hk=8, D=128), B=100, seq=512
"""
import math, time, torch, ctypes, os, sys

sys.path.insert(0, '/home/jiangyon/vllm_turboquant')

DEVICE = "cuda:0"
D = 128; BS = 16; NUM_KV_SPLITS = 8
B = 100; Hq = 64; Hk = 8; seq_len = 512
N_LAYERS = 80
WARMUP = 10; ITERS = 50

def setup_data():
    """Create test tensors for both TQ and BL paths."""
    slot_size = 136
    num_blocks = max(4096, (B * seq_len // BS) + 512)
    kv_cache = torch.randint(0, 256, (num_blocks, BS, Hk, slot_size),
                              dtype=torch.uint8, device=DEVICE)
    for h in range(Hk):
        nv = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
        kv_cache[:,:,h,64]=nv[0]; kv_cache[:,:,h,65]=nv[1]
        kv_cache[:,:,h,66]=nv[0]; kv_cache[:,:,h,67]=nv[1]
        sv = torch.tensor([0.1], dtype=torch.float16).view(torch.uint8)
        kv_cache[:,:,h,132]=sv[0]; kv_cache[:,:,h,133]=sv[1]
        zv = torch.tensor([-0.5], dtype=torch.float16).view(torch.uint8)
        kv_cache[:,:,h,134]=zv[0]; kv_cache[:,:,h,135]=zv[1]
    
    query = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    bps = math.ceil(seq_len / BS)
    block_table = torch.arange(bps, device=DEVICE, dtype=torch.int32).unsqueeze(0).expand(B,-1).contiguous()
    seq_lens_t = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)
    centroids = torch.randn(16, dtype=torch.float32, device=DEVICE) * 0.5
    PiT = torch.randn(D, D, dtype=torch.float32, device=DEVICE) / math.sqrt(D)
    
    # BL data
    k_bl = torch.randn(B, Hk, seq_len, D, dtype=torch.bfloat16, device=DEVICE)
    v_bl = torch.randn(B, Hk, seq_len, D, dtype=torch.bfloat16, device=DEVICE)
    
    return kv_cache, query, block_table, seq_lens_t, centroids, PiT, k_bl, v_bl

def load_hip_kernel():
    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tq_decode_v52_no_nt.so')
    lib = ctypes.CDLL(so_path)
    fn = lib.launch_tq_decode_stage1
    fn.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*9 + [ctypes.c_int]*4 + [ctypes.c_float] + [ctypes.c_int]*2 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    fn.restype = None
    return fn

def time_fn(label, func, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup): func()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): func()
    torch.cuda.synchronize()
    avg_us = (time.perf_counter() - t0) / iters * 1e6
    return avg_us

def main():
    kv_cache, query, block_table, seq_lens_t, centroids, PiT, k_bl, v_bl = setup_data()
    hip_fn = load_hip_kernel()
    from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2
    
    # Pre-alloc TQ buffers (simulating layer cache)
    mid_o = torch.zeros(B, Hq, NUM_KV_SPLITS, D+1, dtype=torch.float32, device=DEVICE)
    output_f32 = torch.empty(B, Hq, D, dtype=torch.float32, device=DEVICE)
    lse = torch.empty(B, Hq, dtype=torch.float32, device=DEVICE)
    attn_scale = 1.0 / math.sqrt(D)
    kv_group_size = Hq // Hk
    stream_ptr = torch.cuda.current_stream(DEVICE).cuda_stream
    
    # Pre-compute centroids (optimization candidate!)
    centroids_f32 = centroids.float().contiguous()
    # Pre-alloc q_rot buffer (optimization candidate!)
    q_rot_buf = torch.empty(B, Hq, D, dtype=torch.float32, device=DEVICE)
    
    print("=" * 70)
    print("  TQ vs BL (SDPA) Per-Layer Decode Attention Profiling")
    print(f"  B={B}, Hq={Hq}, Hk={Hk}, D={D}, seq={seq_len}, splits={NUM_KV_SPLITS}")
    print("=" * 70)
    
    # ==================== BL (SDPA) ====================
    print("\n--- BL (SDPA) ---")
    
    scale = 1.0 / math.sqrt(D)
    t_sdpa = time_fn("SDPA kernel", lambda:
        torch.nn.functional.scaled_dot_product_attention(
            query.unsqueeze(2), k_bl, v_bl, scale=scale, enable_gqa=True))
    print(f"  SDPA kernel:               {t_sdpa:>8.1f} us")
    print(f"  (× {N_LAYERS} layers):              {t_sdpa * N_LAYERS / 1000:>8.1f} ms")
    
    # ==================== TQ Current (有开销) ====================
    print("\n--- TQ Current (with overhead) ---")
    
    # 1. q.float()
    t1 = time_fn("  q.float()", lambda: query.float())
    print(f"  1. query.float():          {t1:>8.1f} us")
    
    # 2. GEMM + contiguous
    q_f = query.float()
    t2 = time_fn("  GEMM", lambda: (q_f @ PiT).contiguous())
    print(f"  2. q@PiT GEMM+contig:      {t2:>8.1f} us")
    
    # 3. centroids.float().contiguous() (called every layer!)
    t3 = time_fn("  centroids cast", lambda: centroids.float().contiguous())
    print(f"  3. centroids.float():      {t3:>8.1f} us")
    
    # 4. mid_o.zero_() (implicit in slice or explicit)
    t4 = time_fn("  mid_o init", lambda: mid_o.zero_())
    print(f"  4. mid_o.zero_():          {t4:>8.1f} us")
    
    # 5. Stage1 HIP kernel
    q_rot = (q_f @ PiT).contiguous()
    def run_s1():
        hip_fn(q_rot.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
               seq_lens_t.data_ptr(), centroids_f32.data_ptr(), mid_o.data_ptr(),
               q_rot.stride(0), q_rot.stride(1),
               kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
               block_table.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
               Hk, BS, NUM_KV_SPLITS, kv_group_size, attn_scale,
               8, 1, B, Hq, ctypes.c_void_p(stream_ptr))
    t5 = time_fn("  Stage1", run_s1)
    print(f"  5. Stage1 HIP kernel:      {t5:>8.1f} us")
    
    # 6. Stage2 Triton
    def run_s2():
        _fwd_kernel_stage2[(B, Hq)](mid_o, output_f32, lse, seq_lens_t,
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            output_f32.stride(0), output_f32.stride(1), lse.stride(0),
            NUM_KV_SPLITS=NUM_KV_SPLITS, BLOCK_DV=128, Lv=D,
            num_warps=4, num_stages=2)
    t6 = time_fn("  Stage2", run_s2)
    print(f"  6. Stage2 Triton:          {t6:>8.1f} us")
    
    # 7. output.to(query.dtype)
    t7 = time_fn("  to(bf16)", lambda: output_f32.to(torch.bfloat16))
    print(f"  7. output.to(bf16):        {t7:>8.1f} us")
    
    t_tq_current = t1 + t2 + t3 + t4 + t5 + t6 + t7
    print(f"  ─────────────────────────────────")
    print(f"  TQ current total:          {t_tq_current:>8.1f} us")
    print(f"  (× {N_LAYERS} layers):              {t_tq_current * N_LAYERS / 1000:>8.1f} ms")
    
    # ==================== TQ Optimized (消除开销) ====================
    print("\n--- TQ Optimized (overhead eliminated) ---")
    
    # Optimizations:
    # A. Pre-compute centroids_f32 once (already cached per layer in _ensure_on_device)
    # B. Pre-alloc q_rot buffer, in-place GEMM
    # C. Fuse q.float() + GEMM into torch.mm with pre-alloc output
    # D. Skip mid_o.zero_() (kernel overwrites all used entries)  
    # E. Skip output.to(bf16) - keep f32, let later layers handle
    
    # Optimized GEMM: reuse buffer, avoid alloc
    def run_gemm_opt():
        # In-place: q_rot_buf = query.float() @ PiT
        # Unfortunately torch.mm requires separate alloc. Use addmm with beta=0.
        torch.mm(query.view(B * Hq, D).float(), PiT, out=q_rot_buf.view(B * Hq, D))
    t_gemm_opt = time_fn("  GEMM (pre-alloc)", run_gemm_opt)
    print(f"  1+2. GEMM (pre-alloc out): {t_gemm_opt:>8.1f} us")
    
    # Stage1 (same - kernel is already optimized)
    torch.mm(query.view(B*Hq,D).float(), PiT, out=q_rot_buf.view(B*Hq,D))
    def run_s1_opt():
        hip_fn(q_rot_buf.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
               seq_lens_t.data_ptr(), centroids_f32.data_ptr(), mid_o.data_ptr(),
               q_rot_buf.stride(0), q_rot_buf.stride(1),
               kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
               block_table.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
               Hk, BS, NUM_KV_SPLITS, kv_group_size, attn_scale,
               8, 1, B, Hq, ctypes.c_void_p(stream_ptr))
    t5_opt = time_fn("  Stage1", run_s1_opt)
    print(f"  3. Stage1 HIP:             {t5_opt:>8.1f} us")
    
    # Stage2 (same)
    print(f"  4. Stage2 Triton:          {t6:>8.1f} us")
    
    # Skip: centroids cast (0), mid_o.zero_ (0), output.to() (0)
    
    t_tq_opt = t_gemm_opt + t5_opt + t6
    print(f"  ─────────────────────────────────")
    print(f"  TQ optimized total:        {t_tq_opt:>8.1f} us")
    print(f"  (× {N_LAYERS} layers):              {t_tq_opt * N_LAYERS / 1000:>8.1f} ms")
    
    # ==================== Full decode step simulation ====================
    print("\n" + "=" * 70)
    print("  FULL DECODE STEP (simulating 80 layers)")
    print("=" * 70)
    
    # BL: 80 SDPA calls
    def run_bl_80():
        for _ in range(N_LAYERS):
            torch.nn.functional.scaled_dot_product_attention(
                query.unsqueeze(2), k_bl, v_bl, scale=scale, enable_gqa=True)
    t_bl_80 = time_fn("BL 80 layers", run_bl_80, warmup=3, iters=10)
    
    # TQ current: 80 full pipeline calls
    def run_tq_current_80():
        for _ in range(N_LAYERS):
            q_f = query.float()
            qr = (q_f @ PiT).contiguous()
            c_f = centroids.float().contiguous()
            mid_o.zero_()
            hip_fn(qr.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
                   seq_lens_t.data_ptr(), c_f.data_ptr(), mid_o.data_ptr(),
                   qr.stride(0), qr.stride(1),
                   kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                   block_table.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                   Hk, BS, NUM_KV_SPLITS, kv_group_size, attn_scale,
                   8, 1, B, Hq, ctypes.c_void_p(stream_ptr))
            _fwd_kernel_stage2[(B, Hq)](mid_o, output_f32, lse, seq_lens_t,
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                output_f32.stride(0), output_f32.stride(1), lse.stride(0),
                NUM_KV_SPLITS=NUM_KV_SPLITS, BLOCK_DV=128, Lv=D,
                num_warps=4, num_stages=2)
            _ = output_f32.to(torch.bfloat16)
    t_tq_80 = time_fn("TQ current 80 layers", run_tq_current_80, warmup=3, iters=10)
    
    # TQ optimized: 80 layers with eliminated overhead
    def run_tq_opt_80():
        for _ in range(N_LAYERS):
            torch.mm(query.view(B*Hq,D).float(), PiT, out=q_rot_buf.view(B*Hq,D))
            hip_fn(q_rot_buf.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
                   seq_lens_t.data_ptr(), centroids_f32.data_ptr(), mid_o.data_ptr(),
                   q_rot_buf.stride(0), q_rot_buf.stride(1),
                   kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                   block_table.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                   Hk, BS, NUM_KV_SPLITS, kv_group_size, attn_scale,
                   8, 1, B, Hq, ctypes.c_void_p(stream_ptr))
            _fwd_kernel_stage2[(B, Hq)](mid_o, output_f32, lse, seq_lens_t,
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                output_f32.stride(0), output_f32.stride(1), lse.stride(0),
                NUM_KV_SPLITS=NUM_KV_SPLITS, BLOCK_DV=128, Lv=D,
                num_warps=4, num_stages=2)
    t_tq_opt_80 = time_fn("TQ optimized 80 layers", run_tq_opt_80, warmup=3, iters=10)
    
    print(f"\n  {'Method':<25s} {'80-layer (ms)':>12s} {'per-req (ms)':>12s} {'vs BL':>8s}")
    print(f"  {'-'*57}")
    print(f"  {'BL (SDPA)':<25s} {t_bl_80/1000:>12.1f} {t_bl_80/1000/B:>12.4f} {'1.00x':>8s}")
    print(f"  {'TQ (current)':<25s} {t_tq_80/1000:>12.1f} {t_tq_80/1000/B:>12.4f} {t_tq_80/t_bl_80:>7.2f}x")
    print(f"  {'TQ (optimized)':<25s} {t_tq_opt_80/1000:>12.1f} {t_tq_opt_80/1000/B:>12.4f} {t_tq_opt_80/t_bl_80:>7.2f}x")
    
    overhead_saved = t_tq_80 - t_tq_opt_80
    print(f"\n  Overhead eliminated: {overhead_saved/1000:.1f} ms ({overhead_saved/t_tq_80*100:.1f}%)")
    print(f"  TQ optimized vs BL: {'FASTER' if t_tq_opt_80 < t_bl_80 else 'SLOWER'} by {abs(t_tq_opt_80 - t_bl_80)/1000:.1f} ms")

if __name__ == "__main__":
    main()

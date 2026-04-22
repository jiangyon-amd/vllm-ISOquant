# V62 Fused Stage1+Stage2 Kernel — Design Notes

## Current Baseline Performance (MI355X gfx950)

### Full pipeline: GEMM(q@PiT) + v52_Stage1 + HIP_Stage2_bf16
```
B=4  seq=8192 sp=32: 149 us  (GEMM ~15, S1 ~122, S2 ~12)
B=20 seq=8192 sp=32: 385 us
```

### Stage1-only at different split counts (B=4, seq=8192)
```
splits  tok/sp   S1(us)   S2(us)   total(us)
  16     512    215.9     8.0      224
  32     256    163.0    12.1      175    ← current default
  64     128    153.0    18.8      172
 128      64    146.6    37.9      185
```
Optimal S1+S2 is at splits=32~64. S1 still drops with more splits, but S2 grows linearly.

## Failed Approaches (v61 series) — DO NOT REPEAT

### v61: All splits in one block
- Grid=(B,Hq), 512 threads = 8 wavefronts per block
- Each wavefront handles seq/8 tokens
- **FAILED**: CU underutilization. B=4 → only 256 blocks for 304 CUs.
  Long sequences mean each wavefront runs too long serially.
- Result: 2x slower than baseline.

### v61b: GEMV inline in every split block
- Grid=(B,Hq,splits), same as v52
- Each block does its own q@PiT GEMV before Stage1
- **FAILED**: 32 splits × same (bid,hid) all repeat the full 128×128 GEMV.
  65K blocks each reading 64KB PiT → enormous redundant work.
- Result: 13x slower.

### v61c: Atomic last-split Stage2
- Grid=(B,Hq,splits), Stage1 same as v52
- After Stage1: __threadfence() + atomicAdd to detect last split
- Last split does Stage2 reduce inline
- **FAILED**: __threadfence() on MI355X multi-chiplet costs 100+ us per block.
  Dwarfs the 11 us saved from eliminating Stage2 launch.
- Result: 1.9x slower.

## Key Architectural Constraints (gfx950/MI355X)

1. Wavefront = 64 lanes (v52 treats them as 2×32 "half-warps")
2. __shfl_xor(val, offset, 32) reduces within 32-lane half-warp
3. Max workgroup = 1024 threads
4. Max waves per CU = 32
5. LDS = 64 KB per CU
6. __threadfence() flushes all chiplet caches — VERY expensive
7. atomicAdd on global memory has ~100ns latency per op

## Viable Approach: hipLaunchCooperativeKernel

`hipLaunchCooperativeKernel()` enables grid-wide synchronization via
`cooperative_groups::this_grid().sync()`. This allows:

1. Phase 1: All blocks run Stage1, write to mid_o (exactly as v52)
2. Grid sync (hardware-supported, NOT __threadfence)
3. Phase 2: First (B×Hq) blocks read from mid_o, do Stage2, write bf16 output
4. Remaining blocks skip Phase 2

Benefits:
- Same CU utilization as v52 in Phase 1
- No __threadfence (grid sync is more efficient on CDNA)
- Single kernel launch for both phases
- mid_o stays as a local scratchpad, not a separate buffer allocated by Python

Concerns:
- hipLaunchCooperativeKernel requires all blocks to fit on GPU simultaneously
  Grid = (B, Hq, splits) = (4, 64, 32) = 8192 blocks
  MI355X: 304 CUs × 8 occupancy = 2432 blocks max concurrent
  8192 > 2432 → cooperative launch may FAIL.
- Workaround: reduce splits or batch size to fit

## Alternative: Stage1 Optimization Only

If fusion proves infeasible, optimize v52 Stage1 directly:

### A. Increase BLOCK_KV to 8
Current: BLOCK_KV=4, 2 half-warps → 8 tokens per outer loop iteration
Proposed: BLOCK_KV=8 → 16 tokens per iteration
Expected: ~10-15% reduction in loop overhead

### B. 4 half-warps (128 threads per block)
Current: 64 threads = 2 half-warps = 1 wavefront
Proposed: 128 threads = 4 half-warps = 2 wavefronts
Each processes BLOCK_KV=4 tokens → 16 tokens per iteration
Note: v60 tried this but had compilation issues. Worth retrying.

### C. Better memory access pattern
Current: each KV token loads from scattered cache addresses
Proposed: sort KV tokens by page before processing → coalesced reads

### D. Reduce VGPR for occupancy
v52 uses __launch_bounds__(64, 8) → 8 blocks/CU
With fewer VGPRs, could fit more blocks → better latency hiding.

## Output Requirements

The v62 kernel MUST export:
```c
extern "C" void launch_tq_decode_v62(
    const float* q_rot,           // [B, Hq, D] float32
    const unsigned char* kv_cache,
    const int* block_table,
    const int* seq_lens,
    const float* centroids,       // [16] float32
    float* mid_o,                 // [B, Hq, splits, D+1] scratchpad
    hip_bfloat16* output,         // [B, Hq, D] bf16 output
    int stride_qb, int stride_qh,
    int stride_cb, int stride_cp, int stride_ch,
    int stride_bt,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_heads, int block_size, int num_kv_splits, int kv_group_size,
    float attn_scale,
    int norm_correction,
    int B, int Hq,
    hipStream_t stream
);
```

Correctness: cos_sim > 0.999 vs reference (v52+Stage2).
Performance: beat 149 us for B=4 seq=8192.

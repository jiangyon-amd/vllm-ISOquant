# GEAK Task: V62 Fused Stage1+Stage2 Kernel

## Objective
Optimize `tq_decode_v62_fused.hip` to beat the current 3-kernel baseline.

## Baseline Numbers (v62 starting point = v52 + inline Stage2, 2 internal launches)
```
B= 4 seq= 8192: 206 us  ← PRIMARY TARGET
B=20 seq= 8192: 703 us  ← SECONDARY TARGET
B= 4 seq= 4096: 129 us
B=32 seq=  512: 110 us
```

## What You CAN Do
1. Restructure the kernel to fuse Stage1 and Stage2 into one or fewer launches
2. Optimize Stage1 hot loop (BLOCK_KV, warp count, prefetch, occupancy)
3. Use hipLaunchCooperativeKernel for grid-wide sync (if feasible)
4. Change thread count, block size, loop structure, shared memory layout
5. Change the number of internal splits (currently 32)

## What You CANNOT Do
1. ❌ Use __threadfence() for inter-block sync (costs 100+ us on MI355X)
2. ❌ Put all splits in one block (kills CU utilization for small B)
3. ❌ Inline GEMV in every split block (massive redundant computation)
4. ❌ Change the launcher function signature (must be launch_tq_decode_v62)
5. ❌ Require external scratch buffers beyond mid_o and output

## Build & Test
```bash
cd /shareddata/amd/jiangyon/vllm_turboquant
bash geak_tq_decode/geak_v62_compile.sh
```

Or manually:
```bash
cd geak_tq_decode/hip_kernel
hipcc --offload-arch=gfx950 -shared -fPIC -O3 -ffast-math \
  tq_decode_v62_fused.hip -o tq_decode_v62.so

cd /shareddata/amd/jiangyon/vllm_turboquant
PYTHONPATH=.:$PYTHONPATH TQ_ALLOW_STALE_HIP_SO=1 HIP_VISIBLE_DEVICES=2 \
  python3 geak_tq_decode/hip_kernel/benchmark_v62.py
```

## Success Criteria
- Correctness: ALL cos_sim > 0.999
- Performance: speedup > 1.00x on at least 3 configs
- Stretch: beat 200 us for B=4 seq=8192

## Reference Files
- `tq_decode_v52_no_nt.hip` — proven Stage1 (DO NOT MODIFY)
- `tq_decode_stage2_v2.hip` — proven Stage2 (DO NOT MODIFY)
- `tq_decode_v61_fused_all.hip` — failed: CU underutilization
- `tq_decode_v61c_s1_fused_s2.hip` — failed: __threadfence cost
- `v62_design_notes.md` — detailed analysis and viable approaches

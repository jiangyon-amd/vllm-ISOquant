# GPU Kernel Optimization Checklist — MoE Rotation+Quantization+Sort

## Kernel Info
- **Name**: MoE fused rotation + MXFP4 quantization + sorted scale scatter
- **Tech Stack**: Triton (primary), Gluon (secondary), HIP (experimental)
- **Operation**: x[M,K] @ rotation[RS,RS] → MXFP4 fp4 + E8M0 sorted scales
- **Shape**: M=1..128, K=2048/7168, RS=128, QG=32, topk=8
- **Baseline**: 29.4 µs (2-kernel: gluon_kw8 + moe_mxfp4_sort), 70.5 µs (torch.compile for dense equivalent)
- **Current Best**: 14.5 µs (Triton topk8 constexpr + num_warps=1, M=1 decode)
- **Target**: maximize speedup over 2-kernel baseline

## Status Legend
- `DONE` — Implemented, tested, benchmarked
- `TRIED` — Implemented but no improvement
- `SKIP` — Cannot apply (with reason)

---

## Direction A: Parameter Tuning

### Triton kernel (primary)
| # | Strategy | Status | Result (µs) | Delta | Notes |
|---|----------|--------|-------------|-------|-------|
| A.1 | num_warps sweep (1/2/4/8) | DONE | 14.7 (nw=1) | -3% vs nw=4 | M=1 decode: fewer warps = less sync |
| A.2 | constexpr N_I/TILE_N | DONE | 14.5 | -8% vs runtime | Compiler optimizes div/mod to mul+shift |
| A.3 | MAX_Q tightening | DONE | same | 0% | Already tight enough |
| A.4 | num_stages sweep (1/2/3/4) | TRIED | same | 0% | Latency-bound, no pipeline benefit |
| A.5 | BLOCK_SIZE tuning | SKIP | - | - | Triton kernel has no BLOCK_SIZE param |

### Gluon kernel (secondary)
| # | Strategy | Status | Result (µs) | Delta | Notes |
|---|----------|--------|-------------|-------|-------|
| A.6 | k_width 4→8 | TRIED | +0.1µs | 0% | Latency-bound, MFMA throughput not bottleneck |
| A.7 | constexpr N_I/TILE_N | DONE | 15.0 | -7.6% | Same benefit as Triton |
| A.8 | buffer_load cache hints | PENDING | | | Not yet tried |
| A.9 | warps_per_cta variations | SKIP | - | - | Layout constraint: must match MFMA |

---

## Direction B: Algorithm Transform

| # | Strategy | Status | Result (µs) | Delta | Notes |
|---|----------|--------|-------------|-------|-------|
| B.1 | Kernel fusion: 2-kernel→1-kernel (M>1) | DONE | 19.1 | -35% vs 29.4 | Eliminated sort kernel launch + HBM round-trip |
| B.2 | Scatter via shared memory (no temp row) | TRIED | FAIL | - | Raw/sorted memory overlap corrupted data |
| B.3 | Register caching (input in VGPRs) | N/A | - | - | Triton manages registers automatically |
| B.4 | NT load for residual/streaming data | DONE | implicit | - | Triton generates efficient loads for scatter |
| B.5 | Pre-allocation of output buffers | DONE | 19.1→18.9 | -6µs vs allocating | Eliminates torch.zeros Python overhead |
| B.6 | Tight MAX_Q = f(m_o) | DONE | same | 0% | Fixes correctness for large M, no perf impact |
| B.7 | Direct scatter (bypass gl.convert_layout) | DONE (Gluon) | -1µs | -6% | Workaround for Triton 3.5 data loss bug |

---

## Direction C: Implementation Path Switch

| # | Strategy | Status | Result (µs) | Delta | Notes |
|---|----------|--------|-------------|-------|-------|
| C.1 | Triton topk8 vs Generic | DONE | 14.5 vs 15.7 | -8% | constexpr specialization |
| C.2 | Triton vs Gluon | DONE | 14.5 vs 15.0 | Tri faster | Gluon has convert_layout overhead |
| C.3 | Triton vs HIP MoE | DONE | 14.5 vs 14.6 | ~same | HIP has same perf for topk8 |
| C.4 | Raw HIP (scalar dot) | DONE | 10.5 | -27% | But FP4 correctness issue (MFMA precision) |
| C.5 | All three compared in one benchmark | DONE | Tri best | - | Full sweep: Triton > HIP > Gluon |

---

## Direction D: System-Level

| # | Strategy | Status | Result (µs) | Delta | Notes |
|---|----------|--------|-------------|-------|-------|
| D.1 | Pre-allocate output buffers | DONE | -6µs | -23% | Eliminates torch.empty/zeros per call |
| D.2 | CUDAGraph | PENDING | | | Already registered as custom op, not benchmarked |
| D.3 | Batch multiple rotation calls | PENDING | | | Could process Q/K/V projections in 1 launch |
| D.4 | XCD remapping | PENDING | | | MI355X 8-XCD block distribution |
| D.5 | MoE dispatcher routing fix | DONE | 31.7→15.1 | -2.1x | Found HIP path using wrong 2-kernel fallback |

---

## Direction E: Precision/Datatype

| # | Strategy | Status | Result (µs) | Delta | Notes |
|---|----------|--------|-------------|-------|-------|
| E.1 | v_cvt_scalef32_pk_fp4_f32 (HW quant) | DONE | baseline | - | Already using hardware instruction |
| E.2 | E8M0 scale via bit manipulation | DONE | baseline | - | Already using direct bit ops |
| E.3 | FP8 rotation matrix | PENDING | | | Would halve rotation load bandwidth |
| E.4 | MFMA 16x16x32 for FP8 | PENDING | | | CDNA4 new: 2x throughput |

---

## Anti-Premature-Stop Rules Check

| Rule | Satisfied? | Evidence |
|------|-----------|---------|
| 1: No theoretical ceiling without test | YES | All "latency-bound" claims backed by bandwidth ceiling measurement |
| 2: Parameter failure → algorithm transform | YES | A failed → B.1 fusion gave -35% |
| 3: 2+ implementation paths | YES | Triton, Gluon, HIP all compared |
| 4: 3+ strategies after matching baseline | YES | constexpr, num_warps=1, fusion, pre-alloc after matching |
| 5: Cross-kernel fusion explored | YES | B.1: 2-kernel→1-kernel fusion done |
| 6: All strategies implemented+tested | PARTIAL | D.2-D.4, E.3-E.4 still PENDING |

## Retrospective: What the framework would have caught

Our manual optimization discovered these in order:
1. Triton parameter sweep (A.1-A.4) — found constexpr and num_warps=1
2. Gluon vs Triton comparison (C.2) — confirmed Triton is faster
3. Raw HIP experiment (C.4) — proved 3-5µs framework overhead exists
4. 3-in-1 kernel fusion (B.1) — biggest win at -35%
5. Pre-allocation (D.1) — -23% from eliminating Python overhead
6. Dispatcher bug fix (D.5) — 2x speedup for HIP path

**The framework would have ALSO driven us to explore:**
- D.2: CUDAGraph benchmark (could save 5-6µs launch overhead)
- D.3: Batch rotation (process multiple layers per launch)
- D.4: XCD remapping (better MI355X utilization)
- E.3: FP8 rotation matrix (halve rotation load)
- E.4: FP8 MFMA (2x MFMA throughput)
- A.8: Gluon buffer_load cache hints

These 6 strategies remain PENDING and represent potential further optimization.

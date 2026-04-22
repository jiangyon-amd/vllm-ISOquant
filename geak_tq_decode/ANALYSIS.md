# _tq_decode_stage1 Kernel Analysis for GEAK Optimization

## 1. Profiling Data

### rocprofv3 (MI300X, 4bit_nc, Qwen3-4B)
```
Kernel               Calls   Total_ms  Avg_us   % GPU
_tq_decode_stage1    8,025   2,139     266.6    56.8%   ← TARGET
_fwd_kernel_stage2   8,025     106      13.3     2.8%
_tq_fused_store_mse  7,449      32       4.3     0.8%
```

### Cross-platform comparison
```
                     CUDA H100    ROCm MI300X
avg per call         180.7 us     266.6 us     (1.48x slower)
% of GPU time        46.5%        56.8%
```

## 2. Inner Loop Dissection (per BLOCK_KV=4 tokens, D=128)

### Memory loads per iteration (MSE path, 4-bit keys + 4-bit values):

| Operation          | Loads/iter | Bytes/iter | Description                    |
|-------------------|-----------|-----------|--------------------------------|
| Block table       | 4         | 16        | page_idx lookup                |
| MSE key unpack    | 2×4×128=1024 | 1024   | 2 byte loads × 4 tok × 128 dim |
| Centroid gather   | 4×128=512  | 2048     | indirect load from 16 centroids|
| Norm load         | 2×4=8      | 8        | 2 bytes (fp16) per token       |
| Value unpack (4b) | 4×64=256   | 256      | 1 byte load per 2 values       |
| Scale/zero load   | 4×4=16     | 16       | scale+zero per token           |
| **TOTAL**         | **~1820**  | **~3368** |                                |

### Compute per iteration:
- Key score: 4 × D dot products = 4×128 = 512 FMAs
- Norm correction: 4 × D = 512 FMAs + 4 sqrt + 4 div
- Value accumulate: D FMAs = 128 FMAs
- Softmax: 4 exp + max + rescale

**Memory : Compute ratio ≈ 3368 bytes / 1152 FMAs ≈ 2.9 bytes/FMA**
→ **Memory bandwidth bound** (roofline: MI300X needs < 0.15 bytes/FMA to be compute bound)

## 3. Bottleneck Breakdown

### A. MSE Key Unpack (lines 175-187) — ~40% of iteration
```python
# Current: 2 separate byte loads, combine to 16-bit, shift+mask
mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
mse_raw0 = tl.load(KV_cache_ptr + mse_addrs0, ...)  # byte[i]
mse_raw1 = tl.load(KV_cache_ptr + mse_addrs0 + 1, ...)  # byte[i+1]
raw16 = mse_raw0 | (mse_raw1 << 8)
mse_idx = (raw16 >> mse_bit_shift) & mse_mask
```
**Problem**: 2 loads × BLOCK_KV × D = 1024 loads per iteration.
For 4-bit MSE with D=128: each token has 64 bytes of packed indices.
Loading byte-by-byte means 128 loads per token for key decode alone.

### B. Centroid Gather (lines 190-194) — ~20%
```python
c_vals = tl.load(Centroids_ptr + mse_idx, ...)  # gather from 16 floats
```
**Problem**: Indirect access pattern. But centroids are only 16×4=64 bytes total.
Could be preloaded to registers.

### C. Value Dequant (lines 268-296) — ~25%
```python
# 4-bit: similar byte-level load pattern
val_raw = tl.load(KV_cache_ptr + val_addrs, ...)
v_idx = (val_raw >> vb_shift) & 0xF
# + 4 more loads for scale/zero
```

### D. Norm Load (lines 211-218) — ~5%
```python
n_lo = tl.load(KV_cache_ptr + norm_bases, ...)
n_hi = tl.load(KV_cache_ptr + norm_bases + 1, ...)
vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
```

## 4. Optimization Strategies (Ranked by Expected Impact)

### S1: Vectorized MSE Load (HIGH PRIORITY)
**Current**: Load byte[i] and byte[i+1] separately → combine
**Proposed**: For 4-bit MSE with D=128 → 64 bytes → load as 16 × uint32
Each uint32 contains 8 × 4-bit indices → extract with shift+mask
**Expected**: Reduce MSE key loads from 1024 → 128 (8x fewer loads)

### S2: Preload Centroids (HIGH PRIORITY)
**Current**: `tl.load(Centroids_ptr + mse_idx)` — indirect global load
**Proposed**: Preload all 16 centroids to a register array before the loop.
Use `tl.load(Centroids_ptr + tl.arange(0, 16))` once, then index from registers.
**Expected**: Eliminate all centroid global loads in inner loop

### S3: Vectorized Value Load (MEDIUM)
**Current**: Byte-level loads for 4-bit values
**Proposed**: Load 64 bytes as 16 × uint32, each containing 8 nibbles
**Expected**: Reduce value loads from 256 → 64 (4x fewer)

### S4: Combined Norm + MSE Load (MEDIUM)
**Current**: Separate loads for norms (2 bytes at MSE_BYTES offset)
**Proposed**: Since norms are at a known offset after MSE data,
load MSE + norms together if stride allows
**Expected**: Eliminate 2 extra loads per token

### S5: Increase BLOCK_KV After Memory Optimization (LOW-MEDIUM)
**Current**: BLOCK_KV=4 (chosen because each iteration is memory-heavy)
**Proposed**: After reducing per-iteration memory loads (S1-S4),
try BLOCK_KV=8 or 16 to reduce loop overhead
**Expected**: Better instruction-level parallelism, ~10-20% gain

### S6: Tune num_warps (LOW)
**Current**: num_warps=1
**Proposed**: After reducing register pressure from S1-S3, try num_warps=2
**Expected**: May help hide memory latency with multiple wavefronts

## 5. Correctness Constraints

- Output tensor must match original within atol=1e-3, rtol=1e-3
- Online softmax numerics must be preserved
- Run `python geak_tq_decode/test_correctness.py` after changes

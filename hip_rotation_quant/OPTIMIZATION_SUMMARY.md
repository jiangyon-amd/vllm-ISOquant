# Dense MFMA Rot+Quant Kernel Optimization Summary

## Target Hardware
- AMD MI355X (gfx950, CDNA4)
- ROCm 7.0

## Performance Targets
- M=1: <5µs
- M=4: <6.5µs  
- M=32: <7µs

## Baseline Performance (v1 kernel)
- M=1: 5.9µs
- M=4: 8.1µs
- M=32: 8.4µs

## Optimized Performance (v8 kernel)
- M=1: 4.5µs (24% faster) ✓
- M=4: 6.3µs (22% faster) ✓
- M=32: 6.2µs (26% faster) ✓

## Optimizations Applied

### 1. Vectorized Rotation Load (v8)
- Changed from scalar loads to bf16x8 vectorized loads for rotation matrix
- 256 threads × 8 elements = 2048 elements per iteration
- 8 iterations to load 16384 elements (128×128 rotation matrix)
- Reduces global memory load latency

### 2. Direct FP4 Global Store (v4, v8)
- Eliminated fp4_lds intermediate LDS buffer
- Store FP4 results directly to global memory during quantization
- Saves 2KB LDS and 1 syncthreads barrier

### 3. Optimized LDS Layout
- Rotation matrix stored in row-major format for coalesced access
- Each row is 128 bf16 = 256 bytes = 16 bf16x8 vectors

## Kernel Versions
- v1: Baseline with scalar B loads
- v2: ds_read_tr for B loads (no improvement - MFMA latency dominates)
- v3: 32x32x16 MFMA (abandoned - complex lane mapping)
- v4: Direct FP4 global store
- v5: A prefetching (no additional improvement)
- v6: 2 waves (slower - more MFMA per wave)
- v7: Single wave for M=1 (slower - more work per wave)
- v8: Vectorized rotation load + direct FP4 store (BEST)

## Usage
```cpp
// Use launch_dense_mfma_rot_quant_best() for auto-selection:
// - Uses v8 for raw scales (fastest)
// - Uses v1 for shuffled scales (correct implementation)
launch_dense_mfma_rot_quant_best(fp4_out, scale_out, x, rot, M, K, ...);
```

## Notes
- Shuffled scales require v1 kernel due to cross-block coordination
- v8 achieves 3.2x speedup over Gluon v2 reference (14.5µs)
- All optimizations maintain 100% correctness vs reference

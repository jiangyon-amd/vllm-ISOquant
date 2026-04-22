# MFMA HIP MoE Kernel Debug Status

## 目标
用 raw HIP MFMA 实现 MoE 3-in-1 (rotation + MXFP4 quant + sorted scale scatter)，性能 5-7µs（比 Triton 16µs 快 2-3x）。

## 已确认的事实

### 1. MFMA matmul 本身是正确的
- `test_mfma_only` kernel（`/tmp/test_mfma_only.so`）在 identity rotation 下 100% 正确
- 使用手动 LDS 标量读取加载 B operand（不用 ds_read_tr）
- Output layout 通过 AMD 官方 `amd_matrix_instruction_calculator` 工具确认：
  - `row = (lane_id / 16) * 4 + i`，`col = lane_id % 16`
  - 16x16x32 和 16x16x16 的输出 layout 完全相同

### 2. ds_read_tr16_b64 有问题
- Debug dump 显示只有 lanes 0,4,8,12 有非零 acc 值
- 可能原因：LDS 数据布局不满足 ds_read_tr 的要求（需要特定的 4×16 块排列）
- CK Tile 通过 `LaneGroupTransposeTraits` + `TileDistribution` 管理这个复杂性

### 3. v_cvt scale 需要 `* 0.25f`
- Dense v3 用 `fp4_scale_func(amax) * 0.25f`
- v_cvt 指令期望 `scale = 2^(e8m0-2)`，不是 `2^e8m0`
- 已修复到 v1 和 v2 的源码中

### 4. 量化代码仍有 bug
- Identity rotation 下，MFMA 输出正确（test_mfma_only 验证），但 v2 kernel 的 FP4 = [1,0,0,0,1,0,0,0,...]
- 只有每 4 个 byte 有非零值，其他全 0
- 说明 v_cvt 的 even/odd pairing 或 FP4 store offset 有问题
- scale `* 0.25f` 修复后输出没有变化——说明问题不在 scale

### 5. MFMA 和 Triton tl.dot 的数值不同
- 两者都用 MFMA 但累加顺序不同（不同 K-tiling）
- 无法位精确匹配——FP4 会有 ~12% 差异
- 这是预期行为，不影响模型精度

## 当前文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `mfma_rot_quant_moe_sort.hip` | 编译通过，FP4 错误 | ds_read_tr 版本 + scale*0.25 修复 |
| `mfma_rot_quant_moe_sort_v2.hip` | 编译通过，FP4 错误 | 手动 LDS 读取版本 + scale*0.25 修复 |
| `mfma_rot_quant_moe_sort.so` | 已编译 | v1 的 .so |
| `mfma_rot_quant_moe_sort_v2.so` | 已编译 | v2 的 .so |
| `debug_mfma_dump.hip/.so` | 已编译 | 用于 dump MFMA per-lane acc 值 |
| `debug_mfma_moe.py` | 可用 | 用于对比 Triton vs HIP FP4 输出 |

## 性能（如果正确性修复后）

| M | HIP MFMA | Triton | Speedup |
|---|----------|--------|---------|
| 1 | 5.4 µs | 16.1 µs | 3.0x |
| 4 | 7.0 µs | 16.3 µs | 2.3x |
| 16 | 6.8 µs | 16.3 µs | 2.4x |
| 128 | 7.2 µs | 16.3 µs | 2.3x |

## 下一步

1. **在 v2 kernel 内部添加 raw acc dump**：在量化循环之前把 acc[nt][i] 写到全局 debug buffer
2. **对比 v2 的 acc 值 vs test_mfma_only 的 acc 值**：确认是 MFMA 不同还是量化循环读错了 acc
3. **如果 acc 相同**：逐行检查量化循环的 v_cvt pairing 和 store offset
4. **如果 acc 不同**：检查 v2 编译器优化是否改变了 MFMA 行为

## 参考资源

- AMD 官方 MFMA layout 工具：`/data/jiangyon/amd_matrix_instruction_calculator/`
  ```bash
  python3 matrix_calculator.py -a cdna3 -i v_mfma_f32_16x16x32_fp8_fp8 -D --register-layout
  python3 matrix_calculator.py -a cdna3 -i v_mfma_f32_16x16x32_fp8_fp8 -B --register-layout
  ```
- MFMA 教程：https://salykova.github.io/matrix-cores-cdna
- CK Tile transpose load：`aiter/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_transpose_load_encoding_hip.hpp`
- CK Tile MFMA wrapper：`aiter/3rdparty/composable_kernel/include/ck_tile/core/arch/mma/mfma/mfma_gfx9_hip.hpp`
- Dense v3 参考（已验证的 MFMA 量化）：`mfma_rot_quant_v3.hip`

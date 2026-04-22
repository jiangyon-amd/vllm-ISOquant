# GEAK Iterative Optimization: V60 Branch Elimination

## 概述

从 V52 (8-warp, GEAK 自动优化) 出发, 通过 5 次 GEAK 手动迭代, 找到 V60 作为最优方案。
V60 的核心优化是**分离主循环和尾部处理, 消除热循环中的 bounds-check 分支**。

## 迭代记录

### 迭代 1: V60 — 分离 main/tail loop ✅
- **策略**: 将 `for` 循环拆分为 full iterations (无 bounds check) + tail (有 bounds check)
- **汇编**: 热循环 s_cbranch 从 24 → 9 (消除 bounds check, 保留 page dedup)
- **VGPRs**: 48 → 56 (可接受)
- **结果**: B=4 seq=8K **+11.6%**, 全场景无回退
- **采纳**: ✅ 部署到所有三个 warp 变体

### 迭代 2: V61 — 完全无分支 ❌  
- **策略**: 去除 page_idx dedup 分支 + 编译期 norm_correction
- **VGPRs**: 85 (过高, occupancy 下降)
- **结果**: B=4 +6.6% (弱于 V60), B=32 **-44%** (严重回退)
- **放弃**: 寄存器压力大于分支消除收益

### 迭代 3: V62 — BLOCK_KV=8 ❌
- **策略**: 每迭代处理 8 tokens (原 4), 减半循环次数
- **VGPRs**: 96 (临界)
- **结果**: B=4 -1.1%, B=32 **-129%** (灾难性回退)
- **放弃**: 8 token 的数据缓冲区消耗过多寄存器

### 迭代 4: V63 — Stage1+Stage2 Fused ✅/❌
- **策略**: Grid=(B,Hq) 无 split, 每 block 处理全序列, 直接输出 bf16
- **VGPRs**: 58 (良好)
- **结果**: B=100 seq=128 **+83.5%**, B=4 seq=8K **-153%**
- **结论**: 仅适用于大 B 短 seq 场景, 作为补充路径

### 迭代 5: V64 — 逐 token 流水线 ❌
- **策略**: 每次只处理 1 token, load→compute 交替
- **VGPRs**: 48 (最优)
- **结果**: 全场景 -4% ~ -12%
- **放弃**: 更多 exp() 调用 (2/token vs 1.25/token), 抵消流水线收益

## 最终结果: V60 部署

```
vllm/v1/attention/ops/tq_decode_hip.so        # V60 2-warp (替换 V52)
vllm/v1/attention/ops/tq_decode_4warp_hip.so   # V60 4-warp (替换 V52)
vllm/v1/attention/ops/tq_decode_8warp_hip.so   # V60 8-warp (替换 V52)
```

## 源码

```
geak_tq_decode/hip_kernel/tq_decode_v60_packed_load.hip  # V60 8-warp
geak_tq_decode/hip_kernel/tq_decode_v60_2warp.hip        # V60 2-warp
geak_tq_decode/hip_kernel/tq_decode_v60_4warp.hip        # V60 4-warp
geak_tq_decode/hip_kernel/tq_decode_v61_branchless.hip   # V61 (rejected)
geak_tq_decode/hip_kernel/tq_decode_v62_bkv8.hip         # V62 (rejected)
geak_tq_decode/hip_kernel/tq_decode_v63_fused.hip        # V63 (fused, conditional)
geak_tq_decode/hip_kernel/tq_decode_v64_pipeline.hip     # V64 (rejected)
```

## GEAK 学到的规则

1. **VGPR 56 是 8-warp 的安全上限**: 超过 60 开始影响 occupancy, 超过 80 严重回退
2. **BLOCK_KV=4 是最优**: 4 tokens 的临时数据 (scores, p, values) 正好fit 寄存器文件
3. **分支消除要 VGPR-neutral**: V60 (分离 main/tail) 只加 8 VGPRs, V61 (强制无分支) 加 37
4. **批量 softmax 优于逐 token**: 共享 block_max 和 rescale 减少 exp() 调用
5. **Fusion 有场景限制**: 小 Grid 的 fused kernel 反而因 CU 利用率低而回退

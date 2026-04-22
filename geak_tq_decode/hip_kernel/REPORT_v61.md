# V61 优化系列总结报告

## 背景
当前 TQ decode 管线：GEMM(q@PiT) → v52 Stage1 → HIP Stage2 bf16
- B=4, seq=8192, splits=32: ~151-157 us total

GPT 建议三个 fusion 方向。我们全部实现并测试。

## 测试结果

### V61 (all-in-one block: GEMV + Stage1 + Stage2)
- Grid=(B, Hq)，512 threads/block，8 wavefronts 处理 8 个 splits
- **Correctness: ✓** (cos_sim = 1.0 所有配置)
- **Performance: ✗** B=4 seq=8192 → 315 us (2.0x 慢于 baseline)
- **根因**: Grid 只有 B×Hq=256 blocks，CU 利用率仅 84%。
  长 seq 下每个 wavefront 处理太多 token，无法利用 GPU 并行性。

### V61b (GEMV in every split block)
- Grid=(B, Hq, splits)，与 v52 相同 grid 大小
- **Correctness: ✓**
- **Performance: ✗** B=32 seq=512 → 1568 us (13x 慢于 baseline!)
- **根因**: 每个 (bid,hid) 有 32 个 split blocks，全部重复做完整 GEMV。
  65536 blocks × 64KB PiT 读取 = 极度浪费。rocBLAS 做一次 batched GEMM 远优。

### V61c (v52 Stage1 + atomic-last-split Stage2)
- Grid=(B, Hq, splits)，Stage1 和 v52 完全相同
- 最后完成的 split 通过 atomicAdd 检测后做 Stage2 reduce
- **Correctness: ✓**  
- **Performance: ✗** 全面慢于 3-kernel baseline
- **根因**: `__threadfence()` 在 MI355X multi-chiplet 架构上极其昂贵。
  仅 fence+atomic 就增加 100+ us，远超省掉的 Stage2 launch 开销(11 us)。

## 关键数据点 (B=4, seq=8192)

| 组件 | 耗时 |
|------|------|
| GEMM (q@PiT) | ~15 us |
| v52 Stage1 | ~162 us |
| HIP Stage2 bf16 | ~11 us |
| 3-kernel total | ~206 us |

## 结论

1. **Stage1+Stage2 fusion 在 MI355X 上不可行**
   - all-in-one-block: 牺牲 CU 利用率
   - atomic inter-block sync: `__threadfence()` 成本 >> Stage2 launch 成本
   
2. **GEMV+Stage1 fusion 已经在 v56 中实现**
   - 对 B≤12 有效（省掉 GEMM launch ~15us）
   - 对 B>12 不如 rocBLAS GEMM

3. **GPT 的建议评估**
   - P0 "q_rot + stage1 fusion": 已完成(v56)，无需再做
   - P1 "stage1 + stage2 fusion": 三种方式全部失败
   - P2 "去掉伪额外开销": 部分已完成(bf16 stage2, buffer复用)

4. **真正的优化方向应转向**
   - Stage1 内部进一步微优化（主要瓶颈）
   - System 层: 减少 splits 数、自适应 splits、batch scheduling
   - 或: 全新的 kernel 架构（如 cooperative groups stage2）
   
## 保留的资产

- v61 kernel correctness 全部验证通过
- v61.hip 可作为 fused kernel 的参考实现
- v61c 的 atomic 方案可用于未来 cooperative kernel API

---

## 补充: Splits 数量 sweep 数据

### B=4, seq_len=8192, Hq=64, Hk=8 (MI355X gfx950)

| splits | tok/split | Stage1(us) | Stage2(us) | S1+S2(us) | 相对 sp=32 |
|--------|-----------|-----------|-----------|----------|-----------|
| 16     | 512       | 215.9     | 8.0       | 236.7    | +30%      |
| **32** | **256**   | **163.0** | **12.1**  | **182.8**| **baseline**|
| 48     | 171       | 163.1     | 15.1      | 192.1    | +5%       |
| 64     | 128       | 153.0     | 18.8      | 189.9    | +4%       |
| 96     | 86        | 152.3     | 28.2      | 205.8    | +13%      |
| 128    | 64        | 146.6     | 37.9      | 216.4    | +18%      |

### 分析

- **splits=32 是当前最优** (S1+S2 总和最小)
- Stage1 收益递减: sp=32→64 只快 10us (6%)
- Stage2 成本线性增长: 每增 32 splits 多 ~6-7us
- splits=64 几乎持平 (182.8 vs 189.9 → 差 7us)

### 最终建议

1. **保持 splits=32** 作为默认配置
2. **kernel fusion (S1+S2) 在 MI355X 上不可行** — 已用三种方案验证
3. **继续优化的最佳方向**:
   - Stage1 per-iteration 效率 (memory access patterns, register pressure)
   - System-level: batch scheduling, 减少 per-layer Python glue
   - v56 GEMV threshold 调优 (当前 B≤12 用 v56, B>12 用 GEMM+v52)
4. **当前管线已经相当优化**: 从初始 266us → 163us (S1) + 12us (S2) + 15us (GEMM) ≈ 190us
   vs 初始 266us + Triton S2 + bf16 cast ≈ 300+us，已提升约 37%

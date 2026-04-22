# Review 对齐文档 — v56/v52/v62 认知修正

## 1. v56 的真正问题：不是"该不该融合"，而是"融合位置"

### 问题描述
v56 把 GEMV（q @ PiT）inline 到 Grid=(B, Hq, splits) 的每个 block 里。
同一个 (bid, hid) 有 splits 个 blocks，它们计算的 q_rot 完全相同。

### 冗余量化分析
```
场景           Grid 总量    同一q_rot重复次数   GEMV wall-clock
B=1  sp=32     2,048       32                 ~4us (全并行)
B=4  sp=32     8,192       32                 ~4us (大部分并行)
B=12 sp=32    24,576       32                 ~12us (部分串行)
B=32 sp=32    65,536       32                 ~108us (27 waves × 4us)
```
- B≤4：2048~8192 blocks 在 2432 CU slots 内几乎全并行 → GEMV wall-clock ≈ 4us
- B≥16：blocks 超过 CU 容量 → GEMV 冗余变成 wall-clock 代价
- 因此 `_V56_BATCH_THRESHOLD = 12` 的存在是合理的

### 正确的认知
v56 的 GEMV fusion 在 B≤12 + seq≤2048 范围内是**有效的**（省掉 ~15us GEMM launch），
但因为位置放错（per-split 而非 per-head），天然无法扩展到大 B。

## 2. v56 Stage1 落后于 v52 的优化

v52 在 v56 之后做了多轮 GEAK 优化，以下特性在 v56 中缺失：

| 优化项 | v52 ✓ | v56 ✗ | 预估影响 |
|--------|-------|-------|---------|
| Software pipelining (prefetch prologue + load-after-consume) | ✓ | ✗ | ~10-15% |
| Block table lookup dedup (同 page 复用 block_num) | ✓ | ✗ | ~3-5% |
| Bit shift for page_idx (代替 / 和 %) | ✓ | ✗ | ~2-3% |
| launch_bounds(64, 8) vs (64, 10) | 8 | 10 | 编译器差异 |

**结论**: 如果现在拿 v56 和 GEMM+v52 对比，v56 会因为 Stage1 本体弱而显得"GEMV fusion 收益不大"。
这是不公平的比较——应该先把 v56 的 Stage1 body 更新到 v52 水平再评估。

## 3. 配置/策略/测试漂移一览

### 3a. Benchmark 脚本 Hq 不一致
- `benchmark_tq_decode.py`: Hq=32, Hk=8 (Qwen3-4B)
- `benchmark_v62.py` / `benchmark_v61.py`: Hq=64, Hk=8 (Qwen2.5-72B)
- **影响**: 同一个 (B, seq_len) 在两个脚本里的 grid size 差 2x，数字不可直接比较

### 3b. v56 只覆盖 B≤12 + seq≤2048
- PRIMARY TARGET (B=4, seq=8192) 实际走 GEMM+v52，不走 v56
- v56 从未在长上下文下被评估过
- 当前阈值选择基于 "MI355X sweep"，注释里写 thr=12 时 417 tok/s

### 3c. _resolve_num_kv_splits 的 adaptive 逻辑
- seq≥4096 时直接用 eager_cap (=32)
- seq≥1024 时建议 16
- 但 benchmark_tq_decode.py 用 `allow_adaptive_kv_splits=False`，全部强制 32
- **影响**: 实际 serving 路径的 splits 可能和 benchmark 不一致

### 3d. .so 文件时间戳
- tq_decode_hip.so (v52): 2026-04-22 06:46 ← 和源文件基本同步
- tq_decode_v56_hip.so: 2026-04-21 07:25 ← 比 v52 源文件早 23 小时
- **v56 .so 是用旧版 v56 源码编译的**，没有任何 v52 后续优化

## 4. 修正方案

### Step 1: 修正 benchmark 脚本参数对齐
所有 benchmark 统一支持 `--Hq` 和 `--Hk` 参数，默认使用 Qwen2.5-72B 配置 (Hq=64, Hk=8)。

### Step 2: 创建 v56b = v56 架构 + v52 Stage1 优化
- 保留 v56 的 GEMV inline 架构
- 将 Stage1 body 升级到 v52 水平 (software pipelining, page dedup, bit shift)
- 保持 `launch_bounds(64, 8)` 与 v52 一致

### Step 3: 重新评估 v56b 阈值
- 用统一 benchmark 对比: GEMM+v52 vs v56b
- 在 B={1,4,8,12,16,20,32} × seq={512,2048,4096,8192} 全矩阵 sweep
- 找到真正的 crossover point

### Step 4: 决策
- 如果 v56b 在 B≤N + seq≤M 范围内稳定快于 GEMM+v52，更新阈值
- 如果 v56b 只在极小范围有优势，考虑简化成单一 GEMM+v52 路径

---

## 5. Crossover 实测数据 (v56b with v52 body optimizations)

### Full matrix (Hq=64, Hk=8, splits=32)
```
B\seq |  512  | 2048  | 4096  | 8192  | winner
-----|-------|-------|-------|-------|--------
  1  | -5.6  | -6.7  | -4.0  | -4.6  | v56b
  4  | -5.5  | -6.3  | -4.4  | -2.4  | v56b
  5  |       |       | +10.9 |       | v52
  8  |       | +18.5 | +21.9 |       | v52
 12  |       | +30.1 | +31.9 |       | v52
 32  |+165.8 | +88.0 |       |       | v52
```
(Δ = v56b_time − ref_time, negative = v56b wins)

### Fine crossover sweep (seq=4096)
```
B=1: -4.0us  B=2: -0.6us  B=3: -1.2us  B=4: -4.4us
B=5: +10.9us  B=6: +17.6us  B=7: +19.8us  B=8: +21.9us
```
**Crossover: B=5** (first point where v52 wins)

### 修正建议 (数据驱动)
```
当前:  _V56_BATCH_THRESHOLD = 12, v56_max_seq_len = 2048
修正:  _V56_BATCH_THRESHOLD = 4,  v56_max_seq_len = 0 (disabled)
```

理由:
1. B=5 开始 v52 赢 → 保守设 threshold=4
2. seq_len 不影响 crossover — GEMV cost 与 seq 无关 → 去掉 seq gate
3. v56_max_seq_len=2048 错误地阻止了 B≤4 seq>2048 使用 v56b
   而这恰好是 v56b 最有效的区域 (B=1 seq=8192 省 4.6us)

### Correctness
所有配置 cos_sim = 1.00000 ✓

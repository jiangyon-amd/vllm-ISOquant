# Multi-Warp Adaptive Stage1 Dispatch — 端到端测试报告

## 概述

实现了 Stage1 kernel 的工作负载自适应分发机制，根据 batch size 和 sequence length
自动选择最优的 kernel 变体。

## Kernel 变体

| 变体 | 线程数 | launch_bounds | blocks/CU | 适用场景 |
|------|--------|---------------|-----------|---------|
| v52 (2-warp) | 64 | (64, 8) | 8 | B≥20 短seq，B≥32 任意 |
| 4-warp | 128 | (128, 4) | 4 | B 5-20, seq≥1024 |
| 8-warp | 256 | (256, 2) | 2 | B≤4, seq≥1024 |

## 分发逻辑

```
if seq >= 1024:
    if B <= 4:   → 8-warp  (10% Stage1 提升)
    elif B <= 20: → 4-warp  (1-2% Stage1 提升)
else:
    → v52 2-warp  (最高占用率)
```

## E2E 性能 (GEMM + Stage1 + Stage2, CUDA Events)

| B | seq | v52 (us) | adaptive (us) | kernel | 提升 |
|---|-----|----------|---------------|--------|------|
| 4 | 2048 | 90.3 | 86.1 | 8warp | **-4.7%** |
| 4 | 4096 | 140.0 | 128.6 | 8warp | **-8.2%** |
| **4** | **8192** | **217.2** | **201.7** | **8warp** | **-7.1%** |
| 4 | 9216 | 237.3 | 219.2 | 8warp | **-7.6%** |
| 20 | 2048 | 220.0 | 218.3 | 4warp | -0.7% |
| 20 | 4096 | 387.9 | 382.4 | 4warp | -1.4% |
| 20 | 8192 | 718.8 | 706.2 | 4warp | -1.7% |
| 32 | 128 | 91.6 | 91.8 | v52 | +0.3% |
| 32 | 512 | 123.9 | 123.4 | v52 | -0.4% |
| 100 | 128 | 191.3 | 191.4 | v52 | +0.1% |

## 正确性

所有 (kernel, B, seq) 组合 vs v52 参考输出: **全部 PASS**
max_diff < 5e-4 (bf16 精度)

## 部署文件

```
vllm/v1/attention/ops/tq_decode_hip.so      # v52 2-warp (baseline)
vllm/v1/attention/ops/tq_decode_4warp_hip.so # 4-warp (NEW)
vllm/v1/attention/ops/tq_decode_8warp_hip.so # 8-warp (NEW)
vllm/v1/attention/ops/triton_turboquant_decode.py  # 分发逻辑 (MODIFIED)
```

## 源码

```
geak_tq_decode/hip_kernel/tq_decode_v52_no_nt.hip    # v52 baseline
geak_tq_decode/hip_kernel/tq_decode_v52_4warp.hip     # 4-warp (NEW)
geak_tq_decode/hip_kernel/tq_decode_v52_8warp.hip     # 8-warp (GEAK generated)
```

## 编译命令

```bash
TMPDIR=/shareddata/amd/jiangyon/tmp_hipcc hipcc --offload-arch=gfx950 -shared -fPIC -O3 -ffast-math \
  geak_tq_decode/hip_kernel/tq_decode_v52_no_nt.hip -o vllm/v1/attention/ops/tq_decode_hip.so
TMPDIR=/shareddata/amd/jiangyon/tmp_hipcc hipcc --offload-arch=gfx950 -shared -fPIC -O3 -ffast-math \
  geak_tq_decode/hip_kernel/tq_decode_v52_4warp.hip -o vllm/v1/attention/ops/tq_decode_4warp_hip.so
TMPDIR=/shareddata/amd/jiangyon/tmp_hipcc hipcc --offload-arch=gfx950 -shared -fPIC -O3 -ffast-math \
  geak_tq_decode/hip_kernel/tq_decode_v52_8warp.hip -o vllm/v1/attention/ops/tq_decode_8warp_hip.so
```

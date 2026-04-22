# Qwen3-8B Fused (Gluon v2) 用到的修改文件

在 **fused-rotation-quant-kernel** 分支上，跑 **Qwen3-8B** 且使用 **Fused rotation+quant（Gluon v2）** 时，以下为实际参与该路径的修改文件（自 Felix 合并点 `fbb7a2710` 的 diff 内）。

---

## 一、必用（核心路径）

| 文件 | 作用 |
|------|------|
| `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` | Dense 量化方案：注册 `fused_rotation_mxfp4_quant` 自定义 op，在 forward 里调用 Gluon v2（或 HIP）；`use_fused_rotation_quant`、`use_online_rotation` 逻辑 |
| `vllm/model_executor/layers/quantization/quark/fused_rotation_quant_gluon.py` | **Gluon v2 内核**：Triton/gluon 的 fused rotation + MXFP4 量化，被 `quark_ocp_mx` 的 custom op 调用 |
| `vllm/model_executor/layers/quantization/quark/transform.py` | Quark 0.11 rotation 配置解析、`OrthogonalTransform`、`rotation_weight_loader`、`setup_transform`（推断 `use_online_rotation` 等），加载 rotation 权重 |
| `vllm/model_executor/models/qwen2.py` | Qwen3 复用 Qwen2 结构；此处修改避免对 `input_rotation` 做 stacked mapping 的错误替换（保证 rotation 权重名正确加载） |

以上 4 个文件构成：**模型/权重加载（qwen2 + transform）→ 方案与 fused op（quark_ocp_mx）→ 内核（fused_rotation_quant_gluon）**。

---

## 二、会用到（环境 / 依赖）

| 文件 | 作用 |
|------|------|
| `vllm/envs.py` | 环境变量（如与 AITER/ROCm 相关）；若存在 `VLLM_DISABLE_FUSED_ROT_QUANT` 等也会在这里读 |
| `vllm/_aiter_ops.py` | 被 `quark_ocp_mx` 等引用；8B Dense 的 **fused 路径**主要用 custom op + Gluon v2，aiter 用于后续 FP4 GEMM 等。其中 **MoE 用到的 rotation 参数** 与 8B Dense 无关，但模块本身在加载时会被用到 |

---

## 三、8B Fused Gluon v2 不会用到的修改文件

以下均为自 `fbb7a2710` 的 diff 中与 **8B Dense Fused Gluon v2** 无关的项（仅列类型，不列全 57 个文件名）：

- **MoE 专用**：`qwen3_moe.py`、`fused_rotation_mxfp4_quant_moe_sort.py`、`fused_rotation_mxfp4_quant_moe_gluon.py`、`fused_rotation_quant_gluon_kw8.py`、`quark_moe.py`、`fused_moe_rotation.py`，以及 MoE 相关 HIP/Triton 等
- **Dense 其他内核/可选路径**：`fused_rotation_quant_hip.py`、`fused_rotation_quant_mfma_hip.py`（HIP 路径仅在设 `VLLM_MOE_HIP_MFMA=1` 时用，8B 默认 Gluon v2）
- **脚本/文档/测试**：`QUICKSTART.md`、`scripts/*`、根目录下 `bench_*.py` / `test_*.py`、`hip_rotation_quant/*` 等
- **空文件/旧实现**：`fused_rotation_persistent.py`、`quark_ocp_mx_fused_all.py`、`fused_rotation_mxfp4_quant_moe.py`、`fused_rotation_mxfp4_quant.py` 等

---

## 四、最小集合（仅跑 8B Fused Gluon v2 所需）

若只关心「能跑通 Qwen3-8B Fused (Gluon v2)」的**最小修改集合**，只需保证这 **4 个文件** 与当前实现一致即可：

1. `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`
2. `vllm/model_executor/layers/quantization/quark/fused_rotation_quant_gluon.py`
3. `vllm/model_executor/layers/quantization/quark/transform.py`
4. `vllm/model_executor/models/qwen2.py`

再加上 **envs.py**、**_aiter_ops.py** 后，即与当前 repo 中 8B Fused Gluon v2 的完整依赖一致。

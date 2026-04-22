# quark_ocp_mx.py 中不必要的新增/改动

基于 **fbb7a2710 → fused-rotation-quant-kernel** 的 diff，仅针对 **Qwen3-8B Fused (Gluon v2)** 路径。

**说明**：以下“建议删除/修正”仅对 **fused-rotation-quant-kernel** 分支有效；当前 **fused-rotation-clean** 上该文件实现不同（无 `_fused_rot_quant_v15`，op 名也不同）。

---

## 一、建议删除（死代码）

| 位置 | 代码 | 说明 |
|------|------|------|
| try 块末尾（约第 98 行） | `_fused_rot_quant_v15 = None  # No longer used` | 未引用，可删。注释里写的 “Keep v16 direct reference for non-custom-op paths” 实际指向的是 v15，且 v15 已不再使用。 |

---

## 二、建议修正（注释/命名）

| 位置 | 当前 | 建议 |
|------|------|------|
| 文件顶部注释 | `# Fused rotation + MXFP4 quantization kernel (Gluon v13)` | 改为 **Gluon v2**，与 `fused_rotation_quant_gluon` 一致。 |

---

## 三、条件写法问题（冗余，应避免）

**位置**：`gemm_with_dynamic_quant` 内判断 tuned decode 的分支条件。

**当前写法（不推荐）**：

```python
if rocm_use_aiter_fp4_asm_gemm and M < 32 and M <= 64 and rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(N, K):
```

**问题**：`M < 32` 已经蕴含 `M <= 64`，再写 `M <= 64` 是冗余条件，既无意义又容易让人误解为“有两个独立范围约束”，这种写法不应保留。

**应改为**：

```python
if rocm_use_aiter_fp4_asm_gemm and M < 32 and rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(N, K):
```

---

## 四、魔法数字过多（应用常量替代）

**位置**：`gemm_with_dynamic_quant` 内 “Fused rotation+quant if rotation provided” 整段。

**问题**：`32`、`64`、`256`、`8`、`2`、`255` 等直接写在表达式里，可读性差、难维护，也不利于与 `ocp_mx_utils` 的约定一致。

**建议**：在文件顶部或该 try 块内定义语义化常量，并用常量重写该段；同时去掉冗余的 `M <= 64`。

**1. 常量定义（可放在本文件 try 块前或 ocp_mx_utils）**

```python
# MXFP4 / scale layout (与 OCP_MX_BLOCK_SIZE=32 一致)
SCALE_GROUP_SIZE = 32
FP4_ELEMS_PER_BYTE = 2
SCALE_COL_ALIGN = 8
SCALE_ROW_TILE = 256
DECODE_MAX_M = 32  # decode path: M < DECODE_MAX_M 用 raw scales
```

**2. 改写后的 fused rotation+quant 分支（示意）**

```python
# Fused rotation+quant if rotation provided
if rotation is not None and rotation_size > 0 and _has_fused_triton_rot_quant:
    x_2d = x.reshape(-1, x.shape[-1])
    K_in = x.shape[-1]
    n_scale_groups = K_in // SCALE_GROUP_SIZE
    fp4_cols = K_in // FP4_ELEMS_PER_BYTE

    is_tuned_decode = (
        rocm_use_aiter_fp4_asm_gemm
        and M < DECODE_MAX_M
        and rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(N, K)
    )
    if is_tuned_decode:
        # Tuned decode: raw scales (no shuffle)
        fp4_buf = torch.empty((M, fp4_cols), dtype=torch.uint8, device=x.device)
        sc_buf = torch.empty((M, n_scale_groups), dtype=torch.uint8, device=x.device)
        x_q, x_s = _fused_rot_quant_gluon(
            x_2d, rotation, rotation_size,
            fp4_out=fp4_buf, scales_out=sc_buf, shuffle_scales=False,
        )
    else:
        # Prefill or untuned: shuffled scales
        sn_pad = (n_scale_groups + SCALE_COL_ALIGN - 1) // SCALE_COL_ALIGN * SCALE_COL_ALIGN
        sm_pad = (M + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE * SCALE_ROW_TILE
        fp4_buf = torch.empty((M, fp4_cols), dtype=torch.uint8, device=x.device)
        sc_buf = torch.empty((sm_pad, sn_pad), dtype=torch.uint8, device=x.device)
        x_q, x_s = _fused_rot_quant_gluon(
            x_2d, rotation, rotation_size,
            fp4_out=fp4_buf, scales_out=sc_buf, shuffle_scales=True,
        )
    x_q = x_q.view(torch.float4_e2m1fn_x2)
    x_s = x_s.view(torch.float8_e8m0fnu)
    x_scales = x_s
    x = x_q
```

这样所有魔法数字都通过常量表达，条件也收束为单一的 `is_tuned_decode`，便于维护和审查。

---

## 五、保留（均为 8B Fused Gluon v2 所需）

- **import os**：用于 `os.environ.get("VLLM_DISABLE_FUSED_ROT_QUANT")`。
- **整块 custom op 注册**（`_fused_rot_quant_op` / `_fused_rot_quant_fake` / `direct_register_custom_op` / `_fused_rot_quant`）：Dense fused 入口，必须保留。
- **gemm_with_dynamic_quant** 中 `rotation` / `rotation_size` 及 fused 分支（含 tuned vs shuffled）：fused 路径在此完成 rot+quant 并进 GEMM，必须保留。
- **gemm_with_dynamic_quant_fake** 的 `rotation` / `rotation_size`：CUDAGraph fake 形状，必须保留。
- **weight / weight_scale 的 view**（`w = weight.view(...)` 等）：与 fused 后 fp4 输入兼容，必须保留。
- **`use_fused_rotation_quant`** 及 `VLLM_DISABLE_FUSED_ROT_QUANT` 判断：控制是否走 fused，必须保留。
- **forward 里 fused 分支**（`torch.ops.vllm.gemm_with_dynamic_quant(..., rotation=..., rotation_size=...)`）：8B fused 调用点，必须保留。

---

## 六、环境变量写法（可选加固）

当前：

```python
and not os.environ.get("VLLM_DISABLE_FUSED_ROT_QUANT")
```

- 未设置：`get` 为 `None`，`not None` 为 True → 不禁用，正确。
- 设为 `"1"`：`not "1"` 为 False → 禁用，正确。
- 设为 `"0"`：`not "0"` 为 False → 也会禁用；若希望 “仅 1 禁用”，可改为：

```python
and os.environ.get("VLLM_DISABLE_FUSED_ROT_QUANT", "0") != "1"
```

按需决定是否采用。

---

## 七、小结

- **必删**：`_fused_rot_quant_v15 = None  # No longer used` 一行。
- **建议改**：顶部注释 “Gluon v13” → “Gluon v2”；条件里去掉冗余的 `M <= 64`。
- **建议改**：fused rotation+quant 分支用命名常量替代魔法数字，并用单一 `is_tuned_decode` 表达条件（见第四节）。
- **可选**：统一用 `!= "1"` 判断 `VLLM_DISABLE_FUSED_ROT_QUANT`。

其他 diff 均为 8B Fused Gluon v2 所需，不建议删减。

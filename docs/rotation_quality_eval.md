# Rotation Model Quality Evaluation Guide

本文档记录如何正确评估 rotation 量化模型（MXFP4 + online rotation）的质量，
对比不同 kernel 实现（Separated / MoE-Fused / Both-Fused）是否影响模型输出质量。

## 1. 工具

使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (v0.4.11+)。

```bash
pip install lm-eval[api]   # 需要安装 tenacity 等 API 依赖
```

## 2. 评估任务与参数

| 任务 | 类型 | Few-shot | 主要指标 |
|------|------|----------|----------|
| `arc_challenge` | loglikelihood (多选) | **25-shot** | acc_norm |
| `hellaswag` | loglikelihood (多选) | **10-shot** | acc_norm |
| `gsm8k` | generate_until (生成) | **5-shot** (默认) | exact_match (strict) |

### ⚠️ 常见错误

1. **必须指定 few-shot 数**：默认 0-shot 分数会非常低（arc_challenge 从 0.65 降到 0.50）
2. **loglikelihood 任务必须用 `local-completions`**：`local-chat-completions` 不支持 loglikelihood，会报 `NotImplementedError`
3. **gsm8k 也应使用 `local-completions`**：chat-completions 模式下模型可能输出 thinking 标签导致格式不匹配（strict-match 从 0.88 降到 0.02）
4. **limit 不宜太小**：limit=200 时 stderr≈±0.035，差异难以区分；建议 limit≥500（stderr≈±0.02）

## 3. 启动 vLLM Server

```bash
# 环境变量控制 fused kernel 模式
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1

# --- Separated (baseline) ---
export VLLM_USE_FUSED_ROTATION_QUANT=0
export VLLM_MOE_FUSED_ROTATION=0

# --- MoE-Fused (HIP MFMA 3-in-1 kernel) ---
# export VLLM_USE_FUSED_ROTATION_QUANT=0
# export VLLM_MOE_FUSED_ROTATION=1

# --- Both-Fused (attn Gluon + MoE HIP MFMA) ---
# export VLLM_USE_FUSED_ROTATION_QUANT=1
# export VLLM_MOE_FUSED_ROTATION=1

python -m vllm.entrypoints.openai.api_server \
    --model /path/to/qwen3-30b-mxfp4-trained-r128-vllm-v2 \
    --port 8300 --trust-remote-code \
    --max-model-len 4096 --gpu-memory-utilization 0.35 \
    --host 0.0.0.0 --disable-log-requests
```

### Server 日志验证

启动后检查 server log 确认 kernel 模式正确：

```
# Separated:     attn: fused=0 sep=48 | moe: fused=0 sep=48
# MoE-Fused:     attn: fused=0 sep=48 | moe: fused=48 sep=0
# Both-Fused:    attn: fused=48 sep=0 | moe: fused=48 sep=0
```

关键词：
- `"Using fused Triton"` → attn fused 层数
- `"Using separated"` → attn separated 层数
- `"mode=fused"` → MoE fused 层数
- `"mode=separated"` → MoE separated 层数

## 4. 运行 lm_eval

### 4.1 loglikelihood 任务（arc_challenge, hellaswag）

```bash
python -m lm_eval \
    --model local-completions \
    --model_args "model=MODEL_PATH,base_url=http://localhost:8300/v1/completions,num_concurrent=4,tokenized_requests=False" \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --limit 500 \
    --output_path ./results/arc_challenge \
    --log_samples
```

```bash
python -m lm_eval \
    --model local-completions \
    --model_args "model=MODEL_PATH,base_url=http://localhost:8300/v1/completions,num_concurrent=4,tokenized_requests=False" \
    --tasks hellaswag \
    --num_fewshot 10 \
    --limit 500 \
    --output_path ./results/hellaswag \
    --log_samples
```

### 4.2 generate_until 任务（gsm8k）

```bash
python -m lm_eval \
    --model local-completions \
    --model_args "model=MODEL_PATH,base_url=http://localhost:8300/v1/completions,num_concurrent=4,tokenized_requests=False" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --limit 500 \
    --output_path ./results/gsm8k \
    --log_samples
```

> **注意**：gsm8k 也用 `local-completions` 而非 `local-chat-completions`。
> 后者会导致 thinking 模型的输出格式不匹配，strict-match 接近 0。

### 4.3 一键运行所有任务

```bash
# 可以合并跑，但需注意 num_fewshot 只能全局设置
# 推荐分开跑各 task 以设置不同的 num_fewshot

for task_cfg in "arc_challenge:25" "hellaswag:10" "gsm8k:5"; do
    task=${task_cfg%%:*}
    nshot=${task_cfg##*:}
    echo "Running $task ($nshot-shot)..."
    python -m lm_eval \
        --model local-completions \
        --model_args "model=MODEL_PATH,base_url=http://localhost:8300/v1/completions,num_concurrent=4,tokenized_requests=False" \
        --tasks $task \
        --num_fewshot $nshot \
        --limit 500 \
        --output_path ./results/${task} \
        --log_samples
done
```

## 5. 解读结果

### 输出格式

```
|    Tasks    |Version|Filter|n-shot| Metric |   |Value|   |Stderr|
|-------------|------:|------|-----:|--------|---|----:|---|-----:|
|arc_challenge|      1|none  |    25|acc     |↑  |0.636|±  |0.0215|
|             |       |none  |    25|acc_norm|↑  |0.652|±  |0.0213|
```

### 参考基线（Qwen3-30B-A3B MXFP4, limit=500）

| 指标 | RTN | Separated | MoE-Fused | Both-Fused |
|------|:---:|:---------:|:---------:|:----------:|
| arc_challenge acc_norm (25-shot) | 0.652 | 0.650 | 0.664 | 0.654 |
| hellaswag acc_norm (10-shot) | 0.648 | 0.658 | 0.670 | 0.674 |
| gsm8k strict-match (5-shot) | 0.874 | 0.892 | 0.886 | 0.880 |

### 如何判断质量是否有损

- **看 stderr 列**：limit=500 时 stderr ≈ ±0.02
- **差异 < 2×stderr**：统计上无显著差异
- 上表中所有模式的差异都在 1-2 stderr 内 → **质量无损**

### RTN vs Rotation 模型

RTN 模型和 Rotation 模型是不同的量化方法，分数差异反映的是**量化策略本身**的差异，
不是 kernel 实现的问题。Kernel 评估应关注**同一模型**不同模式（Separated vs Fused）的对比。

## 6. 自动化脚本

完整的自动化评估脚本：`bench_lm_eval_v2.py`

```bash
# 运行（~30min，4 modes × 3 tasks）
nohup python3 bench_lm_eval_v2.py > bench_results/lm_eval_v2.log 2>&1 &
```

该脚本会自动：
1. 按顺序启动 4 种模式的 server
2. 对每个 server 运行 arc_challenge (25-shot) + hellaswag (10-shot) + gsm8k (5-shot)
3. 保存 raw 输出到 `lm_eval_{mode}_{task}_raw.txt`
4. 汇总结果

## 7. Troubleshooting

| 问题 | 原因 | 解决 |
|------|------|------|
| `NotImplementedError: Loglikelihood is not supported for chat completions` | 用了 `local-chat-completions` 跑 arc/hellaswag | 改用 `local-completions` |
| `ModuleNotFoundError: tenacity` | 缺少 API 依赖 | `pip install lm-eval[api]` 或 `pip install tenacity` |
| 分数异常低（acc ≈ 0.50） | 没设 few-shot，默认 0-shot | 加 `--num_fewshot 25` (arc) / `10` (hellaswag) |
| gsm8k strict-match ≈ 0 | chat 模式输出含 thinking 标签 | 用 `local-completions` 而非 chat |
| `UnboundLocalError: outputs` | lm_eval API 连接不稳定 | 增大 timeout，或减少 num_concurrent |
| Server TIMEOUT | cleanup 杀掉了前一个 server 后等待不足 | 增大 cleanup 后的 sleep 时间 |

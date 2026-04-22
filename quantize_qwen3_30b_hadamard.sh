#!/bin/bash
# Quantize Qwen3-30B-A3B with MXFP4 + Hadamard rotation (R1 128 online + R2)
# Using the llm_ptq quantize_quark.py script from Quark examples

set -e

export HF_HOME=/data/jiangyon/hf_cache
export TRANSFORMERS_CACHE=/data/jiangyon/hf_cache

MODEL_DIR="/data/jiangyon/hf_cache/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
ROTATION_CONFIG="/data/jiangyon/vllm_rotation/quark_examples/examples/torch/language_modeling/rotation/qwen3_moe_30b_hadamard_r1_128_online_r2.json"
QUANTIZE_SCRIPT="/data/jiangyon/vllm_rotation/quark_examples/examples/torch/language_modeling/llm_ptq/quantize_quark.py"

OUTPUT_DIR="/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-hadamard-r128"
LOG_FILE="/data/jiangyon/vllm_rotation/$(date +"%Y-%m-%d_%H-%M-%S")_qwen3_30b_hadamard_r1_128_online_r2.log"

export SHORT_CFG_NAME="qwen3_30b_hadamard_r1_128_online_r2"
export EVAL_TASKS="piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard"

echo "=== Quantizing Qwen3-30B-A3B with MXFP4 + Hadamard Rotation ==="
echo "Model: ${MODEL_DIR}"
echo "Rotation Config: ${ROTATION_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Start time: $(date)"

CUDA_VISIBLE_DEVICES=0 python ${QUANTIZE_SCRIPT} \
    --model_dir ${MODEL_DIR} \
    --quant_scheme mxfp4 \
    --quant_algo rotation \
    --quant_algo_config_file rotation ${ROTATION_CONFIG} \
    --eval_batch_size 16 \
    --model_export hf_format \
    --output_dir ${OUTPUT_DIR} \
    --tasks ${EVAL_TASKS} \
    --model_attn_implementation sdpa \
    2>&1 | tee ${LOG_FILE}

echo "=== Done! ==="
echo "End time: $(date)"

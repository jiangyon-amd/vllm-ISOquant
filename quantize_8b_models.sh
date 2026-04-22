#!/bin/bash
set -e

MODEL_DIR="/data/jiangyon/hf_cache/Qwen3-8B"
QUARK_SCRIPT="/data/jiangyon/vllm_rotation/quark_examples/examples/torch/language_modeling/llm_ptq/quantize_quark.py"
ROTATION_CONFIG="/data/jiangyon/vllm_rotation/quark_examples/examples/torch/language_modeling/rotation/qwen3_hadamard_r1_128_online_r2.json"

export HIP_VISIBLE_DEVICES=3
export HF_HOME=/data/jiangyon/.cache/huggingface
export HF_DATASETS_CACHE=/data/jiangyon/.cache/huggingface/datasets
export TRITON_CACHE_DIR=/data/jiangyon/.triton
export TORCH_EXTENSIONS_DIR=/data/jiangyon/.cache/torch_extensions
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR

echo "============================================"
echo "Step 1: Hadamard Rotation MXFP4 Quantization"
echo "============================================"
HADAMARD_OUT="/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-hadamard"
rm -rf "$HADAMARD_OUT"

python "$QUARK_SCRIPT" \
  --model_dir "$MODEL_DIR" \
  --output_dir "$HADAMARD_OUT" \
  --quant_scheme mxfp4 \
  --quant_algo_config_file rotation "$ROTATION_CONFIG" \
  --num_calib_data 128 \
  --seq_len 2048 \
  --model_export hf_format

echo ""
echo "============================================"
echo "Step 2: RTN (Round-to-Nearest) MXFP4 Quantization"
echo "============================================"
RTN_OUT="/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-rtn"
rm -rf "$RTN_OUT"

python "$QUARK_SCRIPT" \
  --model_dir "$MODEL_DIR" \
  --output_dir "$RTN_OUT" \
  --quant_scheme mxfp4 \
  --num_calib_data 128 \
  --seq_len 2048 \
  --model_export hf_format

echo ""
echo "============================================"
echo "Done! Models exported to:"
echo "  Hadamard: $HADAMARD_OUT"
echo "  RTN:      $RTN_OUT"
echo "============================================"

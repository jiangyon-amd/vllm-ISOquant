#!/bin/bash
set -e

WORKSPACE="/data/jiangyon/vllm_rotation"
QUARK_DIR="$WORKSPACE/quark_examples/examples/torch/language_modeling"
MODEL_NAME="Qwen/Qwen3-30B-A3B"
HF_CACHE="/data/jiangyon/hf_cache"

echo "=== Step 1: Download Qwen3-30B-A3B ==="
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('$MODEL_NAME', cache_dir='$HF_CACHE')
print(f'Downloaded to: {path}')
"

echo ""
echo "=== Step 2: RTN Quantization (no rotation) ==="
cd "$QUARK_DIR/llm_ptq"
HIP_VISIBLE_DEVICES=4 python3 quantize_quark.py \
    --model_dir "$MODEL_NAME" \
    --quant_scheme mxfp4 \
    --output_dir "$WORKSPACE/qwen3-30b-mxfp4-rtn" \
    --no_eval \
    2>&1 | tee "$WORKSPACE/bench_results/rtn_quantize.log"

echo ""
echo "=== Step 3: Hadamard Rotation Quantization ==="
cd "$QUARK_DIR/llm_ptq"
HIP_VISIBLE_DEVICES=4 python3 quantize_quark.py \
    --model_dir "$MODEL_NAME" \
    --quant_scheme mxfp4 \
    --quant_algo rotation \
    --quant_algo_config_file rotation ../rotation/qwen3_moe_30b_hadamard_r1_128_online_r2.json \
    --output_dir "$WORKSPACE/qwen3-30b-mxfp4-rotation-vllm" \
    --no_eval \
    2>&1 | tee "$WORKSPACE/bench_results/rotation_quantize.log"

echo ""
echo "=== Done ==="
echo "RTN model: $WORKSPACE/qwen3-30b-mxfp4-rtn"
echo "Rotation model: $WORKSPACE/qwen3-30b-mxfp4-rotation-vllm"
ls -lh "$WORKSPACE/qwen3-30b-mxfp4-rtn/"*.safetensors 2>/dev/null | head -3
ls -lh "$WORKSPACE/qwen3-30b-mxfp4-rotation-vllm/"*.safetensors 2>/dev/null | head -3

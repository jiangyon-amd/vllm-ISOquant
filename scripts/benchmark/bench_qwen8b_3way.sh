#!/bin/bash
# 3-Way Qwen3-8B Dense: RTN / Sep / Attn-Fuse
# CUDAGraph ON, ShareGPT, c=1,4,16,32, 2x warmup
set -e

GPU=${1:-5}
PORT=8201
D=/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
RTN=/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-rtn
ROT=/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-trained-r128-vllm
SA="--port $PORT --trust-remote-code --disable-log-requests --max-model-len 4096 --gpu-memory-utilization 0.40 --host 0.0.0.0 --quantization quark"

kill_all() {
    fuser -k ${PORT}/tcp 2>/dev/null || true
    pkill -9 -f "port.*$PORT" 2>/dev/null || true
    sleep 15
}

bench() {
    local NAME=$1 MODEL=$2
    for i in $(seq 1 60); do sleep 5; curl -s http://localhost:$PORT/health >/dev/null && break; done
    echo "  Ready, CUDAGraph warmup 45s..."
    sleep 45

    # 2x warmup (as requested)
    echo "  Warmup 1/2..."
    HIP_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.cli.main bench serve --backend vllm --base-url http://localhost:$PORT --model "$MODEL" --dataset-name sharegpt --dataset-path "$D" --num-prompts 50 --request-rate 8 >/dev/null 2>&1 || true
    echo "  Warmup 2/2..."
    HIP_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.cli.main bench serve --backend vllm --base-url http://localhost:$PORT --model "$MODEL" --dataset-name sharegpt --dataset-path "$D" --num-prompts 100 --request-rate 16 >/dev/null 2>&1 || true
    echo "  Benchmarking..."

    for RR in 1 4 16 32; do
        OUT=$(HIP_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.cli.main bench serve --backend vllm --base-url http://localhost:$PORT --model "$MODEL" --dataset-name sharegpt --dataset-path "$D" --num-prompts 200 --request-rate $RR 2>&1)
        TPOT=$(echo "$OUT"|grep "Mean TPOT"|awk '{print $NF}')
        TTFT=$(echo "$OUT"|grep "Mean TTFT"|awk '{print $NF}')
        TPUT=$(echo "$OUT"|grep "Output token throughput"|awk '{print $NF}')
        echo "  $NAME c=$RR: TPOT=${TPOT}ms TTFT=${TTFT}ms tput=${TPUT}tok/s"
    done
    kill_all
}

echo "============================================================"
echo "  Qwen3-8B Dense: RTN / Sep / Attn-Fuse"
echo "  $(date) GPU=$GPU CUDAGraph=ON ShareGPT 200 prompts"
echo "============================================================"

# RTN
echo ""; echo "========== RTN =========="
kill_all
HIP_VISIBLE_DEVICES=$GPU VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 \
    python3 -m vllm.entrypoints.openai.api_server --model "$RTN" $SA >/dev/null 2>&1 &
bench "RTN" "$RTN"

# Sep
echo ""; echo "========== Sep =========="
kill_all
HIP_VISIBLE_DEVICES=$GPU VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 VLLM_MOE_FUSED_ROTATION=0 \
    python3 -m vllm.entrypoints.openai.api_server --model "$ROT" $SA >/dev/null 2>&1 &
bench "Sep" "$ROT"

# Attn-Fuse
echo ""; echo "========== Attn-Fuse =========="
kill_all
HIP_VISIBLE_DEVICES=$GPU VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 VLLM_USE_FUSED_ROTATION_QUANT=1 \
    python3 -m vllm.entrypoints.openai.api_server --model "$ROT" $SA >/dev/null 2>&1 &
bench "Attn" "$ROT"

echo ""; echo "Done $(date)"

#!/bin/bash
# 4-Way Serving Benchmark: RTN / Sep / Attn-Fuse / Both-Fuse
# with concurrency c=1,4,16,32 using ShareGPT dataset
# Usage: sudo bash scripts/benchmark/bench_4way_concurrency.sh [GPU_ID]
set -e

GPU=${1:-4}
PORT=8200
D=/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
RTN=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rtn
ROT=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2
MEM=0.25  # conservative memory to avoid OOM
SA="--port $PORT --trust-remote-code --disable-log-requests --max-model-len 4096 --gpu-memory-utilization $MEM --host 0.0.0.0 --quantization quark"

kill_all() {
    # Kill ALL vllm/python processes on this GPU
    fuser -k ${PORT}/tcp 2>/dev/null || true
    pkill -9 -f "api_server.*port.*$PORT" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "resource_tracker" 2>/dev/null || true
    sleep 15
}

bench() {
    local NAME=$1 MODEL=$2

    for i in $(seq 1 90); do
        sleep 5
        curl -s http://localhost:$PORT/health > /dev/null 2>&1 && break
        [ $i -eq 90 ] && { echo "  TIMEOUT"; return 1; }
    done
    echo "  Ready, CUDAGraph warmup 50s..."
    sleep 50

    # Warmup
    HIP_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.cli.main bench serve \
        --backend vllm --base-url http://localhost:$PORT --model "$MODEL" \
        --dataset-name sharegpt --dataset-path "$D" --num-prompts 32 --request-rate 8 > /dev/null 2>&1 || true
    HIP_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.cli.main bench serve \
        --backend vllm --base-url http://localhost:$PORT --model "$MODEL" \
        --dataset-name sharegpt --dataset-path "$D" --num-prompts 64 --request-rate 16 > /dev/null 2>&1 || true
    echo "  Benchmarking..."

    for RR in 1 4 16 32; do
        OUT=$(HIP_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.cli.main bench serve \
            --backend vllm --base-url http://localhost:$PORT --model "$MODEL" \
            --dataset-name sharegpt --dataset-path "$D" --num-prompts 100 --request-rate $RR 2>&1)
        TPOT=$(echo "$OUT"|grep "Mean TPOT"|awk '{print $NF}')
        TTFT=$(echo "$OUT"|grep "Mean TTFT"|awk '{print $NF}')
        TPUT=$(echo "$OUT"|grep "Output token throughput"|awk '{print $NF}')
        echo "  $NAME c=$RR: TPOT=${TPOT}ms TTFT=${TTFT}ms tput=${TPUT}tok/s"
    done

    kill_all
}

echo "============================================================"
echo "  4-Way Concurrency Benchmark"
echo "  $(date) GPU=$GPU CUDAGraph=ON mem=$MEM"
echo "============================================================"

# 1. RTN
echo ""; echo "========== RTN =========="
kill_all
HIP_VISIBLE_DEVICES=$GPU VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 \
    python3 -m vllm.entrypoints.openai.api_server --model "$RTN" $SA > /dev/null 2>&1 &
bench "RTN" "$RTN"

# 2. Sep
echo ""; echo "========== Sep =========="
kill_all
HIP_VISIBLE_DEVICES=$GPU VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 VLLM_MOE_FUSED_ROTATION=0 \
    python3 -m vllm.entrypoints.openai.api_server --model "$ROT" $SA > /dev/null 2>&1 &
bench "Sep" "$ROT"

# 3. Attn-Fuse
echo ""; echo "========== Attn-Fuse =========="
kill_all
HIP_VISIBLE_DEVICES=$GPU VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 VLLM_MOE_FUSED_ROTATION=0 VLLM_USE_FUSED_ROTATION_QUANT=1 \
    python3 -m vllm.entrypoints.openai.api_server --model "$ROT" $SA > /dev/null 2>&1 &
bench "Attn" "$ROT"

# 4. Both-Fuse
echo ""; echo "========== Both-Fuse =========="
kill_all
HIP_VISIBLE_DEVICES=$GPU VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 VLLM_MOE_FUSED_ROTATION=1 VLLM_USE_FUSED_ROTATION_QUANT=1 \
    python3 -m vllm.entrypoints.openai.api_server --model "$ROT" $SA > /dev/null 2>&1 &
bench "Both" "$ROT"

echo ""; echo "Done $(date)"

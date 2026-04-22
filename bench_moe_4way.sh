#!/bin/bash
# MoE 4-Way E2E Benchmark: RTN / Separated / Fused-Triton / HIP-MFMA
# Model: Qwen3-30B-A3B MoE MXFP4
# GPU: MI355X, CUDAGraph PIECEWISE

set -e

GPU=0
PORT=8200
DATASET=/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
RTN_MODEL=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rtn
ROT_MODEL=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2
RESULTS=/data/jiangyon/vllm_rotation/bench_results/moe_4way_$(date +%Y%m%d_%H%M%S).txt

mkdir -p /data/jiangyon/vllm_rotation/bench_results

BASE_ENV="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 HIP_VISIBLE_DEVICES=$GPU"
SERVER_ARGS="--port $PORT --trust-remote-code --disable-log-requests --max-model-len 4096 --gpu-memory-utilization 0.35 --host 0.0.0.0"

run_mode() {
    local MODE=$1
    local MODEL=$2
    local ENV_EXTRA=$3

    echo "========================================"
    echo "Mode: $MODE"
    echo "========================================"

    # Start server
    eval "$BASE_ENV $ENV_EXTRA python3 -m vllm.entrypoints.openai.api_server --model $MODEL $SERVER_ARGS" &
    SERVER_PID=$!
    
    # Wait for ready
    for i in $(seq 1 60); do
        sleep 5
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "Server ready after ${i}x5s"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "TIMEOUT"
            kill $SERVER_PID 2>/dev/null
            return 1
        fi
    done

    # Correctness
    echo "--- Correctness ---"
    R1=$(curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
        \"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"只回答数字：1+1等于几？\"}],
        \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")
    echo "  1+1 = $R1"

    R2=$(curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
        \"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"只回答数字：3*3等于几？\"}],
        \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")
    echo "  3*3 = $R2"

    # Warmup bench (discard)
    echo "--- Warmup ---"
    vllm bench serve --backend vllm --base-url http://localhost:$PORT \
        --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
        --num-prompts 32 --request-rate 8 > /dev/null 2>&1
    echo "  done"

    # Benchmark
    echo "--- Benchmark ---"
    for c in 1 2 4 8 16 32; do
        RESULT=$(vllm bench serve --backend vllm --base-url http://localhost:$PORT \
            --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
            --num-prompts 128 --request-rate $c 2>&1)
        TPOT=$(echo "$RESULT" | grep "Mean TPOT" | awk '{print $NF}')
        TTFT=$(echo "$RESULT" | grep "Mean TTFT" | awk '{print $NF}')
        THROUGHPUT=$(echo "$RESULT" | grep "Output token throughput" | head -1 | awk '{print $NF}')
        echo "  c=$c: TPOT=${TPOT}ms TTFT=${TTFT}ms throughput=${THROUGHPUT}tok/s"
    done

    # Stop server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 3
    echo ""
}

echo "MoE 4-Way Benchmark — $(date)" | tee $RESULTS
echo "GPU: $GPU (MI355X)" | tee -a $RESULTS
echo "" | tee -a $RESULTS

# 1. RTN (no rotation)
run_mode "RTN" "$RTN_MODEL" "" 2>&1 | tee -a $RESULTS

# 2. Separated (rotation matmul in Python, no fused kernel)
run_mode "Separated" "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=0" 2>&1 | tee -a $RESULTS

# 3. Fused (gluon_kw8 + moe_mxfp4_sort, default)
run_mode "Fused-Default" "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=1" 2>&1 | tee -a $RESULTS

# 4. HIP MFMA (new)
run_mode "HIP-MFMA" "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=1 VLLM_MOE_HIP_MFMA=1" 2>&1 | tee -a $RESULTS

echo "Results saved to $RESULTS"

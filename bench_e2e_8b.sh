#!/bin/bash
# Qwen3-8B (Dense) E2E: RTN / Fused-Gluon / Fused-HIP
# Dense kernel 选择: VLLM_MOE_HIP_MFMA=1 → HIP MFMA, 否则 → Gluon v2
set -e

GPU=${1:-1}
PORT=8200
DATASET=/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
RTN_MODEL=/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-rtn
ROT_MODEL=/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-trained-r128-vllm
BASE_ENV="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 HIP_VISIBLE_DEVICES=$GPU"
ARGS="--port $PORT --trust-remote-code --disable-log-requests --max-model-len 4096 --gpu-memory-utilization 0.35 --host 0.0.0.0"

run_mode() {
    local NAME=$1 MODEL=$2 ENV=$3

    echo "=========================================="
    echo " $NAME"
    echo "=========================================="

    eval "$BASE_ENV $ENV python3 -m vllm.entrypoints.openai.api_server --model $MODEL $ARGS" > /tmp/vllm_8b_${NAME}.log 2>&1 &
    local PID=$!

    local ready=0
    for i in $(seq 1 60); do
        sleep 5
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            if [ $ready -eq 0 ]; then
                ready=1
                echo "  Health OK, waiting for CUDAGraph..."
                sleep 30
            fi
            local test=$(curl -s http://localhost:$PORT/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"temperature\":0}" 2>/dev/null)
            if echo "$test" | grep -q "choices"; then
                echo "  Server fully ready"
                break
            fi
        fi
        if [ $i -eq 60 ]; then echo "  TIMEOUT"; kill $PID 2>/dev/null; return 1; fi
    done

    local R=$(curl -s http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"只回答数字：1+1等于几？\"}],\"max_tokens\":16,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
        2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "?")
    echo "  Correctness: 1+1=$R"

    echo "  Warmup..."
    vllm bench serve --backend vllm --base-url http://localhost:$PORT --model $MODEL \
        --dataset-name sharegpt --dataset-path $DATASET --num-prompts 32 --request-rate 8 > /dev/null 2>&1
    vllm bench serve --backend vllm --base-url http://localhost:$PORT --model $MODEL \
        --dataset-name sharegpt --dataset-path $DATASET --num-prompts 64 --request-rate 16 > /dev/null 2>&1
    echo "  Warmup done"

    echo "  TPOT (ms) | c=1   c=2   c=4   c=8   c=16  c=32"
    echo "  ---------|--------------------------------------"
    local line="  $NAME     |"
    for c in 1 2 4 8 16 32; do
        local OUT=$(vllm bench serve --backend vllm --base-url http://localhost:$PORT --model $MODEL \
            --dataset-name sharegpt --dataset-path $DATASET --num-prompts 128 --request-rate $c 2>&1)
        local TPOT=$(echo "$OUT" | grep "Mean TPOT" | awk '{print $NF}')
        line="${line} ${TPOT}"
    done
    echo "$line"

    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    sleep 10
    echo ""
}

echo "Qwen3-8B (Dense) E2E Benchmark $(date)"
echo "  RTN = no rotation; Gluon = fused gluon v2; HIP = fused HIP MFMA"
echo ""

run_mode "RTN"    "$RTN_MODEL" ""
run_mode "Gluon"  "$ROT_MODEL" ""
run_mode "HIP"    "$ROT_MODEL" "VLLM_MOE_HIP_MFMA=1"

echo "=== Done ==="

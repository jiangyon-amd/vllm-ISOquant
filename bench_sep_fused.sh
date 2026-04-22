#!/bin/bash
# Separated + MoE-Fused only (RTN already done: TPOT=7.63ms @ c=1)
cd /data/jiangyon/vllm_rotation

GPU=6
PORT=8300
DATASET=/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
ROT_MODEL=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2
LOGDIR=/data/jiangyon/vllm_rotation/bench_results
mkdir -p $LOGDIR

BASE_ENV="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 HIP_VISIBLE_DEVICES=$GPU"
SERVER_ARGS="--port $PORT --trust-remote-code --disable-log-requests --max-model-len 4096 --gpu-memory-utilization 0.85 --host 0.0.0.0"
ROUNDS=2

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_mode() {
    local MODE=$1 MODEL=$2 ENV_EXTRA=$3
    
    log "======== [$MODE] ========"
    log "  Env: $ENV_EXTRA"
    
    eval "$BASE_ENV $ENV_EXTRA python3 -m vllm.entrypoints.openai.api_server --model $MODEL $SERVER_ARGS" \
        > $LOGDIR/server_verify_${MODE}.log 2>&1 &
    local SPID=$!
    
    for i in $(seq 1 90); do
        sleep 5
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            log "  Server ready (${i}x5s)"
            break
        fi
        if ! kill -0 $SPID 2>/dev/null; then
            log "  Server DIED!"
            tail -20 $LOGDIR/server_verify_${MODE}.log
            return 1
        fi
        if [ $((i % 12)) -eq 0 ]; then log "  Waiting ($i/90)..."; fi
    done
    
    log "  Correctness:"
    R=$(curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
        \"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"只回答数字：1+1等于几？\"}],
        \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")
    log "    1+1=$R"
    
    log "  Warming up..."
    vllm bench serve --backend vllm --base-url http://localhost:$PORT \
        --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
        --num-prompts 64 --request-rate 16 > /dev/null 2>&1 || true
    
    for r in $(seq 1 $ROUNDS); do
        log "  --- Round $r/$ROUNDS ---"
        for c in 1 4 16 32; do
            RESULT=$(vllm bench serve --backend vllm --base-url http://localhost:$PORT \
                --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
                --num-prompts 128 --request-rate $c 2>&1)
            TPOT=$(echo "$RESULT" | grep "Mean TPOT" | awk '{print $NF}')
            TTFT=$(echo "$RESULT" | grep "Mean TTFT" | awk '{print $NF}')
            THR=$(echo "$RESULT" | grep "Output token throughput" | head -1 | awk '{print $NF}')
            log "    c=$c: TPOT=${TPOT}ms  TTFT=${TTFT}ms  tok/s=${THR}"
        done
    done
    
    kill $SPID 2>/dev/null
    wait $SPID 2>/dev/null || true
    sleep 5
    log ""
}

log "=== Sep + MoE-Fused Verify (GPU $GPU) ==="
log "  (RTN baseline: TPOT=7.63ms @ c=1)"
log ""

run_mode "separated" "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=0 VLLM_USE_FUSED_ROTATION_QUANT=0"
run_mode "moe-fused" "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=1 VLLM_USE_FUSED_ROTATION_QUANT=1"

log "=== ALL DONE ==="

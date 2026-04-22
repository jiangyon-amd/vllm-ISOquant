#!/bin/bash
# 3-Way E2E Benchmark: RTN / Separated / MoE-Fused(HIP MFMA)
# 验证 HIP MFMA kernel 在 E2E 中的实际性能
cd /data/jiangyon/vllm_rotation

GPU=3
PORT=8300
DATASET=/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
RTN_MODEL=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rtn
ROT_MODEL=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2
LOGDIR=/data/jiangyon/vllm_rotation/bench_results
LOGFILE=$LOGDIR/verify_3way_$(date +%Y%m%d_%H%M%S).log
mkdir -p $LOGDIR

BASE_ENV="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 HIP_VISIBLE_DEVICES=$GPU"
SERVER_ARGS="--port $PORT --trust-remote-code --disable-log-requests --max-model-len 4096 --gpu-memory-utilization 0.85 --host 0.0.0.0"
ROUNDS=2

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_mode() {
    local MODE=$1 MODEL=$2 ENV_EXTRA=$3
    
    log "======== [$MODE] ========"
    log "  Model: $(basename $MODEL)"
    log "  Env: $ENV_EXTRA"
    
    # Start server
    eval "$BASE_ENV $ENV_EXTRA python3 -m vllm.entrypoints.openai.api_server --model $MODEL $SERVER_ARGS" \
        > $LOGDIR/server_verify_${MODE}.log 2>&1 &
    local SPID=$!
    
    # Wait
    for i in $(seq 1 90); do
        sleep 5
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            log "  Server ready (${i}x5s)"
            break
        fi
        if ! kill -0 $SPID 2>/dev/null; then
            log "  Server DIED!"
            tail -15 $LOGDIR/server_verify_${MODE}.log
            return 1
        fi
        if [ $((i % 12)) -eq 0 ]; then log "  Waiting ($i/90)..."; fi
    done
    
    # Correctness
    log "  Correctness:"
    for Q in "只回答数字：1+1等于几？" "只回答数字：3*3等于几？"; do
        R=$(curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
            \"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"$Q\"}],
            \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}
        }" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")
        log "    $Q -> $R"
    done
    
    # Warmup
    log "  Warming up..."
    vllm bench serve --backend vllm --base-url http://localhost:$PORT \
        --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
        --num-prompts 64 --request-rate 16 > /dev/null 2>&1
    
    # Multi-round benchmark
    declare -A TPOT_SUM
    for c in 1 4 16 32; do TPOT_SUM[$c]=0; done
    
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
            TPOT_SUM[$c]=$(python3 -c "print(${TPOT_SUM[$c]}+${TPOT})")
        done
    done
    
    log "  === Averages ($ROUNDS rounds) ==="
    for c in 1 4 16 32; do
        AVG=$(python3 -c "print(f'{${TPOT_SUM[$c]}/$ROUNDS:.2f}')")
        log "  Avg c=$c: TPOT=${AVG}ms"
    done
    
    # Kill
    kill $SPID 2>/dev/null
    wait $SPID 2>/dev/null || true
    sleep 5
    log ""
}

log "=== 3-Way E2E Verify (GPU $GPU, $ROUNDS rounds) ===" 
log ""

# 1. RTN (baseline, no rotation)
run_mode "rtn" "$RTN_MODEL" ""

# 2. Separated (rotation done in Python, before MoE)
run_mode "separated" "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=0 VLLM_USE_FUSED_ROTATION_QUANT=0"

# 3. MoE-Fused (HIP MFMA kernel inside aiter fused_moe)
run_mode "moe-fused" "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=1 VLLM_USE_FUSED_ROTATION_QUANT=1"

log "=== ALL DONE ==="

#!/bin/bash
# 3-Way E2E Benchmark: RTN / Separated / MoE-Fused(HIP MFMA)
# GPU 4, Port 8400
set -e
cd /data/jiangyon/vllm_rotation

# Fix cache permission issues
export TRITON_CACHE_DIR=/tmp/jiangyon_triton_cache
export VLLM_CACHE_ROOT=/tmp/jiangyon_vllm_cache
export VLLM_NO_USAGE_STATS=1
export XDG_CONFIG_HOME=/tmp/jiangyon_config
mkdir -p $TRITON_CACHE_DIR $VLLM_CACHE_ROOT $XDG_CONFIG_HOME

GPU=4
PORT=8400
DATASET=/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
RTN_MODEL=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rtn
ROT_MODEL=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2
LOGDIR=/data/jiangyon/vllm_rotation/bench_results/3way_gpu4_$(date +%Y%m%d_%H%M%S)
mkdir -p $LOGDIR

ROUNDS=2

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOGDIR/run.log; }

wait_server() {
    local SPID=$1
    for i in $(seq 1 120); do
        sleep 5
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            log "  Server ready (${i}x5s = $((i*5))s)"
            return 0
        fi
        if ! kill -0 $SPID 2>/dev/null; then
            log "  Server DIED!"
            tail -20 $LOGDIR/server_${MODE}.log
            return 1
        fi
        if [ $((i % 12)) -eq 0 ]; then log "  Still waiting ($((i*5))s)..."; fi
    done
    log "  Server timeout!"
    return 1
}

correctness_check() {
    local MODEL=$1
    log "  Correctness check:"
    for Q in "只回答数字：1+1等于几？" "只回答数字：3*3等于几？" "What is the capital of France? Answer in one word."; do
        R=$(curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
            -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"$Q\"}], \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}}" \
            2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")
        log "    Q: $Q"
        log "    A: $R"
    done
}

run_mode() {
    local MODE=$1 MODEL=$2 ENV_EXTRA=$3

    log ""
    log "========================================================"
    log "  MODE: $MODE"
    log "  Model: $(basename $MODEL)"
    log "  Env: $ENV_EXTRA"
    log "========================================================"

    # Start server
    eval "TRITON_CACHE_DIR=/tmp/jiangyon_triton_cache \
        VLLM_ROCM_USE_AITER=1 \
        VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 \
        HIP_VISIBLE_DEVICES=$GPU \
        $ENV_EXTRA \
        python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $PORT \
        --trust-remote-code \
        --disable-log-requests \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.80 \
        --host 0.0.0.0" \
        > $LOGDIR/server_${MODE}.log 2>&1 &
    local SPID=$!
    log "  Server PID: $SPID"

    if ! wait_server $SPID; then
        kill $SPID 2>/dev/null; wait $SPID 2>/dev/null || true
        return 1
    fi

    # Verify config from server log
    log "  Server config check:"
    grep -o "enforce_eager.*" $LOGDIR/server_${MODE}.log | head -1 | while read l; do log "    $l"; done
    grep -o "CUDAGraphMode.*" $LOGDIR/server_${MODE}.log | head -1 | while read l; do log "    $l"; done
    grep "Using separated\|Using fused\|mode=separated\|mode=fused\|HIP MFMA\|fused_rotation" $LOGDIR/server_${MODE}.log | head -3 | while read l; do log "    $l"; done

    # Correctness
    correctness_check $MODEL

    # Warmup (64 requests)
    log "  Warming up (64 reqs, rate=16)..."
    vllm bench serve --backend vllm --base-url http://localhost:$PORT \
        --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
        --num-prompts 64 --request-rate 16 > /dev/null 2>&1 || true
    sleep 2

    # Benchmark
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
            echo "$MODE,$r,$c,$TPOT,$TTFT,$THR" >> $LOGDIR/results.csv
        done
    done

    # Kill server and ALL child processes, wait for GPU memory release
    kill $SPID 2>/dev/null
    sleep 2
    # Kill any remaining children on this port/model
    pgrep -f "api_server.*$PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
    pgrep -f "EngineCore" 2>/dev/null | xargs kill -9 2>/dev/null || true
    wait $SPID 2>/dev/null || true
    sleep 5
    # Force GPU memory release
    HIP_VISIBLE_DEVICES=$GPU python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true
    sleep 15
    # Verify GPU is free
    log "  Waiting for GPU memory release..."
    for w in $(seq 1 12); do
        FREE=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[$GPU\]" | grep "Used" | awk -F: '{print $2}' | tr -d ' ')
        if [ -n "$FREE" ] && [ "$FREE" -lt 1000000000 ]; then
            log "  GPU $GPU memory freed (used: $FREE bytes)"
            break
        fi
        sleep 5
    done
    log "  Server stopped."
}

# Header
log "========================================================"
log "  3-Way E2E Benchmark"
log "  GPU: $GPU | Port: $PORT | Rounds: $ROUNDS"
log "  Log: $LOGDIR"
log "========================================================"

echo "mode,round,concurrency,tpot_ms,ttft_ms,throughput_tps" > $LOGDIR/results.csv

# 1. RTN (baseline)
run_mode "rtn" "$RTN_MODEL" ""

# 2. Separated (rotation via matmul, no fused kernel)
run_mode "separated" "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=0 VLLM_USE_FUSED_ROTATION_QUANT=0"

# 3. MoE-Fused (HIP MFMA kernel)
run_mode "moe-fused" "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=1 VLLM_USE_FUSED_ROTATION_QUANT=1"

log ""
log "========================================================"
log "  ALL DONE — Results: $LOGDIR/results.csv"
log "========================================================"

# Summary
python3 -c "
import csv
from collections import defaultdict

data = defaultdict(lambda: defaultdict(list))
with open('$LOGDIR/results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        mode = row['mode']
        c = int(row['concurrency'])
        tpot = float(row['tpot_ms'])
        data[mode][c].append(tpot)

print()
print('=' * 70)
print('  SUMMARY: Average TPOT (ms) over $ROUNDS rounds')
print('=' * 70)
print(f\"  {'c':>3} | {'RTN':>10} | {'Separated':>10} | {'MoE-Fused':>10} | {'Sep/RTN':>10} | {'Fused/Sep':>10}\")
print('  ' + '-' * 67)
for c in [1, 4, 16, 32]:
    rtn = sum(data['rtn'][c]) / len(data['rtn'][c]) if data['rtn'][c] else 0
    sep = sum(data['separated'][c]) / len(data['separated'][c]) if data['separated'][c] else 0
    fus = sum(data['moe-fused'][c]) / len(data['moe-fused'][c]) if data['moe-fused'][c] else 0
    sr = f'+{(sep/rtn-1)*100:.1f}%' if rtn else 'N/A'
    fs = f'{(fus/sep-1)*100:+.1f}%' if sep else 'N/A'
    print(f'  {c:>3} | {rtn:>9.2f}ms | {sep:>9.2f}ms | {fus:>9.2f}ms | {sr:>10} | {fs:>10}')
print('=' * 70)
" 2>&1 | tee -a $LOGDIR/run.log

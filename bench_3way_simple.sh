#!/bin/bash
# Simplified 3-Way E2E Benchmark
# Runs each mode sequentially with proper cleanup
cd /data/jiangyon/vllm_rotation

export TRITON_CACHE_DIR=/tmp/jiangyon_triton_cache
export VLLM_CACHE_ROOT=/tmp/jiangyon_vllm_cache
export VLLM_NO_USAGE_STATS=1
export XDG_CONFIG_HOME=/tmp/jiangyon_config
mkdir -p $TRITON_CACHE_DIR $VLLM_CACHE_ROOT $XDG_CONFIG_HOME

GPU=4
PORT=8400
DATASET=/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
RTN_MODEL=qwen3-30b-mxfp4-rtn
ROT_MODEL=qwen3-30b-mxfp4-trained-r128-vllm-v2
OUTDIR=bench_results/3way_simple_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUTDIR

ts() { date +%H:%M:%S; }

cleanup_gpu() {
    echo "[$(ts)] Cleaning up GPU..."
    kill -9 $(pgrep -f "api_server|EngineCore" 2>/dev/null) 2>/dev/null
    sleep 20
    echo "[$(ts)] GPU cleanup done"
}

start_server() {
    local MODEL=$1 ENV_EXTRA=$2 MODE=$3
    echo "[$(ts)] Starting server: $MODE"
    eval "VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 HIP_VISIBLE_DEVICES=$GPU $ENV_EXTRA \
        python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL --port $PORT --trust-remote-code --disable-log-requests \
        --max-model-len 4096 --gpu-memory-utilization 0.80 --host 0.0.0.0" \
        > $OUTDIR/server_${MODE}.log 2>&1 &
    echo $!
}

wait_ready() {
    for i in $(seq 1 120); do
        sleep 5
        curl -s http://localhost:$PORT/health > /dev/null 2>&1 && echo "[$(ts)] Server ready ($((i*5))s)" && return 0
        kill -0 $1 2>/dev/null || { echo "[$(ts)] Server died!"; return 1; }
    done
    echo "[$(ts)] Timeout!"; return 1
}

run_bench() {
    local MODE=$1 MODEL=$2

    # Warmup
    echo "[$(ts)] Warmup..."
    vllm bench serve --backend vllm --base-url http://localhost:$PORT \
        --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
        --num-prompts 64 --request-rate 16 > /dev/null 2>&1
    sleep 2

    # Benchmark 2 rounds
    for r in 1 2; do
        echo "[$(ts)] Round $r"
        for c in 1 4 16 32; do
            RESULT_FILE=$OUTDIR/bench_${MODE}_r${r}_c${c}.txt
            vllm bench serve --backend vllm --base-url http://localhost:$PORT \
                --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
                --num-prompts 128 --request-rate $c > $RESULT_FILE 2>&1

            # Parse various possible output formats
            TPOT=$(grep -oP '(?:Mean TPOT|TPOT \(ms\):)\s*[\d.]+' $RESULT_FILE | grep -oP '[\d.]+$' | head -1)
            TTFT=$(grep -oP '(?:Mean TTFT|TTFT \(ms\):)\s*[\d.]+' $RESULT_FILE | grep -oP '[\d.]+$' | head -1)
            THR=$(grep -oP '(?:Output token throughput|Throughput).*?[\d.]+' $RESULT_FILE | grep -oP '[\d.]+$' | head -1)

            # Fallback: search more broadly
            if [ -z "$TPOT" ]; then
                TPOT=$(grep -i "tpot" $RESULT_FILE | grep -oP '[\d.]+' | head -1)
            fi
            if [ -z "$TTFT" ]; then
                TTFT=$(grep -i "ttft" $RESULT_FILE | grep -oP '[\d.]+' | head -1)
            fi

            echo "[$(ts)]   c=$c: TPOT=${TPOT:-N/A}ms TTFT=${TTFT:-N/A}ms thr=${THR:-N/A}"
            echo "$MODE,$r,$c,${TPOT:-0},${TTFT:-0},${THR:-0}" >> $OUTDIR/results.csv
        done
    done
}

correctness_check() {
    local MODEL=$1
    for Q in "只回答数字：1+1等于几？" "What is the capital of France? One word."; do
        A=$(curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
            2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "ERR")
        echo "[$(ts)]   Q: $Q → A: $A"
    done
}

echo "mode,round,concurrency,tpot_ms,ttft_ms,throughput_tps" > $OUTDIR/results.csv

echo "========================================================"
echo " 3-Way E2E Benchmark | GPU $GPU | $(date)"
echo " Output: $OUTDIR"
echo "========================================================"

# ============ 1. RTN ============
echo ""
echo "======== RTN ========"
SPID=$(start_server "$RTN_MODEL" "" "rtn")
wait_ready $SPID || exit 1
correctness_check "$RTN_MODEL"
run_bench "rtn" "$RTN_MODEL"
cleanup_gpu

# ============ 2. Separated ============
echo ""
echo "======== SEPARATED ========"
SPID=$(start_server "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=0 VLLM_USE_FUSED_ROTATION_QUANT=0" "separated")
wait_ready $SPID || exit 1
correctness_check "$ROT_MODEL"
run_bench "separated" "$ROT_MODEL"
cleanup_gpu

# ============ 3. MoE-Fused ============
echo ""
echo "======== MOE-FUSED ========"
SPID=$(start_server "$ROT_MODEL" "VLLM_MOE_FUSED_ROTATION=1 VLLM_USE_FUSED_ROTATION_QUANT=1" "moe-fused")
wait_ready $SPID || exit 1
correctness_check "$ROT_MODEL"
run_bench "moe-fused" "$ROT_MODEL"
cleanup_gpu

# ============ Summary ============
echo ""
echo "========================================================"
echo " SUMMARY"
echo "========================================================"
python3 << 'PYEOF'
import csv
from collections import defaultdict

data = defaultdict(lambda: defaultdict(list))
with open("OUTDIR/results.csv".replace("OUTDIR", "$OUTDIR")) as f:
    for row in csv.DictReader(f):
        mode = row["mode"]
        c = int(row["concurrency"])
        tpot = float(row["tpot_ms"])
        if tpot > 0:
            data[mode][c].append(tpot)

print(f"  {'c':>3} | {'RTN':>10} | {'Separated':>10} | {'MoE-Fused':>10} | {'Sep/RTN':>10} | {'Fused/Sep':>10}")
print("  " + "-" * 67)
for c in [1, 4, 16, 32]:
    rtn = sum(data["rtn"][c]) / len(data["rtn"][c]) if data["rtn"][c] else 0
    sep = sum(data["separated"][c]) / len(data["separated"][c]) if data["separated"][c] else 0
    fus = sum(data["moe-fused"][c]) / len(data["moe-fused"][c]) if data["moe-fused"][c] else 0
    sr = f"+{(sep/rtn-1)*100:.1f}%" if rtn > 0 else "N/A"
    fs = f"{(fus/sep-1)*100:+.1f}%" if sep > 0 else "N/A"
    print(f"  {c:>3} | {rtn:>9.2f}ms | {sep:>9.2f}ms | {fus:>9.2f}ms | {sr:>10} | {fs:>10}")
PYEOF
echo "========================================================"
echo " Done: $OUTDIR/results.csv"
echo "========================================================"

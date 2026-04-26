#!/bin/bash
###############################################################################
#  3-Way E2E Benchmark: Baseline vs Aditi-v3 vs Ours (v136)
#  Model: Qwen2.5-72B-Instruct, TP=1
#  Workload: Input=8192, Output=1024, MaxConcurrency=32
#
#  Key insight: Python resolves `vllm` from CWD (sys.path[0]='').
#  So we `cd` into the correct workspace to switch codebases.
#
#  Configuration A: Baseline (Triton-only decode, no HIP kernel)
#     → CWD = our repo, TQ_DISABLE_HIP_SO=1 forces Triton path
#  Configuration B: Aditi v3 (unified Triton attention kernel)
#     → CWD = Aditi's repo, VLLM_TQ_DECODE_V3=1, TQ_DISABLE_HIP_SO=1
#  Configuration C: Ours (v136 unified HIP kernel, bf16 Q)
#     → CWD = our repo, default HIP path
###############################################################################
set -uo pipefail

GPU=${1:-0}
PORT=8300
MODEL="/shareddata/amd/jiangyon/models/Qwen2.5-72B-Instruct"
MAX_MODEL_LEN=16384
GPU_UTIL=0.90
INPUT_LEN=8192
OUTPUT_LEN=1024
MAX_CONCURRENCY=32
NUM_PROMPTS=100
SEED=42

OUR_REPO="/shareddata/amd/jiangyon/vllm_turboquant"
ADITI_REPO="/shareddata/amd/jiangyon/vllm_aditi"

RESULT_DIR="${OUR_REPO}/profiling/bench_3way_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

export HIP_VISIBLE_DEVICES=$GPU
export TQ_ALLOW_STALE_HIP_SO=1

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  3-Way E2E: Baseline vs Aditi-v3 vs Ours-v136          ║"
echo "║  Model: Qwen2.5-72B-Instruct, TP=1, GPU=$GPU              ║"
echo "║  Input=$INPUT_LEN, Output=$OUTPUT_LEN, MaxC=$MAX_CONCURRENCY                 ║"
echo "║  Results: $RESULT_DIR"
echo "╚══════════════════════════════════════════════════════════╝"

stop_server() {
    pkill -9 -f "api_server.*${PORT}" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    sleep 10
}

start_server() {
    local LABEL=$1
    local WORK_DIR=$2
    shift 2
    local ENV_VARS=("$@")
    local LOGFILE="${RESULT_DIR}/${LABEL}_server.log"

    stop_server

    echo ""
    echo "┌──────────────────────────────────────────────────────┐"
    echo "│  Starting: $LABEL"
    echo "│  CWD: $WORK_DIR"
    echo "│  ENV: ${ENV_VARS[*]:-<none>}"
    echo "└──────────────────────────────────────────────────────┘"

    # Launch server from the correct working directory
    (
        cd "$WORK_DIR"
        # Export environment variables
        for ev in "${ENV_VARS[@]}"; do
            export "$ev"
        done
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --kv-cache-dtype turboquant_4bit_nc \
            --gpu-memory-utilization "$GPU_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --enforce-eager \
            --no-enable-log-requests \
            --port "$PORT" \
            > "$LOGFILE" 2>&1
    ) &
    SERVER_PID=$!
    echo "  PID: $SERVER_PID, log: $LOGFILE"

    # Wait for ready
    local MAX_WAIT=600
    local WAITED=0
    while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "  ERROR: Server failed to start in ${MAX_WAIT}s"
            tail -30 "$LOGFILE"
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
            return 1
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "  ERROR: Server process died"
            tail -30 "$LOGFILE"
            return 1
        fi
        echo "  Waiting... (${WAITED}s)"
    done
    echo "  ✓ Server ready after ${WAITED}s"

    # Warm-up: run a few requests to get past any Triton compilation / JIT
    echo "  Running warm-up (5 prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --model "$MODEL" \
        --base-url "http://localhost:$PORT" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len 32 \
        --num-prompts 5 \
        --request-rate inf \
        --max-concurrency 2 \
        --seed 0 \
        > "${RESULT_DIR}/${LABEL}_warmup.log" 2>&1
    echo "  ✓ Warm-up done"
}

run_benchmark() {
    local LABEL=$1
    local LOGFILE="${RESULT_DIR}/${LABEL}_bench.log"

    echo "  Running benchmark (${NUM_PROMPTS} prompts, max_concurrency=${MAX_CONCURRENCY})..."

    python3 -m vllm.entrypoints.cli.main bench serve \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --model "$MODEL" \
        --base-url "http://localhost:$PORT" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --seed "$SEED" \
        2>&1 | tee "$LOGFILE"

    echo "  ✓ Benchmark complete. Results: $LOGFILE"
}

# ═══════════════════════════════════════════════════════════
# Config A: Baseline (Triton path, no HIP kernel)
# ═══════════════════════════════════════════════════════════
echo ""
echo "═══ A: BASELINE (Triton Stage1, no HIP) ═══"
start_server "A_baseline" "$OUR_REPO" "TQ_DISABLE_HIP_SO=1"
if [ $? -eq 0 ]; then
    run_benchmark "A_baseline"
fi
stop_server

# ═══════════════════════════════════════════════════════════
# Config B: Aditi v3 (unified Triton attention, from her repo)
# ═══════════════════════════════════════════════════════════
echo ""
echo "═══ B: ADITI v3 (Unified Triton Attention) ═══"
start_server "B_aditi_v3" "$ADITI_REPO" "VLLM_TQ_DECODE_V3=1" "TQ_DISABLE_HIP_SO=1"
if [ $? -eq 0 ]; then
    run_benchmark "B_aditi_v3"
fi
stop_server

# ═══════════════════════════════════════════════════════════
# Config C: Ours (v136 unified HIP kernel)
# ═══════════════════════════════════════════════════════════
echo ""
echo "═══ C: OURS v136 (Unified HIP Kernel, bf16 Q) ═══"
start_server "C_ours_v136" "$OUR_REPO" "PLACEHOLDER=1"
if [ $? -eq 0 ]; then
    run_benchmark "C_ours_v136"
fi
stop_server

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    RESULTS SUMMARY                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Results directory: $RESULT_DIR"
echo ""

for LABEL in A_baseline B_aditi_v3 C_ours_v136; do
    BENCHLOG="${RESULT_DIR}/${LABEL}_bench.log"
    if [ -f "$BENCHLOG" ]; then
        echo "═══════════════════════════════════════════"
        echo "  $LABEL"
        echo "═══════════════════════════════════════════"
        # Extract key metrics
        grep -iE "request throughput|output token throughput|total token throughput|successful|TPOT|ITL|TTFT|e2e" "$BENCHLOG" 2>/dev/null | head -20
        echo ""
    else
        echo "═══ $LABEL ═══ (no results)"
        echo ""
    fi
done

echo "Benchmark complete!"

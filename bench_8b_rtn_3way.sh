#!/bin/bash
# E2E benchmark: 8B RTN model, 3 modes (separated, gluon_fused, geak_opt)
# Concurrency: 1, 2, 4, 8, 16, 32 on GPU4

set -e

MODEL="/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-rtn"
DATASET="/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
PORT=8100
GPU_ID=3  # GPU4 (0-indexed)
NUM_PROMPTS=100
RESULTS_DIR="/data/jiangyon/vllm_rotation/bench_results/8b_rtn_3way"
mkdir -p "$RESULTS_DIR"

export HF_HOME=/data/jiangyon/.cache/huggingface
export HF_DATASETS_CACHE=/data/jiangyon/.cache/huggingface/datasets
export TRITON_CACHE_DIR=/data/jiangyon/.triton

wait_server() {
    echo "  Waiting for server to be ready..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
            echo "  Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server failed to start within 120s"
    return 1
}

kill_server() {
    pkill -f "vllm.entrypoints.openai.api_server.*$PORT" 2>/dev/null || true
    sleep 3
    pkill -9 -f "vllm.entrypoints.openai.api_server.*$PORT" 2>/dev/null || true
    sleep 2
}

run_benchmark() {
    local mode=$1
    local env_vars=$2
    local concurrencies="1 2 4 8 16 32"

    echo ""
    echo "========================================================"
    echo "  Mode: $mode"
    echo "  Env:  $env_vars"
    echo "========================================================"

    kill_server

    # Start server
    echo "  Starting vLLM server..."
    eval "export $env_vars"
    HIP_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port $PORT \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16 \
        --disable-log-requests \
        > "$RESULTS_DIR/${mode}_server.log" 2>&1 &
    SERVER_PID=$!

    if ! wait_server; then
        echo "  Server failed, skipping $mode"
        kill_server
        return
    fi

    # Run benchmarks at each concurrency
    for conc in $concurrencies; do
        echo "  Benchmarking concurrency=$conc ..."
        python -m vllm.entrypoints.cli.main bench serve \
            --backend vllm \
            --model "$MODEL" \
            --dataset-name sharegpt \
            --dataset-path "$DATASET" \
            --num-prompts $NUM_PROMPTS \
            --request-rate inf \
            --max-concurrency $conc \
            --endpoint /v1/completions \
            --port $PORT \
            --save-result \
            --result-dir "$RESULTS_DIR" \
            --result-filename "${mode}_c${conc}.json" \
            2>&1 | tee "$RESULTS_DIR/${mode}_c${conc}.log" | grep -E "Throughput|Request throughput|Output token throughput|Mean TTFT|Mean ITL|Median"
    done

    kill_server
    echo "  Done: $mode"
}

echo "============================================================"
echo "8B RTN MXFP4 E2E Benchmark — 3-way comparison"
echo "Model: $MODEL"
echo "GPU: GPU4 (HIP_VISIBLE_DEVICES=$GPU_ID)"
echo "Concurrency: 1, 2, 4, 8, 16, 32"
echo "============================================================"

# Mode 1: Separated (default - gluon rot+quant, then separate moe_mxfp4_sort)
run_benchmark "separated" "VLLM_MOE_GLUON_SORTED_SCALE_FUSION=0 VLLM_MOE_GLUON_KW8=1"

# Mode 2: Gluon Fused (gluon rot+quant+sort in one kernel)
run_benchmark "gluon_fused" "VLLM_MOE_GLUON_SORTED_SCALE_FUSION=1 VLLM_MOE_GLUON_KW8=1"

# Mode 3: GEAK Optimized (same as gluon_fused, but with GEAK kernel opts already in source)
# The GEAK optimizations are already applied to the source code:
# - Adaptive MAX_Q, k_width=4, buffer_store, num_stages=3
# This is the same as gluon_fused since opts are in the shared source file
run_benchmark "geak_opt" "VLLM_MOE_GLUON_SORTED_SCALE_FUSION=1 VLLM_MOE_GLUON_KW8=1"

echo ""
echo "============================================================"
echo "All benchmarks complete! Results in: $RESULTS_DIR"
echo "============================================================"

# Print summary
echo ""
echo "=== SUMMARY ==="
for mode in separated gluon_fused geak_opt; do
    echo ""
    echo "--- $mode ---"
    for conc in 1 2 4 8 16 32; do
        f="$RESULTS_DIR/${mode}_c${conc}.json"
        if [ -f "$f" ]; then
            python3 -c "
import json
d = json.load(open('$f'))
tput = d.get('request_throughput', d.get('completed', 0))
otput = d.get('output_throughput', 0)
ttft = d.get('mean_ttft_ms', 0)
itl = d.get('mean_itl_ms', 0)
print(f'  c={int(\"$conc\"):2d}: req_tput={tput:.2f} req/s  out_tput={otput:.1f} tok/s  TTFT={ttft:.1f}ms  ITL={itl:.2f}ms')
" 2>/dev/null || echo "  c=$conc: (no result)"
        fi
    done
done

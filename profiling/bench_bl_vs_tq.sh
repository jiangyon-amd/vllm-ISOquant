#!/bin/bash
###############################################################################
#  BL vs TQ-v136 Sweep: Vary output_len = {128, 256, 512, 1024, 2048, 4096}
#  Model: Qwen2.5-72B-Instruct, TP=1, Input=8192, MaxConcurrency=32
#  
#  BL  = Standard bf16 KV cache (--kv-cache-dtype auto)
#  TQ  = TurboQuant 4-bit KV cache + HIP MFMA v136 kernel
###############################################################################
set -uo pipefail

GPU=${1:-4}
PORT=8300
MODEL="/shareddata/amd/jiangyon/models/Qwen2.5-72B-Instruct"
OUR_REPO="/shareddata/amd/jiangyon/vllm_turboquant"
MAX_MODEL_LEN=16384
GPU_UTIL=0.90
INPUT_LEN=8192
MAX_CONCURRENCY=32
NUM_PROMPTS=100
SEED=42

OUTPUT_LENS=(128 256 512 1024 2048 4096)

RESULT_DIR="${OUR_REPO}/profiling/bench_bl_vs_tq_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

export HIP_VISIBLE_DEVICES=$GPU
export TQ_ALLOW_STALE_HIP_SO=1

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BL vs TQ-v136 Sweep: Output={128..4096}                   ║"
echo "║  Model: Qwen2.5-72B-Instruct, TP=1, GPU=$GPU                  ║"
echo "║  Input=$INPUT_LEN, MaxC=$MAX_CONCURRENCY, ${NUM_PROMPTS} prompts each        ║"
echo "║  Results: $RESULT_DIR"
echo "╚══════════════════════════════════════════════════════════════╝"

stop_server() {
    pkill -9 -f "api_server.*${PORT}" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    sleep 15
}

start_server() {
    local LABEL=$1
    local KV_DTYPE=$2
    local EXTRA_ENVS=$3
    local LOGFILE="${RESULT_DIR}/${LABEL}_server.log"

    stop_server

    echo ""
    echo "┌──────────────────────────────────────────────────────┐"
    echo "│  Starting: $LABEL (kv_dtype=$KV_DTYPE)"
    echo "└──────────────────────────────────────────────────────┘"

    (
        cd "$OUR_REPO"
        eval "$EXTRA_ENVS"
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --kv-cache-dtype "$KV_DTYPE" \
            --gpu-memory-utilization "$GPU_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --enforce-eager \
            --no-enable-log-requests \
            --port "$PORT" \
            > "$LOGFILE" 2>&1
    ) &
    SERVER_PID=$!
    echo "  PID: $SERVER_PID"

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
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "  Waiting... (${WAITED}s)"
        fi
    done
    echo "  ✓ Server ready after ${WAITED}s"

    # Warm-up
    echo "  Running warm-up..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --model "$MODEL" \
        --base-url "http://localhost:$PORT" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len 32 \
        --num-prompts 3 \
        --request-rate inf \
        --max-concurrency 2 \
        --seed 0 \
        > "${RESULT_DIR}/${LABEL}_warmup.log" 2>&1
    echo "  ✓ Warm-up done"
}

run_benchmark() {
    local LABEL=$1
    local OUTPUT_LEN=$2
    local LOGFILE="${RESULT_DIR}/${LABEL}_out${OUTPUT_LEN}_bench.log"

    echo "  Benchmarking output_len=$OUTPUT_LEN (${NUM_PROMPTS} prompts)..."

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

    echo "  ✓ Done: $LOGFILE"
}

# ═══════════════════════════════════════════════════════════
# Phase 1: Baseline (standard bf16 KV cache)
# ═══════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  PHASE 1: BASELINE (kv_cache_dtype=auto)"
echo "═══════════════════════════════════════════════════════"
start_server "BL" "auto" ""
if [ $? -eq 0 ]; then
    for OLEN in "${OUTPUT_LENS[@]}"; do
        run_benchmark "BL" "$OLEN"
    done
fi
stop_server

# ═══════════════════════════════════════════════════════════
# Phase 2: TQ v136 (TurboQuant HIP MFMA kernel)
# ═══════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  PHASE 2: TQ v136 (kv_cache_dtype=turboquant_4bit_nc)"
echo "═══════════════════════════════════════════════════════"
start_server "TQ" "turboquant_4bit_nc" ""
if [ $? -eq 0 ]; then
    for OLEN in "${OUTPUT_LENS[@]}"; do
        run_benchmark "TQ" "$OLEN"
    done
fi
stop_server

# ═══════════════════════════════════════════════════════════
# Generate Summary Table
# ═══════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  GENERATING RESULTS TABLE"
echo "═══════════════════════════════════════════════════════"

python3 << PYEOF
import re, os

result_dir = "$RESULT_DIR"
output_lens = [128, 256, 512, 1024, 2048, 4096]

def extract(logfile, metric):
    """Extract metric value from benchmark log."""
    try:
        with open(logfile) as f:
            for line in f:
                if metric in line:
                    # Extract the number at end of line
                    m = re.search(r'([\d.]+)\s*$', line.strip())
                    if m:
                        return float(m.group(1))
    except FileNotFoundError:
        pass
    return None

print()
header = (
    f"| {'Output Len':>10} "
    f"| {'BL Mean TTFT':>15} "
    f"| {'TQ Mean TTFT':>15} "
    f"| {'BL Mean TPOT':>15} "
    f"| {'TQ Mean TPOT':>15} "
    f"| {'BL Throughput':>15} "
    f"| {'TQ Throughput':>15} "
    f"| {'TQ/BL %':>10} |"
)
sep = "|".join(["-" * 12] + ["-" * 17] * 6 + ["-" * 12]) 
sep = "| " + sep + " |"

print(header)
print(sep)

rows_md = []
for olen in output_lens:
    bl_log = os.path.join(result_dir, f"BL_out{olen}_bench.log")
    tq_log = os.path.join(result_dir, f"TQ_out{olen}_bench.log")

    bl_ttft = extract(bl_log, "Mean TTFT")
    tq_ttft = extract(tq_log, "Mean TTFT")
    bl_tpot = extract(bl_log, "Mean TPOT")
    tq_tpot = extract(tq_log, "Mean TPOT")
    bl_tps  = extract(bl_log, "Output token throughput")
    tq_tps  = extract(tq_log, "Output token throughput")

    ratio = f"{tq_tps/bl_tps*100:.2f}%" if bl_tps and tq_tps else "N/A"

    def fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"

    print(
        f"| {olen:>10} "
        f"| {fmt(bl_ttft):>15} "
        f"| {fmt(tq_ttft):>15} "
        f"| {fmt(bl_tpot):>15} "
        f"| {fmt(tq_tpot):>15} "
        f"| {fmt(bl_tps):>15} "
        f"| {fmt(tq_tps):>15} "
        f"| {ratio:>10} |"
    )

    rows_md.append({
        "olen": olen, "bl_ttft": bl_ttft, "tq_ttft": tq_ttft,
        "bl_tpot": bl_tpot, "tq_tpot": tq_tpot,
        "bl_tps": bl_tps, "tq_tps": tq_tps, "ratio": ratio,
    })

# Save markdown table
with open(os.path.join(result_dir, "RESULTS.md"), "w") as f:
    f.write("# BL vs TQ-v136 Benchmark Results\n\n")
    f.write("- **Model**: Qwen2.5-72B-Instruct, TP=1\n")
    f.write("- **GPU**: MI300X (gfx950)\n")
    f.write("- **Input**: 8192 tokens, MaxConcurrency=32, 100 prompts\n")
    f.write("- **BL**: Standard bf16 KV cache (kv_cache_dtype=auto)\n")
    f.write("- **TQ**: TurboQuant 4-bit + HIP MFMA v136 kernel\n\n")
    f.write("| Output Len | BL Mean TTFT (ms) | TQ Mean TTFT (ms) | BL Mean TPOT (ms) | TQ Mean TPOT (ms) | BL Throughput (tok/s) | TQ Throughput (tok/s) | TQ/BL % |\n")
    f.write("| ---------- | ----------------- | ------------------ | ------------------ | ------------------ | --------------------- | --------------------- | ------- |\n")
    for r in rows_md:
        def fmt(v): return f"{v:.2f}" if v else "N/A"
        f.write(f"| {r['olen']} | {fmt(r['bl_ttft'])} | {fmt(r['tq_ttft'])} | {fmt(r['bl_tpot'])} | {fmt(r['tq_tpot'])} | {fmt(r['bl_tps'])} | {fmt(r['tq_tps'])} | {r['ratio']} |\n")

print(f"\nResults saved to {result_dir}/RESULTS.md")
PYEOF

echo ""
echo "Benchmark complete!"

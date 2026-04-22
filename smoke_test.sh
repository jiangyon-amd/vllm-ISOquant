#!/bin/bash
# Quick smoke test: verify MoE-Fused server correctness + quick TPOT
set -e
cd /data/jiangyon/vllm_rotation

GPU=5
PORT=8300
MODEL=/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2
DATASET=/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json

cleanup() {
    echo "[$(date +%H:%M:%S)] Killing server..."
    kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
    sleep 3
    echo "[$(date +%H:%M:%S)] Done"
}
trap cleanup EXIT

echo "[$(date +%H:%M:%S)] Starting MoE-Fused server on GPU $GPU (mem 0.85)..."
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 \
VLLM_MOE_FUSED_ROTATION=1 \
VLLM_USE_FUSED_ROTATION_QUANT=1 \
HIP_VISIBLE_DEVICES=$GPU \
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL --port $PORT --trust-remote-code \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --disable-log-requests \
  > /tmp/smoke_server.log 2>&1 &
SERVER_PID=$!

for i in $(seq 1 90); do
    sleep 5
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] Server ready (${i}x5s)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] Server DIED!"
        tail -30 /tmp/smoke_server.log
        exit 1
    fi
    if [ $((i % 12)) -eq 0 ]; then echo "[$(date +%H:%M:%S)] Still waiting ($i/90)..."; fi
done

echo ""
echo "=== Correctness ==="
for Q in "只回答数字：1+1等于几？" "只回答数字：3*3等于几？" "What is 2+3? Answer with a number only."; do
    R=$(curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
        \"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"$Q\"}],
        \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")
    echo "  $Q -> $R"
done

echo ""
echo "=== Warmup ==="
vllm bench serve --backend vllm --base-url http://localhost:$PORT \
    --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
    --num-prompts 32 --request-rate 8 > /dev/null 2>&1
echo "  done"

echo ""
echo "=== Quick Benchmark ==="
for c in 1 4 16; do
    RESULT=$(vllm bench serve --backend vllm --base-url http://localhost:$PORT \
        --model $MODEL --dataset-name sharegpt --dataset-path $DATASET \
        --num-prompts 64 --request-rate $c 2>&1)
    TPOT=$(echo "$RESULT" | grep "Mean TPOT" | awk '{print $NF}')
    TTFT=$(echo "$RESULT" | grep "Mean TTFT" | awk '{print $NF}')
    THR=$(echo "$RESULT" | grep "Output token throughput" | head -1 | awk '{print $NF}')
    echo "  c=$c: TPOT=${TPOT}ms  TTFT=${TTFT}ms  throughput=${THR}tok/s"
done

echo ""
echo "[$(date +%H:%M:%S)] SMOKE TEST COMPLETE"

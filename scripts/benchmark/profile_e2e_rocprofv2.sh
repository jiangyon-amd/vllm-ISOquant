#!/bin/bash
# E2E rocprofv2 profile: HIP vs Gluon, check sc.zero_() effect and raw scale usage
# Usage: ./profile_e2e_rocprofv2.sh [gpu_id] [model]
#   gpu_id: default 1
#   model: 8b (dense) or 30b (MoE), default 8b for dense rotation layers

set -e
GPU_ID=${1:-1}
MODEL=${2:-8b}
PYTHONPATH="/data/jiangyon/vllm_rotation"
PORT=8200

# Model paths
case "$MODEL" in
  8b)  MODEL_PATH="/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-hadamard-r128" ;;
  30b) MODEL_PATH="/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2" ;;
  *)   echo "Unknown model: $MODEL (use 8b or 30b)"; exit 1 ;;
esac

OUT_DIR="/tmp/rocprof_e2e_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
echo "Output: $OUT_DIR"

BASE_ENV="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 HIP_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=$PYTHONPATH"

cleanup() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 3
}

run_profile() {
  local mode=$1
  local env_extra=$2
  local out_csv="$OUT_DIR/results_${mode}.csv"

  cleanup
  echo ""
  echo "=== [$mode] Starting server with rocprofv2 ..."

  # rocprofv2 runs server; we send requests from another process, then kill server
  (
    export $BASE_ENV
    export $env_extra
    rocprofv2 --kernel-trace -d "$OUT_DIR" -o "kernel_${mode}" python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --port $PORT --trust-remote-code --disable-log-requests \
      --max-model-len 4096 --gpu-memory-utilization 0.15 --host 0.0.0.0
  ) &
  ROCPROF_PID=$!

  # Wait for server
  for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "[$mode] Server ready after ${i}s"
      break
    fi
    sleep 1
  done

  # Get model id from /v1/models (vLLM may use full path)
  MODEL_ID=$(curl -s "http://localhost:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || echo "$MODEL_PATH")
  [ -z "$MODEL_ID" ] && MODEL_ID="$MODEL_PATH"

  # Warmup + requests to trigger CUDAGraph
  echo "[$mode] Warmup (model=$MODEL_ID)..."
  for i in 1 2 3 4 5; do
    curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL_ID"'","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null
  done
  sleep 2

  echo "[$mode] Benchmark requests (c=1)..."
  python3 << PYEOF
import urllib.request, json
model_id = """$MODEL_ID"""
for _ in range(20):
    req = urllib.request.Request('http://localhost:$PORT/v1/chat/completions',
        data=json.dumps({'model':model_id,'messages':[{'role':'user','content':'1+1=?'}],'max_tokens':8,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}).encode(),
        headers={'Content-Type':'application/json'})
    urllib.request.urlopen(req, timeout=30)
print('done')
PYEOF

  sleep 2
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  wait $ROCPROF_PID 2>/dev/null || true
  sleep 3

  # rocprofv2 writes results_<output_name>.csv, e.g. results_kernel_hip.csv
  sleep 2
  if [ -f "$OUT_DIR/results_kernel_${mode}.csv" ]; then
    cp "$OUT_DIR/results_kernel_${mode}.csv" "$out_csv"
  else
    echo "Warning: no CSV found for $mode (expected $OUT_DIR/results_kernel_${mode}.csv)"
  fi
}

# Run HIP and Gluon
run_profile "hip"    "VLLM_MOE_HIP_MFMA=1"
run_profile "gluon"  "VLLM_MOE_HIP_MFMA=0"

cleanup

# Parse and summarize (Python for proper CSV handling)
echo ""
echo "=== Kernel Summary ==="
python3 << PARSE_EOF
import csv, os, glob
out_dir = "$OUT_DIR"
for mode in ["hip", "gluon"]:
    csv_path = f"{out_dir}/results_{mode}.csv"
    if not os.path.exists(csv_path):
        csv_path = f"{out_dir}/results_kernel_{mode}.csv"
    if not os.path.exists(csv_path):
        files = sorted(glob.glob(f"{out_dir}/*.csv"))
        csv_path = next((f for f in files if mode in f), files[0] if files else None)
    if not csv_path:
        print(f"\n--- {mode} --- No CSV found")
        continue
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = (float(row["End_Timestamp"]) - float(row["Start_Timestamp"])) / 1000
                rows.append((t, row["Kernel_Name"]))
            except (KeyError, ValueError):
                pass
    print(f"\n--- {mode} ---")
    mfma = [(t,n) for t,n in rows if "mfma_rot_quant" in n]
    gluon_rows = [(t,n) for t,n in rows if "fused_rot_quant" in n or "gluon" in n.lower()]
    zero_rows = [(t,n) for t,n in rows if "zero" in n.lower() or "memset" in n.lower()]
    if mfma:
        total = sum(t for t,_ in mfma)
        print(f"mfma_rot_quant_dense: count={len(mfma)}, total={total:.1f}µs, avg={total/len(mfma):.2f}µs")
    else:
        print("mfma_rot_quant_dense: (not found - gluon path)")
    if gluon_rows:
        total = sum(t for t,_ in gluon_rows)
        print(f"gluon/fused_rot_quant: count={len(gluon_rows)}, total={total:.1f}µs")
    if zero_rows:
        total = sum(t for t,_ in zero_rows)
        print(f"zero/fill kernels: count={len(zero_rows)}, total={total:.1f}µs")
    print("Top 5 kernels by duration:")
    for t, name in sorted(rows, key=lambda x: -x[0])[:5]:
        short = name[:80] + "..." if len(name) > 80 else name
        print(f"  {t:.1f}µs  {short}")
PARSE_EOF

echo ""
echo "Results in: $OUT_DIR"
echo ""
echo "Raw scale: M<32 decode uses shuffle_scales=False when is_triton_gemm_afp4wfp4_presh_ws_tuned(N,K)."
echo "HIP path uses shuffled only; gluon handles both raw (tuned decode) and shuffled."

#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${TQ_GEAK_CONTAINER:-rotation_online_rjy_clone}"
WORKDIR="${GEAK_WORK_DIR:-/shareddata/amd/jiangyon/vllm_turboquant}"
SCRIPT_PATH="$WORKDIR/geak_tq_decode/benchmark_tq_decode.py"

if [ ! -s "$SCRIPT_PATH" ]; then
  SCRIPT_PATH="/shareddata/amd/jiangyon/vllm_turboquant/geak_tq_decode/benchmark_tq_decode.py"
fi

docker exec "$CONTAINER_NAME" bash -lc "
export PYTHONPATH=\"$WORKDIR:\$PYTHONPATH\"
cd \"$WORKDIR\"
TMPDIR=/shareddata/amd/jiangyon/tmp_hipcc TRITON_CACHE_DIR=/shareddata/amd/jiangyon/triton_cache HIP_VISIBLE_DEVICES=3 TQ_ALLOW_STALE_HIP_SO=1 python3 \"$SCRIPT_PATH\" --sweep --max-num-kv-splits 32 --eager-max-num-kv-splits 32
"

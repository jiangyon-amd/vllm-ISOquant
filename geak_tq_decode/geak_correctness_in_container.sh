#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${TQ_GEAK_CONTAINER:-rotation_online_rjy_clone}"
WORKDIR="${GEAK_WORK_DIR:-/shareddata/amd/jiangyon/vllm_turboquant}"
SCRIPT_PATH="$WORKDIR/geak_tq_decode/test_correctness.py"

if [ ! -s "$SCRIPT_PATH" ]; then
  SCRIPT_PATH="/shareddata/amd/jiangyon/vllm_turboquant/geak_tq_decode/test_correctness.py"
fi

docker exec "$CONTAINER_NAME" bash -lc "
export PYTHONPATH=\"$WORKDIR:\$PYTHONPATH\"
cd \"$WORKDIR/geak_tq_decode/hip_kernel\"
TMPDIR=/shareddata/amd/jiangyon/tmp_hipcc hipcc --offload-arch=gfx950 -shared -fPIC -O3 -ffast-math -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 -o \"$WORKDIR/vllm/v1/attention/ops/tq_decode_hip.so\" tq_decode_v52_no_nt.hip
cd \"$WORKDIR\"
TMPDIR=/shareddata/amd/jiangyon/tmp_hipcc TRITON_CACHE_DIR=/shareddata/amd/jiangyon/triton_cache HIP_VISIBLE_DEVICES=3 TQ_ALLOW_STALE_HIP_SO=1 python3 \"$SCRIPT_PATH\"
"

#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${TQ_GEAK_CONTAINER:-rotation_online_rjy_clone}"
WORKDIR="${GEAK_WORK_DIR:-/shareddata/amd/jiangyon/vllm_turboquant}"

docker exec "$CONTAINER_NAME" bash -lc "
export PYTHONPATH=\"$WORKDIR:\$PYTHONPATH\"
cd \"$WORKDIR/geak_tq_decode/hip_kernel\"
TMPDIR=/shareddata/amd/jiangyon/tmp_hipcc hipcc --offload-arch=gfx950 -shared -fPIC -O3 -ffast-math -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 -o \"$WORKDIR/vllm/v1/attention/ops/tq_decode_hip.so\" tq_decode_v52_no_nt.hip
"

#!/bin/bash
set -euo pipefail

WORKDIR="${GEAK_WORK_DIR:-/shareddata/amd/jiangyon/vllm_turboquant}"
SRC="/shareddata/amd/jiangyon/vllm_turboquant/geak_tq_decode/hip_kernel/tq_decode_v52_no_nt.hip"
DST="$WORKDIR/geak_tq_decode/hip_kernel/tq_decode_v52_no_nt.hip"

cp "$SRC" "$DST"

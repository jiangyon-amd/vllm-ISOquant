#!/bin/bash
set -e
cd /shareddata/amd/jiangyon/vllm_turboquant/geak_tq_decode/hip_kernel

echo "=== Compiling v62 ==="
TMPDIR=/shareddata/amd/jiangyon/tmp_hipcc \
hipcc --offload-arch=gfx950 -shared -fPIC -O3 -ffast-math \
  tq_decode_v62_fused.hip -o tq_decode_v62.so

echo "=== Running benchmark ==="
cd /shareddata/amd/jiangyon/vllm_turboquant
PYTHONPATH=/shareddata/amd/jiangyon/vllm_turboquant:$PYTHONPATH \
TQ_ALLOW_STALE_HIP_SO=1 \
python3 geak_tq_decode/hip_kernel/benchmark_v62.py

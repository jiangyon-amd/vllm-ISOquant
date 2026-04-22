#!/bin/bash
# Run GEAK optimization on TQ Store kernel
# Usage: bash geak_tq_store/run_geak.sh

set -euo pipefail
cd /home/jiangyon/vllm_turboquant

TASK=$(cat geak_tq_store/task_fuse_tq_store.txt)

echo "=== GEAK: Optimizing TQ Store (fused HIP kernel) ==="
echo "=== Current: 33us (N=2,H=2), 249us (N=8192,H=8) ==="
echo "=== Target: <20us (decode), <150us (prefill) ==="
echo ""

mini --config geak_tq_store/geak_config.yaml \
     --task "$TASK" \
     --yolo

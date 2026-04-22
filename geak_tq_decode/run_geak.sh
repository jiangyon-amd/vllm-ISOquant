#!/bin/bash
# Run GEAK optimization on _tq_decode_stage1 kernel
# Usage: bash geak_tq_decode/run_geak.sh

set -euo pipefail
cd /home/jiangyon/vllm_turboquant

TASK=$(cat geak_tq_decode/task_optimize_tq_decode.txt)

echo "=== GEAK: Optimizing _tq_decode_stage1 ==="
echo "=== Target: reduce B=100/seq=512 from ~377us to <250us ==="
echo ""

# Option 1: Single agent (interactive)
mini --config geak_tq_decode/geak_config.yaml \
     --task "$TASK" \
     --yolo

# Option 2: Parallel agents (uncomment to use)
# mini --config geak_tq_decode/geak_config.yaml \
#      --num-parallel 2 \
#      --repo /home/jiangyon/vllm_turboquant \
#      --task "$TASK" \
#      --gpu-ids 0,1 \
#      --metric "Extract us/call for B=100 seq=512 (lower is better)" \
#      --yolo

#!/bin/bash
# Run GEAK agent for GQA fusion optimization
set -euo pipefail
cd /home/jiangyon/vllm_turboquant

TASK=$(cat geak_tq_decode/task_gqa_fusion.txt)

echo "=== GEAK: GQA Fusion Optimization ==="
echo "=== Target: 72B (Hq=64) from 241.9us → ~134us ==="
echo ""

mini --config geak_tq_decode/geak_gqa_config.yaml \
     --task "$TASK" \
     --yolo

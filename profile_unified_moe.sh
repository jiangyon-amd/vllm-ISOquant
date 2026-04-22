#!/bin/bash
# Profile unified MoE kernel with rocprof
# Usage: ./profile_unified_moe.sh [gpu_id]

GPU=${1:-1}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="$SCRIPT_DIR/bench_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRACE_DIR="$RESULT_DIR/rocprof_unified_${TIMESTAMP}"

mkdir -p "$TRACE_DIR"

export HIP_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "=== Unified MoE Kernel Profiling ==="
echo "GPU: $GPU"
echo "Output: $TRACE_DIR"
echo ""

# Step 1: Warmup compilation (don't profile this)
echo "[1/3] Warmup (compile kernels)..."
python "$SCRIPT_DIR/test_unified_moe_kernel.py" --test 2>&1 | tail -5

# Step 2: Profile with rocprof hip-trace
echo ""
echo "[2/3] rocprof --hip-trace ..."
rocprof --hip-trace \
    -o "$TRACE_DIR/trace.csv" \
    python "$SCRIPT_DIR/test_unified_moe_kernel.py" --rocprof 2>&1 | tee "$TRACE_DIR/rocprof.log"

# Step 3: Analyze trace
echo ""
echo "[3/3] Analyzing kernel times..."
if [ -f "$TRACE_DIR/trace.csv" ]; then
    echo ""
    echo "=== Kernel Summary ==="
    # Extract kernel dispatch times from hip-trace CSV
    python3 << 'PYEOF'
import csv
import sys
from collections import defaultdict

trace_dir = sys.argv[1] if len(sys.argv) > 1 else "."
csv_path = f"{trace_dir}/trace.csv"

kernels = defaultdict(list)
try:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", row.get("KernelName", ""))
            dur = float(row.get("DurationNs", row.get("Duration(ns)", 0)))
            if dur > 0 and ("rot_quant" in name.lower() or "moe" in name.lower()
                           or "sort" in name.lower() or "unified" in name.lower()):
                # Shorten kernel name
                short = name.split("(")[0][-60:]
                kernels[short].append(dur / 1000)  # ns → µs
except Exception as e:
    print(f"Could not parse trace: {e}")
    sys.exit(0)

if not kernels:
    print("No MoE-related kernels found in trace.")
    sys.exit(0)

print(f"{'Kernel':<60} {'Count':>6} {'Avg(µs)':>10} {'Min':>8} {'Max':>8} {'Total(ms)':>10}")
print("-" * 110)
for name, times in sorted(kernels.items(), key=lambda kv: -sum(kv[1])):
    n = len(times)
    avg = sum(times) / n
    mn = min(times)
    mx = max(times)
    total = sum(times) / 1000
    print(f"{name:<60} {n:>6} {avg:>10.1f} {mn:>8.1f} {mx:>8.1f} {total:>10.2f}")
PYEOF "$TRACE_DIR"
fi

echo ""
echo "Full trace: $TRACE_DIR/trace.csv"
echo "Import into chrome://tracing or Perfetto for timeline view."

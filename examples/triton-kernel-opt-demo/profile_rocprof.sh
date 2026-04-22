#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <variant> <output-dir> [m] [n] [k] [best-config-json]"
  exit 1
fi

VARIANT="$1"
OUT_DIR="$2"
M="${3:-1024}"
N="${4:-1024}"
K="${5:-1024}"
BEST_CONFIG_OVERRIDE="${6:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${SCRIPT_DIR}/benchmark.py"
BEST_CONFIG_JSON="${OUT_DIR}/best_config.json"

mkdir -p "${OUT_DIR}"

if [[ -n "${BEST_CONFIG_OVERRIDE}" ]]; then
  echo "Using provided best config: ${BEST_CONFIG_OVERRIDE}"
  PROFILE_CONFIG_JSON="${BEST_CONFIG_OVERRIDE}"
else
  echo "Running one unprofiled steady-state sweep to select and cache the best config."
  python3 "${BENCHMARK}" \
    --variant "${VARIANT}" \
    --m "${M}" \
    --n "${N}" \
    --k "${K}" \
    --select-best-config \
    --selection-warmup 20 \
    --selection-iters 100 \
    --selection-repeats 3 \
    --best-config-out "${BEST_CONFIG_JSON}" \
    --json-out "${OUT_DIR}/config_selection.json" >/dev/null
  PROFILE_CONFIG_JSON="${BEST_CONFIG_JSON}"
fi

CMD=(
  python3 "${BENCHMARK}"
  --variant "${VARIANT}"
  --m "${M}"
  --n "${N}"
  --k "${K}"
  --warmup 0
  --iters 20
  --fixed-config-in "${PROFILE_CONFIG_JSON}"
)

if command -v rocprofv3 >/dev/null 2>&1; then
  echo "Selected profiler: rocprofv3"
  rocprofv3 \
    --kernel-trace \
    --summary \
    --output-format csv \
    --output-directory "${OUT_DIR}" \
    --output-file trace \
    -- "${CMD[@]}"
elif command -v rocprof >/dev/null 2>&1; then
  echo "Selected profiler: rocprof"
  rocprof \
    --stats \
    -o "${OUT_DIR}/trace.csv" \
    "${CMD[@]}"
else
  echo "Neither rocprofv3 nor rocprof is available."
  exit 2
fi

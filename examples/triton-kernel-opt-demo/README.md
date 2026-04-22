# Triton Kernel Optimization Demo

This demo shows the exact three-round workflow expected by the `triton-kernel-optimization` skill.

## Files

- `baseline_matmul.py`: round-1 baseline kernel
- `optimized_matmul_round2.py`: round-2 version with grouped program ordering
- `optimized_matmul_round3.py`: round-3 version with grouped ordering plus a full-tile fast path for aligned shapes
- `benchmark.py`: benchmark-driven CLI for config selection, correctness, and timing
- `profile_rocprof.sh`: steady-state profiler wrapper with `rocprofv3 -> rocprof` fallback
- `analyze_rocprof.py`: lightweight profiler output summary
- `check_steady_state_profile.py`: verifies that the profiler trace only captures the steady-state kernel path

## Benchmark CLI

Use `benchmark.py` as the main CLI entry point.

Example:

```bash
python3 examples/triton-kernel-opt-demo/benchmark.py \
  --variant baseline \
  --m 1024 --n 1024 --k 1024 \
  --select-best-config \
  --selection-warmup 20 \
  --selection-iters 100 \
  --selection-repeats 3 \
  --best-config-out examples/triton-kernel-opt-demo/logs/round1_best_config.json
```

Default correctness tolerance:

- `rtol=2e-2`
- `atol=2e-2`

Reference path:

- `torch.matmul(a.float(), b.float()).to(torch.bfloat16)`

Important:

- After each code change, first run `--select-best-config` and save the resulting `best_config.json`.
- Reuse the same `best_config.json` for both the correctness benchmark and the profiler run.
- `profile_rocprof.sh` profiles the steady-state fixed-config path and does not repeat correctness checks on its own.
- `profile_rocprof.sh` accepts an optional sixth argument for an existing `best_config.json`; use that form when you want benchmark and profiler evidence to share the exact same selected config.
- `torch.cuda.Event` loop timing and kernel-trace timing are related but not identical. The benchmark captures repeated dispatch behavior, while the profiler summary isolates kernel execution time only.

## Round Sequence

### Round 1: Generate the baseline kernel

```bash
python3 examples/triton-kernel-opt-demo/benchmark.py \
  --variant baseline \
  --m 1024 --n 1024 --k 1024 \
  --select-best-config \
  --selection-warmup 20 \
  --selection-iters 100 \
  --selection-repeats 3 \
  --best-config-out examples/triton-kernel-opt-demo/logs/round1_best_config.json \
  --json-out examples/triton-kernel-opt-demo/logs/round1_selection.json

python3 examples/triton-kernel-opt-demo/benchmark.py \
  --variant baseline \
  --m 1024 --n 1024 --k 1024 \
  --check \
  --fixed-config-in examples/triton-kernel-opt-demo/logs/round1_best_config.json \
  --json-out examples/triton-kernel-opt-demo/logs/round1_benchmark.json

bash examples/triton-kernel-opt-demo/profile_rocprof.sh \
  baseline \
  examples/triton-kernel-opt-demo/logs/round1_profile \
  1024 1024 1024 \
  examples/triton-kernel-opt-demo/logs/round1_best_config.json

python3 examples/triton-kernel-opt-demo/analyze_rocprof.py \
  --input examples/triton-kernel-opt-demo/logs/round1_profile \
  --variant baseline \
  --json-out examples/triton-kernel-opt-demo/logs/round1_profile_summary.json

python3 examples/triton-kernel-opt-demo/check_steady_state_profile.py \
  --summary-json examples/triton-kernel-opt-demo/logs/round1_profile_summary.json \
  --max-target-dispatches 50
```

Round-1 main idea:

- generate a correct BF16 `tl.dot` baseline

### Round 2: Apply one focused optimization

```bash
python3 examples/triton-kernel-opt-demo/benchmark.py \
  --variant round2 \
  --m 1024 --n 1024 --k 1024 \
  --select-best-config \
  --selection-warmup 20 \
  --selection-iters 100 \
  --selection-repeats 3 \
  --best-config-out examples/triton-kernel-opt-demo/logs/round2_best_config.json \
  --json-out examples/triton-kernel-opt-demo/logs/round2_selection.json

python3 examples/triton-kernel-opt-demo/benchmark.py \
  --variant round2 \
  --m 1024 --n 1024 --k 1024 \
  --check \
  --fixed-config-in examples/triton-kernel-opt-demo/logs/round2_best_config.json \
  --json-out examples/triton-kernel-opt-demo/logs/round2_benchmark.json

bash examples/triton-kernel-opt-demo/profile_rocprof.sh \
  round2 \
  examples/triton-kernel-opt-demo/logs/round2_profile \
  1024 1024 1024 \
  examples/triton-kernel-opt-demo/logs/round2_best_config.json

python3 examples/triton-kernel-opt-demo/analyze_rocprof.py \
  --input examples/triton-kernel-opt-demo/logs/round2_profile \
  --variant round2 \
  --json-out examples/triton-kernel-opt-demo/logs/round2_profile_summary.json

python3 examples/triton-kernel-opt-demo/check_steady_state_profile.py \
  --summary-json examples/triton-kernel-opt-demo/logs/round2_profile_summary.json \
  --max-target-dispatches 50
```

Round-2 main idea:

- switch from linear program ordering to grouped ordering for better locality

### Round 3: Apply one more focused optimization

```bash
python3 examples/triton-kernel-opt-demo/benchmark.py \
  --variant round3 \
  --m 1024 --n 1024 --k 1024 \
  --select-best-config \
  --selection-warmup 20 \
  --selection-iters 100 \
  --selection-repeats 3 \
  --best-config-out examples/triton-kernel-opt-demo/logs/round3_best_config.json \
  --json-out examples/triton-kernel-opt-demo/logs/round3_selection.json

python3 examples/triton-kernel-opt-demo/benchmark.py \
  --variant round3 \
  --m 1024 --n 1024 --k 1024 \
  --check \
  --fixed-config-in examples/triton-kernel-opt-demo/logs/round3_best_config.json \
  --json-out examples/triton-kernel-opt-demo/logs/round3_benchmark.json

bash examples/triton-kernel-opt-demo/profile_rocprof.sh \
  round3 \
  examples/triton-kernel-opt-demo/logs/round3_profile \
  1024 1024 1024 \
  examples/triton-kernel-opt-demo/logs/round3_best_config.json

python3 examples/triton-kernel-opt-demo/analyze_rocprof.py \
  --input examples/triton-kernel-opt-demo/logs/round3_profile \
  --variant round3 \
  --json-out examples/triton-kernel-opt-demo/logs/round3_profile_summary.json

python3 examples/triton-kernel-opt-demo/check_steady_state_profile.py \
  --summary-json examples/triton-kernel-opt-demo/logs/round3_profile_summary.json \
  --max-target-dispatches 50
```

Round-3 main idea:

- keep grouped ordering and add a full-tile fast path that removes boundary masks when the selected tile fully covers the input shape

## Profiler Notes

- On AMD/ROCm, the wrapper prefers `rocprofv3`.
- If `rocprofv3` is not available, it falls back to `rocprof`.
- If neither tool is available, the wrapper exits with an error.
- The wrapper keeps autotune and config selection outside the profiled run.
- The steady-state guard should report a target dispatch count that stays close to the benchmark iteration count.

## Required Decision Discipline

After each round, record:

- correctness result
- benchmark result
- profiler output path
- profiler summary
- keep / revert / revise decision

Do not skip the profiler step after generating or modifying code.

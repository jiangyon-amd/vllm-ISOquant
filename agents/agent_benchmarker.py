"""Benchmarker Agent — runs GPU benchmarks calling the actual HIP kernel via ctypes.

Input:  compiled .so + workload configs
Output: benchmark results {config_key → gpu_time_us}
"""
from __future__ import annotations

import json
import os
import subprocess
import re

from agents.state import CandidateResult, CandidateStatus, PipelineState
from agents.target_registry import (
    BENCH_GENERATORS,
    get_target,
    resolve_workload_pack,
)


class BenchmarkerAgent:
    """Runs standardized GPU benchmarks for kernel performance measurement."""

    def __init__(self, warmup_iters: int = 10, bench_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters

    def run(self, state: PipelineState,
            candidate: CandidateResult | None = None) -> dict[str, float]:
        """Run benchmarks for all workload configs.

        Returns config_key → gpu_time_us map.
        """
        active = candidate or state.current
        target_name = (
            candidate.resolved_target(state.target_kernel)
            if candidate is not None else state.target_kernel
        )
        ver = candidate.candidate_id if candidate is not None else f"v{state.iteration}"
        out_dir = (
            os.path.join(state.run_dir, f"round_{state.round_index:02d}", candidate.candidate_id, "benchmark")
            if candidate is not None else
            os.path.join(state.run_dir, ver, "benchmark")
        )
        os.makedirs(out_dir, exist_ok=True)
        workload_configs = resolve_workload_pack(
            get_target(target_name),
            candidate.workload_pack if candidate is not None else "",
        )

        so_path = active.so_path
        if not so_path or not os.path.exists(so_path):
            print(f"[Benchmarker] WARNING: .so not found at {so_path}")
            print("[Benchmarker] Falling back to currently deployed kernel")
            so_path = self._find_deployed_so(target_name)

        if not so_path:
            print("[Benchmarker] ERROR: No .so found anywhere")
            return {}

        # Generate target-specific benchmark script
        script_path = os.path.join(out_dir, "bench_run.py")
        gen = BENCH_GENERATORS.get(target_name)
        if gen is None:
            print(f"[Benchmarker] ERROR: no benchmark generator for target '{target_name}'")
            return {}
        script_text = gen(
            so_path,
            workload_configs,
            self.warmup_iters,
            self.bench_iters,
        )
        with open(script_path, "w") as f:
            f.write(script_text)

        # Execute
        results = self._execute_benchmark(script_path)

        # Save
        results_path = os.path.join(out_dir, "bench_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        active.benchmark_results = results
        if candidate is not None:
            candidate.status = CandidateStatus.BENCHMARKED.value
        return results

    def _find_deployed_so(self, target: str = "") -> str:
        """Find the currently deployed .so file."""
        if target:
            try:
                tc = get_target(target)
                if os.path.exists(tc.deployed_so):
                    return tc.deployed_so
            except ValueError:
                pass
        candidates = [
            "vllm/v1/attention/ops/tq_decode_stage2_hip.so",
            "vllm/v1/attention/ops/tq_decode_split_hip.so",
            "vllm/v1/attention/ops/tq_decode_fused_hip.so",
        ]
        base = os.path.dirname(os.path.dirname(__file__))
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.exists(p):
                return p
        return ""

    def _execute_benchmark(self, script_path: str) -> dict[str, float]:
        """Run the benchmark script and parse results.

        Tries Docker execution first (for GPU access), falls back to local.
        """
        try:
            from agents.docker_exec import run_script_in_docker
            result = run_script_in_docker(script_path, timeout=300)
        except (ImportError, FileNotFoundError):
            import sys
            python = sys.executable or "python3"
            result = subprocess.run(
                [python, script_path],
                capture_output=True, text=True, timeout=300,
                env={**os.environ, "HIP_VISIBLE_DEVICES": "0"},
            )
        except Exception as e:
            print(f"[Benchmarker] Execution failed: {e}")
            return {}

        # Always print stdout for visibility
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if not line.startswith("{"):
                    print(f"  {line}")

        if result.returncode != 0:
            print(f"[Benchmarker] Script failed (exit={result.returncode}):")
            if result.stderr:
                print(f"  stderr: {result.stderr[:1000]}")
            if result.stdout:
                print(f"  stdout: {result.stdout[:500]}")
            # Still try to parse partial results
            pass

        # Parse ===RESULTS=== JSON line
        output = result.stdout
        found_marker = False
        for line in output.split("\n"):
            if line.strip() == "===RESULTS===":
                found_marker = True
                continue
            if found_marker:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        return {k: float(v) for k, v in data.items()}
                except (json.JSONDecodeError, ValueError):
                    continue

        # Fallback: parse individual lines
        results = {}
        for line in output.split("\n"):
            m = re.search(r"B=(\d+)\s+seq=(\d+)\s+Hq=(\d+)\s+splits=(\d+)\s+time=([\d.]+)", line)
            if m:
                key = f"B{m.group(1)}_seq{m.group(2)}_Hq{m.group(3)}_s{m.group(4)}"
                results[key] = float(m.group(5))

        return results

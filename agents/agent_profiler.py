"""Profiler Agent — runs rocprofv3 kernel trace and extracts GPU metrics.

Input:  kernel .so path + benchmark script
Output: kernel metrics dict {config_key → gpu_time_us}
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agents.state import PipelineState


@dataclass
class ProfileResult:
    """Structured profiling result."""
    kernel_name: str
    config_key: str
    gpu_time_us: float
    grid_size: tuple[int, int, int]
    block_size: tuple[int, int, int]
    vgpr_count: int = 0
    sgpr_count: int = 0


class ProfilerAgent:
    """Wraps rocprofv3 --kernel-trace to extract GPU kernel timings."""

    def __init__(self, rocprofv3_bin: str = "rocprofv3"):
        self.rocprofv3 = rocprofv3_bin

    def run(self, state: PipelineState, benchmark_script: str) -> dict[str, float]:
        """Execute profiling and return config_key → gpu_time_us map.

        Steps:
        1. Run rocprofv3 --kernel-trace on the benchmark script
        2. Parse the SQLite output for kernel dispatch times
        3. Map kernels to config keys via grid dimensions
        4. Return average GPU times per config
        """
        out_dir = os.path.join(state.run_dir, f"v{state.iteration}", "profile")
        os.makedirs(out_dir, exist_ok=True)

        # Run rocprofv3
        db_path = self._run_rocprofv3(benchmark_script, out_dir)

        if db_path is None:
            print("[Profiler] WARNING: rocprofv3 failed, using fallback timing")
            return self._fallback_timing(state, benchmark_script)

        # Parse results
        results = self._parse_db(db_path, state)

        # Save results
        results_path = os.path.join(out_dir, "profile_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _run_rocprofv3(self, script: str, out_dir: str) -> str | None:
        """Run rocprofv3 and return path to output DB."""
        out_prefix = os.path.join(out_dir, "trace")
        cmd = [
            self.rocprofv3,
            "--kernel-trace",
            "-o", out_prefix,
            "--",  # rocprofv3 requires -- before the application
            "python", script,
        ]
        print(f"[Profiler] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=300,  # 5 min timeout
                env={**os.environ, "HIP_VISIBLE_DEVICES": "0"},
            )
        except subprocess.TimeoutExpired:
            print("[Profiler] rocprofv3 timed out")
            return None
        except FileNotFoundError:
            print(f"[Profiler] {self.rocprofv3} not found")
            return None

        if result.returncode != 0:
            print(f"[Profiler] rocprofv3 failed: {result.stderr[:500]}")
            return None

        # Find the output DB file
        for f in Path(out_dir).rglob("*.db"):
            return str(f)
        for f in Path(out_dir).rglob("*.rpd"):
            return str(f)

        print("[Profiler] No output DB found")
        return None

    def _parse_db(self, db_path: str,
                  state: PipelineState) -> dict[str, float]:
        """Parse rocprofv3 SQLite DB for kernel dispatch times.

        rocprofv3 schema (tables have UUID suffixes):
          rocpd_kernel_dispatch_<UUID>:
            kernel_id, grid_size_x/y/z, workgroup_size_x/y/z, start, end
          rocpd_string_<UUID>:
            id, string   (kernel_id → string gives the kernel name)
        """
        results: dict[str, list[float]] = {}

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Find actual table names (they have UUID suffixes)
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'rocpd_kernel_dispatch_%'"
            )
            kd_tables = [r[0] for r in cursor.fetchall()]
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'rocpd_info_kernel_symbol_%'"
            )
            ks_tables = [r[0] for r in cursor.fetchall()]

            if not kd_tables or not ks_tables:
                print("[Profiler] No kernel dispatch/symbol tables in DB")
                conn.close()
                return {}

            kd_table = kd_tables[0]
            ks_table = ks_tables[0]

            # kernel_id in dispatch → id in kernel_symbol gives kernel name
            query = f"""
                SELECT ks.display_name AS kernel_name,
                       k.grid_size_x, k.grid_size_y, k.grid_size_z,
                       k.workgroup_size_x, k.workgroup_size_y,
                       k.workgroup_size_z,
                       (k.end - k.start) AS duration_ns
                FROM {kd_table} k
                JOIN {ks_table} ks ON k.kernel_id = ks.id
                WHERE ks.display_name LIKE '%tq_decode%'
                ORDER BY k.start
            """
            cursor.execute(query)

            for row in cursor.fetchall():
                kernel_name = row[0] if isinstance(row[0], str) else ""
                grid_x = row[1]
                wg_x = row[4]
                duration_ns = row[7]

                # grid_size = grid_dim × block_dim in rocprofv3
                B = grid_x // wg_x if wg_x > 0 else grid_x

                config_key = self._match_config(B, kernel_name, state)
                if config_key:
                    gpu_us = duration_ns / 1000.0
                    results.setdefault(config_key, []).append(gpu_us)

            conn.close()
        except Exception as e:
            print(f"[Profiler] DB parse error: {e}")
            return {}

        # Average (skip first warm-up call per config)
        averaged = {}
        for key, times in results.items():
            if len(times) > 1:
                times = times[1:]  # skip warmup
            averaged[key] = sum(times) / len(times)

        return averaged

    def _match_config(self, B: int, kernel_name: str,
                      state: PipelineState) -> str | None:
        """Match a profiled kernel launch to a workload config."""
        for cfg in state.workload_configs:
            if cfg["B"] == B:
                return state.config_key(cfg)
        # Approximate match (within 10%)
        for cfg in state.workload_configs:
            if abs(cfg["B"] - B) <= max(1, cfg["B"] * 0.1):
                return state.config_key(cfg)
        return None

    def _fallback_timing(self, state: PipelineState,
                         script: str) -> dict[str, float]:
        """When rocprofv3 is unavailable, run the benchmark script and
        parse stdout for timing lines like 'B=128 ... time=48.6us'."""
        import sys
        python = sys.executable or "python3"
        try:
            result = subprocess.run(
                [python, script],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "HIP_VISIBLE_DEVICES": "0"},
            )
            output = result.stdout
        except Exception as e:
            print(f"[Profiler] Fallback failed: {e}")
            return {}

        results = {}
        for line in output.split("\n"):
            # Parse lines like: B=128 seq=128 ... time=48.6
            m = re.search(r"B=(\d+).*?seq=(\d+).*?time[=:]\s*([\d.]+)", line)
            if m:
                B, seq = int(m.group(1)), int(m.group(2))
                t = float(m.group(3))
                for cfg in state.workload_configs:
                    if cfg["B"] == B and cfg["seq"] == seq:
                        results[state.config_key(cfg)] = t
                        break
        return results

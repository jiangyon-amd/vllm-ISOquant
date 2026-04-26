"""Compiler Agent — compiles HIP kernels to .so shared libraries.

Input:  .hip source path
Output: .so path (or None on failure)
"""
from __future__ import annotations

import os
import subprocess

from agents.state import CandidateResult, CandidateStatus, PipelineState
from agents.target_registry import get_target


class CompilerAgent:
    """Compiles HIP kernel source to shared library."""

    def __init__(self, hipcc: str = "hipcc",
                 arch: str = "gfx950",
                 extra_flags: list[str] | None = None):
        self.hipcc = hipcc
        self.arch = arch
        self.extra_flags = extra_flags or [
            "-O3",
            "-ffast-math",
            "-mllvm", "--amdgpu-early-inline-all=true",
            "-mllvm", "--amdgpu-function-calls=false",
        ]

    def run(self, state: PipelineState,
            candidate: CandidateResult | None = None) -> str | None:
        """Compile the current iteration's kernel.

        Returns path to .so on success, None on failure.
        """
        active = candidate or state.current
        target_name = (
            candidate.resolved_target(state.target_kernel)
            if candidate is not None else state.target_kernel
        )
        target_cfg = get_target(target_name)
        if not target_cfg.supports_compile:
            if candidate is not None:
                candidate.status = CandidateStatus.COMPILED.value
            return candidate.so_path if candidate is not None else ""

        hip_src = active.hip_source
        if not hip_src or not os.path.exists(hip_src):
            print(f"[Compiler] Source not found: {hip_src}")
            return None

        ver = candidate.candidate_id if candidate is not None else f"v{state.iteration}"
        out_dir = (
            os.path.join(state.run_dir, f"round_{state.round_index:02d}", candidate.candidate_id)
            if candidate is not None else
            os.path.join(state.run_dir, ver)
        )
        os.makedirs(out_dir, exist_ok=True)

        # Step 1: Compile to .o
        # Use target name from state for output naming
        target_base = target_name
        obj_path = os.path.join(out_dir, f"{target_base}_{ver}.o")
        so_path = os.path.join(out_dir, f"{target_base}_{ver}.so")

        # Try Docker first (correct GPU libraries), then local hipcc
        use_docker = False
        try:
            from agents.docker_exec import compile_hip_in_docker, CONTAINER_NAME
            use_docker = True
        except ImportError:
            pass

        docker_ok = False
        if use_docker:
            extra = " ".join(self.extra_flags)
            print(f"[Compiler] Compiling via Docker ({CONTAINER_NAME})")
            try:
                result = compile_hip_in_docker(
                    hip_src, so_path, arch=self.arch, extra_flags=extra
                )
                if result.returncode == 0:
                    docker_ok = True
                    if result.stderr:
                        print(f"[Compiler] Warnings:\n{result.stderr[:500]}")
                else:
                    print(f"[Compiler] Docker compile failed:\n{result.stderr[:1000]}")
            except Exception as e:
                print(f"[Compiler] Docker unavailable: {e}")

        if not docker_ok:
            compile_cmd = [
                self.hipcc,
                "-c",
                f"--offload-arch={self.arch}",
                *self.extra_flags,
                "-fPIC",
                "-o", obj_path,
                hip_src,
            ]

            print(f"[Compiler] Compiling: {' '.join(compile_cmd)}")
            try:
                result = subprocess.run(
                    compile_cmd, capture_output=True, text=True, timeout=120,
                )
            except FileNotFoundError:
                print(f"[Compiler] {self.hipcc} not found")
                return None
            except subprocess.TimeoutExpired:
                print("[Compiler] Compilation timed out")
                return None

            if result.returncode != 0:
                print(f"[Compiler] Compile failed:\n{result.stderr[:1000]}")
                err_path = os.path.join(out_dir, "compile_error.txt")
                with open(err_path, "w") as f:
                    f.write(result.stderr)
                return None

            if result.stderr:
                print(f"[Compiler] Warnings:\n{result.stderr[:500]}")

            # Step 2: Link to .so
            link_cmd = [
                self.hipcc,
                "-shared",
                f"--offload-arch={self.arch}",
                "-o", so_path,
                obj_path,
            ]

            print(f"[Compiler] Linking: {' '.join(link_cmd)}")
            try:
                result = subprocess.run(
                    link_cmd, capture_output=True, text=True, timeout=60,
                )
            except Exception as e:
                print(f"[Compiler] Link failed: {e}")
                return None

            if result.returncode != 0:
                print(f"[Compiler] Link failed:\n{result.stderr[:500]}")
                return None

        # Verify .so exists and is non-empty
        if not os.path.exists(so_path) or os.path.getsize(so_path) == 0:
            print("[Compiler] Output .so is empty or missing")
            return None

        active.so_path = so_path
        if candidate is not None:
            candidate.artifacts["so_path"] = so_path
            candidate.status = CandidateStatus.COMPILED.value
        print(f"[Compiler] Success: {so_path} ({os.path.getsize(so_path)} bytes)")

        # Extract register info if available
        self._extract_register_info(hip_src, state, candidate=candidate)

        return so_path

    def _extract_register_info(self, hip_src: str,
                               state: PipelineState,
                               candidate: CandidateResult | None = None) -> None:
        """Try to extract VGPR/SGPR counts from compiler output."""
        ver = candidate.candidate_id if candidate is not None else f"v{state.iteration}"
        out_dir = (
            os.path.join(state.run_dir, f"round_{state.round_index:02d}", candidate.candidate_id)
            if candidate is not None else
            os.path.join(state.run_dir, ver)
        )
        target_base = (
            candidate.resolved_target(state.target_kernel)
            if candidate is not None else state.target_kernel
        )
        asm_path = os.path.join(out_dir, f"{target_base}_{ver}.s")
        active = candidate or state.current

        # Use local hipcc for assembly extraction
        try:
            asm_cmd = [
                self.hipcc,
                "-S",
                f"--offload-arch={self.arch}",
                *self.extra_flags,
                "-o", asm_path,
                hip_src,
            ]
            result = subprocess.run(
                asm_cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and os.path.exists(asm_path):
                with open(asm_path) as f:
                    asm = f.read()
                # Parse .vgpr_count, .sgpr_count
                import re
                vgpr = re.search(r'\.vgpr_count:\s*(\d+)', asm)
                sgpr = re.search(r'\.sgpr_count:\s*(\d+)', asm)
                spill = re.search(r'\.vgpr_spill_count:\s*(\d+)', asm)
                if vgpr:
                    active.vgpr_count = int(vgpr.group(1))
                    print(f"[Compiler] VGPRs: {active.vgpr_count}")
                if sgpr:
                    active.sgpr_count = int(sgpr.group(1))
                    print(f"[Compiler] SGPRs: {active.sgpr_count}")
                if spill:
                    spill_count = int(spill.group(1))
                    if spill_count > 0:
                        print(f"[Compiler] WARNING: {spill_count} VGPR spills!")
        except Exception:
            pass  # Assembly extraction is best-effort

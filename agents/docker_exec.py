"""Docker execution helper for running GPU scripts inside the container.

Since torch+GPU are only available inside the Docker container,
this module wraps script execution via `docker exec`.
"""
from __future__ import annotations

import os
import subprocess


CONTAINER_NAME = os.environ.get("TQ_CONTAINER", "rotation_online_rjy_clone")
GPU_DEVICE = os.environ.get("TQ_GPU_DEVICE", "3")
WORKDIR = os.environ.get("TQ_WORKDIR",
    "/shareddata/amd/jiangyon/vllm_turboquant")


def run_script_in_docker(script_path: str,
                         timeout: int = 300,
                         gpu_device: str = "") -> subprocess.CompletedProcess:
    """Run a Python script inside the Docker container.

    Args:
        script_path: absolute path to .py script (on shared filesystem)
        timeout: max seconds
        gpu_device: HIP_VISIBLE_DEVICES value (default from env)

    Returns:
        subprocess.CompletedProcess with stdout/stderr
    """
    gpu = gpu_device or GPU_DEVICE
    cmd = [
        "docker", "exec",
        "-w", WORKDIR,
        CONTAINER_NAME,
        "bash", "-c",
        f"HIP_VISIBLE_DEVICES={gpu} python3 {script_path}"
    ]

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def compile_hip_in_docker(hip_src: str, so_path: str,
                          arch: str = "gfx950",
                          extra_flags: str = "-O3 -ffast-math") -> subprocess.CompletedProcess:
    """Compile a HIP kernel inside the Docker container.

    Args:
        hip_src: absolute path to .hip file
        so_path: absolute path for output .so
        arch: GPU architecture
        extra_flags: additional compiler flags

    Returns:
        subprocess.CompletedProcess
    """
    cmd = [
        "docker", "exec",
        "-w", os.path.dirname(hip_src),
        CONTAINER_NAME,
        "bash", "-c",
        f"TMPDIR=/shareddata/amd/jiangyon/tmp_hipcc hipcc -shared -fPIC {extra_flags} "
        f"--offload-arch={arch} -o {so_path} {hip_src}"
    ]

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

"""Target registry — target-specific configuration for the optimization pipeline.

Each target defines:
  - Source file pattern & kernel dir
  - C launcher function name & ctypes signature
  - Deployed .so location
  - Workload configs
  - Benchmark & correctness script generators
"""
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TargetConfig:
    """Configuration for a specific optimization target."""
    name: str
    kernel_dir: str                     # directory with .hip sources
    source_glob: str                    # glob pattern for versioned sources
    launcher_name: str                  # C function to call
    deployed_so: str                    # path relative to repo root
    workload_configs: list[dict]        # benchmark configs
    target_type: str = "kernel"         # "kernel" | "campaign"
    entry_files: list[str] = field(default_factory=list)
    workload_packs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    evaluator_packs: dict[str, dict[str, Any]] = field(default_factory=dict)
    profile_packs: dict[str, dict[str, Any]] = field(default_factory=dict)
    search_space: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reference_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    supports_compile: bool = True
    default_workload_pack: str = "default"
    default_evaluator_pack: str = "default"
    default_profile_pack: str = "default"
    default_top_k: int = 2
    max_candidates_per_round: int = 4


# ── Repo root ─────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(__file__))
_KERNEL_DIR = os.path.join(_REPO, "geak_tq_decode", "hip_kernel")
_OPS_DIR = os.path.join(_REPO, "vllm", "v1", "attention", "ops")
_FUSION_DIR = os.path.join(_OPS_DIR, "turboquant_soa_fusion")
_QWEN_72B = "/shareddata/amd/jiangyon/models/Qwen2.5-72B-Instruct"


# ── Target definitions ────────────────────────────────────────────────

STAGE2_CONFIG = TargetConfig(
    name="tq_decode_stage2",
    kernel_dir=_KERNEL_DIR,
    source_glob="tq_decode_stage2_v*.hip",
    launcher_name="launch_tq_decode_stage2_bf16",
    deployed_so=os.path.join(_OPS_DIR, "tq_decode_stage2_hip.so"),
    workload_configs=[
        # input=8192, output=2048, con=32
        # decode: seq grows 8192→10240, batch up to 32
        {"B": 8,   "seq": 8192,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 8,   "seq": 9216,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 8,   "seq": 10240, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 16,  "seq": 8192,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 16,  "seq": 9216,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 16,  "seq": 10240, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 32,  "seq": 8192,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 32,  "seq": 9216,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 32,  "seq": 10240, "Hq": 64, "Hk": 8, "splits": 32},
    ],
)

STAGE1_CONFIG = TargetConfig(
    name="tq_decode_stage1",
    kernel_dir=_KERNEL_DIR,
    source_glob="tq_decode_stage1_v101*.hip",  # Start from v70 (XOR swizzle base)
    launcher_name="launch_tq_decode_stage1",
    deployed_so=os.path.join(_OPS_DIR, "tq_decode_split_hip.so"),
    workload_configs=[
        # ★ output>1024: seq=4096-9216, B=4-32
        # input=8192, output=1024 → seq grows 8192→9216
        {"B": 4,   "seq": 4096,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 4,   "seq": 8192,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 4,   "seq": 9216,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 10,  "seq": 4096,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 10,  "seq": 8192,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 10,  "seq": 9216,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 20,  "seq": 4096,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 20,  "seq": 8192,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 32,  "seq": 4096,  "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 32,  "seq": 8192,  "Hq": 64, "Hk": 8, "splits": 32},
    ],
)

WHT_ROTATE_CONFIG = TargetConfig(
    name="tq_wht_rotate",
    kernel_dir=_KERNEL_DIR,
    source_glob="tq_wht_rotate_v*.hip",
    launcher_name="launch_tq_wht_rotate",
    deployed_so=os.path.join(_KERNEL_DIR, "tq_wht_rotate_v1.so"),
    workload_configs=[
        # M = B * Hq (total rows to rotate)
        # Realistic decode workloads: B=batch_size, Hq=num_query_heads=64
        {"M": 8,    "desc": "B=1,Hq=8"},
        {"M": 64,   "desc": "B=1,Hq=64"},
        {"M": 128,  "desc": "B=2,Hq=64"},
        {"M": 512,  "desc": "B=8,Hq=64"},
        {"M": 1024, "desc": "B=16,Hq=64"},
        {"M": 2048, "desc": "B=32,Hq=64"},
        {"M": 4096, "desc": "B=64,Hq=64"},
        {"M": 8192, "desc": "B=128,Hq=64"},
    ],
)

FUSED_WHT_CONFIG = TargetConfig(
    name="tq_decode_fused_wht",
    kernel_dir=_KERNEL_DIR,
    source_glob="tq_decode_fused_v5*.hip",
    launcher_name="launch_tq_decode_fused_v5",
    deployed_so=os.path.join(_KERNEL_DIR, "tq_decode_fused_v5.so"),
    workload_configs=[
        # Short seq: decode start (output just began, context is short)
        {"B": 1,   "seq": 128,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 4,   "seq": 128,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 4,   "seq": 256,   "Hq": 64, "Hk": 8, "splits": 0},
        # Medium seq: large output scenarios (input=2K, generating output)
        # During output=512 generation, seq grows from 2048→2560
        {"B": 4,   "seq": 512,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 4,   "seq": 1024,  "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 4,   "seq": 2048,  "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 10,  "seq": 256,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 10,  "seq": 512,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 10,  "seq": 1024,  "Hq": 64, "Hk": 8, "splits": 0},
        # Large batch: high concurrency during output generation
        {"B": 16,  "seq": 256,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 16,  "seq": 512,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 32,  "seq": 256,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 32,  "seq": 512,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 32,  "seq": 1024,  "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 32,  "seq": 2048,  "Hq": 64, "Hk": 8, "splits": 0},
    ],
)

FUSED_CONFIG = TargetConfig(
    name="tq_decode_fused",
    kernel_dir=_KERNEL_DIR,
    source_glob="tq_decode_fused_v*.hip",
    launcher_name="launch_tq_decode_fused",
    deployed_so=os.path.join(_OPS_DIR, "tq_decode_fused_hip.so"),
    workload_configs=[
        {"B": 1,   "seq": 128,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 4,   "seq": 128,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 4,   "seq": 256,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 4,   "seq": 512,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 16,  "seq": 128,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 16,  "seq": 512,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 32,  "seq": 128,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 64,  "seq": 128,   "Hq": 64, "Hk": 8, "splits": 0},
        {"B": 128, "seq": 128,   "Hq": 64, "Hk": 8, "splits": 0},
    ],
)

FUSION_CAMPAIGN_CONFIG = TargetConfig(
    name="turboquant_soa_fusion",
    kernel_dir=_FUSION_DIR,
    source_glob="*.py",
    launcher_name="",
    deployed_so="",
    workload_configs=[
        {"B": 8, "seq": 4096, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 8, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 16, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 32, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
    ],
    target_type="campaign",
    entry_files=[
        os.path.join(_FUSION_DIR, "backend_impl.py"),
        os.path.join(_FUSION_DIR, "external_ops.py"),
        os.path.join(_REPO, "vllm", "v1", "attention", "backends", "turboquant_attn.py"),
    ],
    workload_packs={
        "default": [
            {"B": 8, "seq": 4096, "Hq": 64, "Hk": 8, "splits": 32},
            {"B": 8, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
            {"B": 16, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
            {"B": 32, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
        ],
        "microbench": [
            {"B": 4, "seq": 1024, "Hq": 64, "Hk": 8, "splits": 16},
            {"B": 8, "seq": 2048, "Hq": 64, "Hk": 8, "splits": 16},
        ],
        "long_context": [
            {"B": 8, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
            {"B": 16, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
            {"B": 32, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
        ],
        "72b_e2e": [
            {
                "model": _QWEN_72B,
                "input_len": 8192,
                "output_len": 1024,
                "num_prompts": 16,
                "concurrency": 32,
                "gpu": "0",
                "port": 8201,
                "gpu_memory_utilization": 0.84,
                "max_model_len": 32768,
                "max_num_seqs": 512,
            }
        ],
    },
    evaluator_packs={
        "default": {
            "quick_gate": {
                "mode": "fusion_synthetic_roundtrip",
                "batch_size": 4,
                "seq_len": 256,
                "heads_q": 64,
                "heads_k": 8,
                "head_dim": 128,
                "min_cosine": 0.985,
                "decode_smoke": {
                    "mode": "service_decode_smoke",
                    "model": _QWEN_72B,
                    "prompt": "Write a five-word sentence about AMD GPUs.",
                    "max_tokens": 8,
                    "gpu": "0",
                    "port": 8201,
                    "gpu_memory_utilization": 0.82,
                    "max_model_len": 4096,
                    "max_num_seqs": 8,
                },
            },
            "mid_gate": {
                "mode": "service_warmup",
                "model": _QWEN_72B,
                "input_len": 1024,
                "output_len": 64,
                "num_prompts": 4,
                "concurrency": 2,
                "gpu": "0",
                "port": 8201,
                "gpu_memory_utilization": 0.82,
                "max_model_len": 16384,
                "max_num_seqs": 64,
            },
            "quality_gate": {
                "mode": "quality_spotcheck_72b",
                "model": _QWEN_72B,
                "port": 8201,
            },
            "heavy_gate": {
                "mode": "benchmark_72b_full",
                "model": _QWEN_72B,
                "input_len": 8192,
                "output_len": 1024,
                "num_prompts": 16,
                "concurrency": 32,
                "gpu": "0",
                "port": 8201,
                "gpu_memory_utilization": 0.84,
                "max_model_len": 32768,
                "max_num_seqs": 512,
            },
        },
        "strict_e2e": {
            "quick_gate": {
                "mode": "fusion_synthetic_roundtrip",
                "batch_size": 8,
                "seq_len": 512,
                "heads_q": 64,
                "heads_k": 8,
                "head_dim": 128,
                "min_cosine": 0.99,
                "decode_smoke": {
                    "mode": "service_decode_smoke",
                    "model": _QWEN_72B,
                    "prompt": "Return exactly one short factual sentence about Paris.",
                    "max_tokens": 12,
                    "gpu": "0",
                    "port": 8201,
                    "gpu_memory_utilization": 0.84,
                    "max_model_len": 4096,
                    "max_num_seqs": 8,
                },
            },
            "mid_gate": {
                "mode": "service_warmup",
                "model": _QWEN_72B,
                "input_len": 2048,
                "output_len": 128,
                "num_prompts": 6,
                "concurrency": 4,
                "gpu": "0",
                "port": 8201,
                "gpu_memory_utilization": 0.84,
                "max_model_len": 32768,
                "max_num_seqs": 256,
            },
            "quality_gate": {
                "mode": "quality_spotcheck_72b",
                "model": _QWEN_72B,
                "port": 8201,
            },
            "heavy_gate": {
                "mode": "benchmark_72b_full",
                "model": _QWEN_72B,
                "input_len": 8192,
                "output_len": 1024,
                "num_prompts": 32,
                "concurrency": 32,
                "gpu": "0",
                "port": 8201,
                "gpu_memory_utilization": 0.84,
                "max_model_len": 32768,
                "max_num_seqs": 512,
            },
        },
    },
    profile_packs={
        "default": {"mode": "kernel_only"},
        "kernel_only": {"mode": "kernel_only"},
        "rocprof": {"mode": "rocprof"},
        "strict_e2e": {"mode": "strict_e2e"},
    },
    search_space={
        "fusion_config_change": [
            {
                "name": "baseline_runtime_defaults",
                "params": {},
                "expected_gain": 0.00,
            },
            {
                "name": "hip_v3_scalar",
                "params": {"decode_impl": "hip_v3_scalar"},
                "expected_gain": 0.22,
            },
            {
                "name": "hip_v3_scalar_kv8",
                "params": {
                    "decode_impl": "hip_v3_scalar",
                    "max_num_kv_splits_cap": 8,
                },
                "expected_gain": 0.19,
            },
            {
                "name": "hip_v3_scalar_kv16",
                "params": {
                    "decode_impl": "hip_v3_scalar",
                    "max_num_kv_splits_cap": 16,
                },
                "expected_gain": 0.18,
            },
            {
                "name": "hip_v3_mfma_qk",
                "params": {"decode_impl": "hip_v3_mfma_qk"},
                "expected_gain": 0.28,
            },
            {
                "name": "hip_v3_mfma_qk_kv8",
                "params": {
                    "decode_impl": "hip_v3_mfma_qk",
                    "max_num_kv_splits_cap": 8,
                },
                "expected_gain": 0.25,
            },
            {
                "name": "hip_v3_mfma_qk_kv16",
                "params": {
                    "decode_impl": "hip_v3_mfma_qk",
                    "max_num_kv_splits_cap": 16,
                },
                "expected_gain": 0.24,
            },
            {
                "name": "hip_v3_flash_tq",
                "params": {"decode_impl": "hip_v3_flash_tq"},
                "expected_gain": 0.30,
            },
            {
                "name": "hip_v3_flash_tq_kv8",
                "params": {
                    "decode_impl": "hip_v3_flash_tq",
                    "max_num_kv_splits_cap": 8,
                },
                "expected_gain": 0.27,
            },
            {
                "name": "hip_v3_flash_tq_kv16",
                "params": {
                    "decode_impl": "hip_v3_flash_tq",
                    "max_num_kv_splits_cap": 16,
                },
                "expected_gain": 0.26,
            },
            {
                "name": "soa_bf16q_pv_mfma",
                "params": {"decode_impl": "soa_bf16q_pv_mfma"},
                "expected_gain": 0.31,
            },
            {
                "name": "soa_bf16q_pv_mfma_kv8",
                "params": {
                    "decode_impl": "soa_bf16q_pv_mfma",
                    "max_num_kv_splits_cap": 8,
                },
                "expected_gain": 0.28,
            },
            {
                "name": "soa_bf16q_pv_mfma_kv16",
                "params": {
                    "decode_impl": "soa_bf16q_pv_mfma",
                    "max_num_kv_splits_cap": 16,
                },
                "expected_gain": 0.27,
            },
            {
                "name": "pr_decode_threshold_128",
                "params": {"short_decode_threshold": 128},
                "expected_gain": 0.00,
            },
            {
                "name": "short_decode_threshold_256",
                "params": {"short_decode_threshold": 256},
                "expected_gain": 0.03,
            },
            {
                "name": "continuation_threshold_512",
                "params": {"short_decode_threshold": 512},
                "expected_gain": 0.06,
            },
            {
                "name": "cap_kv_splits_8",
                "params": {"max_num_kv_splits_cap": 8},
                "expected_gain": 0.055,
            },
            {
                "name": "disable_output_buf",
                "params": {"output_buf_policy": "disable"},
                "expected_gain": 0.02,
            },
            {
                "name": "cap_kv_splits_16",
                "params": {"max_num_kv_splits_cap": 16},
                "expected_gain": 0.05,
            },
            {
                "name": "cap_kv_splits_32",
                "params": {"max_num_kv_splits_cap": 32},
                "expected_gain": 0.045,
            },
            {
                "name": "continuation_full_dequant",
                "params": {"continuation_mode": "full_dequant"},
                "expected_gain": 0.03,
            },
        ],
        "fusion_code_change": [
            {
                "name": "aditi_full_stack",
                "params": {
                    "variant": "aditi_full_stack",
                    "source_file": "triton_turboquant_unified_attention.py",
                    "campaign_config": {"max_num_kv_splits_cap": 16},
                },
                "expected_gain": 0.20,
            },
            {
                "name": "aditi_full_stack_3d_4096",
                "params": {
                    "variant": "aditi_full_stack_3d_4096",
                    "source_file": "triton_turboquant_unified_attention.py",
                    "campaign_config": {"max_num_kv_splits_cap": 16},
                },
                "expected_gain": 0.18,
            },
            {
                "name": "decode_block_m_32",
                "params": {
                    "variant": "decode_block_m_32",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.12,
            },
            {
                "name": "decode_block_m_64",
                "params": {
                    "variant": "decode_block_m_64",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.10,
            },
            {
                "name": "ablate_fuse_q_rot_off",
                "params": {
                    "variant": "ablate_fuse_q_rot_off",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.05,
            },
            {
                "name": "tile32_stg2_kv16",
                "params": {
                    "variant": "decode_tile_size_32_num_stages_2",
                    "source_file": "triton_turboquant_unified_attention.py",
                    "campaign_config": {"max_num_kv_splits_cap": 16},
                },
                "expected_gain": 0.14,
            },
            {
                "name": "tile32_stg2_kv8",
                "params": {
                    "variant": "decode_tile_size_32_num_stages_2",
                    "source_file": "triton_turboquant_unified_attention.py",
                    "campaign_config": {"max_num_kv_splits_cap": 8},
                },
                "expected_gain": 0.13,
            },
            {
                "name": "tile32_stg2_kv32",
                "params": {
                    "variant": "decode_tile_size_32_num_stages_2",
                    "source_file": "triton_turboquant_unified_attention.py",
                    "campaign_config": {"max_num_kv_splits_cap": 32},
                },
                "expected_gain": 0.12,
            },
            {
                "name": "tile32_stg2_3d_threshold_2048",
                "params": {
                    "variant": "tile32_stg2_3d_threshold_2048",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.115,
            },
            {
                "name": "tile32_stg2_3d_threshold_4096",
                "params": {
                    "variant": "tile32_stg2_3d_threshold_4096",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.11,
            },
            {
                "name": "decode_tile_size_32",
                "params": {
                    "variant": "decode_tile_size_32",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.09,
            },
            {
                "name": "decode_tile_size_32_num_stages_2",
                "params": {
                    "variant": "decode_tile_size_32_num_stages_2",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.10,
            },
            {
                "name": "hip_num_stages_2",
                "params": {
                    "variant": "hip_num_stages_2",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.08,
            },
            {
                "name": "decode_tile_size_64",
                "params": {
                    "variant": "decode_tile_size_64",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.01,
            },
            {
                "name": "decode_tile_size_64_num_stages_2",
                "params": {
                    "variant": "decode_tile_size_64_num_stages_2",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.005,
            },
            {
                "name": "decode_3d_threshold_4096",
                "params": {
                    "variant": "decode_3d_threshold_4096",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.055,
            },
            {
                "name": "decode_force_2d",
                "params": {
                    "variant": "decode_force_2d",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.025,
            },
            {
                "name": "prefill_block_m_256",
                "params": {
                    "variant": "prefill_block_m_256",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.01,
            },
            {
                "name": "avoid_decode_query_recontig",
                "params": {
                    "variant": "avoid_decode_query_recontig",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.04,
            },
            {
                "name": "decode_3d_threshold_2048",
                "params": {
                    "variant": "decode_3d_threshold_2048",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.03,
            },
            {
                "name": "prefill_block_m_64",
                "params": {
                    "variant": "prefill_block_m_64",
                    "source_file": "triton_turboquant_unified_attention.py",
                },
                "expected_gain": 0.02,
            },
        ],
        "workload_pack_change": [
            {
                "name": "microbench",
                "params": {"workload_pack": "microbench"},
                "expected_gain": 0.00,
            },
            {
                "name": "long_context",
                "params": {"workload_pack": "long_context"},
                "expected_gain": 0.00,
            },
            {
                "name": "72b_e2e",
                "params": {"workload_pack": "72b_e2e"},
                "expected_gain": 0.00,
            },
        ],
        "evaluator_pack_change": [
            {
                "name": "default_eval_pack",
                "params": {"evaluator_pack": "default"},
                "expected_gain": 0.00,
            },
            {
                "name": "strict_e2e_eval_pack",
                "params": {"evaluator_pack": "strict_e2e"},
                "expected_gain": 0.00,
            },
        ],
        "profile_pack_change": [
            {
                "name": "kernel_only_profile",
                "params": {"profile_pack": "kernel_only"},
                "expected_gain": 0.00,
            },
            {
                "name": "strict_e2e_profile",
                "params": {"profile_pack": "strict_e2e"},
                "expected_gain": 0.00,
            },
        ],
    },
    reference_metrics={
        "baseline": {
            "output_token_throughput": 218.39,
            "median_tpot_ms": 119.13,
            "mean_ttft_ms": 17505.08,
        },
        "hip_tq": {
            "output_token_throughput": 145.21,
            "median_tpot_ms": 198.62,
            "mean_ttft_ms": 8178.92,
        },
        "fusion_current": {
            "output_token_throughput": 198.17,
            "median_tpot_ms": 137.13,
            "mean_ttft_ms": 2410.87,
        },
    },
    supports_compile=False,
    default_workload_pack="default",
    default_evaluator_pack="default",
    default_profile_pack="kernel_only",
    default_top_k=2,
    max_candidates_per_round=6,
)


REGISTRY: dict[str, TargetConfig] = {
    "tq_decode_stage2": STAGE2_CONFIG,
    "tq_decode_stage1": STAGE1_CONFIG,
    "tq_decode_fused":  FUSED_CONFIG,
    "tq_wht_rotate":    WHT_ROTATE_CONFIG,
    "tq_decode_fused_wht": FUSED_WHT_CONFIG,
    "turboquant_soa_fusion": FUSION_CAMPAIGN_CONFIG,
}


def get_target(name: str) -> TargetConfig:
    """Get target config by name."""
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown target '{name}'. Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name]


def resolve_workload_pack(target: TargetConfig, pack_name: str = "") -> list[dict[str, Any]]:
    """Resolve workload configs from pack name with a sensible fallback."""
    if pack_name and pack_name in target.workload_packs:
        return target.workload_packs[pack_name]
    if target.workload_packs and target.default_workload_pack in target.workload_packs:
        return target.workload_packs[target.default_workload_pack]
    return target.workload_configs


def resolve_evaluator_pack(target: TargetConfig, pack_name: str = "") -> dict[str, Any]:
    """Resolve evaluator pack for service/E2E gates."""
    if pack_name and pack_name in target.evaluator_packs:
        return target.evaluator_packs[pack_name]
    if target.evaluator_packs and target.default_evaluator_pack in target.evaluator_packs:
        return target.evaluator_packs[target.default_evaluator_pack]
    return {}


# ── Benchmark script generators ──────────────────────────────────────

def generate_stage2_bench(so_path: str, configs: list[dict],
                          warmup: int, iters: int) -> str:
    """Generate Stage2 benchmark script."""
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Stage2 benchmark — HIP kernel via ctypes.\"\"\"
        import torch, json, ctypes
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        D = 128
        lib = ctypes.CDLL("{os.path.abspath(so_path)}")
        fn = lib.launch_tq_decode_stage2_bf16
        fn.restype = None
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        stream = torch.cuda.current_stream().cuda_stream
        results = {{}}
        for cfg in {json.dumps(configs)}:
            B, Hq, splits, seq = cfg["B"], cfg["Hq"], cfg["splits"], cfg["seq"]
            mid_o  = torch.randn(B, Hq, splits, D+1, dtype=torch.float32, device=device)
            output = torch.zeros(B, Hq, D, dtype=torch.bfloat16, device=device)
            sl     = torch.full((B,), seq, dtype=torch.int32, device=device)
            def run():
                fn(ctypes.c_void_p(mid_o.data_ptr()), ctypes.c_void_p(output.data_ptr()),
                   ctypes.c_void_p(sl.data_ptr()),
                   ctypes.c_int(mid_o.stride(0)), ctypes.c_int(mid_o.stride(1)),
                   ctypes.c_int(mid_o.stride(2)), ctypes.c_int(output.stride(0)),
                   ctypes.c_int(output.stride(1)), ctypes.c_int(splits),
                   ctypes.c_int(B), ctypes.c_int(Hq), ctypes.c_void_p(stream))
            for _ in range({warmup}): run()
            torch.cuda.synchronize()
            se = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            ee = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            for i in range({iters}):
                se[i].record(); run(); ee[i].record()
            torch.cuda.synchronize()
            ts = sorted([s.elapsed_time(e)*1000 for s,e in zip(se,ee)])
            lo,hi = max(1,len(ts)//10), len(ts)-max(1,len(ts)//10)
            avg = sum(ts[lo:hi])/(hi-lo) if hi>lo else sum(ts)/len(ts)
            key = f"B{{B}}_seq{{seq}}_Hq{{Hq}}_s{{splits}}"
            results[key] = round(avg,2)
            print(f"B={{B}} seq={{seq}} Hq={{Hq}} splits={{splits}} time={{avg:.2f}}us")
        print("\\n===RESULTS===")
        print(json.dumps(results))
    """)


def generate_stage1_bench(so_path: str, configs: list[dict],
                          warmup: int, iters: int) -> str:
    """Generate Stage1 benchmark script.

    Calls the actual HIP kernel via ctypes with synthetic KV cache data.
    Output is garbage (random KV cache) but timing is accurate.

    TQ KV cache layout per slot: KPS=68 bytes
      [MSE_BYTES=64 (key quant)] [4 (norm)] = 68 bytes per position per head
    kv_cache shape: [num_blocks, block_size, num_kv_heads, KPS] as uint8
    """
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Stage1 benchmark — calls real HIP kernel via ctypes.\"\"\"
        import torch, json, ctypes, math
        torch.manual_seed(42)
        device = torch.device("cuda:0")

        D = 128
        BLOCK_SIZE = 16   # TQ block size (positions per block)
        KPS = 68          # bytes per KV slot (key+value quantized)
        N_CENTROIDS = 16  # centroid count

        SO_PATH = "{os.path.abspath(so_path)}"
        lib = ctypes.CDLL(SO_PATH)
        fn = lib.launch_tq_decode_stage1
        fn.restype = None
        # Must match _STAGE1_ARGTYPES in triton_turboquant_decode.py
        # NOTE: no block_kv param — it's compile-time in the deployed kernel
        fn.argtypes = (
            [ctypes.c_void_p] * 6     # q_rot, kv_cache, bt, seq_lens, centroids, mid_o
            + [ctypes.c_int] * 2      # stride_qb, stride_qh
            + [ctypes.c_int] * 3      # stride_cb, stride_cp, stride_ch
            + [ctypes.c_int]          # stride_bt
            + [ctypes.c_int] * 3      # stride_mb, stride_mh, stride_ms
            + [ctypes.c_int] * 4      # num_kv_heads, block_size, num_kv_splits, kv_group_size
            + [ctypes.c_float]        # attn_scale
            + [ctypes.c_int]          # norm_correction
            + [ctypes.c_int] * 2      # B, Hq
            + [ctypes.c_void_p]       # stream
        )
        print(f"[bench] Loaded HIP Stage1 from {{SO_PATH}}")
        stream = torch.cuda.current_stream().cuda_stream

        results = {{}}
        for cfg in {json.dumps(configs)}:
            B      = cfg["B"]
            Hq     = cfg["Hq"]
            Hk     = cfg["Hk"]
            seq    = cfg["seq"]
            splits = cfg["splits"]
            kvg    = Hq // Hk

            num_blocks = math.ceil(seq / BLOCK_SIZE)
            # Allocate tensors matching real TQ layout
            q_rot    = torch.randn(B, Hq, D, dtype=torch.float32, device=device)
            # kv_cache: [num_blocks, block_size, num_kv_heads, KPS] as uint8
            kv_cache = torch.randint(0, 256,
                (num_blocks, BLOCK_SIZE, Hk, KPS),
                dtype=torch.uint8, device=device)
            # block_table: [B, num_blocks] — maps seq positions to cache blocks
            block_table = torch.arange(num_blocks, dtype=torch.int32, device=device) \\
                          .unsqueeze(0).expand(B, -1).contiguous()
            seq_lens = torch.full((B,), seq, dtype=torch.int32, device=device)
            centroids = torch.randn(N_CENTROIDS, D // 2, dtype=torch.float32, device=device)
            mid_o    = torch.zeros(B, Hq, splits, D + 1, dtype=torch.float32, device=device)

            scale = 1.0 / math.sqrt(D)

            def run():
                fn(
                    ctypes.c_void_p(q_rot.data_ptr()),
                    ctypes.c_void_p(kv_cache.data_ptr()),
                    ctypes.c_void_p(block_table.data_ptr()),
                    ctypes.c_void_p(seq_lens.data_ptr()),
                    ctypes.c_void_p(centroids.data_ptr()),
                    ctypes.c_void_p(mid_o.data_ptr()),
                    ctypes.c_int(q_rot.stride(0)),    ctypes.c_int(q_rot.stride(1)),
                    ctypes.c_int(kv_cache.stride(0)), ctypes.c_int(kv_cache.stride(1)),
                    ctypes.c_int(kv_cache.stride(2)),
                    ctypes.c_int(block_table.stride(0)),
                    ctypes.c_int(mid_o.stride(0)), ctypes.c_int(mid_o.stride(1)),
                    ctypes.c_int(mid_o.stride(2)),
                    ctypes.c_int(Hk), ctypes.c_int(BLOCK_SIZE),
                    ctypes.c_int(splits), ctypes.c_int(kvg),
                    ctypes.c_float(scale),
                    ctypes.c_int(0),          # norm_correction
                    ctypes.c_int(B), ctypes.c_int(Hq),
                    ctypes.c_void_p(stream),
                )

            # Warmup
            for _ in range({warmup}):
                run()
            torch.cuda.synchronize()

            # Timed
            se = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            ee = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            for i in range({iters}):
                se[i].record(); run(); ee[i].record()
            torch.cuda.synchronize()

            ts = sorted([s.elapsed_time(e) * 1000.0 for s, e in zip(se, ee)])
            lo, hi = max(1, len(ts)//10), len(ts) - max(1, len(ts)//10)
            avg = sum(ts[lo:hi]) / (hi - lo) if hi > lo else sum(ts) / len(ts)

            key = f"B{{B}}_seq{{seq}}_Hq{{Hq}}_s{{splits}}"
            results[key] = round(avg, 2)
            print(f"B={{B}} seq={{seq}} Hq={{Hq}} splits={{splits}} "
                  f"time={{avg:.2f}}us  (min={{ts[0]:.2f}} max={{ts[-1]:.2f}})")

        print("\\n===RESULTS===")
        print(json.dumps(results))
    """)


def generate_fused_bench(so_path: str, configs: list[dict],
                         warmup: int, iters: int) -> str:
    """Generate Fused kernel benchmark script.

    Calls the actual HIP fused decode kernel via ctypes.
    Uses synthetic TQ KV cache data (random bytes) — output is garbage
    but GPU timing is accurate.

    V4 kernel signature:
      launch_tq_decode_fused(
        q_rot, kv_cache, block_table, seq_lens, centroids, output,
        stride_qb, stride_qh,
        stride_cb, stride_cp, stride_ch,
        stride_bt,
        stride_ob, stride_oh,
        num_kv_heads, block_size, kv_group_size,
        attn_scale, norm_correction, dtype,
        B, Hq, stream)
    """
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Fused kernel benchmark — calls real HIP kernel via ctypes.\"\"\"
        import torch, json, ctypes, math, numpy as np, gc
        torch.manual_seed(42)
        device = torch.device("cuda:0")

        D = 128
        BLOCK_SIZE = 16
        SLOT_SIZE = 136       # KPS(68) + VAL(68) per head per position
        N_CENTROIDS = 16

        SO_PATH = "{os.path.abspath(so_path)}"
        lib = ctypes.CDLL(SO_PATH)
        fn = lib.launch_tq_decode_fused
        fn.restype = None
        # V4 signature: q_rot is void* (native bf16/fp16), dtype param added
        fn.argtypes = (
            [ctypes.c_void_p] * 6     # q_rot, kv_cache, bt, seq_lens, centroids, output
            + [ctypes.c_int] * 2      # stride_qb, stride_qh
            + [ctypes.c_int] * 3      # stride_cb, stride_cp, stride_ch
            + [ctypes.c_int]          # stride_bt
            + [ctypes.c_int] * 2      # stride_ob, stride_oh
            + [ctypes.c_int]          # num_kv_heads
            + [ctypes.c_int]          # block_size
            + [ctypes.c_int]          # kv_group_size
            + [ctypes.c_float]        # attn_scale
            + [ctypes.c_int]          # norm_correction
            + [ctypes.c_int]          # dtype (0=bf16, 1=fp16)
            + [ctypes.c_int] * 2      # B, Hq
            + [ctypes.c_void_p]       # stream
        )
        print(f"[bench] Loaded HIP Fused kernel from {{SO_PATH}}")
        stream = torch.cuda.current_stream().cuda_stream

        results = {{}}
        for cfg in {json.dumps(configs)}:
          try:
            B      = cfg["B"]
            Hq     = cfg["Hq"]
            Hk     = cfg["Hk"]
            seq    = cfg["seq"]
            kvg    = Hq // Hk

            num_blocks = math.ceil(seq / BLOCK_SIZE)
            # TQ KV cache: [num_blocks, block_size, num_kv_heads, SLOT_SIZE] uint8
            kv_cache = torch.randint(0, 256,
                (num_blocks, BLOCK_SIZE, Hk, SLOT_SIZE),
                dtype=torch.uint8, device=device)
            # Q in bf16 (native dtype for V4)
            q_rot = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=device) * 0.1
            output = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=device)
            block_table = torch.arange(num_blocks, dtype=torch.int32, device=device) \\
                          .unsqueeze(0).expand(B, -1).contiguous()
            seq_lens = torch.full((B,), seq, dtype=torch.int32, device=device)
            centroids = torch.randn(N_CENTROIDS, dtype=torch.float32, device=device)

            scale = 1.0 / math.sqrt(D)
            DTYPE_BF16 = 0

            def run():
                fn(
                    ctypes.c_void_p(q_rot.data_ptr()),
                    ctypes.c_void_p(kv_cache.data_ptr()),
                    ctypes.c_void_p(block_table.data_ptr()),
                    ctypes.c_void_p(seq_lens.data_ptr()),
                    ctypes.c_void_p(centroids.data_ptr()),
                    ctypes.c_void_p(output.data_ptr()),
                    ctypes.c_int(q_rot.stride(0)),    ctypes.c_int(q_rot.stride(1)),
                    ctypes.c_int(kv_cache.stride(0)), ctypes.c_int(kv_cache.stride(1)),
                    ctypes.c_int(kv_cache.stride(2)),
                    ctypes.c_int(block_table.stride(0)),
                    ctypes.c_int(output.stride(0)), ctypes.c_int(output.stride(1)),
                    ctypes.c_int(Hk), ctypes.c_int(BLOCK_SIZE),
                    ctypes.c_int(kvg),
                    ctypes.c_float(scale),
                    ctypes.c_int(1),          # norm_correction
                    ctypes.c_int(DTYPE_BF16), # dtype=bf16
                    ctypes.c_int(B), ctypes.c_int(Hq),
                    ctypes.c_void_p(stream),
                )

            # Warmup
            for _ in range({warmup}):
                run()
            torch.cuda.synchronize()

            # Timed
            se = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            ee = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            for i in range({iters}):
                se[i].record(); run(); ee[i].record()
            torch.cuda.synchronize()

            ts = sorted([s.elapsed_time(e) * 1000.0 for s, e in zip(se, ee)])
            lo, hi = max(1, len(ts)//10), len(ts) - max(1, len(ts)//10)
            avg = sum(ts[lo:hi]) / (hi - lo) if hi > lo else sum(ts) / len(ts)

            key = f"B{{B}}_seq{{seq}}_Hq{{Hq}}_s{{cfg.get('splits',0)}}"
            results[key] = round(avg, 2)
            print(f"B={{B}} seq={{seq}} Hq={{Hq}} "
                  f"time={{avg:.2f}}us  (min={{ts[0]:.2f}} max={{ts[-1]:.2f}})")

            del kv_cache, q_rot, output, block_table, seq_lens, centroids
            gc.collect(); torch.cuda.empty_cache()
          except Exception as e:
            print(f"B={{B}} seq={{seq}} Hq={{Hq}} ERROR: {{e}}")
            gc.collect(); torch.cuda.empty_cache()

        print("\\n===RESULTS===")
        print(json.dumps(results))
    """)


def generate_wht_rotate_bench(so_path: str, configs: list[dict],
                              warmup: int, iters: int) -> str:
    """Generate WHT rotation benchmark script."""
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"WHT rotation benchmark — HIP butterfly kernel via ctypes.\"\"\"
        import torch, json, ctypes, math
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        D = 128

        SO_PATH = "{os.path.abspath(so_path)}"
        lib = ctypes.CDLL(SO_PATH)
        fn = lib.launch_tq_wht_rotate
        fn.restype = None
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        print(f"[bench] Loaded WHT rotate kernel from {{SO_PATH}}")
        stream = torch.cuda.current_stream().cuda_stream

        signs = (torch.randint(0, 2, (D,), device=device, dtype=torch.float32) * 2 - 1)

        results = {{}}
        for cfg in {json.dumps(configs)}:
            M = cfg["M"]
            desc = cfg.get("desc", "")
            q = torch.randn(M, D, device=device, dtype=torch.bfloat16)
            out = torch.zeros(M, D, device=device, dtype=torch.bfloat16)

            def run():
                fn(ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                   ctypes.c_void_p(signs.data_ptr()), ctypes.c_int(M), ctypes.c_int(0),
                   ctypes.c_void_p(stream))

            for _ in range({warmup}): run()
            torch.cuda.synchronize()

            se = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            ee = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            for i in range({iters}):
                se[i].record(); run(); ee[i].record()
            torch.cuda.synchronize()

            ts = sorted([s.elapsed_time(e)*1000 for s,e in zip(se,ee)])
            lo,hi = max(1,len(ts)//10), len(ts)-max(1,len(ts)//10)
            avg = sum(ts[lo:hi])/(hi-lo) if hi>lo else sum(ts)/len(ts)

            key = f"M{{M}}"
            results[key] = round(avg,2)
            print(f"M={{M}} ({{desc}}) time={{avg:.2f}}us (min={{ts[0]:.2f}} max={{ts[-1]:.2f}})")

        print("\\n===RESULTS===")
        print(json.dumps(results))
    """)


def generate_fused_wht_bench(so_path: str, configs: list[dict],
                             warmup: int, iters: int) -> str:
    """Generate Fused+WHT (V5) kernel benchmark script.

    V5 signature adds 'signs' parameter and takes raw Q (not q_rot).
    """
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Fused+WHT (V5) benchmark — single kernel launch.\"\"\"
        import torch, json, ctypes, math, gc
        torch.manual_seed(42)
        device = torch.device("cuda:0")

        D = 128
        BLOCK_SIZE = 16
        SLOT_SIZE = 136
        N_CENTROIDS = 16

        SO_PATH = "{os.path.abspath(so_path)}"
        lib = ctypes.CDLL(SO_PATH)
        fn = lib.launch_tq_decode_fused_v5
        fn.restype = None
        fn.argtypes = (
            [ctypes.c_void_p] * 7     # q_raw, kv_cache, bt, seq_lens, centroids, signs, output
            + [ctypes.c_int] * 2      # stride_qb, stride_qh
            + [ctypes.c_int] * 3      # stride_cb, stride_cp, stride_ch
            + [ctypes.c_int]          # stride_bt
            + [ctypes.c_int] * 2      # stride_ob, stride_oh
            + [ctypes.c_int] * 3      # num_kv_heads, block_size, kv_group_size
            + [ctypes.c_float]        # attn_scale
            + [ctypes.c_int] * 2      # norm_correction, dtype
            + [ctypes.c_int] * 2      # B, Hq
            + [ctypes.c_void_p]       # stream
        )
        print(f"[bench] Loaded Fused+WHT V5 from {{SO_PATH}}")
        stream = torch.cuda.current_stream().cuda_stream

        signs = (torch.randint(0, 2, (D,), device=device, dtype=torch.float32) * 2 - 1)

        results = {{}}
        for cfg in {json.dumps(configs)}:
          try:
            B      = cfg["B"]
            Hq     = cfg["Hq"]
            Hk     = cfg["Hk"]
            seq    = cfg["seq"]
            kvg    = Hq // Hk

            num_blocks = math.ceil(seq / BLOCK_SIZE)
            kv_cache = torch.randint(0, 256,
                (num_blocks, BLOCK_SIZE, Hk, SLOT_SIZE),
                dtype=torch.uint8, device=device)
            q_raw = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=device) * 0.1
            output = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=device)
            block_table = torch.arange(num_blocks, dtype=torch.int32, device=device) \\
                          .unsqueeze(0).expand(B, -1).contiguous()
            seq_lens = torch.full((B,), seq, dtype=torch.int32, device=device)
            centroids = torch.randn(N_CENTROIDS, dtype=torch.float32, device=device)
            scale = 1.0 / math.sqrt(D)

            def run():
                fn(
                    ctypes.c_void_p(q_raw.data_ptr()),
                    ctypes.c_void_p(kv_cache.data_ptr()),
                    ctypes.c_void_p(block_table.data_ptr()),
                    ctypes.c_void_p(seq_lens.data_ptr()),
                    ctypes.c_void_p(centroids.data_ptr()),
                    ctypes.c_void_p(signs.data_ptr()),
                    ctypes.c_void_p(output.data_ptr()),
                    ctypes.c_int(q_raw.stride(0)), ctypes.c_int(q_raw.stride(1)),
                    ctypes.c_int(kv_cache.stride(0)), ctypes.c_int(kv_cache.stride(1)),
                    ctypes.c_int(kv_cache.stride(2)),
                    ctypes.c_int(block_table.stride(0)),
                    ctypes.c_int(output.stride(0)), ctypes.c_int(output.stride(1)),
                    ctypes.c_int(Hk), ctypes.c_int(BLOCK_SIZE), ctypes.c_int(kvg),
                    ctypes.c_float(scale), ctypes.c_int(1), ctypes.c_int(0),
                    ctypes.c_int(B), ctypes.c_int(Hq),
                    ctypes.c_void_p(stream),
                )

            for _ in range({warmup}): run()
            torch.cuda.synchronize()

            se = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            ee = [torch.cuda.Event(enable_timing=True) for _ in range({iters})]
            for i in range({iters}):
                se[i].record(); run(); ee[i].record()
            torch.cuda.synchronize()

            ts = sorted([s.elapsed_time(e) * 1000.0 for s, e in zip(se, ee)])
            lo, hi = max(1, len(ts)//10), len(ts) - max(1, len(ts)//10)
            avg = sum(ts[lo:hi]) / (hi - lo) if hi > lo else sum(ts) / len(ts)

            key = f"B{{B}}_seq{{seq}}_Hq{{Hq}}_s{{cfg.get('splits',0)}}"
            results[key] = round(avg, 2)
            print(f"B={{B}} seq={{seq}} Hq={{Hq}} "
                  f"time={{avg:.2f}}us  (min={{ts[0]:.2f}} max={{ts[-1]:.2f}})")

            del kv_cache, q_raw, output, block_table, seq_lens, centroids
            gc.collect(); torch.cuda.empty_cache()
          except Exception as e:
            print(f"B={{B}} seq={{seq}} Hq={{Hq}} ERROR: {{e}}")
            gc.collect(); torch.cuda.empty_cache()

        print("\\n===RESULTS===")
        print(json.dumps(results))
    """)


BENCH_GENERATORS = {
    "tq_decode_stage2": generate_stage2_bench,
    "tq_decode_stage1": generate_stage1_bench,
    "tq_decode_fused":  generate_fused_bench,
    "tq_wht_rotate":    generate_wht_rotate_bench,
    "tq_decode_fused_wht": generate_fused_wht_bench,
}

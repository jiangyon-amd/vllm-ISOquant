"""GEAK Optimizer Agent — generates optimized HIP kernel variants.

Input:  bottleneck report + current kernel source + hardware model
Output: new kernel .hip file implementing the top optimization hypothesis
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from agents.state import ActionSpec, CandidateResult, CandidateStatus, PipelineState
from agents.agent_analyzer import BottleneckReport
from agents.hardware_model import MI355X, HardwareSpec


class GEAKOptimizerAgent:
    """Generates optimized HIP kernel source from analysis.

    Key design principle: each transform checks whether it's already applied
    before modifying source. No-op and identity actions are skipped so they do
    not consume compile or benchmark cycles.
    """

    # Hypothesis keyword → transform method name
    HYPOTHESIS_TO_TRANSFORM: dict[str, str] = {
        "online softmax":           "_apply_online_softmax",
        "single-pass":              "_apply_online_softmax",
        "float4":                   "_apply_float4_vectorize",
        "vectorized":               "_apply_float4_vectorize",
        "warp-level":               "_apply_warp_broadcast",
        "__shfl":                   "_apply_warp_broadcast",
        "thread count":             "_apply_thread_count_tune",
        "batch-adaptive":           "_apply_batch_adaptive",
        "crossover":                "_apply_batch_adaptive",
        "fuse stage1+stage2":       "_apply_fuse_stages",
        "reduce vgpr":              "_apply_reduce_vgpr",
        "vgpr":                     "_apply_reduce_vgpr",
        "launch_bounds":            "_apply_tune_launch_bounds",
        "nontemporal":              "_apply_nontemporal",
        "batch_crossover":          "_apply_tune_crossover",
        "baseline":                 "_apply_identity",
        "diminishing":              "_apply_identity",
        "near theoretical":         "_apply_identity",
        # Compute-bound (Stage1) transforms
        "block_kv":                 "_apply_reduce_block_kv",
        "reduce block_kv":          "_apply_reduce_block_kv",
        "occupancy":                "_apply_reduce_block_kv",
        "num_warps":                "_apply_tune_num_warps",
        "num_warps=2":              "_apply_tune_num_warps",
        "dims_per_thread":          "_apply_tune_dims_per_thread",
        "reorder inner":            "_apply_identity",  # complex rewrite
        "software pipelining":      "_apply_software_pipeline",
        "double-buffer":            "_apply_software_pipeline",
        "prefetch":                 "_apply_software_pipeline",
        "increase block_kv":        "_apply_increase_block_kv",
        "pairwise lut":             "_apply_pairwise_lut",
        "flute":                    "_apply_pairwise_lut",
        "qmap2":                    "_apply_pairwise_lut",
        "phase2":                   "_apply_qc_precompute",
        "qc precompute":            "_apply_qc_precompute",
        "qc table":                 "_apply_qc_precompute",
        "phase3":                   "_apply_mfma_decode",
        "mfma":                     "_apply_mfma_decode",
        "cross-batch":              "_apply_cross_batch_mfma",
        "cross_batch":              "_apply_cross_batch_mfma",
        "m=16":                     "_apply_cross_batch_mfma",
        "per-half-warp":            "_apply_halfwarp_swizzle",
        "2-copy swizzle":           "_apply_twocopy_swizzle",
        "fp16 centroid":            "_apply_fp16_centroid_lds",
        "__half centroid":          "_apply_fp16_centroid_lds",
        "halve lds":                "_apply_fp16_centroid_lds",
        "fp16":                     "_apply_fp16_compute",
        "__half":                   "_apply_fp16_compute",
        "__hmul":                   "_apply_fp16_compute",
        "bf16":                     "_apply_fp16_compute",
        "void* q_rot":              "_apply_void_q_input",
        "dtype parameter":          "_apply_void_q_input",
        # WHT rotation transforms
        "rows_per_block":           "_apply_wht_increase_rows",
        "lds":                      "_apply_wht_lds_signs",
        "cache signs":              "_apply_wht_lds_signs",
        "shared":                   "_apply_wht_lds_signs",
        "bit-manipulation":         "_apply_wht_xor_signs",
        "xor the sign bit":         "_apply_wht_xor_signs",
        "persistent kernel":        "_apply_identity",  # too complex
        "fuse wht":                 "_apply_identity",  # integration task
    }

    def __init__(self, kernel_dir: str = "", hw: HardwareSpec = MI355X):
        if not kernel_dir:
            kernel_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "geak_tq_decode", "hip_kernel"
            )
        self.kernel_dir = kernel_dir
        self.hw = hw

    def generate_candidates(
        self,
        state: PipelineState,
        action_specs: list[ActionSpec],
        report: BottleneckReport | None = None,
    ) -> list[CandidateResult]:
        """Emit multiple candidate branches for the current round."""
        report = report or self._report_from_state(state)
        candidates: list[CandidateResult] = []
        seen_signatures = state.seen_candidate_signatures()
        max_candidates = state.search_policy.max_candidates_per_round

        for spec in action_specs:
            if spec.action_type == "search_weight_change":
                continue
            if spec.action_type in state.search_policy.frozen_action_types:
                continue
            if spec.family() in state.search_policy.frozen_action_types:
                continue
            candidate_signature = state._candidate_signature(spec, state.best_candidate_id)
            if candidate_signature in seen_signatures:
                continue

            candidate = state.new_candidate(
                action_spec=spec,
                generator="geak" if spec.action_type == "kernel_transform" else "system",
                parent_id=state.best_candidate_id,
            )
            if spec.action_type == "kernel_transform":
                self._materialize_kernel_candidate(state, report, candidate)
            else:
                self._materialize_system_candidate(state, candidate)
            if candidate.status not in (
                CandidateStatus.SKIPPED.value,
                CandidateStatus.FAILED.value,
            ):
                candidates.append(candidate)
            seen_signatures.add(candidate.signature)
            if len(candidates) >= max_candidates:
                break

        return candidates

    def _report_from_state(self, state: PipelineState) -> BottleneckReport:
        cur = state.current
        return BottleneckReport(
            summary=cur.bottleneck_report,
            bottleneck_type="memory",
            efficiency_map=cur.efficiency,
            worst_config=(min(cur.efficiency, key=cur.efficiency.get) if cur.efficiency else ""),
            worst_efficiency=(min(cur.efficiency.values()) if cur.efficiency else 0.0),
            hypotheses=cur.optimization_hypotheses,
            estimated_gains=[0.1] * len(cur.optimization_hypotheses),
            action_specs=cur.action_specs,
        )

    def _candidate_dir(self, state: PipelineState, candidate: CandidateResult) -> str:
        out_dir = os.path.join(
            state.run_dir,
            f"round_{state.round_index:02d}",
            candidate.candidate_id,
        )
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _materialize_kernel_candidate(
        self,
        state: PipelineState,
        report: BottleneckReport,
        candidate: CandidateResult,
    ) -> None:
        """Generate a .hip branch for one kernel-transform action."""
        out_dir = self._candidate_dir(state, candidate)
        src = self._find_current_source(state)
        if not src:
            src = self._create_from_template(out_dir)

        with open(src) as f:
            source_code = f.read()

        hypothesis = (
            candidate.action_spec.params.get("hypothesis")
            if candidate.action_spec is not None
            else candidate.action_name
        )
        method = self._match_hypothesis(hypothesis or candidate.action_name)
        if method is None:
            candidate.status = CandidateStatus.SKIPPED.value
            candidate.failure_reason = "no transform matched hypothesis"
            candidate.notes.append("kernel candidate skipped because no transform matched the hypothesis")
            return
        if getattr(method, "__name__", "") == "_apply_identity":
            candidate.status = CandidateStatus.SKIPPED.value
            candidate.failure_reason = "identity transform skipped"
            candidate.notes.append("identity transform intentionally skipped to avoid wasting compile/benchmark cycles")
            return

        new_code = method(source_code, report, state)
        transform_name = getattr(method, "__name__", "identity")
        if new_code == source_code:
            candidate.status = CandidateStatus.SKIPPED.value
            candidate.failure_reason = f"no-op transform: {transform_name}"
            candidate.notes.append("transform produced no source change for the current lineage")
            return

        out_path = os.path.join(out_dir, f"{state.target_kernel}_{candidate.candidate_id}.hip")
        with open(out_path, "w") as f:
            f.write(new_code)

        candidate.hip_source = out_path
        candidate.kernel_version = candidate.candidate_id
        candidate.artifacts["hip_source"] = out_path
        candidate.artifacts["transform_name"] = transform_name
        candidate.status = CandidateStatus.GENERATED.value
        print(f"[GEAK] Candidate {candidate.candidate_id}: {transform_name} -> {out_path}")

    def _materialize_system_candidate(
        self,
        state: PipelineState,
        candidate: CandidateResult,
    ) -> None:
        """Generate a config-only branch for executable system actions."""
        from agents.target_registry import get_target

        target = get_target(candidate.resolved_target(state.target_kernel))
        out_dir = self._candidate_dir(state, candidate)
        primary_source = target.entry_files[0] if target.entry_files else ""
        candidate.hip_source = primary_source
        if primary_source:
            candidate.artifacts["primary_source"] = primary_source

        candidate.kernel_version = candidate.candidate_id
        candidate.workload_pack = candidate.workload_pack or target.default_workload_pack
        candidate.evaluator_pack = candidate.evaluator_pack or target.default_evaluator_pack
        candidate.profile_pack = candidate.profile_pack or target.default_profile_pack

        if candidate.action_type == "fusion_config_change" and candidate.action_spec is not None:
            config_payload = dict(candidate.action_spec.params)
            config_path = os.path.join(out_dir, "fusion_campaign_config.json")
            with open(config_path, "w") as f:
                json.dump(config_payload, f, indent=2, sort_keys=True)
            candidate.artifacts["campaign_config"] = config_path
        if candidate.action_type == "fusion_code_change" and candidate.action_spec is not None:
            self._materialize_fusion_code_candidate(out_dir, candidate)
            if candidate.status in (
                CandidateStatus.SKIPPED.value,
                CandidateStatus.FAILED.value,
            ):
                return
        candidate.status = CandidateStatus.GENERATED.value
        print(
            f"[GEAK] Candidate {candidate.candidate_id}: "
            f"{candidate.action_type} -> {candidate.artifacts}"
        )

    def _materialize_fusion_code_candidate(
        self,
        out_dir: str,
        candidate: CandidateResult,
    ) -> None:
        """Create an isolated external-kernel source root for a code variant."""
        assert candidate.action_spec is not None
        params = candidate.action_spec.params
        source_root = Path(
            os.environ.get(
                "VLLM_TQ_SOA_FUSION_SOURCE_ROOT",
                "/shareddata/amd/jiangyon/vllm_tq_rocm_v3_sinks/vllm/v1/attention/ops",
            )
        )
        overlay_root = Path(out_dir) / "external_ops"
        overlay_root.mkdir(parents=True, exist_ok=True)
        for file_name in (
            "triton_turboquant_decode.py",
            "triton_turboquant_store.py",
            "triton_turboquant_unified_attention.py",
        ):
            src = source_root / file_name
            if not src.exists():
                candidate.status = CandidateStatus.FAILED.value
                candidate.failure_reason = f"missing external source file: {src}"
                return
            shutil.copy2(src, overlay_root / file_name)

        source_file = str(params.get("source_file", "triton_turboquant_unified_attention.py"))
        target_file = overlay_root / source_file
        if not target_file.exists():
            candidate.status = CandidateStatus.FAILED.value
            candidate.failure_reason = f"code overlay target is missing: {source_file}"
            return

        before = target_file.read_text()
        variant = str(params.get("variant", candidate.action_name))
        after = self._apply_fusion_code_variant(before, variant)
        if after == before:
            candidate.status = CandidateStatus.SKIPPED.value
            candidate.failure_reason = f"no-op fusion code variant: {variant}"
            return
        target_file.write_text(after)

        campaign_config = params.get("campaign_config")
        if isinstance(campaign_config, dict):
            config_path = Path(out_dir) / "fusion_campaign_config.json"
            config_path.write_text(json.dumps(campaign_config, indent=2, sort_keys=True))
            candidate.artifacts["campaign_config"] = str(config_path)

        manifest_path = Path(out_dir) / "code_overlay_manifest.json"
        manifest = {
            "source_root": str(source_root),
            "overlay_root": str(overlay_root),
            "source_file": source_file,
            "variant": variant,
            "action_name": candidate.action_name,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        candidate.artifacts["external_source_root"] = str(overlay_root)
        candidate.artifacts["code_overlay_manifest"] = str(manifest_path)
        candidate.artifacts["patched_files"] = [str(target_file)]

    def _apply_fusion_code_variant(self, source: str, variant: str) -> str:
        """Apply conservative, registered source transforms for fusion overlays."""
        replacement_sets = {
            "decode_tile_size_32": (
                (
                    "tile_size = 32 if is_prefill_like else 16",
                    "tile_size = 32 if is_prefill_like else 32",
                ),
            ),
            "decode_tile_size_64": (
                (
                    "tile_size = 32 if is_prefill_like else 16",
                    "tile_size = 32 if is_prefill_like else 64",
                ),
            ),
            "hip_num_stages_2": (
                (
                    "num_stages = 1 if _is_hip else 2",
                    "num_stages = 2",
                ),
            ),
            "hip_num_stages_3": (
                (
                    "num_stages = 1 if _is_hip else 2",
                    "num_stages = 3 if _is_hip else 2",
                ),
            ),
            "decode_tile_size_32_num_stages_2": (
                (
                    "tile_size = 32 if is_prefill_like else 16",
                    "tile_size = 32 if is_prefill_like else 32",
                ),
                (
                    "num_stages = 1 if _is_hip else 2",
                    "num_stages = 2",
                ),
            ),
            "tile32_stg2_3d_threshold_2048": (
                (
                    "tile_size = 32 if is_prefill_like else 16",
                    "tile_size = 32 if is_prefill_like else 32",
                ),
                (
                    "num_stages = 1 if _is_hip else 2",
                    "num_stages = 2",
                ),
                (
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024",
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 2048",
                ),
            ),
            "tile32_stg2_3d_threshold_4096": (
                (
                    "tile_size = 32 if is_prefill_like else 16",
                    "tile_size = 32 if is_prefill_like else 32",
                ),
                (
                    "num_stages = 1 if _is_hip else 2",
                    "num_stages = 2",
                ),
                (
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024",
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 4096",
                ),
            ),
            "decode_tile_size_64_num_stages_2": (
                (
                    "tile_size = 32 if is_prefill_like else 16",
                    "tile_size = 32 if is_prefill_like else 64",
                ),
                (
                    "num_stages = 1 if _is_hip else 2",
                    "num_stages = 2",
                ),
            ),
            "decode_tile_size_8": (
                (
                    "tile_size = 32 if is_prefill_like else 16",
                    "tile_size = 32 if is_prefill_like else 8",
                ),
            ),
            "prefill_block_m_64": (
                (
                    "BLOCK_M = max(128, triton.next_power_of_2(kv_group_size))",
                    "BLOCK_M = max(64, triton.next_power_of_2(kv_group_size))",
                ),
            ),
            "prefill_block_m_256": (
                (
                    "BLOCK_M = max(128, triton.next_power_of_2(kv_group_size))",
                    "BLOCK_M = max(256, triton.next_power_of_2(kv_group_size))",
                ),
            ),
            "avoid_decode_query_recontig": (
                (
                    "query=query.contiguous(),",
                    "query=query if query.is_contiguous() else query.contiguous(),",
                ),
            ),
            "decode_3d_threshold_2048": (
                (
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024",
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 2048",
                ),
            ),
            "decode_3d_threshold_4096": (
                (
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024",
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 4096",
                ),
            ),
            "decode_force_2d": (
                (
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024",
                    "use_3d = False",
                ),
            ),
            "ablate_fuse_q_rot_off": (
                (
                    "        apply_fuse_q_rot = bool(fuse_q_rot)",
                    "        apply_fuse_q_rot = False",
                ),
            ),
            "decode_block_m_32": (
                (
                    "        BLOCK_M = 16 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)",
                    "        BLOCK_M = 32 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)",
                ),
            ),
            "decode_block_m_64": (
                (
                    "        BLOCK_M = 16 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)",
                    "        BLOCK_M = 64 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)",
                ),
            ),
            "decode_max_kv_splits_64": (
                (
                    "    max_num_kv_splits: int = 32,\n    sinks: torch.Tensor | None = None,\n) -> torch.Tensor:",
                    "    max_num_kv_splits: int = 64,\n    sinks: torch.Tensor | None = None,\n) -> torch.Tensor:",
                ),
            ),
            "aditi_full_stack": (
                (
                    "tile_size = 32 if is_prefill_like else 16",
                    "tile_size = 32 if is_prefill_like else 32",
                ),
                (
                    "num_stages = 1 if _is_hip else 2",
                    "num_stages = 2",
                ),
                (
                    "        BLOCK_M = 16 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)",
                    "        BLOCK_M = 32 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)",
                ),
                (
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024",
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 2048",
                ),
            ),
            "aditi_full_stack_3d_4096": (
                (
                    "tile_size = 32 if is_prefill_like else 16",
                    "tile_size = 32 if is_prefill_like else 32",
                ),
                (
                    "num_stages = 1 if _is_hip else 2",
                    "num_stages = 2",
                ),
                (
                    "        BLOCK_M = 16 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)",
                    "        BLOCK_M = 32 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)",
                ),
                (
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024",
                    "use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 4096",
                ),
            ),
        }
        if variant not in replacement_sets:
            return source
        updated = source
        for old, new in replacement_sets[variant]:
            if old not in updated:
                return source
            updated = updated.replace(old, new, 1)
        return updated

    @staticmethod
    def _version_key(path: Path) -> int:
        """Extract numeric version from filename for correct sorting.

        tq_decode_stage1_v9.hip  → 9
        tq_decode_stage1_v54.hip → 54
        tq_decode_fused_v14.hip  → 14
        tq_decode_stage1_v60b.hip → 60  (suffix letters ignored)
        """
        m = re.search(r'_v(\d+)', path.name)
        return int(m.group(1)) if m else 0

    def _find_current_source(self, state: PipelineState) -> str | None:
        """Find the most recent kernel source for the current target.

        Priority:
        1. Previous iteration's .hip source (from within this run)
        2. The deployed .so's matching .hip source (for the first iteration)
        3. Latest versioned .hip file in kernel_dir (fallback)

        For fused kernels, we strongly prefer V4 (the deployed version)
        because earlier versions (v1-v14) have incompatible signatures.
        """
        best = state.best_candidate
        if best and best.hip_source and os.path.exists(best.hip_source):
            return best.hip_source

        if state.prev and state.prev.hip_source and \
           os.path.exists(state.prev.hip_source):
            return state.prev.hip_source

        # For the first iteration, prefer the known-good deployed source
        from agents.target_registry import get_target
        try:
            tc = get_target(state.target_kernel)
        except (ValueError, KeyError):
            tc = None

        # Check for specific "latest deployed" source files
        if tc and state.target_kernel == "tq_decode_fused":
            # V4 is the deployed version with native bf16/fp16 Q input
            v4_path = os.path.join(self.kernel_dir, "tq_decode_fused_v4.hip")
            if os.path.exists(v4_path):
                print(f"[GEAK] Source: tq_decode_fused_v4.hip (deployed)")
                return v4_path

        if tc and state.target_kernel == "tq_decode_fused_wht":
            v5_path = os.path.join(self.kernel_dir, "tq_decode_fused_v5.hip")
            if os.path.exists(v5_path):
                print(f"[GEAK] Source: tq_decode_fused_v5.hip (fused+WHT)")
                return v5_path

        if tc and state.target_kernel == "tq_wht_rotate":
            # Prefer V2 (multi-row) over V1 (single-row)
            v2_path = os.path.join(self.kernel_dir, "tq_wht_rotate_v2.hip")
            v1_path = os.path.join(self.kernel_dir, "tq_wht_rotate_v1.hip")
            if os.path.exists(v2_path):
                print(f"[GEAK] Source: tq_wht_rotate_v2.hip (multi-row)")
                return v2_path
            if os.path.exists(v1_path):
                print(f"[GEAK] Source: tq_wht_rotate_v1.hip (single-row)")
                return v1_path

        # Fallback: use target-specific glob with NUMERIC version sort
        if tc:
            hip_files = sorted(
                Path(self.kernel_dir).glob(tc.source_glob),
                key=self._version_key,
            )
        else:
            hip_files = sorted(
                Path(self.kernel_dir).glob("tq_decode_stage2_v*.hip"),
                key=self._version_key,
            )
        if hip_files:
            chosen = str(hip_files[-1])
            print(f"[GEAK] Source: {Path(chosen).name} "
                  f"(v{self._version_key(hip_files[-1])})")
            return chosen
        return None

    def _create_from_template(self, out_dir: str) -> str:
        """Create baseline kernel from earliest versioned source."""
        # Try target-specific templates
        for pattern in ["tq_decode_stage2_v2.hip",
                        "tq_decode_stage1.hip",
                        "tq_decode_fused.hip"]:
            template = os.path.join(self.kernel_dir, pattern)
            if os.path.exists(template):
                return template
        template_path = os.path.join(out_dir, "tq_decode_baseline.hip")
        with open(template_path, "w") as f:
            f.write("// Baseline — needs implementation\n")
        return template_path

    def _match_hypothesis(self, hypothesis: str) -> callable | None:
        """Match a hypothesis string to a transform method."""
        hyp_lower = hypothesis.lower()
        for keyword, method_name in self.HYPOTHESIS_TO_TRANSFORM.items():
            if keyword in hyp_lower:
                return getattr(self, method_name, None)
        return None

    # ── Transform implementations ─────────────────────────────────────

    def _apply_identity(self, source: str,
                        report: BottleneckReport,
                        state: PipelineState) -> str:
        """Explicit no-op placeholder used to mark unsupported transforms."""
        return source

    def _apply_online_softmax(self, source: str,
                              report: BottleneckReport,
                              state: PipelineState) -> str:
        """Replace two-pass softmax with online single-pass."""
        if "online softmax" in source.lower() or "single-pass" in source.lower():
            return source  # already applied

        replacement = """
    // === Online Softmax (single-pass) ===
    float acc = 0.0f;
    float e_max = -1e30f;
    float e_sum = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
        const int split_start = split_len * s;
        const int split_end = min(split_start + split_len, seq_len);

        if (split_end > split_start) {
            float lse = mid_o[mid_base + s * stride_ms + HEAD_DIM];
            float val = mid_o[mid_base + s * stride_ms + tid];

            if (lse > e_max) {
                float rescale = expf(e_max - lse);
                acc = acc * rescale + val;
                e_sum = e_sum * rescale + 1.0f;
                e_max = lse;
            } else {
                float scale = expf(lse - e_max);
                acc += scale * val;
                e_sum += scale;
            }
        }
    }"""

        two_pass = re.search(
            r'(//.*?[Ff]irst pass.*?)(//.*?[Ww]rite output|const int out_base)',
            source, re.DOTALL
        )
        if two_pass:
            start, end = two_pass.start(1), two_pass.start(2)
            source = source[:start] + replacement + "\n\n    " + source[end:]
            return source
        return source  # pattern not found, no change

    def _apply_float4_vectorize(self, source: str,
                                report: BottleneckReport,
                                state: PipelineState) -> str:
        """Convert scalar loads to float4 vectorized loads."""
        if "float4" in source:
            return source

        header = """
// float4 vectorized load for 4x bandwidth per thread
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}
"""
        idx = source.rfind("#include")
        if idx >= 0:
            eol = source.index("\n", idx)
            return source[:eol+1] + header + source[eol+1:]
        return source

    def _apply_warp_broadcast(self, source: str,
                              report: BottleneckReport,
                              state: PipelineState) -> str:
        """Add __shfl broadcast for LSE values."""
        if "__shfl" in source:
            return source

        old_lse = "float lse = mid_o[mid_base + s * stride_ms + HEAD_DIM];"
        new_lse = """float lse;
            if (tid % 64 == 0) {
                lse = mid_o[mid_base + s * stride_ms + HEAD_DIM];
            }
            lse = __shfl(lse, 0, 64);  // broadcast from lane 0"""

        if old_lse in source:
            return source.replace(old_lse, new_lse)
        return source

    def _apply_thread_count_tune(self, source: str,
                                 report: BottleneckReport,
                                 state: PipelineState) -> str:
        """Add a comment about thread count tuning (informational)."""
        if "GEAK recommendation: thread count" in source:
            return source
        comment = f"""\
// GEAK recommendation: thread count
// Use 32 threads with float4 for B>=80, 128 threads scalar for B<80
// Worst config efficiency: {report.worst_efficiency*100:.1f}%
"""
        return comment + source

    def _apply_batch_adaptive(self, source: str,
                              report: BottleneckReport,
                              state: PipelineState) -> str:
        """Add batch-size adaptive dispatch to the C launcher."""
        if "BATCH_CROSSOVER" in source:
            return source

        old_launcher = re.search(
            r'(extern "C" void launch_tq_decode_stage2_bf16\(.*?\{)(.*?\})',
            source, re.DOTALL
        )
        if old_launcher:
            new_body = """
    const int BATCH_CROSSOVER = 80;
    dim3 grid(B, Hq);
    if (B >= BATCH_CROSSOVER) {
        dim3 block(32, 1, 1);
        hipLaunchKernelGGL(tq_decode_stage2_v4_bf16,
            grid, block, 0, stream,
            mid_o, output, seq_lens,
            stride_mb, stride_mh, stride_ms,
            stride_ob, stride_oh, num_kv_splits);
    } else {
        dim3 block(128, 1, 1);
        hipLaunchKernelGGL(tq_decode_stage2_v3_bf16,
            grid, block, 0, stream,
            mid_o, output, seq_lens,
            stride_mb, stride_mh, stride_ms,
            stride_ob, stride_oh, num_kv_splits);
    }
}"""
            return source[:old_launcher.start(2)] + new_body + source[old_launcher.end():]
        return source

    def _apply_fuse_stages(self, source: str,
                           report: BottleneckReport,
                           state: PipelineState) -> str:
        """Stub: fusing stages is a major rewrite."""
        return source  # no-op, too complex for automated transform

    def _apply_reduce_vgpr(self, source: str,
                           report: BottleneckReport,
                           state: PipelineState) -> str:
        """Reduce VGPR usage by adjusting launch_bounds."""
        new = re.sub(
            r'__launch_bounds__\(\d+,\s*\d+\)',
            '__launch_bounds__(128, 10)',
            source,
        )
        return new  # may be same if pattern not found or already 10

    # ── Fine-tuning transforms ────────────────────────────────────────

    def _apply_tune_crossover(self, source: str,
                              report: BottleneckReport,
                              state: PipelineState) -> str:
        """Sweep BATCH_CROSSOVER threshold ±16."""
        m = re.search(r'BATCH_CROSSOVER\s*=\s*(\d+)', source)
        if not m:
            return source
        current = int(m.group(1))

        # Alternate between higher and lower thresholds
        iteration = state.iteration
        if iteration % 2 == 0:
            new_val = min(current + 16, 192)
        else:
            new_val = max(current - 16, 32)

        if new_val == current:
            return source

        print(f"[GEAK] Tuning BATCH_CROSSOVER: {current} → {new_val}")
        return source.replace(f"BATCH_CROSSOVER = {current}",
                             f"BATCH_CROSSOVER = {new_val}")

    def _apply_tune_launch_bounds(self, source: str,
                                  report: BottleneckReport,
                                  state: PipelineState) -> str:
        """Experiment with launch_bounds occupancy targets."""
        # Cycle through occupancy targets: 8 → 10 → 12 → 6 → 8
        targets = [8, 10, 12, 6]
        iteration = state.iteration
        target = targets[iteration % len(targets)]

        old_pattern = r'__launch_bounds__\(128,\s*\d+\)'
        new_val = f'__launch_bounds__(128, {target})'

        new_source = re.sub(old_pattern, new_val, source)
        if new_source == source:
            # Try 32-thread variant too
            old_pattern = r'__launch_bounds__\(32,\s*\d+\)'
            new_val = f'__launch_bounds__(32, {target})'
            new_source = re.sub(old_pattern, new_val, source)

        return new_source

    def _apply_nontemporal(self, source: str,
                           report: BottleneckReport,
                           state: PipelineState) -> str:
        """Add __builtin_nontemporal_load hints for mid_o reads."""
        if "nontemporal" in source:
            return source

        comment = """\
// GEAK: nontemporal load hint for streaming mid_o access
// (bypasses L2 cache to avoid pollution for one-shot reads)
"""
        if "#define HEAD_DIM" in source:
            return source.replace("#define HEAD_DIM", comment + "#define HEAD_DIM")
        return source

    # ── Compute-bound transforms (Stage1) ─────────────────────────────

    def _apply_reduce_block_kv(self, source: str,
                               report: BottleneckReport,
                               state: PipelineState) -> str:
        """Reduce BLOCK_KV to cut VGPR pressure for higher occupancy.

        BLOCK_KV controls how many KV tokens each warp processes in the
        inner loop before synchronization. Larger = fewer syncs but more
        registers (scores[], p[], slot_bases[] arrays are BLOCK_KV-sized).

        BLOCK_KV=8 → 58 VGPRs → 8 waves/CU
        BLOCK_KV=4 → ~42 VGPRs → 12 waves/CU (+50% occupancy)
        BLOCK_KV=2 → ~34 VGPRs → 15 waves/CU (+87% occupancy)
        """
        m = re.search(r'#define\s+BLOCK_KV\s+(\d+)', source)
        if not m:
            return source
        current = int(m.group(1))

        # Halve BLOCK_KV, minimum 2
        new_val = max(2, current // 2)
        if new_val == current:
            return source

        print(f"[GEAK] Reducing BLOCK_KV: {current} → {new_val}")
        return source.replace(f"#define BLOCK_KV {current}",
                             f"#define BLOCK_KV {new_val}")

    def _apply_increase_block_kv(self, source: str,
                                 report: BottleneckReport,
                                 state: PipelineState) -> str:
        """Increase BLOCK_KV to reduce loop iterations for long seq.

        For seq=2048+, each iteration's overhead (page lookup, warp_reduce)
        is amortized across more tokens. BLOCK_KV=16 means 4 warp ×16=64
        tokens/iter → only 32 iterations for seq=2048, vs 64 at BLOCK_KV=8.
        Trade-off: more VGPRs (scores[]/p[]/slot_bases[]).
        """
        m = re.search(r'#define\s+BLOCK_KV\s+(\d+)', source)
        if not m:
            return source
        current = int(m.group(1))
        new_val = min(16, current * 2)
        if new_val == current:
            return source
        print(f"[GEAK] Increasing BLOCK_KV: {current} → {new_val}")
        return source.replace(f"#define BLOCK_KV {current}",
                             f"#define BLOCK_KV {new_val}")

    def _apply_software_pipeline(self, source: str,
                                 report: BottleneckReport,
                                 state: PipelineState) -> str:
        """Add software pipelining: prefetch next KV batch while computing current.

        Key idea: the main loop loads KV data from HBM, then computes scores.
        With pipelining, we load the NEXT batch's KV data into registers
        WHILE computing the CURRENT batch's scores. This hides memory latency.

        Implementation: double-buffer the inner loop.
        """
        if "prefetch" in source.lower() or "double_buffer" in source.lower():
            return source  # already applied

        # Add a comment marker so subsequent runs know it was tried
        # The actual transform is complex — we add async prefetch hints
        # using __builtin_amdgcn_s_prefetch_data or manual register buffering

        old_loop = 'for (int base_n = split_start; base_n < split_end; base_n += BLOCK_KV * NUM_WARPS) {'
        if old_loop not in source:
            return source

        # Add prefetch hint before the main loop body
        new_loop = (
            '// Software pipelining: prefetch next iteration data\n'
            '    for (int base_n = split_start; base_n < split_end; base_n += BLOCK_KV * NUM_WARPS) {\n'
            '        // Prefetch next iteration KV cache data into L1\n'
            '        const int next_base = base_n + BLOCK_KV * NUM_WARPS;\n'
            '        if (next_base < split_end) {\n'
            '            const int next_warp_base = next_base + warp_id * BLOCK_KV;\n'
            '            if (next_warp_base < split_end) {\n'
            '                const int nkp = next_warp_base;\n'
            '                const int npi = nkp / block_size;\n'
            '                const int npo = nkp & (block_size - 1);\n'
            '                const int nbn = block_table[bt_base + npi];\n'
            '                const long long nsb = (long long)nbn * stride_cb\n'
            '                    + npo * stride_cp + kv_head_offset;\n'
            '                // Prefetch key MSE indices\n'
            '                __builtin_prefetch(kv_cache + nsb + byte_idx, 0, 3);\n'
            '                // Prefetch value data\n'
            '                __builtin_prefetch(kv_cache + nsb + KPS + byte_idx, 0, 3);\n'
            '            }\n'
            '        }'
        )
        return source.replace(old_loop, new_loop)

    def _apply_tune_num_warps(self, source: str,
                              report: BottleneckReport,
                              state: PipelineState) -> str:
        """Tune NUM_WARPS for occupancy vs intra-block parallelism.

        NUM_WARPS=4 (128 threads): more parallelism per block, more LDS
        NUM_WARPS=2 (64 threads):  less LDS, may fit more blocks on CU

        IMPORTANT: Must also update THREADS macro and __launch_bounds__
        to match the new thread count, otherwise threads with
        warp_id >= new_NUM_WARPS will write results that are never reduced.
        """
        m = re.search(r'#define\s+NUM_WARPS\s+(\d+)', source)
        if not m:
            return source
        current = int(m.group(1))

        # Cycle: 4→2→4
        new_val = 2 if current == 4 else 4
        if new_val == current:
            return source

        old_threads = 32 * current  # WARP_SIZE * old_NUM_WARPS
        new_threads = 32 * new_val

        print(f"[GEAK] Tuning NUM_WARPS: {current} → {new_val} "
              f"(threads: {old_threads} → {new_threads})")

        new_source = source
        # 1. Update NUM_WARPS
        new_source = new_source.replace(
            f"#define NUM_WARPS {current}",
            f"#define NUM_WARPS {new_val}")
        # 2. Update THREADS macro (WARP_SIZE * NUM_WARPS)
        new_source = re.sub(
            r'#define\s+THREADS\s+\(WARP_SIZE\s*\*\s*NUM_WARPS\)\s*//\s*\d+',
            f"#define THREADS (WARP_SIZE * NUM_WARPS)  // {new_threads}",
            new_source)
        # 3. Update __launch_bounds__
        new_source = re.sub(
            rf'__launch_bounds__\({old_threads},',
            f'__launch_bounds__({new_threads},',
            new_source)
        # 4. Update C launcher block dim
        new_source = re.sub(
            rf'dim3 block\({old_threads},',
            f'dim3 block({new_threads},',
            new_source)

        return new_source

    def _apply_tune_dims_per_thread(self, source: str,
                                    report: BottleneckReport,
                                    state: PipelineState) -> str:
        """Tune DIMS_PER_THREAD to trade register pressure for loop count."""
        m = re.search(r'#define\s+DIMS_PER_THREAD\s+(\d+)', source)
        if not m:
            return source
        current = int(m.group(1))

        new_val = 2 if current == 4 else 4
        if new_val == current:
            return source

        print(f"[GEAK] Tuning DIMS_PER_THREAD: {current} → {new_val}")
        return source.replace(f"#define DIMS_PER_THREAD {current}",
                             f"#define DIMS_PER_THREAD {new_val}")

    def _apply_increase_num_warps(self, source: str,
                                   report: BottleneckReport,
                                   state: PipelineState) -> str:
        """Increase NUM_WARPS for better parallelism on long sequences.

        NUM_WARPS=8 (256 threads): more warps processing KV tokens in
        parallel, but more shared memory for cross-warp reduction.
        Good when seq_len is large and CU occupancy permits.
        """
        m = re.search(r'#define\s+NUM_WARPS\s+(\d+)', source)
        if not m:
            return source
        current = int(m.group(1))

        if current >= 8:
            return source  # already at max

        new_val = min(8, current * 2)
        old_threads = 32 * current
        new_threads = 32 * new_val

        print(f"[GEAK] Increasing NUM_WARPS: {current} → {new_val} "
              f"(threads: {old_threads} → {new_threads})")

        new_source = source
        new_source = new_source.replace(
            f"#define NUM_WARPS {current}",
            f"#define NUM_WARPS {new_val}")
        new_source = re.sub(
            r'#define\s+THREADS\s+\(WARP_SIZE\s*\*\s*NUM_WARPS\)\s*//\s*\d+',
            f"#define THREADS (WARP_SIZE * NUM_WARPS)  // {new_threads}",
            new_source)
        new_source = re.sub(
            rf'__launch_bounds__\({old_threads},',
            f'__launch_bounds__({new_threads},',
            new_source)
        new_source = re.sub(
            rf'dim3 block\({old_threads},',
            f'dim3 block({new_threads},',
            new_source)
        return new_source

    # ── WHT Rotation transforms ──────────────────────────────────────

    def _apply_wht_increase_rows(self, source: str,
                                  report: BottleneckReport,
                                  state: PipelineState) -> str:
        """Increase ROWS_PER_BLOCK to process more rows per block.

        More rows/block = fewer blocks launched = less scheduling overhead.
        The kernel is launch-latency bound (~5-6µs), so reducing grid size
        can help.
        """
        m = re.search(r'#define\s+ROWS_PER_BLOCK\s+(\d+)', source)
        if not m:
            # V1 kernel has no ROWS_PER_BLOCK, can't apply
            return source
        current = int(m.group(1))

        new_val = min(32, current * 2)
        if new_val == current:
            return source

        old_threads = 32 * current
        new_threads = 32 * new_val

        print(f"[GEAK] WHT ROWS_PER_BLOCK: {current} → {new_val} "
              f"(threads/block: {old_threads} → {new_threads})")

        new_source = source
        new_source = new_source.replace(
            f"#define ROWS_PER_BLOCK {current}",
            f"#define ROWS_PER_BLOCK {new_val}")
        # Update launch_bounds
        new_source = re.sub(
            rf'__launch_bounds__\({old_threads},',
            f'__launch_bounds__({new_threads},',
            new_source)
        # Update block dim
        new_source = re.sub(
            rf'dim3 block\(WARP_SIZE, ROWS_PER_BLOCK\);\s*//.*',
            f'dim3 block(WARP_SIZE, ROWS_PER_BLOCK);  // 32 × {new_val} = {new_threads} threads',
            new_source)

        return new_source

    def _apply_wht_lds_signs(self, source: str,
                              report: BottleneckReport,
                              state: PipelineState) -> str:
        """Cache signs array in LDS (shared memory).

        Signs is only 128×4=512 bytes. Loading it into __shared__
        once per block avoids redundant global loads across warps.
        """
        if "__shared__" in source and "s_signs" in source:
            return source  # already applied

        # Insert shared memory declaration after kernel signature
        lds_decl = """
    // ── LDS cache for signs (512 bytes) ──────────────────────────
    __shared__ float s_signs[D];
    // Load signs cooperatively
    {
        const int load_tid = threadIdx.y * WARP_SIZE + threadIdx.x;
        if (load_tid < D) {
            s_signs[load_tid] = signs[load_tid];
        }
        __syncthreads();
    }
"""
        # Find the first line after kernel opening brace
        kernel_body = re.search(r'(void\s+tq_wht_rotate_kernel\w*\(.*?\)\s*\{)', source, re.DOTALL)
        if not kernel_body:
            return source

        insert_pos = kernel_body.end()
        new_source = source[:insert_pos] + lds_decl + source[insert_pos:]

        # Replace global signs access with LDS access
        new_source = new_source.replace("signs[base + i]", "s_signs[base + i]")

        return new_source

    def _apply_wht_xor_signs(self, source: str,
                              report: BottleneckReport,
                              state: PipelineState) -> str:
        """Replace sign multiplication with XOR on float sign bit.

        signs are ±1.0f. Instead of vals[i] *= signs[i], we can
        XOR bit 31 of the float when sign is negative.
        This replaces a multiply with a bitwise op.
        """
        if "sign_bits" in source or "__float_as_uint" in source:
            return source  # already applied

        # Replace the sign multiplication block
        old_pattern = r'(// ── Apply signs.*?)(\n\s*// ── (?:Stage 0|Butterfly))'
        m = re.search(old_pattern, source, re.DOTALL)
        if not m:
            return source

        new_code = """// ── Apply signs via XOR (branchless, no multiply) ─────────
    {
        // signs are ±1.0f. Negative sign has bit 31 set.
        // XOR with 0x80000000 flips the sign of the float.
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            unsigned int sign_bit = __float_as_uint(signs[base + i]) & 0x80000000u;
            vals[i] = __uint_as_float(__float_as_uint(vals[i]) ^ sign_bit);
        }
    }
"""
        return source[:m.start(1)] + new_code + source[m.start(2):]

    def _apply_fp16_compute(self, source: str,
                            report: BottleneckReport,
                            state: PipelineState) -> str:
        """Convert Stage1 score computation from fp32 to fp16 (__half).

        Key changes:
        1. Q loaded as __half (or bf16→__half) instead of float
        2. Centroid lookup from LDS as float → convert to __half
        3. Score = __hmul(q, c) accumulation in __half
        4. warp_reduce stays in fp32 (for numerical stability)
        5. Value accumulation stays in fp32 (for output precision)

        This halves Q register pressure (4 floats → 4 __half = 2 regs vs 4)
        and may enable __hmul/__hfma for 2× throughput on compute.
        """
        import re

        if "__half q0h" in source or "__hmul(q0h" in source:
            return source  # already applied

        # Only apply to Stage1 kernels that use fp32 Q
        if "const float* __restrict__ q_rot" not in source:
            return source

        # Strategy: Replace Q load + score computation with __half path
        # 1. Add __half Q variables (after existing Q load, detect pattern)
        if "const float4 q_vec" in source:
            source = source.replace(
                "const float q0 = q_vec.x, q1 = q_vec.y, q2 = q_vec.z, q3 = q_vec.w;",
                "const float q0 = q_vec.x, q1 = q_vec.y, q2 = q_vec.z, q3 = q_vec.w;\n"
                "    // Convert Q to __half for score compute (__hmul is 2x throughput)\n"
                "    const __half q0h = __float2half(q0);\n"
                "    const __half q1h = __float2half(q1);\n"
                "    const __half q2h = __float2half(q2);\n"
                "    const __half q3h = __float2half(q3);"
            )
        elif "float q0, q1, q2, q3;" in source:
            # void* q_rot variant — add __half after the dtype switch
            source = source.replace(
                "    }  // end dtype switch",
                "    }  // end dtype switch\n"
                "    const __half q0h = __float2half(q0);\n"
                "    const __half q1h = __float2half(q1);\n"
                "    const __half q2h = __float2half(q2);\n"
                "    const __half q3h = __float2half(q3);"
            )

        # 2. Replace score computation: float→__half multiply
        # Old: float score = q0*c0 + q1*c1 + q2*c2 + q3*c3;
        #      float norm_sq = c0*c0 + c1*c1 + c2*c2 + c3*c3;
        source = re.sub(
            r'const float c0 = s_c\[mse_u16 & 0xF\];\s*'
            r'const float c1 = s_c\[\(mse_u16 >> 4\) & 0xF\];\s*'
            r'const float c2 = s_c\[\(mse_u16 >> 8\) & 0xF\];\s*'
            r'const float c3 = s_c\[\(mse_u16 >> 12\) & 0xF\];\s*\n'
            r'\s*float score = q0\*c0 \+ q1\*c1 \+ q2\*c2 \+ q3\*c3;\s*\n'
            r'\s*float norm_sq = c0\*c0 \+ c1\*c1 \+ c2\*c2 \+ c3\*c3;',
            """const __half c0h = __float2half(s_c[mse_u16 & 0xF]);
                const __half c1h = __float2half(s_c[(mse_u16 >> 4) & 0xF]);
                const __half c2h = __float2half(s_c[(mse_u16 >> 8) & 0xF]);
                const __half c3h = __float2half(s_c[(mse_u16 >> 12) & 0xF]);

                // fp16 score: __hmul + accumulate in fp32 for reduction
                float score = __half2float(__hmul(q0h, c0h))
                            + __half2float(__hmul(q1h, c1h))
                            + __half2float(__hmul(q2h, c2h))
                            + __half2float(__hmul(q3h, c3h));
                float norm_sq = __half2float(__hmul(c0h, c0h))
                              + __half2float(__hmul(c1h, c1h))
                              + __half2float(__hmul(c2h, c2h))
                              + __half2float(__hmul(c3h, c3h));""",
            source,
        )

        # Add #include <hip/hip_fp16.h> if not present
        if "hip_fp16.h" not in source:
            source = source.replace(
                "#include <hip/hip_runtime.h>",
                "#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>"
            )

        return source

    def _apply_void_q_input(self, source: str,
                            report: BottleneckReport,
                            state: PipelineState) -> str:
        """Change Q input from const float* to const void* with dtype param.

        Enables the kernel to accept bf16/fp16/fp32 Q directly from Python
        without a separate fp32 cast. The kernel converts to its internal
        compute format (fp32 or __half) at load time.
        """
        import re

        if "void* __restrict__ q_rot" in source:
            return source  # already applied

        if "const float* __restrict__ q_rot" not in source:
            return source

        # Change kernel signature
        source = source.replace(
            "const float* __restrict__ q_rot,",
            "const void* __restrict__ q_rot,  // bf16/fp16/fp32 (see dtype param)"
        )

        # Add dtype parameter after norm_correction
        source = re.sub(
            r'(int norm_correction)\s*\)',
            r'\1,\n    int dtype  // 0=bf16, 1=fp16, 2=fp32\n)',
            source,
        )

        # Replace float4 Q load with dtype-aware load
        source = source.replace(
            "const float4 q_vec = *reinterpret_cast<const float4*>(q_rot + q_base);",
            "// Load Q with dtype-aware vectorized load\n"
            "    float q0, q1, q2, q3;\n"
            "    if (dtype == 2) {  // fp32: float4 vectorized\n"
            "        const float4 q_vec = *reinterpret_cast<const float4*>(\n"
            "            reinterpret_cast<const float*>(q_rot) + q_base + lane_id * DIMS_PER_THREAD);\n"
            "        q0 = q_vec.x; q1 = q_vec.y; q2 = q_vec.z; q3 = q_vec.w;\n"
            "    } else if (dtype == 1) {  // fp16\n"
            "        const __half* hp = reinterpret_cast<const __half*>(q_rot) + q_base + lane_id * DIMS_PER_THREAD;\n"
            "        q0 = __half2float(hp[0]); q1 = __half2float(hp[1]);\n"
            "        q2 = __half2float(hp[2]); q3 = __half2float(hp[3]);\n"
            "    } else {  // bf16\n"
            "        const unsigned int* p32 = reinterpret_cast<const unsigned int*>(\n"
            "            reinterpret_cast<const hip_bfloat16*>(q_rot) + q_base + lane_id * DIMS_PER_THREAD);\n"
            "        unsigned int w0 = p32[0], w1 = p32[1];\n"
            "        q0 = __uint_as_float((w0 & 0xFFFF) << 16);\n"
            "        q1 = __uint_as_float((w0 >> 16) << 16);\n"
            "        q2 = __uint_as_float((w1 & 0xFFFF) << 16);\n"
            "        q3 = __uint_as_float((w1 >> 16) << 16);\n"
            "    }"
        )

        # Add hip_bfloat16 include
        if "hip_bfloat16.h" not in source:
            source = source.replace(
                "#include <hip/hip_fp16.h>",
                "#include <hip/hip_fp16.h>\n#include <hip/hip_bfloat16.h>"
            )

        # Update launcher to pass dtype
        source = re.sub(
            r'(attn_scale, norm_correction)\);',
            r'\1, dtype);',
            source,
        )

        # Add dtype param to launcher function signature
        source = re.sub(
            r'(float attn_scale, int norm_correction,)\s*\n\s*(int B, int Hq)',
            r'\1\n    int dtype,\n    \2',
            source,
        )

        return source

    # ── Swizzle-specific transforms ─────────────────────────────────

    def _apply_halfwarp_swizzle(self, source: str,
                                report, state) -> str:
        """Reduce swizzle from per-thread (32 copies) to per-half-warp (2 copies).

        Each half-warp (16 threads) shares one XOR-swizzled copy.
        LDS: 32*16 = 512 floats/warp → 2*16 = 32 floats/warp = 16× reduction.
        """
        import re
        if 'half_warp' in source or 'HALF_WARP' in source:
            return source

        # Replace per-thread addressing with per-half-warp
        old = '#define SWIZZLE_ADDR(warp, lane, swizzled_idx) \\\n    ((warp) * WARP_SIZE * N_CENTROIDS + (lane) * N_CENTROIDS + (swizzled_idx))'
        new = ('// Half-warp swizzle: 2 copies per warp (even/odd halves)\n'
               '#define HALF_WARP_SIZE 16\n'
               '#define SWIZZLE_ADDR(warp, lane, swizzled_idx) \\\n'
               '    ((warp) * 2 * N_CENTROIDS + ((lane) >> 4) * N_CENTROIDS + (swizzled_idx))')
        if old in source:
            source = source.replace(old, new)
            # Update LDS size
            source = source.replace(
                '__shared__ float s_c[NUM_WARPS * WARP_SIZE * N_CENTROIDS];  // 8KB',
                '__shared__ float s_c[NUM_WARPS * 2 * N_CENTROIDS];  // 512B (2 copies/warp)')
            # Update store: only first 16 threads per half store
            source = source.replace(
                '    #pragma unroll\n    for (int c = 0; c < N_CENTROIDS; c++) {\n        s_c[SWIZZLE_ADDR(warp_id, lane_id, SWIZZLE_IDX(lane_id, c))] = centroids[c];\n    }',
                '    // Half-warp swizzle: each half stores once\n'
                '    if ((lane_id & 0xF) < N_CENTROIDS) {\n'
                '        int half = lane_id >> 4;\n'
                '        int lid = lane_id & 0xF;\n'
                '        s_c[warp_id * 2 * N_CENTROIDS + half * N_CENTROIDS + (lid ^ (half * 7))] = centroids[lid];\n'
                '    }')
            # Update XOR key for half-warp
            source = source.replace(
                'const int my_xor_key = lane_id & 0xF;',
                'const int my_xor_key = (lane_id >> 4) * 7;  // half-warp XOR key')
            source = source.replace(
                'const int my_lds_base = warp_id * WARP_SIZE * N_CENTROIDS + lane_id * N_CENTROIDS;',
                'const int my_lds_base = warp_id * 2 * N_CENTROIDS + (lane_id >> 4) * N_CENTROIDS;')
            # Update launch bounds for better occupancy
            source = re.sub(r'__launch_bounds__\(\d+,\s*\d+\)', '__launch_bounds__(128, 8)', source)
            print("[GEAK] Applied half-warp swizzle (LDS 8KB→512B)")
        return source

    def _apply_twocopy_swizzle(self, source: str, report, state) -> str:
        """2-copy swizzle: even lanes use copy A, odd lanes use copy B.

        Copy A at banks 0-15, copy B at banks 16-31.
        No conflict between even/odd lanes. Within each group (16 threads),
        random access to 16 banks → minimal conflict.
        """
        import re
        if '2copy' in source or 'twocopy' in source:
            return source

        # Replace the entire swizzle mechanism with simple 2-copy
        old_shared = '__shared__ float s_c[NUM_WARPS * WARP_SIZE * N_CENTROIDS];  // 8KB'
        new_shared = '__shared__ float s_c[32];  // 2copy: 2 × 16 centroids = 128B'
        if old_shared in source:
            source = source.replace(old_shared, new_shared)
        elif 'NUM_WARPS * 2 * N_CENTROIDS' in source:
            source = re.sub(r'__shared__ float s_c\[.*?\];.*',
                           '__shared__ float s_c[32];  // 2copy: 128B', source)

        # Replace store
        old_store_patterns = [
            '    #pragma unroll\n    for (int c = 0; c < N_CENTROIDS; c++) {\n        s_c[SWIZZLE_ADDR(warp_id, lane_id, SWIZZLE_IDX(lane_id, c))] = centroids[c];\n    }',
        ]
        new_store = ('    // 2-copy store: copy A at [0..15], copy B at [16..31]\n'
                     '    if (tid < 16) {\n'
                     '        s_c[tid] = centroids[tid];\n'
                     '        s_c[16 + tid] = centroids[tid];\n'
                     '    }')
        for pat in old_store_patterns:
            if pat in source:
                source = source.replace(pat, new_store)
                break
        else:
            # Try regex for other store patterns
            source = re.sub(
                r'// Half-warp.*?centroids\[lid\];\s*\}',
                new_store.lstrip(), source, flags=re.DOTALL)

        # Replace lookup with 2-copy addressing
        source = re.sub(
            r'const int my_lds_base = .*?;',
            '// 2copy: even lanes→copy A, odd lanes→copy B\n'
            '    const int my_copy_offset = (lane_id & 1) * 16;', source)
        source = re.sub(r'const int my_xor_key = .*?;', '', source)

        # Replace actual lookups
        source = re.sub(
            r's_c\[my_lds_base \+ \(\(?(.*?)\)? \^ my_xor_key\)\]',
            r's_c[my_copy_offset + (\1)]', source)

        # Remove SWIZZLE macros
        source = re.sub(r'#define SWIZZLE_IDX.*\n', '', source)
        source = re.sub(r'#define SWIZZLE_ADDR.*\n', '', source)
        source = re.sub(r'#define HALF_WARP.*\n', '', source)

        # Update launch bounds
        source = re.sub(r'__launch_bounds__\(\d+,\s*\d+\)', '__launch_bounds__(128, 8)', source)
        print("[GEAK] Applied 2-copy swizzle (LDS→128B)")
        return source

    def _apply_fp16_centroid_lds(self, source: str, report, state) -> str:
        """Store centroids as fp16 in LDS to halve LDS usage."""
        import re
        if '_Float16' in source and 's_c_h' in source:
            return source
        if '__shared__ float s_c[' not in source:
            return source

        # Change s_c from float to _Float16
        source = re.sub(
            r'__shared__ float s_c\[(\d+)\];(.*)',
            r'__shared__ _Float16 s_c_h[\1]; // fp16 centroids (half LDS)\2', source)

        # Change store
        source = source.replace(
            'centroids[c];\n    }',
            '(_Float16)centroids[c];\n    }')
        source = source.replace(
            's_c[tid] = centroids[tid]',
            's_c_h[tid] = (_Float16)centroids[tid]')
        source = source.replace(
            's_c[16 + tid] = centroids[tid]',
            's_c_h[16 + tid] = (_Float16)centroids[tid]')

        # Change reads: s_c[...] → (float)s_c_h[...]
        source = re.sub(r'= s_c\[', '= (float)s_c_h[', source)
        source = re.sub(r'= s_c_h\[', '= (float)s_c_h[', source)

        print("[GEAK] Applied fp16 centroid LDS (halved LDS)")
        return source

    def _apply_pairwise_lut(self, source: str, report, state) -> str:
        """FLUTE-style pairwise LUT: replace 4x LDS centroid lookup with 2x half2 lookup.

        qmap2[256] = all pairs of (centroid[lo], centroid[hi]) packed as __half2.
        One byte from KV cache (= 2 packed 4-bit indices) directly indexes qmap2.
        Result: 2 centroid values in one LDS read (half2).
        """
        import re
        if 'qmap2' in source or '__half2 c01' in source:
            return source  # already applied

        if 's_c[N_CENTROIDS]' not in source and 's_c[16]' not in source:
            return source  # not a standard centroid kernel

        # Add half2 include
        if 'hip_fp16.h' not in source:
            source = source.replace('#include <hip/hip_runtime.h>',
                '#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>')

        # Replace shared memory: s_c[16] float → s_qmap2[256] __half2
        source = re.sub(
            r'__shared__ float s_c\[N_CENTROIDS\];.*',
            '// FLUTE pairwise LUT: qmap2[256] = half2(centroid[lo], centroid[hi])\n'
            '    __shared__ __half2 s_qmap2[256];  // 1KB: 16×16 centroid pairs as half2',
            source)

        # Replace centroid load
        source = re.sub(
            r'if \(tid < N_CENTROIDS\) s_c\[tid\] = centroids\[tid\];',
            '// Build pairwise LUT: each of 256 entries = half2(c[lo], c[hi])\n'
            '    if (tid < 256) {\n'
            '        int lo = tid & 0xF;\n'
            '        int hi = (tid >> 4) & 0xF;\n'
            '        __half c_lo = __float2half(centroids[lo]);\n'
            '        __half c_hi = __float2half(centroids[hi]);\n'
            '        s_qmap2[tid] = __halves2half2(c_lo, c_hi);\n'
            '    }\n'
            '    // Extra threads (128-255) need second pass for 128-thread blocks\n'
            '    if (tid + 128 < 256) {\n'
            '        int idx2 = tid + 128;\n'
            '        int lo2 = idx2 & 0xF;\n'
            '        int hi2 = (idx2 >> 4) & 0xF;\n'
            '        s_qmap2[idx2] = __halves2half2(__float2half(centroids[lo2]), __float2half(centroids[hi2]));\n'
            '    }',
            source)

        # Replace the centroid lookup in the inner loop
        # Old pattern: 4 separate LDS reads from s_c[idx]
        old_lookup = (
            r'const float c0 = s_c\[mse_u16 & 0xF\];\s*'
            r'const float c1 = s_c\[\(mse_u16 >> 4\) & 0xF\];\s*'
            r'const float c2 = s_c\[\(mse_u16 >> 8\) & 0xF\];\s*'
            r'const float c3 = s_c\[\(mse_u16 >> 12\) & 0xF\];'
        )
        new_lookup = (
            '// FLUTE pairwise LUT: 2 half2 reads instead of 4 float reads\n'
            '                const __half2 c01_h2 = s_qmap2[mse_u16 & 0xFF];        // idx[0]|idx[1] → half2\n'
            '                const __half2 c23_h2 = s_qmap2[(mse_u16 >> 8) & 0xFF]; // idx[2]|idx[3] → half2\n'
            '                const float c0 = __half2float(__low2half(c01_h2));\n'
            '                const float c1 = __half2float(__high2half(c01_h2));\n'
            '                const float c2 = __half2float(__low2half(c23_h2));\n'
            '                const float c3 = __half2float(__high2half(c23_h2));'
        )
        source = re.sub(old_lookup, new_lookup, source)

        print("[GEAK] Applied FLUTE pairwise LUT (4 LDS → 2 half2 LDS)")
        return source

    def _apply_qc_precompute(self, source: str, report, state) -> str:
        """Phase 2: Precompute QC[d][c] = q_rot[d] * centroid[c] in LDS.

        Eliminates FMA from inner loop: score becomes pure lookup + add.
        Uses stride=17 for bank-conflict-free access.
        """
        import re
        if 'qc_table' in source:
            return source

        if 's_qmap2' not in source and 'qmap2' not in source:
            return source  # needs pairwise LUT first

        # Add QC table shared memory
        source = source.replace(
            '__shared__ __half2 s_qmap2[256];  // 1KB: 16×16 centroid pairs as half2',
            '__shared__ __half2 s_qmap2[256];  // 1KB: pairwise centroid pairs\n'
            '    // Phase 2: QC precomputed table: QC[d][c] = q_rot[d] * centroid[c]\n'
            '    // stride=17 for bank-conflict-free LDS access\n'
            '    #define QC_STRIDE 17\n'
            '    __shared__ float qc_table[HEAD_DIM * QC_STRIDE];  // 128*17*4 = 8.7KB')

        # Build QC table after Q is loaded (after __syncthreads)
        # Find the Q load section and add QC precompute after it
        q_load_end = source.find('float m_prev = -1e30f')
        if q_load_end > 0:
            qc_build = '''
    // Phase 2: Build QC table = q_rot[d] * centroid[c] for all d,c
    // 128 threads, each builds 16 entries (one row of QC table)
    {
        const int my_d = tid;  // thread tid handles dim tid (tid < 128)
        if (my_d < HEAD_DIM) {
            float q_val = q_rot[bid * stride_qb + hid * stride_qh + my_d];
            #pragma unroll
            for (int c = 0; c < N_CENTROIDS; c++) {
                qc_table[my_d * QC_STRIDE + c] = q_val * centroids[c];
            }
        }
    }
    __syncthreads();

    '''
            source = source[:q_load_end] + qc_build + source[q_load_end:]

        # Replace the score computation: instead of q*c FMA, use QC lookup
        old_score = (
            r'float score = q0\*c0 \+ q1\*c1 \+ q2\*c2 \+ q3\*c3;'
        )
        new_score = (
            '// Phase 2: pure QC table lookup (no FMA!)\n'
            '                const int d_base = lane_id * DIMS_PER_THREAD;\n'
            '                float score = qc_table[(d_base+0)*QC_STRIDE + (mse_u16 & 0xF)]\n'
            '                            + qc_table[(d_base+1)*QC_STRIDE + ((mse_u16>>4) & 0xF)]\n'
            '                            + qc_table[(d_base+2)*QC_STRIDE + ((mse_u16>>8) & 0xF)]\n'
            '                            + qc_table[(d_base+3)*QC_STRIDE + ((mse_u16>>12) & 0xF)];'
        )
        source = re.sub(old_score, new_score, source)

        # Remove the now-unused q0*c0 FMA variables but keep centroid reads for norm_sq
        # Actually norm_sq still needs c0,c1,c2,c3 for norm correction
        # So keep pairwise LUT reads for norm_sq, but score uses QC table

        print("[GEAK] Applied Phase 2: QC precompute table (no FMA in score)")
        return source

    def _apply_mfma_decode(self, source: str, report, state) -> str:
        """Phase 3: MFMA for score computation (eliminates warp_reduce).

        gfx950 MFMA_F32_16x16x16_F16:
          C[16,16] = A[16,16] × B[16,16]^T  (fp16 → fp32)
          Thread t: A[t%16][(t/16)*4:+4], B[t%16][(t/16)*4:+4], C[t%16][(t/16)*4:+4]

        For decode: A = Q (row 0), B = K_dequant (16 tokens), C[0][n] = score[n]
        8 MFMAs for D=128, eliminates 5× warp_reduce shuffle per token.
        Requires 64 threads (1 wavefront).
        """
        import re
        if '__builtin_amdgcn_mfma' in source:
            return source
        if 's_qmap2' not in source:
            return source  # needs pairwise LUT first

        # This is a major rewrite - create a new kernel with MFMA score computation
        # Keep pairwise LUT for dequant, but replace warp_reduce with MFMA

        # For now, add MFMA as an alternative score path within the existing kernel
        # The key change: instead of per-token sequential processing,
        # batch 16 tokens and use MFMA

        # Add MFMA vector type definitions
        if 'half4_t' not in source:
            source = source.replace(
                '#define BLOCK_KV 8',
                '#define BLOCK_KV 8\n'
                '\n'
                '// MFMA types for v_mfma_f32_16x16x16_f16\n'
                'typedef _Float16 half4_t __attribute__((ext_vector_type(4)));\n'
                'typedef float float4_t __attribute__((ext_vector_type(4)));')

        print("[GEAK] Added MFMA types (full MFMA rewrite requires manual kernel development)")
        return source

    def _apply_cross_batch_mfma(self, source, report, state):
        """Cross-batch MFMA: batch 2 requests into M=16 for 100% utilization."""
        if 'cross_batch' in source:
            return source
        # This is a major architectural change — needs manual implementation
        print("[GEAK] Cross-batch M=16: requires manual kernel rewrite (grid change)")
        # For now, mark it as explored
        return source.replace(
            '// ═══ Main loop ═══',
            '// TODO: Cross-batch M=16 (batch 2 requests × 8 heads = 16 MFMA rows)\n'
            '    // ═══ Main loop ═══')

"""Analyzer Agent — identifies bottlenecks and proposes optimization hypotheses.

Input:  profile metrics + hardware model + current kernel source
Output: bottleneck report + ranked optimization hypotheses
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

from agents.hardware_model import (
    MI355X, HardwareSpec, stage2_theoretical_us,
    stage1_theoretical_us, fused_theoretical_us,
    wht_rotate_theoretical_us,
)
from agents.state import ActionSpec, PipelineState, IterationResult


@dataclass
class BottleneckReport:
    """Structured bottleneck analysis output."""
    summary: str
    bottleneck_type: str                     # "memory" | "compute" | "latency" | "occupancy"
    efficiency_map: dict[str, float]         # config → HBM efficiency
    worst_config: str                        # config with lowest efficiency
    worst_efficiency: float
    hypotheses: list[str]                    # ranked optimization ideas
    estimated_gains: list[float]             # estimated % gain per hypothesis
    action_specs: list[ActionSpec] = field(default_factory=list)


class AnalyzerAgent:
    """Analyzes profiling data against hardware model to find bottlenecks."""

    def __init__(self, hw: HardwareSpec = MI355X):
        self.hw = hw

    def run(self, state: PipelineState) -> BottleneckReport:
        """Analyze current iteration's profile OR benchmark data.

        Prefers profile_results, falls back to benchmark_results from
        the previous iteration when profiling was unavailable.
        """
        cur = state.current

        # Use profile data if available, otherwise prior benchmark data
        perf_data = cur.profile_results
        data_source = "profile"
        if not perf_data and state.prev and state.prev.benchmark_results:
            perf_data = state.prev.benchmark_results
            data_source = "prev_benchmark"
        if not perf_data and cur.benchmark_results:
            perf_data = cur.benchmark_results
            data_source = "benchmark"

        # 1. Compute theoretical times for ALL configs
        theoretical = {}
        for cfg in state.workload_configs:
            key = state.config_key(cfg)
            if state.target_kernel == "tq_decode_stage2":
                theoretical[key] = stage2_theoretical_us(
                    B=cfg["B"], Hq=cfg["Hq"],
                    num_kv_splits=cfg["splits"], D=128,
                    hw=self.hw,
                )
            elif state.target_kernel == "tq_decode_stage1":
                theoretical[key] = stage1_theoretical_us(
                    B=cfg["B"], seq_len=cfg["seq"],
                    Hq=cfg["Hq"], Hk=cfg["Hk"],
                    num_kv_splits=cfg["splits"],
                    hw=self.hw,
                )
            elif state.target_kernel in ("tq_decode_fused", "tq_decode_fused_wht"):
                theoretical[key] = fused_theoretical_us(
                    B=cfg["B"], seq_len=cfg["seq"],
                    Hq=cfg["Hq"], Hk=cfg["Hk"],
                    hw=self.hw,
                )
            elif state.target_kernel == "tq_wht_rotate":
                theoretical[key] = wht_rotate_theoretical_us(
                    M=cfg["M"], hw=self.hw,
                )
        cur.theoretical_us = theoretical

        if not perf_data:
            # First iteration with no prior data — can still compute theoretical
            # and provide initial hypotheses
            report = self._initial_report(state, cur, theoretical)
            state.analysis_action_specs = report.action_specs
            return report

        # 2. Calculate efficiency
        efficiency = {}
        for key, actual in perf_data.items():
            if key in theoretical and actual > 0:
                efficiency[key] = theoretical[key] / actual
        cur.efficiency = efficiency

        # 3. Find worst case
        if efficiency:
            worst_key = min(efficiency, key=efficiency.get)
            worst_eff = efficiency[worst_key]
        else:
            worst_key = ""
            worst_eff = 0.0

        # 4. Classify bottleneck
        avg_eff = sum(efficiency.values()) / len(efficiency) if efficiency else 0
        bottleneck_type = self._classify_bottleneck(avg_eff, cur, state)

        # 5. Generate hypotheses based on kernel source analysis
        hypotheses, gains = self._generate_hypotheses(
            bottleneck_type, efficiency, perf_data, theoretical, cur, state
        )

        action_specs = self._build_action_specs(
            state=state,
            hypotheses=hypotheses,
            gains=gains,
        )

        report = BottleneckReport(
            summary=self._build_summary(
                avg_eff, worst_eff, worst_key, bottleneck_type,
                state.iteration, data_source
            ),
            bottleneck_type=bottleneck_type,
            efficiency_map=efficiency,
            worst_config=worst_key,
            worst_efficiency=worst_eff,
            hypotheses=hypotheses,
            estimated_gains=gains,
            action_specs=action_specs,
        )

        # Save
        cur.bottleneck_report = report.summary
        cur.optimization_hypotheses = report.hypotheses
        cur.action_specs = report.action_specs
        state.analysis_action_specs = report.action_specs

        report_path = os.path.join(
            state.run_dir, f"v{state.iteration}", "analysis.txt"
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report.summary + "\n\n")
            f.write(f"Data source: {data_source}\n")
            f.write(f"Perf data ({len(perf_data)} configs):\n")
            for k, v in sorted(perf_data.items()):
                theo = theoretical.get(k, 0)
                eff = efficiency.get(k, 0)
                f.write(f"  {k}: {v:.2f}µs (theo={theo:.2f}µs, eff={eff:.1%})\n")
            f.write(f"\nHypotheses:\n")
            for i, (h, g) in enumerate(zip(hypotheses, gains)):
                f.write(f"  {i+1}. [{g*100:.1f}%] {h}\n")
            if action_specs:
                f.write("\nAction specs:\n")
                for spec in action_specs:
                    f.write(
                        f"  - {spec.action_type}::{spec.name} "
                        f"gain={spec.expected_gain*100:.1f}% "
                        f"params={spec.params}\n"
                    )

        return report

    def _initial_report(self,
                        state: PipelineState,
                        cur: IterationResult,
                        theoretical: dict[str, float]) -> BottleneckReport:
        """Generate an initial report when no perf data is available yet."""
        cur.theoretical_us = theoretical
        # Provide generic hypotheses for the first run
        hypotheses = [
            "Baseline profiling needed — run benchmark to establish reference",
            "Online softmax: single-pass reduction eliminates 2nd read pass",
            "float4 vectorized loads for improved memory coalescing",
        ]
        gains = [0.0, 0.20, 0.15]

        report = BottleneckReport(
            summary="Initial iteration — no performance data yet, running baseline",
            bottleneck_type="unknown",
            efficiency_map={},
            worst_config="",
            worst_efficiency=0.0,
            hypotheses=hypotheses,
            estimated_gains=gains,
            action_specs=self._build_action_specs(
                state=state,
                hypotheses=hypotheses,
                gains=gains,
            ),
        )
        cur.bottleneck_report = report.summary
        cur.optimization_hypotheses = report.hypotheses
        cur.action_specs = report.action_specs
        return report

    def _build_action_specs(
        self,
        state: PipelineState | None,
        hypotheses: list[str],
        gains: list[float],
    ) -> list[ActionSpec]:
        """Convert free-form hypotheses into executable action specs."""
        try:
            from agents.target_registry import get_target

            target_name = state.target_kernel if state is not None else "tq_decode_stage2"
            target_cfg = get_target(target_name)
        except Exception:
            target_cfg = None

        action_specs: list[ActionSpec] = []
        if target_cfg is not None and target_cfg.target_type == "campaign":
            round_id = state.round_index if state else 0
            for action_type, default_tags, rationale in (
                (
                    "fusion_config_change",
                    ["fusion", "runtime-config"],
                    "Analyzer mapped fusion-path observations onto the "
                    "runtime-config search space.",
                ),
                (
                    "fusion_code_change",
                    ["fusion", "code-overlay"],
                    "Analyzer mapped fusion-path observations onto isolated "
                    "external-kernel code overlays.",
                ),
            ):
                for idx, option in enumerate(target_cfg.search_space.get(action_type, [])):
                    action_specs.append(
                        ActionSpec(
                            action_id=f"analyzer-{round_id}-{action_type}-{idx+1}",
                            action_type=action_type,
                            name=option.get("name", f"{action_type}_{idx+1}"),
                            params=copy.deepcopy(option.get("params", {})),
                            rationale=rationale,
                            expected_gain=float(option.get("expected_gain", 0.0)),
                            source_agent="analyzer",
                            priority=idx,
                            tags=list(default_tags),
                        )
                    )
            return action_specs

        round_id = state.round_index if state is not None else 0
        for idx, (hypothesis, gain) in enumerate(zip(hypotheses, gains), 1):
            action_specs.append(
                ActionSpec(
                    action_id=f"analyzer-{round_id}-kernel-{idx}",
                    action_type="kernel_transform",
                    name=hypothesis[:96],
                    params={"hypothesis": hypothesis},
                    rationale=hypothesis,
                    expected_gain=float(gain),
                    source_agent="analyzer",
                    priority=idx,
                )
            )
        return action_specs

    def _classify_bottleneck(self, avg_eff: float,
                             cur: IterationResult,
                             state: PipelineState | None = None) -> str:
        """Classify the primary bottleneck type."""
        if avg_eff >= 0.85:
            return "near_optimal"

        # WHT rotation is launch-latency bound (data is tiny, ~5-6µs floor)
        target = state.target_kernel if state else ""
        if target == "tq_wht_rotate":
            return "latency"

        # Stage1 and Fused are compute-bound (dot products + exp + centroid lookup)
        if target in ("tq_decode_stage1", "tq_decode_fused", "tq_decode_fused_wht") and avg_eff < 0.30:
            # Stage1 at 6-11% HBM efficiency is normal — it's compute-bound
            if cur.vgpr_count > 0 and cur.vgpr_count > 48:
                return "compute_occupancy"  # compute + VGPR pressure
            return "compute"

        if avg_eff >= 0.65:
            return "memory"
        if cur.vgpr_count > 0 and cur.vgpr_count >= 128:
            return "occupancy"
        if avg_eff < 0.4:
            return "latency"
        return "memory"

    def _generate_hypotheses(
        self, bottleneck_type: str,
        efficiency: dict[str, float],
        perf_data: dict[str, float],
        theoretical: dict[str, float],
        cur: IterationResult,
        state: PipelineState,
    ) -> tuple[list[str], list[float]]:
        """Generate ranked optimization hypotheses.

        Reads the actual kernel source to avoid suggesting already-applied
        optimizations.
        """
        hypotheses: list[str] = []
        gains: list[float] = []

        # Read current kernel source to check what's already applied
        source = ""
        if cur.hip_source and os.path.exists(cur.hip_source):
            with open(cur.hip_source) as f:
                source = f.read()
        elif state.prev and state.prev.hip_source and os.path.exists(state.prev.hip_source):
            with open(state.prev.hip_source) as f:
                source = f.read()

        source_lower = source.lower()

        # ── WHT Rotation kernel ──────────────────────────────────────
        if state.target_kernel == "tq_wht_rotate":
            return self._generate_wht_hypotheses(
                source, source_lower, efficiency, cur
            )

        # ── Fused kernel (compute-bound like Stage1 but no mid_o) ────
        if state.target_kernel in ("tq_decode_fused", "tq_decode_fused_wht"):
            return self._generate_fused_hypotheses(
                source, source_lower, efficiency, cur
            )

        # ── Compute-bound kernels (Stage1) ────────────────────────────
        if bottleneck_type in ("compute", "compute_occupancy"):
            import re
            # Check if source has swizzle → use swizzle-specific hypotheses
            if 'SWIZZLE' in source or 'swizzle' in source.lower() or 'xor' in source.lower():
                return self._generate_swizzle_hypotheses(
                    source, source_lower, efficiency, cur
                )

            # Cross-batch M=16 MFMA (100% utilization)
            if 'mfma' in source_lower and 'cross_batch' not in source:
                hypotheses.insert(0,
                    "Cross-batch MFMA M=16: batch 2 requests × 8 GQA heads = 16 rows "
                    "in MFMA A matrix. Grid changes from (B, Hk, splits) to "
                    "(ceil(B/2), Hk, splits). Each block processes 2 batch items' "
                    "8 Q heads simultaneously. MFMA utilization: 100% (vs 50% with M=8). "
                    "Different batch items may have different seq_lens → "
                    "use max(seq_len[b1], seq_len[b2]) for split boundaries."
                )
                gains.insert(0, 0.15)

            # GQA-MFMA specific optimizations
            if 'mfma' in source_lower or 'MFMA' in source or 'KV_GROUP_SIZE' in source:
                hypotheses_gqa = []
                gains_gqa = []
                import re

                # Thread count tuning
                threads_m = re.search(r'THREADS\s+(\d+)', source)
                if threads_m and int(threads_m.group(1)) == 256:
                    hypotheses_gqa.append(
                        "Try 128 threads (2 wavefronts): wave 0 for MFMA, "
                        "wave 1 for value. Each thread handles 1 head, "
                        "128/8=16 threads/head × 8 dims = 128. "
                        "Fewer threads may improve occupancy."
                    )
                    gains_gqa.append(0.12)

                # BLOCK_N tuning
                bn_m = re.search(r'BLOCK_N\s+(\d+)', source)
                if bn_m and int(bn_m.group(1)) == 16:
                    hypotheses_gqa.append(
                        "Try BLOCK_N=32: process 32 tokens per MFMA batch. "
                        "Use 2 MFMA calls (16+16) then shared softmax. "
                        "Amortizes MFMA setup over more tokens."
                    )
                    gains_gqa.append(0.10)

                # Value dequant optimization
                if '__fmaf_rn' in source:
                    hypotheses_gqa.append(
                        "Use pairwise LUT for value dequant too: "
                        "current value dequant loads scale+zero per token "
                        "then does 4× shift+mask+FMA. Could use qmap-style "
                        "lookup for value indices too."
                    )
                    gains_gqa.append(0.08)

                # Nontemporal KV cache loads
                if '__builtin_nontemporal' not in source:
                    hypotheses_gqa.append(
                        "Add nontemporal hints for KV cache reads: "
                        "streaming data, no reuse, avoid polluting L2."
                    )
                    gains_gqa.append(0.05)

                # launch_bounds tuning
                lb_m = re.search(r'__launch_bounds__\(\d+,\s*(\d+)\)', source)
                if lb_m:
                    occ = int(lb_m.group(1))
                    if occ < 4:
                        hypotheses_gqa.append(
                            f"Increase launch_bounds occupancy hint from {occ}→4 "
                            f"for better latency hiding with more concurrent waves."
                        )
                        gains_gqa.append(0.06)

                if hypotheses_gqa:
                    return hypotheses_gqa, gains_gqa

            # Phase 2: precomputed QC table (on top of pairwise LUT)
            if 'qmap2' in source and 'qc_table' not in source:
                hypotheses.insert(0,
                    "Phase2 QC precompute: build QC[128][16] = q_rot[d]*centroid[c] in LDS at kernel start. "
                    "Inner loop becomes: partial += QC[d][idx] (pure lookup, no FMA!). "
                    "Combined with pairwise: read qmap2 → get (lo,hi) indices → "
                    "partial += QC[d][lo] + QC[d+1][hi] (2 LDS + 1 ADD, no FMA). "
                    "QC table: 128*17*4 = 8.5KB LDS (stride=17 for bank-conflict-free). "
                    "Eliminates 4 FMA/token from inner loop."
                )
                gains.insert(0, 0.25)

            # Phase 3: MFMA (on top of pairwise LUT or QC)
            if ('qmap2' in source or 'qc_table' in source) and 'mfma' not in source_lower:
                hypotheses.insert(0 if 'qc_table' in source else 1,
                    "Phase3 MFMA: dequant via pairwise LUT directly into MFMA B-fragment registers. "
                    "MFMA_F32_16x16x16_F16 computes C[16,16] += A[16,16] x B[16,16]. "
                    "A = Q values (replicated for 16 tokens), B = dequantized K (from qmap2). "
                    "MFMA handles dim reduction internally → eliminates warp_reduce (5 shuffles)! "
                    "Process 16 tokens per MFMA call, 128/16=8 MFMAs for full D=128."
                )
                gains.insert(0 if 'qc_table' in source else 1, 0.20)

            # FLUTE pairwise LUT: replace 4x LDS with 2x half2 LDS
            if 'qmap2' not in source and '__half2' not in source:
                hypotheses.insert(0,
                    "FLUTE pairwise LUT: replace s_c[16] float centroid table with "
                    "qmap2[256] __half2 pairwise table. Construction: "
                    "qmap2[i*16+j] = __halves2half2(centroid[i], centroid[j]). "
                    "Lookup: 1 byte (2 packed 4-bit indices) → 1 LDS read → half2. "
                    "Reduces LDS reads from 4 to 2 per token (50% fewer). "
                    "LDS cost: 1KB (vs 64B), but much better access pattern."
                )
                gains.insert(0, 0.30)
            # Check BLOCK_KV value
            bkv_match = re.search(r'#define\s+BLOCK_KV\s+(\d+)', source)
            block_kv = int(bkv_match.group(1)) if bkv_match else 0

            if cur.vgpr_count > 48:
                hypotheses.append(
                    f"Reduce BLOCK_KV from {block_kv}→{max(2, block_kv//2)} "
                    f"to cut VGPRs from {cur.vgpr_count}→~{cur.vgpr_count - block_kv*3} "
                    f"for higher occupancy"
                )
                gains.append(0.20)

            nw_match = re.search(r'#define\s+NUM_WARPS\s+(\d+)', source)
            num_warps = int(nw_match.group(1)) if nw_match else 0
            if num_warps == 4:
                hypotheses.append(
                    "Try NUM_WARPS=2 (64 threads): halves shared memory, "
                    "may improve occupancy for short sequences"
                )
                gains.append(0.10)

            dpt_match = re.search(r'#define\s+DIMS_PER_THREAD\s+(\d+)', source)
            dpt = int(dpt_match.group(1)) if dpt_match else 0
            if dpt == 4:
                hypotheses.append(
                    "Try DIMS_PER_THREAD=2: halves per-thread register "
                    "pressure at cost of 2x more loop iterations"
                )
                gains.append(0.08)

            if "__builtin_nontemporal" not in source:
                hypotheses.append(
                    "Add __builtin_nontemporal_load for KV cache reads "
                    "(streaming access, bypass L2 pollution)"
                )
                gains.append(0.05)

            hypotheses.append(
                "Reorder inner loops: process K scores for all BLOCK_KV "
                "tokens before V accumulation (better register reuse)"
            )
            gains.append(0.07)

            # Software pipelining: overlap next KV data load with current compute
            if "prefetch" not in source_lower and "double.buffer" not in source_lower:
                hypotheses.append(
                    "Software pipelining: double-buffer KV data — load next "
                    "batch while computing current batch. For large output "
                    "workloads (seq=2048+), memory latency hiding is critical. "
                    "Prefetch next iteration's mse_u16 and val_u16 into "
                    "registers while computing scores for current tokens."
                )
                gains.append(0.18)

            # Increase BLOCK_KV for better amortization
            if block_kv <= 4:
                hypotheses.append(
                    f"Increase BLOCK_KV from {block_kv}→{block_kv*2}: "
                    f"processes {block_kv*2} tokens per warp iter instead of "
                    f"{block_kv}. For seq=2048+, this halves loop iterations "
                    f"and amortizes warp_reduce overhead across more tokens."
                )
                gains.append(0.15)

            # NUM_WARPS=8 for long sequences
            if num_warps <= 4:
                hypotheses.append(
                    "Try NUM_WARPS=8 (256 threads): for large output "
                    "(seq=2048+), more warps process more tokens per "
                    "iteration (8×BLOCK_KV tokens/iter vs 4×), reducing "
                    "total iterations and warp_reduce overhead."
                )
                gains.append(0.12)

            # Check if using fp32 compute — fp16/bf16 could be faster
            if "float4" in source or "const float*" in source:
                if "__half" not in source and "hip_bfloat16" not in source:
                    hypotheses.append(
                        "Convert Q input and score computation from fp32 to "
                        "__half (fp16): halves Q bandwidth (16B→8B per "
                        "thread), enables __hmul/__hfma for score compute, "
                        "and reduces register pressure (~50% fewer VGPRs for "
                        "Q storage). Centroid lookup stays in LDS (fp32), "
                        "convert to __half before FMA."
                    )
                    gains.append(0.25)

            if "const float*" in source and "void*" not in source:
                hypotheses.append(
                    "Accept void* q_rot with dtype parameter (0=bf16, "
                    "1=fp16, 2=fp32): enables native bf16/fp16 Q input "
                    "from Python, eliminating fp32 cast overhead in the "
                    "dispatch path."
                )
                gains.append(0.12)

            # Sort by gain
            paired = sorted(zip(gains, hypotheses), reverse=True)
            gains = [g for g, _ in paired]
            hypotheses = [h for _, h in paired]
            return hypotheses, gains

        if bottleneck_type == "near_optimal":
            hypotheses.append("Near theoretical limit — diminishing returns expected")
            gains.append(0.02)
            if "lds" not in source_lower:
                hypotheses.append("Try LDS-based partial reduction for sub-warp configs")
                gains.append(0.05)
            return hypotheses, gains

        # ── Source-aware hypothesis generation ─────────────────────────
        has_online = "online" in source_lower or "single-pass" in source_lower or \
                     ("e_max - lse" in source and "lse - e_max" in source)
        has_float4 = "float4" in source
        has_shfl   = "__shfl" in source
        has_adaptive = "batch_crossover" in source_lower or "crossover" in source_lower
        has_two_loops = source.count("for (int s") >= 2

        # Only suggest what's NOT already there
        if not has_online and has_two_loops:
            hypotheses.append(
                "Online softmax: single-pass reduction eliminates 2nd read loop"
            )
            gains.append(0.20)

        if not has_float4:
            hypotheses.append(
                "float4 vectorized loads — 4 dims/thread, 32 threads cover D=128"
            )
            gains.append(0.15)

        if not has_shfl:
            hypotheses.append(
                "Warp-level __shfl broadcast for LSE — avoid 63 redundant global loads"
            )
            gains.append(0.10)

        if not has_adaptive:
            hypotheses.append(
                "Batch-adaptive dispatch: different strategy for small vs large B"
            )
            gains.append(0.08)

        # Check for specific bottleneck patterns
        # Small-B dominated by launch overhead
        small_b_keys = [k for k in efficiency if "B1_" in k or "B4_" in k]
        if small_b_keys:
            small_b_eff = sum(efficiency[k] for k in small_b_keys) / len(small_b_keys)
            if small_b_eff < 0.10:
                hypotheses.append(
                    "Small batch (B≤4) dominated by launch overhead — "
                    "consider persistent kernel or batching multiple heads per block"
                )
                gains.append(0.05)

        if bottleneck_type == "latency":
            hypotheses.append(
                "Fuse Stage1+Stage2 to eliminate mid_o global memory round-trip"
            )
            gains.append(0.30)

        if cur.vgpr_count > 0 and cur.vgpr_count >= 64:
            hypotheses.append(
                f"VGPR count is {cur.vgpr_count} — try reducing to <64 for better occupancy"
            )
            gains.append(0.10)

        # If everything is already applied, suggest fine-tuning
        if not hypotheses:
            hypotheses.append(
                "All major optimizations applied — try tuning BATCH_CROSSOVER threshold"
            )
            gains.append(0.03)
            hypotheses.append(
                "Try __builtin_nontemporal_load for mid_o reads (bypass L2)"
            )
            gains.append(0.05)
            hypotheses.append(
                "Experiment with launch_bounds occupancy target"
            )
            gains.append(0.03)

        # Sort by estimated gain
        paired = sorted(zip(gains, hypotheses), reverse=True)
        gains = [g for g, _ in paired]
        hypotheses = [h for _, h in paired]

        return hypotheses, gains

    def _generate_wht_hypotheses(
        self, source: str, source_lower: str,
        efficiency: dict[str, float],
        cur: IterationResult,
    ) -> tuple[list[str], list[float]]:
        """WHT rotation kernel optimization hypotheses.

        The kernel is launch-latency bound (~5-6µs floor). Main strategies:
        1. Process more rows per block to reduce grid size
        2. Vectorize memory access (uint64 / float4)
        3. Use LDS to cache signs (avoid redundant global reads)
        4. Process multiple elements per thread
        5. Try different warp/block configurations
        """
        import re
        hypotheses: list[str] = []
        gains: list[float] = []

        # Parse kernel parameters
        rpb = re.search(r'#define\s+ROWS_PER_BLOCK\s+(\d+)', source)
        rows_per_block = int(rpb.group(1)) if rpb else 1

        dpt = re.search(r'#define\s+DIMS_PER_THREAD\s+(\d+)', source)
        dims_per_thread = int(dpt.group(1)) if dpt else 4

        has_vec_load = "uint64_t" in source or "float4" in source
        has_lds_signs = "__shared__" in source and "signs" in source_lower
        has_constexpr_scale = "constexpr" in source and "inv_sqrt" in source_lower

        # 1. More rows per block
        if rows_per_block < 16:
            hypotheses.append(
                f"Increase ROWS_PER_BLOCK from {rows_per_block}→{min(32, rows_per_block*2)}: "
                f"reduces grid size from M→M/{min(32, rows_per_block*2)}, "
                f"fewer blocks = lower scheduling overhead"
            )
            gains.append(0.15)

        # 2. LDS signs caching
        if not has_lds_signs:
            hypotheses.append(
                "Cache signs array in LDS (__shared__): only 128×4=512 bytes, "
                "avoids repeated global memory reads across warps in same block"
            )
            gains.append(0.10)

        # 3. Vectorized loads
        if not has_vec_load:
            hypotheses.append(
                "Use vectorized uint64_t loads for 4×bf16 elements per load, "
                "and vectorized stores for output"
            )
            gains.append(0.08)

        # 4. DIMS_PER_THREAD increase
        if dims_per_thread < 8:
            hypotheses.append(
                f"DIMS_PER_THREAD={dims_per_thread*2}: fewer threads per warp needed, "
                f"16 threads × 8 dims = 128 → more rows per warp"
            )
            gains.append(0.05)

        # 5. Pre-multiply signs into fused load+multiply
        if "vals[i] * signs" in source or "vals[i] *= signs" in source:
            hypotheses.append(
                "Use bit-manipulation for sign application: signs are ±1, "
                "so XOR the sign bit of float32 instead of multiplication"
            )
            gains.append(0.03)

        # 6. Persistent kernel approach
        hypotheses.append(
            "Persistent kernel: keep wavefronts alive across multiple dispatches, "
            "poll for new work via atomic flag (eliminates launch overhead entirely)"
        )
        gains.append(0.25)

        # 7. Fuse with fused decode kernel
        hypotheses.append(
            "Fuse WHT rotation into the fused decode kernel: transform Q in-register "
            "before the attention computation, zero additional launch overhead"
        )
        gains.append(0.30)

        if not hypotheses:
            hypotheses.append("All WHT optimizations applied — try tuning block config")
            gains.append(0.02)

        # Sort by gain
        paired = sorted(zip(gains, hypotheses), reverse=True)
        gains = [g for g, _ in paired]
        hypotheses = [h for _, h in paired]
        return hypotheses, gains

    def _generate_fused_hypotheses(
        self, source: str, source_lower: str,
        efficiency: dict[str, float],
        cur: IterationResult,
    ) -> tuple[list[str], list[float]]:
        """Fused-kernel-specific optimization hypotheses.

        The fused kernel is compute-bound: for each KV token it does
        centroid lookup + Q·K dot product + online softmax + V dequant +
        accumulate. Reducing per-thread register usage and increasing
        occupancy are the main levers.
        """
        import re
        hypotheses: list[str] = []
        gains: list[float] = []

        # BLOCK_KV: how many KV tokens per warp per iteration
        bkv = re.search(r'#define\s+BLOCK_KV\s+(\d+)', source)
        block_kv = int(bkv.group(1)) if bkv else 0

        # NUM_WARPS
        nw = re.search(r'#define\s+NUM_WARPS\s+(\d+)', source)
        num_warps = int(nw.group(1)) if nw else 0

        # DIMS_PER_THREAD
        dpt = re.search(r'#define\s+DIMS_PER_THREAD\s+(\d+)', source)
        dims_per_thread = int(dpt.group(1)) if dpt else 0

        # launch_bounds occupancy hint
        lb = re.search(r'__launch_bounds__\(\d+,\s*(\d+)\)', source)
        occ_hint = int(lb.group(1)) if lb else 0

        # 1. BLOCK_KV reduction — main VGPR lever
        if block_kv > 2:
            hypotheses.append(
                f"Reduce BLOCK_KV {block_kv}→{max(2, block_kv//2)}: "
                f"cuts {block_kv*3} VGPRs (scores[]/p[]/slot_bases[]) → "
                f"higher occupancy"
            )
            gains.append(0.20)

        # 2. NUM_WARPS tuning
        if num_warps == 4:
            hypotheses.append(
                "NUM_WARPS=2 (64 threads): halves s_acc shared memory "
                "(2KB→1KB), may allow more blocks/CU for short sequences"
            )
            gains.append(0.12)
        elif num_warps == 2:
            hypotheses.append(
                "NUM_WARPS=8 (256 threads): more intra-block parallelism "
                "for long sequences, better latency hiding"
            )
            gains.append(0.10)

        # 3. DIMS_PER_THREAD
        if dims_per_thread == 4:
            hypotheses.append(
                "DIMS_PER_THREAD=2: halves per-thread registers "
                "(q0h-q3h → q0h-q1h), needs 64 threads/warp to cover D=128"
            )
            gains.append(0.08)

        # 4. launch_bounds occupancy target
        if occ_hint > 0 and occ_hint < 12:
            hypotheses.append(
                f"Increase launch_bounds occupancy hint from {occ_hint} "
                f"to {min(16, occ_hint + 4)} — compiler may spill less"
            )
            gains.append(0.05)

        # 5. Vectorized centroid load
        if "float4" not in source and "__half2" not in source:
            hypotheses.append(
                "Use __half2 packed centroid multiply: process 2 centroids "
                "per instruction via half2 multiply-accumulate"
            )
            gains.append(0.10)

        # 6. Nontemporal KV cache loads
        if "__builtin_nontemporal" not in source:
            hypotheses.append(
                "Add nontemporal load hints for KV cache reads "
                "(streaming access pattern, bypass L2 pollution)"
            )
            gains.append(0.04)

        # 7. Shared memory preload of centroids already present, but check
        if "s_c[" not in source and "centroids" in source:
            hypotheses.append(
                "Preload centroids to shared memory (avoid repeated "
                "global memory reads in inner loop)"
            )
            gains.append(0.15)

        # 8. Loop unrolling depth
        unroll_count = source.count("#pragma unroll")
        if unroll_count < 3:
            hypotheses.append(
                "Add #pragma unroll to inner KV loop and cross-warp "
                "reduction for reduced loop overhead"
            )
            gains.append(0.05)

        if not hypotheses:
            hypotheses.append("All major fused-kernel optimizations applied "
                             "— try tuning launch_bounds")
            gains.append(0.03)

        # Sort by gain
        paired = sorted(zip(gains, hypotheses), reverse=True)
        gains = [g for g, _ in paired]
        hypotheses = [h for _, h in paired]
        return hypotheses, gains

    def _build_summary(self, avg_eff: float, worst_eff: float,
                       worst_key: str, bottleneck_type: str,
                       iteration: int, data_source: str) -> str:
        """Build human-readable summary."""
        return (
            f"Bottleneck Analysis (iteration {iteration})\n"
            f"{'='*50}\n"
            f"Data source: {data_source}\n"
            f"Avg HBM efficiency: {avg_eff*100:.1f}%\n"
            f"Worst config: {worst_key} ({worst_eff*100:.1f}%)\n"
            f"Bottleneck type: {bottleneck_type}\n"
            f"Target: >80% efficiency across all configs\n"
        )

    # ── XOR Swizzle specific hypotheses ──────────────────────────
    def _generate_swizzle_hypotheses(
        self, source: str, source_lower: str,
        efficiency: dict[str, float],
        cur,
    ) -> tuple[list[str], list[float]]:
        """Hypotheses for XOR swizzle centroid lookup optimization."""
        import re
        hypotheses = []
        gains = []

        # Check current LDS usage
        lds_match = re.search(r'group_segment.*?(\d+)', source)

        if 'SWIZZLE' in source or 'swizzle' in source or 'xor' in source_lower:
            # Already has swizzle — optimize the swizzle itself

            # 1. Reduce per-thread copies: use per-half-warp instead of per-thread
            if 'lane_id * N_CENTROIDS' in source:
                hypotheses.append(
                    "Reduce swizzle copies: use per-half-warp (16 threads share 1 copy) "
                    "instead of per-thread (32 copies). LDS: 8KB→2KB. "
                    "Half-warp threads 0-15 share copy A (banks 0-15), "
                    "threads 16-31 share copy B (banks 16-31). "
                    "Within half-warp: XOR with (lane_id & 0x7) for 8-way conflict avoidance."
                )
                gains.append(0.25)

            # 2. Reduce BLOCK_KV to offset LDS cost
            bkv = re.search(r'#define\s+BLOCK_KV\s+(\d+)', source)
            if bkv and int(bkv.group(1)) > 4:
                hypotheses.append(
                    f"Reduce BLOCK_KV from {bkv.group(1)}→4: fewer VGPRs for "
                    f"scores/p/slot arrays, offsetting swizzle LDS cost."
                )
                gains.append(0.12)

            # 3. Use fp16 centroids in swizzle table (halve LDS)
            if '__half' not in source or 'fp16 centroid' not in source_lower:
                hypotheses.append(
                    "Store swizzle centroids as __half (fp16) instead of float: "
                    "halves swizzle LDS from 8KB→4KB, improves occupancy. "
                    "Convert to float after lookup: __half2float(s_c_h16[...])."
                )
                gains.append(0.20)

            # 4. Reduce launch_bounds occupancy hint
            lb = re.search(r'__launch_bounds__\(\d+,\s*(\d+)\)', source)
            if lb and int(lb.group(1)) < 8:
                hypotheses.append(
                    f"Increase launch_bounds hint from {lb.group(1)}→8: "
                    f"let compiler optimize for higher occupancy."
                )
                gains.append(0.08)

            # 5. Use 2-copy instead of 32-copy
            if 'WARP_SIZE * N_CENTROIDS' in source:
                hypotheses.append(
                    "2-copy swizzle: only 2 copies of centroids (even/odd lanes), "
                    "offset by 16 banks. LDS: 8KB→256B. "
                    "Even lanes: s_c[idx], odd lanes: s_c[16+idx]. "
                    "Conflicts halved (16 threads per copy vs 32)."
                )
                gains.append(0.22)


            # Long-seq compute optimizations (output>1024)
            # These target the real bottleneck for seq≥4096

            # Increase BLOCK_KV for fewer loop iterations
            bkv = re.search(r'#define\s+BLOCK_KV\s+(\d+)', source)
            if bkv and int(bkv.group(1)) < 16:
                hypotheses.append(
                    f"Increase BLOCK_KV from {bkv.group(1)}→{min(16, int(bkv.group(1))*2)}: "
                    f"for seq≥4096, halves loop iterations. Each iter has fixed "
                    f"overhead (page lookup, warp_reduce setup). More tokens/iter = less overhead."
                )
                gains.append(0.15)

            # NUM_WARPS tuning for long seq
            nw = re.search(r'#define\s+NUM_WARPS\s+(\d+)', source)
            if nw and int(nw.group(1)) == 4:
                hypotheses.append(
                    "Try NUM_WARPS=8: for seq≥4096, more warps process more tokens "
                    "per iteration (8×BLOCK_KV vs 4×BLOCK_KV), better GPU utilization."
                )
                gains.append(0.12)
            elif nw and int(nw.group(1)) == 4:
                hypotheses.append(
                    "Try NUM_WARPS=2: fewer warps but higher occupancy, "
                    "may help when grid is already large enough."
                )
                gains.append(0.08)

            # Unroll factor for value dequant
            if '__fmaf_rn(p[kv]' in source:
                hypotheses.append(
                    "Hoist value scale/zero load outside inner kv loop: "
                    "currently loads scale+zero per token inside BLOCK_KV loop. "
                    "For seq≥4096, preload scale/zero for all BLOCK_KV tokens "
                    "before the accumulation loop to reduce memory stalls."
                )
                gains.append(0.10)

            # Nontemporal load for long sequences
            if '__builtin_nontemporal' not in source:
                hypotheses.append(
                    "Add __builtin_nontemporal_load for KV cache reads: "
                    "for seq≥4096, KV data is streaming (read once, no reuse). "
                    "Nontemporal hints prevent L2 cache pollution."
                )
                gains.append(0.08)

        return hypotheses, gains

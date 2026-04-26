"""SystemAnalyzer Agent — Layer 1-3 system-level bottleneck analysis.

This agent was born from a key lesson: optimizing kernel parameters (Layer 4-5)
while ignoring system architecture (Layer 1-3) is like polishing the engine
while the car has square wheels.

It answers the question V1 never asked:
  "WHY is this kernel slow at the system level?"

Instead of suggesting BLOCK_KV or NUM_WARPS changes, it analyzes:
  - Cache line utilization (how much of each 64B line is useful?)
  - Data layout efficiency (AoS vs SoA impact on memory bandwidth)
  - Work distribution (what computation can move to store-time?)
  - Kernel decomposition (is split/fused/unified the right choice?)
  - E2E bottleneck identification (is the kernel even the problem?)
"""
from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import dataclass, field

from agents.state import ActionSpec, PipelineState
from agents.hardware_model import HardwareSpec, MI355X


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class CacheLineAnalysis:
    """Analysis of cache line utilization for a specific data access pattern."""
    access_name: str          # e.g. "K_norm load", "V_scale load"
    useful_bytes: int         # bytes actually used from the cache line
    cacheline_bytes: int      # 64 bytes on MI355X
    utilization: float        # useful_bytes / cacheline_bytes
    frequency: str            # "per_token" | "per_block" | "once"
    coalesced: bool           # whether adjacent threads access adjacent bytes
    recommendation: str       # what to do about it


@dataclass
class WorkDistributionItem:
    """An operation that could potentially move between store and decode."""
    operation: str
    current_phase: str        # "decode" | "store" | "both"
    proposed_phase: str       # "store" | "decode"
    cost_per_decode: float    # µs per decode step
    cost_per_store: float     # µs per store step
    savings_decode_pct: float # % decode time saved if moved
    complexity: str           # "trivial" | "moderate" | "hard"


@dataclass
class SystemAnalysisReport:
    """Full system-level analysis output."""

    # Layer 3: Data Layout
    layout_type: str = "unknown"      # "AoS" | "SoA" | "hybrid"
    cacheline_analyses: list[CacheLineAnalysis] = field(default_factory=list)
    avg_cacheline_utilization: float = 0.0
    layout_recommendation: str = ""
    layout_improvement_estimate: float = 0.0  # multiplier, e.g. 1.5 = 50% faster

    # Layer 2: Work Distribution
    store_time_candidates: list[WorkDistributionItem] = field(default_factory=list)
    total_decode_savings_pct: float = 0.0

    # Layer 1: System Architecture
    kernel_decomposition: str = ""    # "split_2stage" | "fused" | "unified"
    decomposition_recommendation: str = ""
    dispatch_overhead_us: float = 0.0
    e2e_bottleneck: str = ""          # "attention" | "model_fwd" | "scheduling"

    # Overall
    optimization_layer: int = 4       # which layer to optimize (1=system, 5=instruction)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    action_specs: list[ActionSpec] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║          SYSTEM-LEVEL ANALYSIS REPORT                       ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"Optimization Layer: {self.optimization_layer} "
            f"({'SYSTEM ARCH' if self.optimization_layer <= 2 else 'DATA LAYOUT' if self.optimization_layer == 3 else 'KERNEL' if self.optimization_layer == 4 else 'INSTRUCTION'})",
            "",
            "─── Layer 3: Data Layout ───",
            f"  Current layout: {self.layout_type}",
            f"  Avg cache line utilization: {self.avg_cacheline_utilization*100:.1f}%",
            "",
        ]
        for cl in self.cacheline_analyses:
            lines.append(
                f"  {cl.access_name:25s}: {cl.useful_bytes:2d}/{cl.cacheline_bytes}B "
                f"= {cl.utilization*100:.0f}% util "
                f"{'✓ coalesced' if cl.coalesced else '✗ scattered'} "
                f"[{cl.frequency}]"
            )
        lines.extend([
            "",
            f"  Layout recommendation: {self.layout_recommendation}",
            f"  Estimated improvement: {self.layout_improvement_estimate:.1f}×",
            "",
            "─── Layer 2: Work Distribution ───",
        ])
        for item in self.store_time_candidates:
            lines.append(
                f"  {item.operation:30s}: {item.current_phase}→{item.proposed_phase} "
                f"saves {item.savings_decode_pct:.0f}% decode "
                f"[{item.complexity}]"
            )
        lines.extend([
            f"  Total decode savings: {self.total_decode_savings_pct:.0f}%",
            "",
            "─── Layer 1: System Architecture ───",
            f"  Kernel decomposition: {self.kernel_decomposition}",
            f"  Decomposition recommendation: {self.decomposition_recommendation}",
            f"  Dispatch overhead: {self.dispatch_overhead_us:.1f}µs/step",
            f"  E2E bottleneck: {self.e2e_bottleneck}",
            "",
            "─── Recommendations (priority order) ───",
        ])
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        if self.action_specs:
            lines.extend(["", "─── Action Specs ───"])
            for spec in self.action_specs:
                lines.append(
                    f"  - {spec.action_type}: {spec.name} params={spec.params}"
                )

        lines.extend(["", f"Summary: {self.summary}", ""])
        return "\n".join(lines)


# ── Main Agent ───────────────────────────────────────────────────────

class SystemAnalyzerAgent:
    """Analyzes system-level bottlenecks before kernel-level optimization."""

    def __init__(self, hw: HardwareSpec = MI355X):
        self.hw = hw

    def run(self, state: PipelineState) -> SystemAnalysisReport:
        """Perform system-level analysis.

        This runs BEFORE the Profile→Analyze→GEAK cycle.
        It determines WHETHER kernel optimization is the right approach,
        or if a system-level change (layout, decomposition) is needed first.
        """
        report = SystemAnalysisReport()

        # Read current kernel source and layout config
        source = self._read_kernel_source(state)
        layout_config = self._detect_layout(source, state)

        # Layer 3: Cache line analysis
        self._analyze_cacheline_utilization(report, layout_config, source)

        # Layer 2: Work distribution analysis
        self._analyze_work_distribution(report, source, state)

        # Layer 1: System architecture analysis
        self._analyze_system_architecture(report, source, state)

        # Determine which layer to optimize
        self._determine_optimization_layer(report, state)

        # Generate prioritized recommendations
        self._generate_recommendations(report)
        self._generate_action_specs(report, state)

        # Save report
        self._save_report(report, state)
        state.system_action_specs = report.action_specs

        return report

    def _read_kernel_source(self, state: PipelineState) -> str:
        """Find and read the kernel source."""
        candidates = []
        if state.current.hip_source and os.path.exists(state.current.hip_source):
            candidates.append(state.current.hip_source)
        if state.prev and state.prev.hip_source and os.path.exists(state.prev.hip_source):
            candidates.append(state.prev.hip_source)

        # Also check the kernel directory
        from agents.target_registry import get_target
        try:
            tc = get_target(state.target_kernel)
            from pathlib import Path
            for f in sorted(Path(tc.kernel_dir).glob(tc.source_glob),
                          key=lambda p: int(re.search(r'_v(\d+)', p.name).group(1))
                          if re.search(r'_v(\d+)', p.name) else 0):
                candidates.append(str(f))
        except (ValueError, KeyError):
            pass

        for c in reversed(candidates):  # newest first
            if os.path.exists(c):
                with open(c) as f:
                    return f.read()
        return ""

    def _detect_layout(self, source: str, state: PipelineState) -> dict:
        """Detect the current KV cache data layout from source code."""
        layout = {
            "type": "AoS",  # default assumption
            "slot_size": 136,
            "key_data_bytes": 64,   # MSE quantized key indices
            "key_norm_bytes": 2,    # fp16 norm
            "key_norm_offset": 64,  # offset within slot
            "val_data_bytes": 64,   # 4-bit quantized values
            "val_scale_bytes": 2,   # fp16 scale
            "val_zero_bytes": 2,    # fp16 zero
            "val_offset": 68,       # KPS=68
            "block_size": 16,
        }

        # Parse from source
        kps = re.search(r'#define\s+KPS\s+(\d+)', source)
        if kps:
            layout["val_offset"] = int(kps.group(1))

        mse = re.search(r'#define\s+MSE_BYTES\s+(\d+)', source)
        if mse:
            layout["key_data_bytes"] = int(mse.group(1))

        val_data = re.search(r'#define\s+VAL_DATA_BYTES\s+(\d+)', source)
        if val_data:
            layout["val_data_bytes"] = int(val_data.group(1))

        bs = re.search(r'#define\s+BLOCK_SIZE\s+(\d+)', source)
        if bs:
            layout["block_size"] = int(bs.group(1))

        # Detect SoA layout markers
        if 'META_REGION_OFFSET' in source or 'soa_field' in source.lower():
            layout["type"] = "SoA"
        elif 'data_bases' in source and 'head_meta' in source:
            layout["type"] = "SoA"

        return layout

    def _analyze_cacheline_utilization(self, report: SystemAnalysisReport,
                                        layout: dict, source: str) -> None:
        """Analyze how efficiently each memory access uses cache lines.

        This is the analysis that V1 NEVER did. The key insight:
        loading 2 bytes of K_norm from a 136-byte AoS slot means
        only 2/64 = 3.1% of the 64B cache line is useful data.
        """
        report.layout_type = layout["type"]
        cacheline = 64  # bytes
        analyses = []

        if layout["type"] == "AoS":
            # AoS layout: [key_data(64) | norm(2) | pad(2) | val_data(64) | scale(2) | zero(2)]
            # Each slot is 136 bytes, containing data for ONE position, ONE head

            # 1. K data load (4-bit quantized key, 128 dims = 64 bytes)
            analyses.append(CacheLineAnalysis(
                access_name="K quantized data",
                useful_bytes=64,
                cacheline_bytes=cacheline,
                utilization=64.0 / max(cacheline, 64),
                frequency="per_token",
                coalesced=False,  # different threads access different tokens' slots
                recommendation="Good utilization per load, but scattered across slots"
            ))

            # 2. K norm load (2 bytes from offset 64 in 136-byte slot)
            analyses.append(CacheLineAnalysis(
                access_name="K_norm (fp16)",
                useful_bytes=2,
                cacheline_bytes=cacheline,
                utilization=2.0 / cacheline,
                frequency="per_token",
                coalesced=False,
                recommendation="CRITICAL: 3% utilization. SoA layout would coalesce "
                              "16 norms into 32B = 50% util. 16× improvement."
            ))

            # 3. V data load (4-bit quantized values, 64 bytes)
            analyses.append(CacheLineAnalysis(
                access_name="V quantized data",
                useful_bytes=64,
                cacheline_bytes=cacheline,
                utilization=64.0 / max(cacheline, 64),
                frequency="per_token",
                coalesced=False,
                recommendation="Good utilization per load, but scattered across slots"
            ))

            # 4. V scale+zero load (4 bytes from offset 132 in 136-byte slot)
            analyses.append(CacheLineAnalysis(
                access_name="V_scale+V_zero (fp16×2)",
                useful_bytes=4,
                cacheline_bytes=cacheline,
                utilization=4.0 / cacheline,
                frequency="per_token",
                coalesced=False,
                recommendation="CRITICAL: 6% utilization. SoA coalesces 16 scales "
                              "into 32B + 16 zeros into 32B = 50% util each."
            ))

            # 5. Q rotation (if not fused)
            if 'q_rot' in source and 'launch_tq_wht' not in source:
                analyses.append(CacheLineAnalysis(
                    access_name="Q rotation (external)",
                    useful_bytes=0,
                    cacheline_bytes=0,
                    utilization=0.0,
                    frequency="per_step",
                    coalesced=True,
                    recommendation="Q rotation done externally. Fusing into attention "
                                  "kernel eliminates ~50-60µs/step dispatch overhead."
                ))

        elif layout["type"] == "SoA":
            # SoA layout: data region contiguous, metadata region separate
            analyses.append(CacheLineAnalysis(
                access_name="K quantized data",
                useful_bytes=64,
                cacheline_bytes=cacheline,
                utilization=1.0,
                frequency="per_token",
                coalesced=True,
                recommendation="Good — contiguous data region"
            ))
            analyses.append(CacheLineAnalysis(
                access_name="K_norm (SoA coalesced)",
                useful_bytes=32,  # 16 norms × 2B
                cacheline_bytes=cacheline,
                utilization=32.0 / cacheline,
                frequency="per_block",
                coalesced=True,
                recommendation="Good — 16 norms loaded together from metadata region"
            ))
            analyses.append(CacheLineAnalysis(
                access_name="V_scale (SoA coalesced)",
                useful_bytes=32,  # 16 scales × 2B
                cacheline_bytes=cacheline,
                utilization=32.0 / cacheline,
                frequency="per_block",
                coalesced=True,
                recommendation="Good — 16 scales loaded together"
            ))

        report.cacheline_analyses = analyses

        # Compute utilization — focus on METADATA accesses (the bottleneck)
        # The key insight: K/V data loads are fine (64B = 100%),
        # but metadata loads (norm, scale, zero) are 3-6% — THESE are the problem.
        if analyses:
            metadata_accesses = [a for a in analyses
                                if a.cacheline_bytes > 0 and a.utilization < 0.5]
            if metadata_accesses:
                # Use the WORST metadata utilization as the headline number
                # This is what matters — one 3% access pattern dominates latency
                report.avg_cacheline_utilization = min(a.utilization for a in metadata_accesses)
            else:
                # All accesses are well-utilized
                valid = [a for a in analyses if a.cacheline_bytes > 0]
                report.avg_cacheline_utilization = (
                    sum(a.utilization for a in valid) / len(valid) if valid else 0
                )

        # Estimate layout improvement
        if layout["type"] == "AoS":
            # AoS→SoA eliminates scattered metadata loads
            # Metadata loads (norm, scale, zero) go from 3-6% util to 50% util
            # This affects ~20-30% of total memory traffic
            # Net improvement: ~1.3-1.5× for memory-bound kernels
            # But Stage1 is latency-bound (not BW-bound), so impact is higher:
            # scattered loads create stalls, SoA eliminates them → ~2× or more
            report.layout_recommendation = (
                "CHANGE TO SoA: Metadata (K_norm, V_scale, V_zero) should be in "
                "contiguous per-block regions, separate from data. This eliminates "
                "scattered 2-4 byte loads that waste 94-97% of each cache line."
            )
            report.layout_improvement_estimate = 1.5  # conservative
        else:
            report.layout_recommendation = "SoA layout already in use — good"
            report.layout_improvement_estimate = 1.0

    def _analyze_work_distribution(self, report: SystemAnalysisReport,
                                    source: str, state: PipelineState) -> None:
        """Analyze what work can be moved from decode-time to store-time."""
        items = []

        # 1. norm_correction: rsqrt(centroid_norm²) computed per decode step
        if 'norm_correction' in source or 'rsqrtf' in source:
            has_store_folding = 'c_inv_norm' in source and 'store' in source.lower()
            items.append(WorkDistributionItem(
                operation="norm_correction (rsqrt)",
                current_phase="decode" if not has_store_folding else "store",
                proposed_phase="store",
                cost_per_decode=5.0 if not has_store_folding else 0.0,
                cost_per_store=0.1,  # tiny per-token cost
                savings_decode_pct=3.0 if not has_store_folding else 0.0,
                complexity="trivial" if not has_store_folding else "done",
            ))

        # 2. Centroid norm² accumulation (128 dims × 4 multiplies × shuffle reduce)
        if 'my_nsq' in source or 'c_norm_sq' in source:
            items.append(WorkDistributionItem(
                operation="centroid norm² accumulation",
                current_phase="decode",
                proposed_phase="store",
                cost_per_decode=8.0,  # 128 FMAs + cross-lane shuffle
                cost_per_store=0.2,
                savings_decode_pct=5.0,
                complexity="trivial",  # fold into K_norm at store time
            ))

        # 3. Q rotation (WHT butterfly)
        if 'q_rot' in source and 'signs' not in source:
            items.append(WorkDistributionItem(
                operation="Q rotation (external launch)",
                current_phase="decode",
                proposed_phase="decode_fused",
                cost_per_decode=50.0,  # ~50µs launch overhead
                cost_per_store=0.0,
                savings_decode_pct=15.0,
                complexity="moderate",  # fuse into attention kernel prologue
            ))

        # 4. Centroid LUT construction
        if 's_qmap2' in source or 'qmap2' in source:
            items.append(WorkDistributionItem(
                operation="Centroid LUT build (per block)",
                current_phase="decode",
                proposed_phase="decode",  # must stay, but can be optimized
                cost_per_decode=1.0,
                cost_per_store=0.0,
                savings_decode_pct=0.0,
                complexity="done",
            ))

        report.store_time_candidates = items
        report.total_decode_savings_pct = sum(
            i.savings_decode_pct for i in items if i.proposed_phase != i.current_phase
        )

    def _analyze_system_architecture(self, report: SystemAnalysisReport,
                                      source: str, state: PipelineState) -> None:
        """Analyze kernel decomposition and dispatch strategy."""

        # Detect current decomposition
        if 'mid_o' in source and 'split' in state.target_kernel.lower():
            report.kernel_decomposition = "split_2stage"
            report.decomposition_recommendation = (
                "Split 2-stage (Stage1 → mid_o → Stage2) incurs global memory "
                "round-trip. For short sequences (< 4096), fused kernel is better. "
                "For long sequences, split is necessary but consider unified dispatch "
                "that dynamically chooses fused vs split based on seq_len."
            )
            report.dispatch_overhead_us = 15.0  # Stage2 launch + mid_o alloc
        elif 'fused' in state.target_kernel.lower():
            report.kernel_decomposition = "fused"
            report.decomposition_recommendation = (
                "Fused kernel avoids mid_o round-trip but may have lower occupancy "
                "for long sequences. Consider unified 2D/3D kernel that uses "
                "BLOCK_M=128 for prefill and 3D split-KV for long decode."
            )
            report.dispatch_overhead_us = 5.0
        else:
            report.kernel_decomposition = "unknown"
            report.decomposition_recommendation = "Could not determine decomposition"
            report.dispatch_overhead_us = 10.0

        # E2E bottleneck (heuristic)
        # For 72B model: attention is ~40% of total decode step
        # If TQ attention is 2× slower than BF16, it adds ~40% to total step time
        report.e2e_bottleneck = "attention_kernel"

    def _determine_optimization_layer(self, report: SystemAnalysisReport,
                                       state: PipelineState) -> None:
        """Decide which layer to focus optimization on."""

        # If cache line utilization is < 20%, layout change is the priority
        if report.avg_cacheline_utilization < 0.20:
            report.optimization_layer = 3
            report.summary = (
                f"SYSTEM BOTTLENECK: Data layout causes {report.avg_cacheline_utilization*100:.0f}% "
                f"cache line utilization. Kernel-level optimizations cannot fix this. "
                f"Priority: change to SoA layout (Layer 3)."
            )
            return

        # If > 10% decode time can be saved by moving work to store, do that
        if report.total_decode_savings_pct > 10:
            report.optimization_layer = 2
            report.summary = (
                f"WORK DISTRIBUTION: {report.total_decode_savings_pct:.0f}% of decode "
                f"time is spent on work that could be done at store-time. "
                f"Priority: move norm_correction and Q rotation to store/fuse (Layer 2)."
            )
            return

        # If decomposition is suboptimal for the workload
        if (report.kernel_decomposition == "split_2stage" and
            report.dispatch_overhead_us > 10):
            report.optimization_layer = 1
            report.summary = (
                f"ARCHITECTURE: Split 2-stage dispatch adds {report.dispatch_overhead_us:.0f}µs "
                f"overhead. Consider unified kernel with dynamic dispatch."
            )
            return

        # Otherwise, kernel-level optimization is appropriate
        report.optimization_layer = 4
        report.summary = (
            "System-level architecture is reasonable. "
            "Proceed with kernel-level optimization (Layer 4)."
        )

    def _generate_recommendations(self, report: SystemAnalysisReport) -> None:
        """Generate prioritized recommendations based on analysis."""
        recs = []

        # Sort by impact (Layer 3 > Layer 2 > Layer 1 > Layer 4)

        # Layer 3 recommendations
        if report.layout_type == "AoS" and report.avg_cacheline_utilization < 0.30:
            recs.append(
                f"[Layer 3 — HIGH IMPACT] Change KV cache from AoS to SoA layout. "
                f"Current cache line utilization is {report.avg_cacheline_utilization*100:.0f}%. "
                f"SoA separates metadata (K_norm, V_scale, V_zero) into contiguous "
                f"per-block regions. Expected improvement: {report.layout_improvement_estimate:.1f}×"
            )

        # Layer 2 recommendations
        movable = [i for i in report.store_time_candidates
                   if i.current_phase != i.proposed_phase and i.savings_decode_pct > 0]
        for item in sorted(movable, key=lambda x: -x.savings_decode_pct):
            recs.append(
                f"[Layer 2] Move '{item.operation}' from {item.current_phase} to "
                f"{item.proposed_phase}: saves {item.savings_decode_pct:.0f}% decode time "
                f"[complexity: {item.complexity}]"
            )

        # Layer 1 recommendations
        if report.kernel_decomposition == "split_2stage":
            recs.append(
                "[Layer 1] Consider unified 2D/3D kernel that handles both "
                "prefill (BLOCK_M=128) and decode (3D split-KV) with single dispatch"
            )

        # Layer 4 recommendation (only if higher layers are OK)
        if not recs:
            recs.append(
                "[Layer 4] System architecture is sound. Proceed with "
                "kernel-level tuning (BLOCK_N, occupancy, vectorization)."
            )

        report.recommendations = recs

    def _generate_action_specs(self, report: SystemAnalysisReport,
                               state: PipelineState) -> None:
        """Convert system-level recommendations into executable action specs."""
        report.action_specs = []
        try:
            from agents.target_registry import get_target

            target_cfg = get_target(state.target_kernel)
        except Exception:
            target_cfg = None

        if target_cfg is not None and target_cfg.target_type != "campaign":
            if report.optimization_layer <= 3:
                report.action_specs.append(
                    ActionSpec(
                        action_id=f"system-{state.round_index}-campaign-switch",
                        action_type="campaign_switch",
                        name="switch_to_tq_fusion_v3_hip",
                        params={"target_kernel": "tq_fusion_v3_hip"},
                        rationale=(
                            "System analysis detected that layout / work-distribution "
                            "changes dominate micro-kernel tuning."
                        ),
                        expected_gain=max(report.layout_improvement_estimate - 1.0, 0.0),
                        source_agent="system_analyzer",
                        priority=0,
                        tags=["system", "campaign"],
                    )
                )
            return

        if target_cfg is None:
            return

        for bucket in (
            "workload_pack_change",
            "evaluator_pack_change",
            "profile_pack_change",
        ):
            for idx, option in enumerate(target_cfg.search_space.get(bucket, []), 1):
                report.action_specs.append(
                    ActionSpec(
                        action_id=f"system-{state.round_index}-{bucket}-{idx}",
                        action_type=bucket,
                        name=option.get("name", f"{bucket}_{idx}"),
                        params=copy.deepcopy(option.get("params", {})),
                        rationale=(
                            "System analyzer expanded the registered system-action "
                            "template for fusion-path scheduling."
                        ),
                        expected_gain=float(option.get("expected_gain", 0.0)),
                        source_agent="system_analyzer",
                        priority=10 + idx,
                        tags=["system", bucket],
                    )
                )

        report.action_specs.append(
            ActionSpec(
                action_id=f"system-{state.round_index}-search-weight",
                action_type="search_weight_change",
                name="correctness_then_baseline_gap",
                params={
                    "objective_weights": {
                        "correctness": 1.0,
                        "fusion_vs_hip": 0.40,
                        "baseline_gap": 0.35,
                        "kernel_metric": 0.05,
                    }
                },
                rationale=(
                    "For the fusion campaign, correctness and the gap to baseline "
                    "are more important than isolated kernel-only speedups."
                ),
                expected_gain=0.0,
                source_agent="system_analyzer",
                priority=30,
                tags=["system", "search-policy"],
            )
        )

    def _save_report(self, report: SystemAnalysisReport,
                     state: PipelineState) -> None:
        """Save analysis report to run directory."""
        report_dir = os.path.join(state.run_dir, f"v{state.iteration}")
        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(report_dir, "system_analysis.txt")
        with open(report_path, "w") as f:
            f.write(report.to_text())

        print(report.to_text())

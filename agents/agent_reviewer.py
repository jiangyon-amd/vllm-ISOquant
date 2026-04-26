"""Reviewer Agent — code review + risk analysis.

Input:  kernel source + perf data + correctness report
Output: risk list + approval/rejection decision
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from agents.state import CandidateResult, CandidateStatus, IterationResult, PipelineState


@dataclass
class ReviewResult:
    """Structured review output."""
    approved: bool = False
    risks: list[dict[str, str]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""


class ReviewerAgent:
    """Automated code reviewer for HIP kernel safety and quality."""

    # Patterns that indicate potential issues
    RISK_PATTERNS = [
        {
            "pattern": r"__syncthreads\s*\(",
            "severity": "LOW",
            "message": "Uses __syncthreads — ensure all threads reach barrier",
            "check": "sync_barrier",
        },
        {
            "pattern": r"atomicAdd|atomicCAS|atomicOr",
            "severity": "MED",
            "message": "Uses atomic operations — potential contention bottleneck",
            "check": "atomic_ops",
        },
        {
            "pattern": r"shared\s+.*\[",
            "severity": "LOW",
            "message": "Uses shared memory — check for bank conflicts",
            "check": "shared_mem",
        },
        {
            "pattern": r"#pragma\s+unroll",
            "severity": "LOW",
            "message": "Manual unroll pragma — verify loop bounds are compile-time constant",
            "check": "unroll",
        },
        {
            "pattern": r"reinterpret_cast",
            "severity": "LOW",
            "message": "reinterpret_cast — ensure alignment requirements are met",
            "check": "reinterpret",
        },
        {
            "pattern": r"__launch_bounds__\(\d+,\s*(\d+)\)",
            "severity": "INFO",
            "message": "launch_bounds specified — verify occupancy target is achievable",
            "check": "launch_bounds",
        },
        {
            "pattern": r"if\s*\(seq_len\s*<=\s*0\)",
            "severity": "LOW",
            "message": "Zero-length sequence guard — good",
            "check": "zero_guard",
        },
        {
            "pattern": r"division by zero|[^/]\s/\s0[^.]|[^/]\s/\s0$",
            "severity": "HIGH",
            "message": "Potential division by zero",
            "check": "div_zero",
        },
        {
            "pattern": r"expf?\(",
            "severity": "LOW",
            "message": "Uses exp — ensure numerical stability (subtract max first)",
            "check": "exp_stability",
        },
    ]

    # Performance anti-patterns
    PERF_PATTERNS = [
        {
            "pattern": r"for\s*\(.*;\s*.*<\s*num_kv_splits\s*;",
            "check": "split_loop",
            "message": "Loop over splits — consider if loop body has redundant loads",
        },
        {
            "pattern": r"mid_o\[.*\+\s*HEAD_DIM\]",
            "check": "lse_load",
            "message": "LSE load inside loop — consider hoisting or broadcasting",
        },
    ]

    def run(self, state: PipelineState,
            candidate: CandidateResult | None = None) -> ReviewResult:
        """Review the current kernel iteration.

        Checks:
        1. Source code safety patterns
        2. Correctness results
        3. Performance regression
        4. Numerical stability
        """
        cur = candidate or state.current
        result = self._review_subject(state, cur)
        if candidate is not None:
            candidate.status = (
                CandidateStatus.REVIEWED.value
                if result.approved else CandidateStatus.FAILED.value
            )
        return result

    def _review_subject(
        self,
        state: PipelineState,
        cur: CandidateResult | IterationResult,
    ) -> ReviewResult:
        result = ReviewResult()

        # 1. Source code analysis
        if cur.hip_source and os.path.exists(cur.hip_source):
            with open(cur.hip_source) as f:
                source = f.read()
            self._check_source(source, result)
        elif isinstance(cur, CandidateResult) and cur.action_type and cur.action_type != "kernel_transform":
            result.recommendations.append(
                "System-action candidate: source-level review is limited; rely on correctness and E2E gates."
            )
        else:
            result.risks.append({
                "severity": "MED",
                "message": "No kernel source found for review",
                "check": "missing_source",
            })

        # 2. Correctness check
        if not cur.correctness_pass:
            result.risks.append({
                "severity": "HIGH",
                "message": f"Correctness FAILED — max_abs={cur.max_abs_error:.6f}",
                "check": "correctness",
            })
            result.approved = False
            result.recommendations.append("Fix correctness before deployment")
        elif cur.max_abs_error > 0.05:
            result.risks.append({
                "severity": "MED",
                "message": f"High abs error ({cur.max_abs_error:.6f}) — borderline for bf16",
                "check": "precision",
            })

        # 3. Performance regression check
        if state.prev and state.prev.benchmark_results and cur.benchmark_results:
            regressions = []
            for key in cur.benchmark_results:
                if key in state.prev.benchmark_results:
                    old = state.prev.benchmark_results[key]
                    new = cur.benchmark_results[key]
                    if new > old * 1.05:  # 5% regression threshold
                        regressions.append(
                            f"{key}: {old:.1f}µs → {new:.1f}µs "
                            f"({(new/old - 1)*100:+.1f}%)"
                        )
            if regressions:
                result.risks.append({
                    "severity": "MED",
                    "message": f"Performance regressions: {'; '.join(regressions)}",
                    "check": "perf_regression",
                })
                result.recommendations.append(
                    "Investigate regressions — may need config-specific dispatch"
                )

        # 4. Efficiency check
        avg_eff = cur.avg_efficiency()
        if avg_eff < 0.50:
            result.risks.append({
                "severity": "MED",
                "message": f"Low avg HBM efficiency: {avg_eff*100:.1f}%",
                "check": "low_efficiency",
            })

        # Decision
        high_risks = [r for r in result.risks
                      if r["severity"] in ("HIGH", "CRITICAL")]
        result.approved = len(high_risks) == 0

        # Summary
        n_high = len(high_risks)
        n_med = len([r for r in result.risks if r["severity"] == "MED"])
        n_low = len([r for r in result.risks if r["severity"] in ("LOW", "INFO")])
        verdict = "APPROVED" if result.approved else "REJECTED"
        result.summary = (
            f"Review: {verdict}\n"
            f"Risks: {n_high} HIGH, {n_med} MED, {n_low} LOW\n"
            f"Avg efficiency: {avg_eff*100:.1f}%\n"
            f"Correctness: {'PASS' if cur.correctness_pass else 'FAIL'}\n"
        )

        # Update state
        cur.review_risks = result.risks
        cur.review_approved = result.approved

        # Save report
        report_dir = (
            os.path.join(
                state.run_dir,
                f"round_{state.round_index:02d}",
                cur.candidate_id,
            )
            if isinstance(cur, CandidateResult) and cur.candidate_id
            else os.path.join(state.run_dir, f"v{state.iteration}")
        )
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "review.txt")
        with open(report_path, "w") as f:
            f.write(result.summary + "\n")
            for r in result.risks:
                f.write(f"  [{r['severity']}] {r['message']}\n")
            if result.recommendations:
                f.write("\nRecommendations:\n")
                for rec in result.recommendations:
                    f.write(f"  • {rec}\n")

        print(f"[Reviewer] {result.summary}")
        return result

    def _check_source(self, source: str, result: ReviewResult) -> None:
        """Check source code against known patterns."""
        for pat in self.RISK_PATTERNS:
            matches = re.findall(pat["pattern"], source)
            if matches and pat["check"] != "zero_guard":  # zero_guard is positive
                result.risks.append({
                    "severity": pat["severity"],
                    "message": f"{pat['message']} ({len(matches)} occurrences)",
                    "check": pat["check"],
                })

        for pat in self.PERF_PATTERNS:
            matches = re.findall(pat["pattern"], source)
            if matches:
                result.recommendations.append(
                    f"{pat['message']} ({len(matches)} occurrences)"
                )

        # Check for online softmax (good pattern)
        has_online = "online" in source.lower() or "single-pass" in source.lower()
        if not has_online:
            # Check if there's a two-pass pattern (find max, then weighted sum)
            has_two_pass = (
                "first pass" in source.lower() or
                ("e_max" in source and source.count("for (int s") >= 2)
            )
            if has_two_pass:
                result.recommendations.append(
                    "Consider online softmax for single-pass reduction"
                )

        # Check for __shfl broadcast
        if "__shfl" not in source and "mid_o[" in source:
            result.recommendations.append(
                "Consider __shfl broadcast for shared scalar loads (LSE)"
            )

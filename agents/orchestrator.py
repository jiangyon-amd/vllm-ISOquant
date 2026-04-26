"""Round-based fan-out/fan-in orchestrator for multi-candidate optimization."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

from agents.agent_analyzer import AnalyzerAgent, BottleneckReport
from agents.agent_benchmarker import BenchmarkerAgent
from agents.agent_campaign_evaluator import CampaignEvaluatorAgent
from agents.agent_compiler import CompilerAgent
from agents.agent_correctness import CorrectnessAgent
from agents.agent_geak import GEAKOptimizerAgent
from agents.agent_profiler import ProfilerAgent
from agents.agent_reviewer import ReviewerAgent
from agents.agent_system_analyzer import SystemAnalyzerAgent
from agents.hardware_model import MI355X
from agents.state import ActionSpec, CandidateResult, CandidateStatus, Phase, PipelineState, create_run
from agents.target_registry import get_target, resolve_workload_pack


class Orchestrator:
    """Round scheduler with multi-candidate fan-out/fan-in semantics."""

    def __init__(
        self,
        state: PipelineState,
        benchmark_script: str = "",
        simulate: bool = False,
    ):
        self.state = state
        self.repo_root = os.path.dirname(os.path.dirname(__file__))
        self.benchmark_script = benchmark_script or self._default_bench_script()
        self.simulate = simulate
        self.system_analyzer = SystemAnalyzerAgent(hw=MI355X)
        self.profiler = ProfilerAgent()
        self.analyzer = AnalyzerAgent(hw=MI355X)
        self.geak = GEAKOptimizerAgent(hw=MI355X)
        self.compiler = CompilerAgent()
        self.benchmarker = BenchmarkerAgent()
        self.correctness = CorrectnessAgent()
        self.reviewer = ReviewerAgent()
        self.campaign_evaluator = CampaignEvaluatorAgent()
        self._system_report = None
        self._analysis_report = None

    def run(self) -> None:
        print("=" * 72)
        print("  Multi-Candidate Optimization Orchestrator V3")
        print(f"  Target: {self.state.target_kernel}")
        print(f"  Run: {self.state.run_id}")
        print(f"  Max rounds: {self.state.max_iterations}")
        print("=" * 72)

        try:
            while True:
                phase_before = self.state.phase
                print(
                    f"\n[{time.strftime('%H:%M:%S')}] "
                    f"phase={self.state.phase.value} "
                    f"round={self.state.round_index} "
                    f"winner_history={len(self.state.iterations)}"
                )
                started = time.time()
                self._step()
                self.state.record_phase_duration(phase_before, time.time() - started)
                self.state.save()

                if self.state.phase == Phase.FINAL_REPORT:
                    break
                if self.state.phase == Phase.FAILED:
                    raise RuntimeError("round exhausted all viable candidates")
                if self.state.hit_max_iterations() and self.state.phase not in (
                    Phase.CONVERGED,
                    Phase.FINAL_REPORT,
                ):
                    self.state.phase = Phase.CONVERGED

        except KeyboardInterrupt:
            print("\n[Orchestrator] Interrupted by user")
            self.state.save()
        except Exception as exc:
            print(f"\n[Orchestrator] Fatal error: {exc}")
            traceback.print_exc()
            self.state.phase = Phase.FAILED
            self.state.save()
        finally:
            self.campaign_evaluator.cleanup()

    def _step(self) -> None:
        handlers = {
            Phase.START: self._handle_start,
            Phase.SYSTEM_ANALYZE: self._handle_system_analyze,
            Phase.ARCH_DECIDE: self._handle_arch_decide,
            Phase.PROFILE: self._handle_profile,
            Phase.ANALYZE: self._handle_analyze,
            Phase.GEAK_OPT: self._handle_geak_opt,
            Phase.COMPILE: self._handle_compile,
            Phase.BENCHMARK: self._handle_benchmark,
            Phase.CORRECTNESS: self._handle_correctness,
            Phase.REVIEW: self._handle_review,
            Phase.DEPLOY: self._handle_deploy,
            Phase.E2E_EVALUATE: self._handle_e2e_evaluate,
            Phase.CONVERGED: self._handle_converged,
        }
        handler = handlers.get(self.state.phase)
        if handler is None:
            raise ValueError(f"no handler for phase {self.state.phase}")
        handler()

    def _handle_start(self) -> None:
        self.state.start_iteration()
        self._seed_current_round()
        self.state.advance(Phase.SYSTEM_ANALYZE)

    def _seed_current_round(self) -> None:
        target = get_target(self.state.target_kernel)
        best = self.state.best_candidate
        if best is not None:
            self.state.current.hip_source = best.primary_source_path()
            self.state.current.so_path = best.so_path
            self.state.current.kernel_version = best.kernel_version
        elif target.entry_files:
            self.state.current.hip_source = target.entry_files[0]
            self.state.current.kernel_version = os.path.basename(target.entry_files[0])
        self.state.workload_configs = resolve_workload_pack(target)
        self.state.top_k = target.default_top_k
        self.state.search_policy.top_k = target.default_top_k
        self.state.search_policy.max_candidates_per_round = target.max_candidates_per_round

    def _handle_system_analyze(self) -> None:
        self._system_report = self.system_analyzer.run(self.state)
        cur = self.state.current
        cur.system_analysis_summary = self._system_report.summary
        cur.optimization_layer = self._system_report.optimization_layer
        cur.cacheline_utilization = self._system_report.avg_cacheline_utilization
        cur.layout_type = self._system_report.layout_type
        cur.action_specs = list(self._system_report.action_specs)
        self.state.advance(Phase.ARCH_DECIDE)

    def _handle_arch_decide(self) -> None:
        executable_actions = self._apply_direct_actions(self.state.system_action_specs)
        self.state.system_action_specs = executable_actions
        self.state.advance(Phase.PROFILE)

    def _handle_profile(self) -> None:
        target = get_target(self.state.target_kernel)
        if target.target_type == "campaign":
            self.state.advance(Phase.ANALYZE)
            return
        if self.simulate:
            self.state.current.profile_results = {"simulated_profile": 1.0}
            self.state.advance(Phase.ANALYZE)
            return
        print("[Profile] baseline/winner profiling")
        profile_data = self.profiler.run(self.state, self.benchmark_script)
        if profile_data:
            self.state.current.profile_results = profile_data
        self.state.advance(Phase.ANALYZE)

    def _handle_analyze(self) -> None:
        self._analysis_report = self.analyzer.run(self.state)
        self.state.current.action_specs = list(
            self.state.system_action_specs + self._analysis_report.action_specs
        )
        self.state.analysis_action_specs = self._apply_direct_actions(
            self._analysis_report.action_specs
        )
        if self.state.is_converged():
            self.state.advance(Phase.CONVERGED)
            return
        self.state.advance(Phase.GEAK_OPT)

    def _handle_geak_opt(self) -> None:
        report = self._analysis_report or BottleneckReport(
            summary=self.state.current.bottleneck_report,
            bottleneck_type="unknown",
            efficiency_map=self.state.current.efficiency,
            worst_config="",
            worst_efficiency=0.0,
            hypotheses=self.state.current.optimization_hypotheses,
            estimated_gains=[0.0] * len(self.state.current.optimization_hypotheses),
            action_specs=[],
        )
        action_specs = self._select_action_specs()
        if not action_specs:
            self.state.advance(Phase.CONVERGED)
            return
        candidates = self.geak.generate_candidates(self.state, action_specs, report=report)
        live_candidates = [
            cand for cand in candidates
            if cand.status not in (CandidateStatus.SKIPPED.value, CandidateStatus.FAILED.value)
        ]
        if not live_candidates:
            self.state.advance(Phase.CONVERGED)
            return
        print(f"[GEAK] emitted {len(live_candidates)} candidate(s)")
        self._write_leaderboard()
        self.state.advance(Phase.COMPILE)

    def _handle_compile(self) -> None:
        pending = [
            cand for cand in self.state.candidate_pool.values()
            if cand.status in (CandidateStatus.QUEUED.value, CandidateStatus.GENERATED.value)
        ]
        if not pending:
            self.state.advance(Phase.BENCHMARK)
            return

        if self.simulate:
            for cand in pending:
                cand.status = CandidateStatus.COMPILED.value
                cand.so_path = cand.so_path or os.path.join(
                    self.state.run_dir,
                    f"round_{self.state.round_index:02d}",
                    cand.candidate_id,
                    f"{cand.candidate_id}.so",
                )
            self._write_leaderboard()
            self.state.advance(Phase.BENCHMARK)
            return

        max_workers = min(4, len(pending))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = {
            pool.submit(self.compiler.run, self.state, cand): cand for cand in pending
        }
        handled: set[object] = set()
        timed_out = False
        try:
            for future in as_completed(
                futures,
                timeout=self._compile_wait_timeout_s(len(pending)),
            ):
                handled.add(future)
                self._collect_compile_future(futures[future], future)
        except FuturesTimeoutError:
            timed_out = True
            for future, cand in futures.items():
                if future in handled:
                    continue
                if future.done():
                    self._collect_compile_future(cand, future)
                    handled.add(future)
                    continue
                cand.status = CandidateStatus.FAILED.value
                cand.failure_reason = "compile phase timed out"
        finally:
            pool.shutdown(wait=not timed_out, cancel_futures=timed_out)

        if not self._surviving_candidates():
            self.state.advance(Phase.FAILED)
            return
        self._write_leaderboard()
        self.state.advance(Phase.BENCHMARK)

    def _handle_benchmark(self) -> None:
        candidates, start_idx = self._ordered_candidates_for_phase(
            self._surviving_candidates(),
            Phase.BENCHMARK,
        )
        self.state.resume_cursor["candidate_ids"] = [c.candidate_id for c in candidates]
        for idx, cand in enumerate(candidates[start_idx:], start=start_idx):
            if cand.status in (
                CandidateStatus.BENCHMARKED.value,
                CandidateStatus.CORRECTNESS_PASSED.value,
                CandidateStatus.REVIEWED.value,
                CandidateStatus.DEPLOYED.value,
            ):
                self.state.resume_cursor["next_index"] = idx + 1
                continue
            self.state.resume_cursor["next_index"] = idx
            target = get_target(cand.resolved_target(self.state.target_kernel))
            if self.simulate:
                self._simulate_benchmark_candidate(cand, target.target_type == "campaign")
                self.state.resume_cursor["next_index"] = idx + 1
                continue
            if target.target_type == "campaign":
                quick_metrics = self.campaign_evaluator.quick_gate(self.state, cand)
                if not quick_metrics or not cand.correctness_pass:
                    cand.score = -1e9
                    self.state.resume_cursor["next_index"] = idx + 1
                    continue
                mid_metrics = self.campaign_evaluator.mid_gate(self.state, cand)
                if not mid_metrics:
                    cand.status = CandidateStatus.FAILED.value
                    cand.score = -1e9
                    self.state.resume_cursor["next_index"] = idx + 1
                    continue
            else:
                results = self.benchmarker.run(self.state, cand)
                if not results:
                    cand.status = CandidateStatus.FAILED.value
                    cand.failure_reason = "benchmark failed"
                    cand.score = -1e9
                    self.state.resume_cursor["next_index"] = idx + 1
                    continue
            cand.score = self._score_candidate(cand)
            self.state.resume_cursor["next_index"] = idx + 1

        if not self._surviving_candidates():
            self.state.advance(Phase.FAILED)
            return
        self._write_leaderboard()
        self.state.advance(Phase.CORRECTNESS)

    def _handle_correctness(self) -> None:
        candidates, start_idx = self._ordered_candidates_for_phase(
            self.state.shortlist_candidates(self.state.top_k),
            Phase.CORRECTNESS,
        )
        if not candidates:
            self.state.advance(Phase.FAILED)
            return

        self.state.resume_cursor["candidate_ids"] = [c.candidate_id for c in candidates]
        for idx, cand in enumerate(candidates[start_idx:], start=start_idx):
            if cand.status == CandidateStatus.FAILED.value:
                self.state.resume_cursor["next_index"] = idx + 1
                continue
            if cand.status in (
                CandidateStatus.CORRECTNESS_PASSED.value,
                CandidateStatus.REVIEWED.value,
                CandidateStatus.DEPLOYED.value,
            ):
                self.state.resume_cursor["next_index"] = idx + 1
                continue
            self.state.resume_cursor["next_index"] = idx
            target = get_target(cand.resolved_target(self.state.target_kernel))
            if self.simulate:
                cand.correctness_pass = True
                cand.status = CandidateStatus.CORRECTNESS_PASSED.value
            elif target.target_type == "campaign":
                metrics = self.campaign_evaluator.quality_gate(self.state, cand)
                if not metrics and cand.status != CandidateStatus.FAILED.value:
                    cand.status = CandidateStatus.FAILED.value
                    cand.failure_reason = cand.failure_reason or "quality gate returned no metrics"
                elif cand.correctness_pass:
                    cand.status = CandidateStatus.CORRECTNESS_PASSED.value
            else:
                self.correctness.run(self.state, cand)
            cand.score = self._score_candidate(cand)
            self.state.resume_cursor["next_index"] = idx + 1

        if not [c for c in candidates if c.status != CandidateStatus.FAILED.value]:
            self.state.advance(Phase.FAILED)
            return
        self._write_leaderboard()
        self.state.advance(Phase.REVIEW)

    def _handle_review(self) -> None:
        shortlisted = self.state.shortlist_candidates(self.state.top_k)
        reviewable, start_idx = self._ordered_candidates_for_phase(
            [c for c in shortlisted if c.status != CandidateStatus.FAILED.value],
            Phase.REVIEW,
        )
        if not reviewable:
            self.state.advance(Phase.FAILED)
            return

        self.state.resume_cursor["candidate_ids"] = [c.candidate_id for c in reviewable]
        for idx, cand in enumerate(reviewable[start_idx:], start=start_idx):
            if cand.status in (
                CandidateStatus.REVIEWED.value,
                CandidateStatus.DEPLOYED.value,
            ):
                self.state.resume_cursor["next_index"] = idx + 1
                continue
            self.state.resume_cursor["next_index"] = idx
            if self.simulate:
                cand.review_approved = True
                cand.review_risks = []
                cand.status = CandidateStatus.REVIEWED.value
            else:
                self.reviewer.run(self.state, cand)
            cand.score = self._score_candidate(cand)
            self.state.resume_cursor["next_index"] = idx + 1

        self._write_leaderboard()
        self.state.advance(Phase.DEPLOY)

    def _handle_deploy(self) -> None:
        candidates = self._deployment_candidates()
        if not candidates:
            notes = ["no deployment candidate survived review/correctness gates"]
            self._complete_round(None, notes, allow_empty_winner=True)
            return

        winner = self.state.active_candidate
        if winner is None or winner.candidate_id not in {cand.candidate_id for cand in candidates}:
            winner = candidates[0]
            self.state.set_active_candidate(winner.candidate_id)
        if winner.status != CandidateStatus.DEPLOYED.value:
            winner.status = CandidateStatus.DEPLOYED.value
            if not self.simulate:
                self._materialize_winner_artifact(winner)
        print(
            f"[Deploy] selected winner {winner.candidate_id} "
            f"score={winner.score:.4f} action={winner.action_type}:{winner.action_name}"
        )
        self.state.advance(Phase.E2E_EVALUATE)

    def _handle_e2e_evaluate(self) -> None:
        shortlisted, _ = self._ordered_candidates_for_phase(
            self._deployment_candidates(),
            Phase.E2E_EVALUATE,
        )
        if not shortlisted:
            notes = ["no candidate remained eligible for heavy E2E evaluation"]
            self._complete_round(None, notes, allow_empty_winner=True)
            return

        self.state.resume_cursor["candidate_ids"] = [c.candidate_id for c in shortlisted]
        chosen: CandidateResult | None = None
        for idx, cand in enumerate(shortlisted):
            if cand.status == CandidateStatus.FAILED.value:
                self.state.resume_cursor["next_index"] = idx + 1
                continue
            self.state.resume_cursor["next_index"] = idx
            target = get_target(cand.resolved_target(self.state.target_kernel))
            self.state.set_active_candidate(cand.candidate_id)
            if self.simulate:
                if target.target_type == "campaign":
                    cand.stage_scores["heavy_gate"] = max(
                        cand.stage_scores.get("quick_gate", 0.0) * 100.0,
                        1.0,
                    )
                    cand.e2e_tok_per_sec[32] = cand.stage_scores["heavy_gate"]
                cand.score = self._score_candidate(cand)
            elif target.target_type == "campaign":
                metrics = (
                    {"output_token_throughput": cand.stage_scores.get("heavy_gate", 0.0)}
                    if "heavy_gate" in cand.stage_scores and cand.status != CandidateStatus.FAILED.value
                    else self.campaign_evaluator.heavy_gate(self.state, cand)
                )
                if not metrics:
                    cand.status = CandidateStatus.FAILED.value
                    cand.failure_reason = cand.failure_reason or "heavy gate failed"
                    cand.score = -1e9
                    self.state.resume_cursor["next_index"] = idx + 1
                    continue
                cand.score = self._score_candidate(cand)
            else:
                cand.score = self._score_candidate(cand)
            if chosen is None or cand.score > chosen.score:
                chosen = cand
            self.state.resume_cursor["next_index"] = idx + 1

        notes = self._update_search_policy_after_round()
        if chosen is None:
            notes.append("all shortlisted candidates failed heavy gate; archived round without winner")
        self._complete_round(chosen, notes, allow_empty_winner=chosen is None)

    def _handle_converged(self) -> None:
        self._generate_final_report()
        self.state.advance(Phase.FINAL_REPORT)

    def _apply_direct_actions(self, action_specs: list[ActionSpec]) -> list[ActionSpec]:
        direct_actions: list[ActionSpec] = []
        for spec in action_specs:
            if spec.action_type == "search_weight_change":
                weights = spec.params.get("objective_weights", {})
                if isinstance(weights, dict):
                    self.state.search_policy.objective_weights.update(weights)
                continue
            direct_actions.append(spec)
        return direct_actions

    def _select_action_specs(self) -> list[ActionSpec]:
        ranked: list[tuple[float, ActionSpec]] = []
        seen: set[str] = set()
        for spec in self.state.system_action_specs + self.state.analysis_action_specs:
            signature = spec.signature()
            if signature in seen:
                continue
            seen.add(signature)
            family_weight = self.state.search_policy.action_type_weights.get(
                spec.family(), 1.0
            )
            score = spec.expected_gain * family_weight - spec.priority * 1e-3
            ranked.append((score, spec))
        ranked.sort(key=lambda item: item[0], reverse=True)
        limit = min(len(ranked), self.state.search_policy.max_candidates_per_round)
        if limit <= 0:
            return []
        if limit == 1 or len(ranked) == 1:
            return [ranked[0][1]]

        explore_slots = min(
            max(1, int(round(limit * self.state.search_policy.exploration_ratio))),
            max(0, limit - 1),
        )
        exploit_slots = max(1, limit - explore_slots)

        selected = ranked[:exploit_slots]
        remaining = ranked[exploit_slots:]
        remaining.sort(
            key=lambda item: (
                self.state.search_policy.history_depth(
                    item[1].action_type or item[1].family()
                ),
                -item[0],
                item[1].priority,
            )
        )
        selected.extend(remaining[:explore_slots])
        selected = selected[:limit]

        # The first campaign round must keep both the current runtime-default
        # path and any explicit PR-original anchor in the leaderboard.
        target_cfg = get_target(self.state.target_kernel)
        if target_cfg.target_type == "campaign" and self.state.round_index <= 1:
            anchor_predicates = (
                lambda spec: spec.name == "baseline_runtime_defaults",
                lambda spec: spec.name.startswith("pr_"),
            )
            for predicate in anchor_predicates:
                if any(predicate(spec) for _, spec in selected):
                    continue
                anchor = next(((score, spec) for score, spec in ranked if predicate(spec)), None)
                if anchor is None:
                    continue
                if len(selected) >= limit:
                    replace_idx = next(
                        (
                            idx
                            for idx in range(len(selected) - 1, -1, -1)
                            if not any(pred(selected[idx][1]) for pred in anchor_predicates)
                        ),
                        len(selected) - 1,
                    )
                    selected[replace_idx] = anchor
                else:
                    selected.append(anchor)

        return [spec for _, spec in selected]

    def _surviving_candidates(self) -> list[CandidateResult]:
        return [
            cand for cand in self.state.candidate_pool.values()
            if cand.status not in (
                CandidateStatus.FAILED.value,
                CandidateStatus.SKIPPED.value,
            )
        ]

    def _deployment_candidates(self) -> list[CandidateResult]:
        candidates = [
            cand for cand in self.state.candidate_pool.values()
            if cand.status != CandidateStatus.FAILED.value
            and (cand.review_approved or cand.correctness_pass)
        ]
        candidates.sort(key=lambda cand: cand.score, reverse=True)
        return candidates[: self.state.top_k]

    def _score_candidate(self, candidate: CandidateResult) -> float:
        if candidate.status == CandidateStatus.FAILED.value:
            return -1e9

        family = (
            candidate.action_spec.family()
            if candidate.action_spec is not None else "system_action"
        )
        family_weight = self.state.search_policy.action_type_weights.get(family, 1.0)
        weights = self.state.search_policy.objective_weights
        score = 0.0

        correctness = 1.0 if candidate.correctness_pass else 0.0
        score += weights.get("correctness", 1.0) * correctness

        target = get_target(candidate.resolved_target(self.state.target_kernel))
        if target.target_type == "campaign":
            refs = target.reference_metrics
            throughput = max(candidate.e2e_tok_per_sec.values()) if candidate.e2e_tok_per_sec else 0.0
            tpot = min(candidate.e2e_tpot_ms.values()) if candidate.e2e_tpot_ms else candidate.benchmark_results.get("median_tpot_ms", 0.0)
            hip_ref = refs.get("hip_tq", {})
            baseline_ref = refs.get("baseline", {})

            fusion_vs_hip_terms = []
            if throughput > 0 and hip_ref.get("output_token_throughput", 0.0) > 0:
                fusion_vs_hip_terms.append(
                    throughput / hip_ref["output_token_throughput"] - 1.0
                )
            if tpot > 0 and hip_ref.get("median_tpot_ms", 0.0) > 0:
                fusion_vs_hip_terms.append(
                    hip_ref["median_tpot_ms"] / tpot - 1.0
                )
            fusion_vs_hip = (
                sum(fusion_vs_hip_terms) / len(fusion_vs_hip_terms)
                if fusion_vs_hip_terms else 0.0
            )

            baseline_terms = []
            if throughput > 0 and baseline_ref.get("output_token_throughput", 0.0) > 0:
                baseline_terms.append(
                    1.0 - abs(throughput - baseline_ref["output_token_throughput"])
                    / baseline_ref["output_token_throughput"]
                )
            if tpot > 0 and baseline_ref.get("median_tpot_ms", 0.0) > 0:
                baseline_terms.append(
                    1.0 - abs(tpot - baseline_ref["median_tpot_ms"])
                    / baseline_ref["median_tpot_ms"]
                )
            baseline_gap = sum(baseline_terms) / len(baseline_terms) if baseline_terms else 0.0

            kernel_metric = candidate.stage_scores.get("quick_gate", 0.0)
            candidate.stage_scores["fusion_vs_hip"] = fusion_vs_hip
            candidate.stage_scores["baseline_gap"] = baseline_gap
            candidate.stage_scores["kernel_metric"] = kernel_metric
            score += weights.get("fusion_vs_hip", 0.35) * fusion_vs_hip
            score += weights.get("baseline_gap", 0.25) * baseline_gap
            score += weights.get("kernel_metric", 0.10) * kernel_metric
        else:
            prev = self.state.prev
            bench_terms = []
            if prev is not None:
                for key, value in candidate.benchmark_results.items():
                    prev_value = prev.benchmark_results.get(key, 0.0)
                    if prev_value > 0 and value > 0:
                        bench_terms.append(prev_value / value - 1.0)
            bench_gain = sum(bench_terms) / len(bench_terms) if bench_terms else 0.0
            kernel_metric = candidate.avg_efficiency()
            candidate.stage_scores["bench_gain"] = bench_gain
            candidate.stage_scores["kernel_metric"] = kernel_metric
            score += weights.get("bench_gain", 0.35) * bench_gain
            score += weights.get("kernel_metric", 0.10) * kernel_metric

        if candidate.review_approved:
            score += 0.05
        if candidate.action_type in self.state.search_policy.frozen_action_types:
            score -= 1.0
        return score * family_weight

    def _materialize_winner_artifact(self, winner: CandidateResult) -> None:
        target = get_target(winner.resolved_target(self.state.target_kernel))
        if target.target_type == "campaign":
            config_path = winner.artifacts.get("campaign_config")
            if config_path and os.path.exists(config_path):
                dst = os.path.join(self.state.run_dir, "best_candidate_config.json")
                shutil.copy2(config_path, dst)
            return

        if not winner.so_path or not os.path.exists(winner.so_path):
            return
        if not target.deployed_so:
            return
        deploy_dir = os.path.dirname(target.deployed_so)
        if not os.path.exists(deploy_dir):
            return
        try:
            shutil.copy2(winner.so_path, target.deployed_so)
        except PermissionError:
            print(f"[Deploy] permission denied for {target.deployed_so}; keeping run-local artifact")

    def _write_round_report(self, winner: CandidateResult | None) -> None:
        round_dir = os.path.join(self.state.run_dir, f"round_{self.state.round_index:02d}")
        os.makedirs(round_dir, exist_ok=True)
        ranked_candidates = sorted(
            self.state.candidate_pool.values(),
            key=lambda cand: cand.score,
            reverse=True,
        )
        payload = {
            "round_index": self.state.round_index,
            "phase": self.state.phase.value,
            "winner_id": winner.candidate_id if winner else "",
            "winner_score": winner.score if winner else None,
            "search_policy": self.state.search_policy.objective_weights,
            "candidates": self._ranked_candidate_payload(ranked_candidates),
        }
        with open(os.path.join(round_dir, "round_report.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _write_leaderboard(self) -> None:
        current = list(self.state.candidate_pool.values())
        archived = list(self.state.archived_candidates)
        merged = archived + current
        merged.sort(key=lambda cand: cand.score, reverse=True)
        payload = self._ranked_candidate_payload(merged)
        with open(os.path.join(self.state.run_dir, "leaderboard.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _ranked_candidate_payload(self, candidates: list[CandidateResult]) -> list[dict]:
        return [
            self._candidate_to_json(candidate, rank=rank)
            for rank, candidate in enumerate(candidates, start=1)
        ]

    def _candidate_to_json(self, candidate: CandidateResult, rank: int) -> dict:
        return {
            "candidate_id": candidate.candidate_id,
            "round_index": candidate.round_index,
            "parent_id": candidate.parent_id,
            "status": candidate.status,
            "score": candidate.score,
            "rank": rank,
            "action_type": candidate.action_type,
            "action_name": candidate.action_name,
            "family": candidate.action_spec.family() if candidate.action_spec else "",
            "signature": candidate.signature,
            "artifacts": candidate.artifacts,
            "benchmark_results": candidate.benchmark_results,
            "e2e_tok_per_sec": candidate.e2e_tok_per_sec,
            "e2e_tpot_ms": candidate.e2e_tpot_ms,
            "stage_scores": candidate.stage_scores,
            "correctness_pass": candidate.correctness_pass,
            "review_approved": candidate.review_approved,
            "failure_reason": candidate.failure_reason,
            "notes": candidate.notes,
        }

    def _update_search_policy_after_round(self) -> list[str]:
        notes: list[str] = []
        family_scores: dict[str, list[float]] = {"system_action": [], "kernel_transform": []}
        exact_scores: dict[str, list[float]] = {}
        self.state.search_policy.decay_action_type_weights()
        for cand in self.state.candidate_pool.values():
            if cand.status == CandidateStatus.FAILED.value:
                continue
            family = cand.action_spec.family() if cand.action_spec else "system_action"
            family_scores.setdefault(family, []).append(cand.score)
            exact_scores.setdefault(cand.action_type or "unknown", []).append(cand.score)

        for family, values in family_scores.items():
            if values:
                self.state.search_policy.action_type_history.setdefault(family, []).append(
                    sum(values) / len(values)
                )
                self.state.search_policy.action_type_history[family] = (
                    self.state.search_policy.action_type_history[family][-8:]
                )

        sys_hist = self.state.search_policy.action_type_history.get("system_action", [])
        ker_hist = self.state.search_policy.action_type_history.get("kernel_transform", [])
        if len(sys_hist) >= 2 and len(ker_hist) >= 2:
            if sys_hist[-1] > ker_hist[-1] and sys_hist[-2] > ker_hist[-2]:
                self.state.search_policy.action_type_weights["system_action"] = round(
                    self.state.search_policy.action_type_weights.get("system_action", 1.0) + 0.15,
                    2,
                )
                self.state.search_policy.action_type_weights["kernel_transform"] = round(
                    max(
                        0.25,
                        self.state.search_policy.action_type_weights.get("kernel_transform", 1.0) - 0.15,
                    ),
                    2,
                )
                notes.append("system_action outperformed kernel_transform for 2 rounds; upweighted system actions")

        for action_type, values in exact_scores.items():
            history = self.state.search_policy.action_type_history.setdefault(action_type, [])
            history.append(sum(values) / len(values))
            self.state.search_policy.action_type_history[action_type] = history[-8:]
            if (
                len(history) >= 2
                and history[-1] <= 0
                and history[-2] <= 0
                and action_type not in self.state.search_policy.frozen_action_types
            ):
                self.state.search_policy.frozen_action_types.append(action_type)
                notes.append(f"froze action type '{action_type}' after 2 no-gain rounds")
        self.state.search_policy.normalize_action_type_weights()
        return notes

    def _generate_final_report(self) -> None:
        report_path = os.path.join(self.state.run_dir, "REPORT.md")
        lines = [
            f"# Multi-Agent Optimization Report: {self.state.target_kernel}",
            "",
            f"- Run ID: `{self.state.run_id}`",
            f"- Status: `{self.state.phase.value}`",
            f"- Rounds: `{self.state.round_index}`",
            f"- Archived candidates: `{len(self.state.archived_candidates)}`",
            f"- Recorded rounds: `{len(self.state.rounds)}`",
            f"- Top-k policy: `{self.state.top_k}`",
            "",
            "## Winner History",
            "",
        ]
        for it in self.state.iterations:
            lines.extend([
                f"### Round {it.round_index}",
                f"- Winner candidate: `{it.winner_candidate_id}`",
                f"- Score: `{it.leaderboard_score:.4f}`",
                f"- Correctness: `{'PASS' if it.correctness_pass else 'FAIL'}`",
                f"- Benchmark keys: `{list(it.benchmark_results.keys())}`",
                "",
            ])

        if self.state.rounds:
            lines.extend([
                "## Round Diagnostics",
                "",
            ])
            for round_summary in self.state.rounds:
                lines.extend([
                    f"### Round {round_summary.round_index}",
                    f"- Winner candidate: `{round_summary.winner_id or 'none'}`",
                    f"- Best score: `{round_summary.best_score:.4f}`",
                    f"- Winner status: `{round_summary.winner_status or 'n/a'}`",
                    f"- Phase timings (s): `{round_summary.phase_durations}`",
                    f"- Notes: `{round_summary.notes}`",
                    "",
                ])

        lines.extend([
            "## Search Policy",
            "",
            f"- Objective weights: `{self.state.search_policy.objective_weights}`",
            f"- Action-type weights: `{self.state.search_policy.action_type_weights}`",
            f"- Frozen action types: `{self.state.search_policy.frozen_action_types}`",
            "",
            "## Outputs",
            "",
            "- `leaderboard.json`",
            "- `round_*/round_report.json`",
            "- `state.json`",
            "",
        ])
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print(f"[Report] written to {report_path}")

    def _default_bench_script(self) -> str:
        candidates = [
            os.path.join(self.repo_root, "profiling", "rocprof_tq_decode.py"),
            os.path.join(self.repo_root, "bench_tq_rocm.py"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""

    def _compile_wait_timeout_s(self, pending_count: int) -> float:
        return max(240.0, 240.0 * max(1, pending_count))

    def _collect_compile_future(self, cand: CandidateResult, future) -> None:
        try:
            so_path = future.result()
        except Exception as exc:
            cand.status = CandidateStatus.FAILED.value
            cand.failure_reason = f"compile exception: {exc}"
            return
        if so_path is None:
            cand.status = CandidateStatus.FAILED.value
            cand.failure_reason = "compile failed"

    def _ordered_candidates_for_phase(
        self,
        candidates: list[CandidateResult],
        phase: Phase,
    ) -> tuple[list[CandidateResult], int]:
        if not candidates:
            return [], 0
        cursor = self.state.resume_cursor
        if cursor.get("phase") != phase.value:
            return candidates, 0

        by_id = {cand.candidate_id: cand for cand in candidates}
        ordered: list[CandidateResult] = []
        seen: set[str] = set()
        for candidate_id in cursor.get("candidate_ids", []):
            cand = by_id.get(candidate_id)
            if cand is None:
                continue
            ordered.append(cand)
            seen.add(candidate_id)
        ordered.extend(cand for cand in candidates if cand.candidate_id not in seen)
        next_index = int(cursor.get("next_index", 0) or 0)
        return ordered, max(0, min(next_index, len(ordered)))

    def _complete_round(
        self,
        chosen: CandidateResult | None,
        notes: list[str],
        allow_empty_winner: bool = False,
    ) -> None:
        self._write_round_report(chosen)
        self._write_leaderboard()
        winner = self.state.finalize_round(
            chosen.candidate_id if chosen else "",
            notes=notes,
            allow_empty_winner=allow_empty_winner,
        )

        last_round = self.state.rounds[-1]
        if winner is not None:
            print(
                f"[Round] winner={winner.candidate_id} "
                f"score={winner.score:.4f} "
                f"correctness={'PASS' if winner.correctness_pass else 'FAIL'}"
            )
        else:
            print(
                f"[Round] winner=none "
                f"best_score={last_round.best_score:.4f} "
                f"notes={last_round.notes}"
            )

        if self.state.is_converged() or self.state.hit_max_iterations():
            self.state.phase = Phase.CONVERGED
            return
        self.state.start_iteration()
        self._seed_current_round()
        self.state.phase = Phase.SYSTEM_ANALYZE
        self.state.resume_cursor["phase"] = Phase.SYSTEM_ANALYZE.value

    def _simulate_benchmark_candidate(
        self,
        candidate: CandidateResult,
        is_campaign: bool,
    ) -> None:
        baseline = max(candidate.action_spec.expected_gain if candidate.action_spec else 0.01, 0.01)
        if is_campaign:
            candidate.correctness_pass = True
            candidate.benchmark_results = {
                "median_tpot_ms": round(10.0 / (1.0 + baseline), 4),
                "mean_ttft_ms": round(20.0 / (1.0 + baseline), 4),
                "median_itl_ms": round(5.0 / (1.0 + baseline), 4),
            }
            candidate.e2e_tok_per_sec[2] = round(100.0 * (1.0 + baseline), 4)
            candidate.e2e_tpot_ms[2] = candidate.benchmark_results["median_tpot_ms"]
        else:
            candidate.benchmark_results = {
                "simulated_gpu_time_us": round(100.0 / (1.0 + baseline), 4),
            }
        candidate.status = CandidateStatus.BENCHMARKED.value
        candidate.score = self._score_candidate(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent optimization orchestrator V3")
    parser.add_argument("--target", default="tq_decode_stage2", help="Target to optimize")
    parser.add_argument("--resume", default=None, help="Path to state.json to resume")
    parser.add_argument("--benchmark-script", default="", help="Profiler benchmark script")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum rounds")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration only")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run the full scheduler with simulated compile/benchmark/e2e results",
    )
    args = parser.parse_args()

    if args.resume:
        print(f"Resuming from {args.resume}")
        state = PipelineState.load(args.resume)
    else:
        state = create_run(target_kernel=args.target)
        state.max_iterations = args.max_iterations

    if args.dry_run:
        print(f"Target: {state.target_kernel}")
        print(f"Run dir: {state.run_dir}")
        print(f"Phase: {state.phase.value}")
        print(f"Top-k: {state.top_k}")
        print(f"Max rounds: {state.max_iterations}")
        print(f"Simulate: {args.simulate}")
        return

    Orchestrator(
        state,
        benchmark_script=args.benchmark_script,
        simulate=args.simulate,
    ).run()


if __name__ == "__main__":
    main()

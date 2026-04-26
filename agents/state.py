"""Optimization state machine + artifact tracking.

Central state shared across all optimization agents. The V3 version keeps the
old iteration summary model for backward compatibility, while adding a
candidate-pool / round-level model so the orchestrator can fan out multiple
branches, rank them, archive them, and resume mid-round.
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

from agents.hardware_model import compute_data_volume_bytes


class Phase(str, Enum):
    """Optimization pipeline phases."""

    START = "START"
    SYSTEM_ANALYZE = "SYSTEM_ANALYZE"
    ARCH_DECIDE = "ARCH_DECIDE"
    PROFILE = "PROFILE"
    ANALYZE = "ANALYZE"
    GEAK_OPT = "GEAK_OPT"
    COMPILE = "COMPILE"
    BENCHMARK = "BENCHMARK"
    CORRECTNESS = "CORRECTNESS"
    REVIEW = "REVIEW"
    DEPLOY = "DEPLOY"
    E2E_EVALUATE = "E2E_EVALUATE"
    CONVERGED = "CONVERGED"
    FINAL_REPORT = "FINAL_REPORT"
    FAILED = "FAILED"

    @staticmethod
    def transitions() -> dict["Phase", list["Phase"]]:
        P = Phase
        return {
            P.START: [P.SYSTEM_ANALYZE],
            P.SYSTEM_ANALYZE: [P.ARCH_DECIDE, P.FAILED],
            P.ARCH_DECIDE: [P.PROFILE, P.GEAK_OPT, P.FAILED],
            P.PROFILE: [P.ANALYZE, P.FAILED],
            P.ANALYZE: [P.GEAK_OPT, P.CONVERGED, P.FAILED],
            P.GEAK_OPT: [P.COMPILE, P.CONVERGED, P.FAILED],
            P.COMPILE: [P.BENCHMARK, P.GEAK_OPT, P.FAILED],
            P.BENCHMARK: [P.CORRECTNESS, P.FAILED],
            P.CORRECTNESS: [P.REVIEW, P.GEAK_OPT, P.FAILED],
            P.REVIEW: [P.DEPLOY, P.GEAK_OPT, P.FAILED],
            P.DEPLOY: [P.E2E_EVALUATE, P.SYSTEM_ANALYZE, P.CONVERGED, P.FAILED],
            P.E2E_EVALUATE: [P.SYSTEM_ANALYZE, P.CONVERGED, P.FAILED],
            P.CONVERGED: [P.FINAL_REPORT],
            P.FINAL_REPORT: [],
            P.FAILED: [],
        }


class CandidateStatus(str, Enum):
    """Lifecycle for one candidate inside a round."""

    QUEUED = "queued"
    GENERATED = "generated"
    COMPILED = "compiled"
    BENCHMARKED = "benchmarked"
    SHORTLISTED = "shortlisted"
    CORRECTNESS_PASSED = "correctness_passed"
    REVIEWED = "reviewed"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ActionSpec:
    """Structured action emitted by analyzers / system analyzers."""

    action_id: str = ""
    action_type: str = ""
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_gain: float = 0.0
    source_agent: str = ""
    priority: int = 0
    tags: list[str] = field(default_factory=list)

    def signature(self) -> str:
        payload = {
            "action_type": self.action_type,
            "name": self.name,
            "params": self.params,
            "source_agent": self.source_agent,
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def family(self) -> str:
        if self.action_type == "kernel_transform":
            return "kernel_transform"
        return "system_action"


@dataclass
class IterationResult:
    """Winner summary for one optimization round."""

    iteration: int = 0
    round_index: int = 0
    kernel_version: str = ""
    hip_source: str = ""
    so_path: str = ""

    profile_results: dict[str, float] = field(default_factory=dict)
    benchmark_results: dict[str, float] = field(default_factory=dict)
    theoretical_us: dict[str, float] = field(default_factory=dict)
    efficiency: dict[str, float] = field(default_factory=dict)

    bottleneck_report: str = ""
    optimization_hypotheses: list[str] = field(default_factory=list)
    action_specs: list[ActionSpec] = field(default_factory=list)

    vgpr_count: int = 0
    sgpr_count: int = 0

    correctness_pass: bool = False
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0

    review_risks: list[dict[str, str]] = field(default_factory=list)
    review_approved: bool = False

    system_analysis_summary: str = ""
    optimization_layer: int = 4
    cacheline_utilization: float = 0.0
    layout_type: str = ""

    e2e_tok_per_sec: dict[int, float] = field(default_factory=dict)
    e2e_tpot_ms: dict[int, float] = field(default_factory=dict)

    winner_candidate_id: str = ""
    leaderboard_score: float = 0.0

    ts_start: float = 0.0
    ts_end: float = 0.0

    def avg_efficiency(self) -> float:
        if not self.efficiency:
            return 0.0
        return sum(self.efficiency.values()) / len(self.efficiency)

    def delta_perf_vs(self, prev: "IterationResult" | None) -> float:
        """Geometric mean speedup over previous iteration.

        Assumes lower benchmark values are better; this remains true for the
        kernel targets and for the quick/mid gate TPOT-centric metrics stored by
        the fusion campaign.
        """
        if prev is None or not prev.benchmark_results:
            return float("inf")
        ratios = []
        for key, new_value in self.benchmark_results.items():
            old_value = prev.benchmark_results.get(key)
            if old_value is None or old_value <= 0 or new_value <= 0:
                continue
            ratios.append(old_value / new_value)
        if not ratios:
            return float("inf")
        from functools import reduce
        import operator

        return reduce(operator.mul, ratios, 1.0) ** (1.0 / len(ratios)) - 1.0


@dataclass
class CandidateResult(IterationResult):
    """Detailed result for one candidate branch within a round."""

    candidate_id: str = ""
    parent_id: str = ""
    generator: str = ""
    action_type: str = ""
    action_name: str = ""
    action_spec: ActionSpec | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    status: str = CandidateStatus.QUEUED.value
    score: float = float("-inf")
    stage_scores: dict[str, float] = field(default_factory=dict)
    failure_reason: str = ""
    workload_pack: str = ""
    evaluator_pack: str = ""
    profile_pack: str = ""
    target_kernel_override: str = ""
    signature: str = ""
    notes: list[str] = field(default_factory=list)

    def resolved_target(self, default_target: str) -> str:
        return self.target_kernel_override or default_target

    def primary_source_path(self) -> str:
        return (
            self.artifacts.get("hip_source")
            or self.artifacts.get("primary_source")
            or self.hip_source
        )


@dataclass
class RoundSummary:
    """Round-level metadata for resume, ranking, and reporting."""

    round_index: int = 0
    phase_started: str = ""
    phase_finished: str = ""
    candidate_ids: list[str] = field(default_factory=list)
    shortlisted_ids: list[str] = field(default_factory=list)
    winner_id: str = ""
    best_score: float = float("-inf")
    objective_weights: dict[str, float] = field(default_factory=dict)
    action_type_weights: dict[str, float] = field(default_factory=dict)
    frozen_action_types: list[str] = field(default_factory=list)
    action_type_scores: dict[str, float] = field(default_factory=dict)
    phase_durations: dict[str, float] = field(default_factory=dict)
    winner_status: str = ""
    notes: list[str] = field(default_factory=list)
    ts_start: float = 0.0
    ts_end: float = 0.0


@dataclass
class SearchPolicy:
    """Mutable search/ranking policy that evolves across rounds."""

    top_k: int = 2
    max_candidates_per_round: int = 4
    objective_weights: dict[str, float] = field(default_factory=lambda: {
        "correctness": 1.0,
        "bench_gain": 0.35,
        "fusion_vs_hip": 0.35,
        "baseline_gap": 0.25,
        "kernel_metric": 0.10,
    })
    action_type_weights: dict[str, float] = field(default_factory=lambda: {
        "kernel_transform": 1.0,
        "system_action": 1.0,
    })
    frozen_action_types: list[str] = field(default_factory=list)
    action_type_history: dict[str, list[float]] = field(default_factory=dict)
    exploration_ratio: float = 0.40

    def remember_scores(self, action_type_scores: dict[str, float]) -> None:
        for action_type, score in action_type_scores.items():
            self.action_type_history.setdefault(action_type, []).append(score)
            self.action_type_history[action_type] = self.action_type_history[action_type][-8:]

    def history_depth(self, action_type: str) -> int:
        return len(self.action_type_history.get(action_type, []))

    def decay_action_type_weights(self, decay: float = 0.20) -> None:
        for action_type, weight in list(self.action_type_weights.items()):
            decayed = 1.0 + (weight - 1.0) * (1.0 - decay)
            self.action_type_weights[action_type] = round(max(0.25, decayed), 3)

    def normalize_action_type_weights(self) -> None:
        if not self.action_type_weights:
            return
        avg = sum(self.action_type_weights.values()) / len(self.action_type_weights)
        if avg <= 0:
            return
        for action_type, weight in list(self.action_type_weights.items()):
            normalized = max(0.25, weight / avg)
            self.action_type_weights[action_type] = round(normalized, 3)


@dataclass
class PipelineState:
    """Full optimization pipeline state — serializable to JSON."""

    target_kernel: str = "tq_decode_stage2"
    run_id: str = ""
    run_dir: str = ""

    phase: Phase = Phase.START
    iteration: int = 0
    round_index: int = 0

    iterations: list[IterationResult] = field(default_factory=list)
    rounds: list[RoundSummary] = field(default_factory=list)

    convergence_window: int = 2
    convergence_threshold: float = 0.03
    efficiency_target: float = 0.80
    max_iterations: int = 20
    stale_round_limit: int = 5
    dedupe_history_window: int = 12

    current: IterationResult = field(default_factory=IterationResult)
    phase_durations: dict[str, float] = field(default_factory=dict)

    candidate_pool: dict[str, CandidateResult] = field(default_factory=dict)
    archived_candidates: list[CandidateResult] = field(default_factory=list)
    active_candidate_id: str = ""
    best_candidate_id: str = ""
    top_k: int = 2
    resume_cursor: dict[str, Any] = field(default_factory=lambda: {
        "phase": "",
        "candidate_ids": [],
        "next_index": 0,
    })
    search_policy: SearchPolicy = field(default_factory=SearchPolicy)
    system_action_specs: list[ActionSpec] = field(default_factory=list)
    analysis_action_specs: list[ActionSpec] = field(default_factory=list)

    workload_configs: list[dict[str, Any]] = field(default_factory=lambda: [
        {"B": 1, "seq": 128, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 4, "seq": 128, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 16, "seq": 128, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 64, "seq": 128, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 128, "seq": 128, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 200, "seq": 128, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 4, "seq": 2048, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 4, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
        {"B": 20, "seq": 8192, "Hq": 64, "Hk": 8, "splits": 32},
    ])

    efficiency_min_bytes: int = 50_000_000

    def advance(self, new_phase: Phase) -> None:
        valid = Phase.transitions().get(self.phase, [])
        if new_phase not in valid:
            raise ValueError(
                f"Invalid transition: {self.phase} -> {new_phase}. Valid: {valid}"
            )
        self.phase = new_phase
        self.resume_cursor["phase"] = new_phase.value

    def start_iteration(self) -> None:
        self.iteration += 1
        self.round_index += 1
        self.current = IterationResult(
            iteration=self.iteration,
            round_index=self.round_index,
            ts_start=time.time(),
        )
        self.phase_durations = {}
        self.candidate_pool = {}
        self.active_candidate_id = ""
        self.system_action_specs = []
        self.analysis_action_specs = []
        self.resume_cursor = {
            "phase": self.phase.value,
            "candidate_ids": [],
            "next_index": 0,
        }

    def finish_iteration(self) -> None:
        self.current.ts_end = time.time()
        self.iterations.append(copy.deepcopy(self.current))

    def _candidate_base(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "round_index": self.round_index,
            "system_analysis_summary": self.current.system_analysis_summary,
            "optimization_layer": self.current.optimization_layer,
            "cacheline_utilization": self.current.cacheline_utilization,
            "layout_type": self.current.layout_type,
            "bottleneck_report": self.current.bottleneck_report,
            "optimization_hypotheses": copy.deepcopy(
                self.current.optimization_hypotheses
            ),
            "action_specs": copy.deepcopy(self.current.action_specs),
            "ts_start": time.time(),
        }

    def _next_candidate_id(self) -> str:
        return f"r{self.round_index:02d}c{len(self.candidate_pool) + 1:02d}"

    def seen_candidate_signatures(self) -> set[str]:
        archived_window = (
            self.archived_candidates[-self.dedupe_history_window:]
            if self.dedupe_history_window > 0 else self.archived_candidates
        )
        signatures = {cand.signature for cand in archived_window if cand.signature}
        signatures.update(
            cand.signature for cand in self.candidate_pool.values() if cand.signature
        )
        return signatures

    def _candidate_lineage(self, parent_id: str = "") -> str:
        if parent_id:
            return parent_id
        if self.best_candidate_id:
            return self.best_candidate_id
        if self.current.kernel_version:
            return self.current.kernel_version
        if self.current.hip_source:
            return os.path.basename(self.current.hip_source)
        return "root"

    def _candidate_signature(self, action_spec: ActionSpec, parent_id: str = "") -> str:
        payload = {
            "action": action_spec.signature(),
            "lineage": self._candidate_lineage(parent_id),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def add_candidate(self, candidate: CandidateResult) -> CandidateResult:
        if not candidate.candidate_id:
            candidate.candidate_id = self._next_candidate_id()
        if candidate.action_spec and not candidate.signature:
            candidate.signature = self._candidate_signature(
                candidate.action_spec,
                candidate.parent_id,
            )
        if not candidate.action_type and candidate.action_spec is not None:
            candidate.action_type = candidate.action_spec.action_type
        if not candidate.action_name and candidate.action_spec is not None:
            candidate.action_name = candidate.action_spec.name
        if not candidate.workload_pack and candidate.action_spec is not None:
            candidate.workload_pack = candidate.action_spec.params.get(
                "workload_pack", ""
            )
        if not candidate.evaluator_pack and candidate.action_spec is not None:
            candidate.evaluator_pack = candidate.action_spec.params.get(
                "evaluator_pack", ""
            )
        if not candidate.profile_pack and candidate.action_spec is not None:
            candidate.profile_pack = candidate.action_spec.params.get(
                "profile_pack", ""
            )
        if not candidate.target_kernel_override and candidate.action_spec is not None:
            candidate.target_kernel_override = candidate.action_spec.params.get(
                "target_kernel", ""
            )
        candidate.iteration = self.iteration
        candidate.round_index = self.round_index
        self.candidate_pool[candidate.candidate_id] = candidate
        self.resume_cursor["candidate_ids"] = list(self.candidate_pool.keys())
        return candidate

    def new_candidate(
        self,
        action_spec: ActionSpec,
        generator: str,
        parent_id: str = "",
    ) -> CandidateResult:
        base = self._candidate_base()
        base["action_specs"] = [copy.deepcopy(action_spec)]
        resolved_parent_id = parent_id or self.best_candidate_id
        candidate = CandidateResult(
            **base,
            candidate_id=self._next_candidate_id(),
            parent_id=resolved_parent_id,
            generator=generator,
            action_type=action_spec.action_type,
            action_name=action_spec.name,
            action_spec=copy.deepcopy(action_spec),
            workload_pack=action_spec.params.get("workload_pack", ""),
            evaluator_pack=action_spec.params.get("evaluator_pack", ""),
            profile_pack=action_spec.params.get("profile_pack", ""),
            target_kernel_override=action_spec.params.get("target_kernel", ""),
            signature=self._candidate_signature(action_spec, resolved_parent_id),
            status=CandidateStatus.QUEUED.value,
        )
        return self.add_candidate(candidate)

    def get_candidate(self, candidate_id: str) -> CandidateResult | None:
        return self.candidate_pool.get(candidate_id)

    @property
    def active_candidate(self) -> CandidateResult | None:
        if not self.active_candidate_id:
            return None
        return self.candidate_pool.get(self.active_candidate_id)

    @property
    def best_candidate(self) -> CandidateResult | None:
        if self.best_candidate_id:
            for cand in self.archived_candidates:
                if cand.candidate_id == self.best_candidate_id:
                    return cand
            if self.best_candidate_id in self.candidate_pool:
                return self.candidate_pool[self.best_candidate_id]
        if self.archived_candidates:
            return max(self.archived_candidates, key=lambda c: c.score)
        if self.candidate_pool:
            return max(self.candidate_pool.values(), key=lambda c: c.score)
        return None

    def set_active_candidate(self, candidate_id: str) -> None:
        if candidate_id not in self.candidate_pool:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        self.active_candidate_id = candidate_id

    def shortlist_candidates(self, k: int | None = None) -> list[CandidateResult]:
        k = k or self.top_k or self.search_policy.top_k
        eligible = [
            cand for cand in self.candidate_pool.values()
            if cand.status not in (CandidateStatus.FAILED.value, CandidateStatus.SKIPPED.value)
        ]
        eligible.sort(
            key=lambda cand: (
                cand.score,
                1 if cand.correctness_pass else 0,
                1 if cand.review_approved else 0,
                cand.candidate_id,
            ),
            reverse=True,
        )
        return eligible[:k]

    def archive_candidate(self, candidate: CandidateResult) -> None:
        archived = copy.deepcopy(candidate)
        archived.status = CandidateStatus.ARCHIVED.value
        if any(c.candidate_id == archived.candidate_id for c in self.archived_candidates):
            return
        self.archived_candidates.append(archived)

    def record_phase_duration(self, phase: Phase, elapsed_s: float) -> None:
        self.phase_durations[phase.value] = (
            self.phase_durations.get(phase.value, 0.0) + max(0.0, elapsed_s)
        )

    def finalize_round(
        self,
        winner_id: str = "",
        notes: list[str] | None = None,
        allow_empty_winner: bool = False,
    ) -> CandidateResult | None:
        best_before_round = self.best_candidate
        winner = None
        if winner_id:
            winner = self.candidate_pool.get(winner_id)
        if winner is None and self.candidate_pool and not allow_empty_winner:
            ranked = self.shortlist_candidates(k=1)
            winner = ranked[0] if ranked else None

        summary = copy.deepcopy(self.current)
        if winner is not None:
            for field_info in fields(IterationResult):
                setattr(summary, field_info.name, copy.deepcopy(getattr(winner, field_info.name)))
            summary.iteration = self.iteration
            summary.round_index = self.round_index
            summary.winner_candidate_id = winner.candidate_id
            summary.leaderboard_score = winner.score
            summary.ts_start = self.current.ts_start or winner.ts_start
            summary.ts_end = time.time()
            if best_before_round is None or winner.score >= best_before_round.score:
                self.best_candidate_id = winner.candidate_id
        else:
            summary.iteration = self.iteration
            summary.round_index = self.round_index
            summary.winner_candidate_id = ""
            summary.leaderboard_score = float("-inf")
            summary.ts_start = self.current.ts_start
            summary.ts_end = time.time()

        action_type_scores: dict[str, list[float]] = {}
        for cand in self.candidate_pool.values():
            action_type_scores.setdefault(cand.action_type or "unknown", []).append(
                cand.score if cand.score != float("-inf") else -1e9
            )

        round_summary = RoundSummary(
            round_index=self.round_index,
            phase_started=Phase.SYSTEM_ANALYZE.value,
            phase_finished=self.phase.value,
            candidate_ids=list(self.candidate_pool.keys()),
            shortlisted_ids=[c.candidate_id for c in self.shortlist_candidates()],
            winner_id=winner.candidate_id if winner is not None else "",
            best_score=winner.score if winner is not None else float("-inf"),
            objective_weights=copy.deepcopy(self.search_policy.objective_weights),
            action_type_weights=copy.deepcopy(self.search_policy.action_type_weights),
            frozen_action_types=copy.deepcopy(self.search_policy.frozen_action_types),
            action_type_scores={
                key: sum(values) / len(values) for key, values in action_type_scores.items()
            },
            phase_durations=copy.deepcopy(self.phase_durations),
            winner_status=winner.status if winner is not None else "",
            notes=notes or [],
            ts_start=self.current.ts_start,
            ts_end=time.time(),
        )
        self.rounds.append(round_summary)
        self.search_policy.remember_scores(round_summary.action_type_scores)

        self.current = summary
        if winner is not None:
            self.finish_iteration()

        for cand in self.candidate_pool.values():
            self.archive_candidate(cand)
        self.candidate_pool = {}
        self.active_candidate_id = ""
        self.phase_durations = {}
        self.resume_cursor = {"phase": self.phase.value, "candidate_ids": [], "next_index": 0}
        return winner

    def is_converged(self) -> bool:
        if self.is_stale():
            return True
        if len(self.iterations) < self.convergence_window:
            return False

        recent = self.iterations[-self.convergence_window:]
        if not self._is_flat_window(recent):
            return False

        latest = self.iterations[-1]
        eligible_eff = self._eligible_efficiency(latest)
        efficiency_met = eligible_eff is not None and eligible_eff >= self.efficiency_target
        if not efficiency_met:
            return False
        if not latest.correctness_pass:
            return False
        for risk in latest.review_risks:
            if risk.get("severity", "").upper() in ("HIGH", "CRITICAL"):
                return False
        return True

    def is_stale(self) -> bool:
        if len(self.rounds) < self.stale_round_limit:
            return False

        recent_rounds = self.rounds[-self.stale_round_limit:]
        if all(not round_summary.winner_id for round_summary in recent_rounds):
            return True

        recent_winners = [
            it for it in self.iterations
            if it.round_index in {round_summary.round_index for round_summary in recent_rounds}
        ]
        if not recent_winners:
            return False

        has_clean_winner = any(
            winner.correctness_pass and not any(
                risk.get("severity", "").upper() in ("HIGH", "CRITICAL")
                for risk in winner.review_risks
            )
            for winner in recent_winners
        )
        if has_clean_winner:
            return False

        if len(recent_winners) < 2:
            return False
        return self._is_flat_window(recent_winners)

    def _eligible_efficiency(self, it: IterationResult) -> float | None:
        eligible: dict[str, float] = {}
        for cfg in self.workload_configs:
            key = self.config_key(cfg)
            total_bytes = compute_data_volume_bytes(self.target_kernel, cfg)
            if total_bytes >= self.efficiency_min_bytes and key in it.efficiency:
                eligible[key] = it.efficiency[key]
        if not eligible:
            return None
        return sum(eligible.values()) / len(eligible)

    def _is_flat_window(self, window: list[IterationResult]) -> bool:
        if len(window) < 2:
            return False
        for i in range(1, len(window)):
            delta = window[i].delta_perf_vs(window[i - 1])
            if abs(delta) >= self.convergence_threshold:
                return False
        return True

    def hit_max_iterations(self) -> bool:
        return self.iteration >= self.max_iterations

    @property
    def prev(self) -> IterationResult | None:
        if not self.iterations:
            return None
        return self.iterations[-1]

    def save(self, path: str | None = None) -> str:
        if path is None:
            path = os.path.join(self.run_dir, "state.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "PipelineState":
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    def _to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["phase"] = self.phase.value
        return data

    @staticmethod
    def _filter_dataclass_payload(
        payload: dict[str, Any],
        dataclass_type: type,
    ) -> dict[str, Any]:
        allowed = dataclass_type.__dataclass_fields__
        return {key: value for key, value in payload.items() if key in allowed}

    @classmethod
    def _restore_action_spec(cls, payload: dict[str, Any] | None) -> ActionSpec | None:
        if not payload:
            return None
        return ActionSpec(**cls._filter_dataclass_payload(dict(payload), ActionSpec))

    @classmethod
    def _restore_iteration(cls, payload: dict[str, Any]) -> IterationResult:
        payload = dict(payload)
        payload["action_specs"] = [
            cls._restore_action_spec(spec) for spec in payload.get("action_specs", [])
        ]
        return IterationResult(**cls._filter_dataclass_payload(payload, IterationResult))

    @classmethod
    def _restore_candidate(cls, payload: dict[str, Any]) -> CandidateResult:
        payload = dict(payload)
        payload["action_specs"] = [
            cls._restore_action_spec(spec) for spec in payload.get("action_specs", [])
        ]
        payload["action_spec"] = cls._restore_action_spec(payload.get("action_spec"))
        return CandidateResult(**cls._filter_dataclass_payload(payload, CandidateResult))

    @classmethod
    def _restore_round(cls, payload: dict[str, Any]) -> RoundSummary:
        return RoundSummary(**cls._filter_dataclass_payload(dict(payload), RoundSummary))

    @classmethod
    def _from_dict(cls, payload: dict[str, Any]) -> "PipelineState":
        payload = dict(payload)
        payload["phase"] = Phase(payload.get("phase", Phase.START.value))
        iterations = [
            cls._restore_iteration(item) for item in payload.pop("iterations", [])
        ]
        rounds = [cls._restore_round(item) for item in payload.pop("rounds", [])]
        current_payload = payload.pop("current", {})
        current = cls._restore_iteration(current_payload) if current_payload else IterationResult()
        candidate_pool_payload = payload.pop("candidate_pool", {})
        archived_payload = payload.pop("archived_candidates", [])
        search_policy_payload = payload.pop("search_policy", {})
        system_specs_payload = payload.pop("system_action_specs", [])
        analysis_specs_payload = payload.pop("analysis_action_specs", [])

        state = cls(**{
            key: value
            for key, value in payload.items()
            if key in cls.__dataclass_fields__
        })
        state.iterations = iterations
        state.rounds = rounds
        state.current = current
        state.candidate_pool = {
            candidate_id: cls._restore_candidate(candidate_payload)
            for candidate_id, candidate_payload in candidate_pool_payload.items()
        }
        state.archived_candidates = [
            cls._restore_candidate(item) for item in archived_payload
        ]
        state.search_policy = SearchPolicy(**search_policy_payload) if search_policy_payload else SearchPolicy()
        state.system_action_specs = [
            cls._restore_action_spec(spec) for spec in system_specs_payload
            if spec is not None
        ]
        state.analysis_action_specs = [
            cls._restore_action_spec(spec) for spec in analysis_specs_payload
            if spec is not None
        ]
        return state

    @staticmethod
    def config_key(cfg: dict[str, Any]) -> str:
        if "M" in cfg and "B" not in cfg:
            return f"M{cfg['M']}"
        return (
            f"B{cfg['B']}_seq{cfg['seq']}_Hq{cfg['Hq']}"
            f"_s{cfg.get('splits', 0)}"
        )


def create_run(target_kernel: str = "tq_decode_stage2", base_dir: str = "") -> PipelineState:
    """Create a new optimization run with a timestamped directory."""

    if not base_dir:
        base_dir = os.path.join(os.path.dirname(__file__), "runs")

    run_id = f"{target_kernel}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    workload_configs = None
    top_k = None
    max_candidates = None
    try:
        from agents.target_registry import get_target

        target_cfg = get_target(target_kernel)
        workload_configs = target_cfg.workload_configs
        top_k = target_cfg.default_top_k
        max_candidates = target_cfg.max_candidates_per_round
    except (ValueError, ImportError, AttributeError):
        pass

    state = PipelineState(
        target_kernel=target_kernel,
        run_id=run_id,
        run_dir=run_dir,
    )
    if workload_configs:
        state.workload_configs = workload_configs
    if top_k:
        state.top_k = top_k
        state.search_policy.top_k = top_k
    if max_candidates:
        state.search_policy.max_candidates_per_round = max_candidates
    return state

from __future__ import annotations

import json
from concurrent.futures import TimeoutError as FuturesTimeoutError

import agents.orchestrator as orchestrator_module
from agents.agent_campaign_evaluator import CampaignEvaluatorAgent
from agents.agent_reviewer import ReviewerAgent
from agents.orchestrator import Orchestrator
from agents.state import ActionSpec, CandidateStatus, Phase, PipelineState, create_run
from agents.target_registry import get_target


def _make_candidate(
    state,
    *,
    action_type: str,
    name: str,
    generator: str,
):
    return state.new_candidate(
        ActionSpec(
            action_id=name,
            action_type=action_type,
            name=name,
        ),
        generator=generator,
    )


def test_e2e_all_heavy_fail_does_not_promote_failed_candidate(tmp_path, monkeypatch):
    state = create_run("turboquant_soa_fusion", base_dir=str(tmp_path))
    state.start_iteration()
    state.phase = Phase.E2E_EVALUATE

    winner = _make_candidate(
        state,
        action_type="fusion_config_change",
        name="winner_cfg",
        generator="system",
    )
    fallback = _make_candidate(
        state,
        action_type="fusion_config_change",
        name="fallback_cfg",
        generator="system",
    )
    for cand, score in ((winner, 1.0), (fallback, 0.8)):
        cand.review_approved = True
        cand.correctness_pass = True
        cand.status = CandidateStatus.REVIEWED.value
        cand.score = score

    state.set_active_candidate(winner.candidate_id)
    orch = Orchestrator(state)

    def always_fail_heavy(_state, cand):
        cand.status = CandidateStatus.FAILED.value
        cand.failure_reason = "heavy gate failed"
        return {}

    monkeypatch.setattr(orch.campaign_evaluator, "heavy_gate", always_fail_heavy)

    orch._handle_e2e_evaluate()

    assert state.rounds[-1].winner_id == ""
    assert len(state.iterations) == 0
    assert state.phase == Phase.SYSTEM_ANALYZE
    assert state.candidate_pool == {}


def test_benchmark_resume_skips_already_processed_candidates(tmp_path, monkeypatch):
    state = create_run("tq_decode_stage2", base_dir=str(tmp_path))
    state.start_iteration()
    state.phase = Phase.BENCHMARK

    candidates = [
        _make_candidate(
            state,
            action_type="kernel_transform",
            name=f"cand_{idx}",
            generator="geak",
        )
        for idx in range(3)
    ]
    for cand in candidates:
        cand.status = CandidateStatus.COMPILED.value
        cand.so_path = __file__

    candidates[0].status = CandidateStatus.BENCHMARKED.value
    candidates[0].benchmark_results = {"simulated_gpu_time_us": 123.0}
    candidates[0].score = 0.1

    state.resume_cursor = {
        "phase": Phase.BENCHMARK.value,
        "candidate_ids": [cand.candidate_id for cand in candidates],
        "next_index": 1,
    }

    called: list[str] = []
    orch = Orchestrator(state)

    def fake_bench(_state, cand):
        called.append(cand.candidate_id)
        cand.status = CandidateStatus.BENCHMARKED.value
        cand.benchmark_results = {"simulated_gpu_time_us": float(len(called))}
        return cand.benchmark_results

    monkeypatch.setattr(orch.benchmarker, "run", fake_bench)

    orch._handle_benchmark()

    assert called == [candidates[1].candidate_id, candidates[2].candidate_id]
    assert candidates[0].benchmark_results == {"simulated_gpu_time_us": 123.0}
    assert state.resume_cursor["next_index"] == len(candidates)
    assert state.phase == Phase.CORRECTNESS


def test_quick_gate_executes_decode_smoke_when_configured(tmp_path, monkeypatch):
    state = create_run("turboquant_soa_fusion", base_dir=str(tmp_path))
    state.start_iteration()
    candidate = _make_candidate(
        state,
        action_type="fusion_config_change",
        name="decode_smoke_cfg",
        generator="system",
    )

    evaluator = CampaignEvaluatorAgent(python_exe="python")
    smoke_called: list[bool] = []

    monkeypatch.setattr(
        evaluator,
        "_run_fusion_synthetic_roundtrip",
        lambda *_args, **_kwargs: {
            "passed": 1.0,
            "key_cosine": 0.99,
            "value_cosine": 0.99,
            "min_cosine": 0.99,
        },
    )

    def fake_decode_smoke(*_args, **_kwargs):
        smoke_called.append(True)
        return {"passed": 1.0, "response_chars": 16.0}

    monkeypatch.setattr(evaluator, "_run_decode_smoke", fake_decode_smoke)

    metrics = evaluator.quick_gate(state, candidate)

    assert metrics["passed"] == 1.0
    assert smoke_called == [True]
    assert candidate.correctness_pass is True
    assert candidate.stage_scores["quick_decode_smoke"] == 1.0


def test_campaign_correctness_sets_status_when_quality_gate_only_returns_metrics(
    tmp_path,
    monkeypatch,
):
    state = create_run("turboquant_soa_fusion", base_dir=str(tmp_path))
    state.start_iteration()
    state.phase = Phase.CORRECTNESS
    candidate = _make_candidate(
        state,
        action_type="fusion_config_change",
        name="quality_only",
        generator="system",
    )
    candidate.score = 0.2
    candidate.status = CandidateStatus.BENCHMARKED.value
    candidate.correctness_pass = True

    orch = Orchestrator(state)

    def quality_metrics_only(_state, cand):
        cand.correctness_pass = True
        return {"passed": 1.0, "fact_hits": 4.0}

    monkeypatch.setattr(orch.campaign_evaluator, "quality_gate", quality_metrics_only)

    orch._handle_correctness()

    assert candidate.status == CandidateStatus.CORRECTNESS_PASSED.value
    assert state.phase == Phase.REVIEW


def test_compile_timeout_marks_pending_candidates_failed(tmp_path, monkeypatch):
    state = create_run("tq_decode_stage2", base_dir=str(tmp_path))
    state.start_iteration()
    state.phase = Phase.COMPILE
    candidate = _make_candidate(
        state,
        action_type="kernel_transform",
        name="compile_timeout",
        generator="geak",
    )
    candidate.status = CandidateStatus.GENERATED.value

    created_executors = []

    class FakeFuture:
        def done(self):
            return False

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.shutdown_args = None
            created_executors.append(self)

        def submit(self, *_args, **_kwargs):
            return FakeFuture()

        def shutdown(self, wait, cancel_futures):
            self.shutdown_args = (wait, cancel_futures)

    def raise_timeout(_futures, timeout=None):
        raise FuturesTimeoutError()

    monkeypatch.setattr(orchestrator_module, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(orchestrator_module, "as_completed", raise_timeout)

    orch = Orchestrator(state)
    orch._handle_compile()

    assert candidate.status == CandidateStatus.FAILED.value
    assert "timed out" in candidate.failure_reason
    assert state.phase == Phase.FAILED
    assert created_executors[0].shutdown_args == (False, True)


def test_state_load_ignores_removed_legacy_fields(tmp_path):
    action_payload = {
        "action_id": "legacy",
        "action_type": "kernel_transform",
        "name": "legacy_transform",
        "executable": True,
        "requires_approval": False,
    }
    state_payload = {
        "target_kernel": "tq_decode_stage2",
        "run_id": "legacy_run",
        "run_dir": str(tmp_path),
        "phase": Phase.START.value,
        "iteration": 1,
        "round_index": 1,
        "fine_tune_index": 3,
        "iterations": [
            {
                "iteration": 1,
                "round_index": 1,
                "action_specs": [action_payload],
                "e2e_vs_baseline": 0.42,
            }
        ],
        "rounds": [],
        "current": {
            "iteration": 1,
            "round_index": 1,
            "action_specs": [action_payload],
            "e2e_vs_baseline": 0.11,
        },
        "candidate_pool": {
            "r01c01": {
                "iteration": 1,
                "round_index": 1,
                "candidate_id": "r01c01",
                "status": CandidateStatus.QUEUED.value,
                "rank": 7,
                "action_specs": [action_payload],
                "action_spec": action_payload,
            }
        },
        "archived_candidates": [],
        "search_policy": {},
    }
    state_path = tmp_path / "legacy_state.json"
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    loaded = PipelineState.load(str(state_path))

    assert loaded.current.action_specs[0].name == "legacy_transform"
    candidate = loaded.candidate_pool["r01c01"]
    assert candidate.action_spec is not None
    assert candidate.action_spec.name == "legacy_transform"
    assert not hasattr(candidate, "rank")
    assert not hasattr(loaded, "fine_tune_index")


def test_reviewer_run_keeps_state_current_object(tmp_path):
    state = create_run("turboquant_soa_fusion", base_dir=str(tmp_path))
    state.start_iteration()
    candidate = _make_candidate(
        state,
        action_type="fusion_config_change",
        name="system_review",
        generator="system",
    )
    candidate.correctness_pass = True

    original_current = state.current
    reviewer = ReviewerAgent()
    result = reviewer.run(state, candidate)

    assert state.current is original_current
    assert result.approved is True
    assert candidate.review_approved is True


def test_shortlist_candidates_does_not_overwrite_processed_status(tmp_path):
    state = create_run("tq_decode_stage2", base_dir=str(tmp_path))
    state.start_iteration()
    reviewed = _make_candidate(
        state,
        action_type="kernel_transform",
        name="reviewed",
        generator="geak",
    )
    reviewed.status = CandidateStatus.REVIEWED.value
    reviewed.correctness_pass = True
    reviewed.review_approved = True
    reviewed.score = 1.0

    benchmarked = _make_candidate(
        state,
        action_type="kernel_transform",
        name="benchmarked",
        generator="geak",
    )
    benchmarked.status = CandidateStatus.BENCHMARKED.value
    benchmarked.correctness_pass = True
    benchmarked.score = 0.5

    shortlisted = state.shortlist_candidates(k=2)

    assert [cand.candidate_id for cand in shortlisted] == [
        reviewed.candidate_id,
        benchmarked.candidate_id,
    ]
    assert reviewed.status == CandidateStatus.REVIEWED.value
    assert benchmarked.status == CandidateStatus.BENCHMARKED.value


def test_soa_bf16q_pv_mfma_candidate_is_registered():
    target = get_target("turboquant_soa_fusion")
    candidates = target.search_space["fusion_config_change"]

    assert any(
        candidate["name"] == "soa_bf16q_pv_mfma"
        and candidate["params"]["decode_impl"] == "soa_bf16q_pv_mfma"
        for candidate in candidates
    )


def test_fusion_env_enables_soa_bf16q_pv_mfma_decode_impl(tmp_path, monkeypatch):
    config_path = tmp_path / "campaign_config.json"
    config_path.write_text(
        json.dumps({"decode_impl": "soa_bf16q_pv_mfma"}),
        encoding="utf-8",
    )
    state = create_run("turboquant_soa_fusion", base_dir=str(tmp_path))
    state.start_iteration()
    candidate = _make_candidate(
        state,
        action_type="fusion_config_change",
        name="soa_bf16q_pv_mfma",
        generator="system",
    )
    candidate.artifacts["campaign_config"] = str(config_path)

    monkeypatch.setenv("VLLM_TQ_SOA_FUSION_DECODE_FLASH_TQ", "1")
    evaluator = CampaignEvaluatorAgent(python_exe="python")
    env = evaluator._fusion_env(candidate, {"gpu": "0"})

    assert env["VLLM_TQ_SOA_FUSION"] == "1"
    assert env["VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA"] == "1"
    assert "VLLM_TQ_SOA_FUSION_DECODE_FLASH_TQ" not in env

"""Campaign evaluator for fusion/system-level optimization targets.

The existing benchmarker/correctness agents are kernel-centric: they expect a
compiled `.so` and use synthetic microbench scripts. The fusion campaign needs
multi-stage gates that can evaluate config branches directly against the live
Python/Triton path:

1. quick gate: synthetic store/dequant roundtrip
2. mid gate: service warmup benchmark
3. heavy gate: 72B serving benchmark
4. quality gate: small prompt spot-check
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from typing import Any

from agents.state import CandidateResult, CandidateStatus, PipelineState
from agents.target_registry import (
    get_target,
    resolve_evaluator_pack,
)


class CampaignEvaluatorAgent:
    """Evaluate fusion-path candidates with quick/mid/heavy gates."""

    QUALITY_PROMPTS = [
        ("The capital of France is", "Paris"),
        ("Albert Einstein was born in the year", "1879"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("The chemical formula for water is", "H2O"),
        ("1+1=2, 2+2=4, 3+3=", "6"),
    ]

    def __init__(self, python_exe: str = ""):
        self.python = python_exe or sys.executable or "python3"
        self.repo_root = os.path.dirname(os.path.dirname(__file__))
        self._active_servers: dict[int, subprocess.Popen[Any]] = {}

    def quick_gate(
        self,
        state: PipelineState,
        candidate: CandidateResult,
    ) -> dict[str, float]:
        target = get_target(candidate.resolved_target(state.target_kernel))
        pack = resolve_evaluator_pack(target, candidate.evaluator_pack)
        cfg = pack.get("quick_gate", {})
        if cfg.get("mode") != "fusion_synthetic_roundtrip":
            return {}
        metrics = self._run_fusion_synthetic_roundtrip(state, candidate, cfg)
        decode_metrics = {}
        decode_cfg = cfg.get("decode_smoke", {})
        if decode_cfg.get("mode") == "service_decode_smoke":
            decode_metrics = self._run_decode_smoke(state, candidate, decode_cfg)
        passed = bool(metrics.get("passed", 0.0) >= 1.0) and bool(
            decode_metrics.get("passed", 1.0) >= 1.0
        )
        candidate.correctness_pass = passed
        candidate.stage_scores["quick_gate"] = metrics.get("min_cosine", 0.0)
        if decode_metrics:
            candidate.stage_scores["quick_decode_smoke"] = decode_metrics.get("passed", 0.0)
        candidate.max_abs_error = max(0.0, 1.0 - metrics.get("min_cosine", 0.0))
        if not passed:
            candidate.status = CandidateStatus.FAILED.value
            candidate.failure_reason = "fusion synthetic roundtrip failed"
        return metrics

    def mid_gate(
        self,
        state: PipelineState,
        candidate: CandidateResult,
    ) -> dict[str, float]:
        target = get_target(candidate.resolved_target(state.target_kernel))
        pack = resolve_evaluator_pack(target, candidate.evaluator_pack)
        cfg = pack.get("mid_gate", {})
        if cfg.get("mode") != "service_warmup":
            return {}
        metrics = self._run_service_benchmark(state, candidate, cfg, stage_name="mid")
        if metrics:
            candidate.benchmark_results = {
                "median_tpot_ms": metrics.get("median_tpot_ms", 0.0),
                "mean_ttft_ms": metrics.get("mean_ttft_ms", 0.0),
                "median_itl_ms": metrics.get("median_itl_ms", 0.0),
            }
            concurrency = int(cfg.get("concurrency", 1))
            if metrics.get("output_token_throughput", 0.0) > 0:
                candidate.e2e_tok_per_sec[concurrency] = metrics["output_token_throughput"]
            if metrics.get("median_tpot_ms", 0.0) > 0:
                candidate.e2e_tpot_ms[concurrency] = metrics["median_tpot_ms"]
            candidate.status = CandidateStatus.BENCHMARKED.value
        else:
            candidate.status = CandidateStatus.FAILED.value
            candidate.failure_reason = "service warmup benchmark failed"
        return metrics

    def quality_gate(
        self,
        state: PipelineState,
        candidate: CandidateResult,
    ) -> dict[str, float]:
        target = get_target(candidate.resolved_target(state.target_kernel))
        pack = resolve_evaluator_pack(target, candidate.evaluator_pack)
        cfg = pack.get("quality_gate", {})
        if cfg.get("mode") != "quality_spotcheck_72b":
            return {}
        metrics = self._run_quality_spotcheck(state, candidate, cfg)
        passed = bool(metrics.get("passed", 0.0) >= 1.0)
        candidate.correctness_pass = passed
        candidate.stage_scores["quality_gate"] = metrics.get("fact_hits", 0.0)
        if not passed:
            candidate.status = CandidateStatus.FAILED.value
            candidate.failure_reason = "quality spot-check failed"
        else:
            candidate.status = CandidateStatus.CORRECTNESS_PASSED.value
        return metrics

    def heavy_gate(
        self,
        state: PipelineState,
        candidate: CandidateResult,
    ) -> dict[str, float]:
        target = get_target(candidate.resolved_target(state.target_kernel))
        pack = resolve_evaluator_pack(target, candidate.evaluator_pack)
        cfg = pack.get("heavy_gate", {})
        if cfg.get("mode") != "benchmark_72b_full":
            return {}
        metrics = self._run_service_benchmark(state, candidate, cfg, stage_name="heavy")
        concurrency = int(cfg.get("concurrency", 1))
        if metrics.get("output_token_throughput", 0.0) > 0:
            candidate.e2e_tok_per_sec[concurrency] = metrics["output_token_throughput"]
        if metrics.get("median_tpot_ms", 0.0) > 0:
            candidate.e2e_tpot_ms[concurrency] = metrics["median_tpot_ms"]
        candidate.stage_scores["heavy_gate"] = metrics.get("output_token_throughput", 0.0)
        return metrics

    def _candidate_dir(self, state: PipelineState, candidate: CandidateResult) -> str:
        path = os.path.join(
            state.run_dir,
            f"round_{state.round_index:02d}",
            candidate.candidate_id,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _fusion_env(
        self,
        candidate: CandidateResult,
        cfg: dict[str, Any],
    ) -> dict[str, str]:
        env = os.environ.copy()
        gpu = str(os.environ.get("TQ_GPU_DEVICE", cfg.get("gpu", "0")))
        env["HIP_VISIBLE_DEVICES"] = gpu
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTORCH_ALLOC_CONF"] = env.get(
            "PYTORCH_ALLOC_CONF",
            "expandable_segments:True",
        )
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            self.repo_root if not pythonpath else f"{self.repo_root}:{pythonpath}"
        )
        env["TQ_ALLOW_STALE_HIP_SO"] = env.get("TQ_ALLOW_STALE_HIP_SO", "1")
        env["VLLM_TQ_FUSION_V3_HIP"] = "1"
        config_path = candidate.artifacts.get("campaign_config", "")
        if config_path:
            env["VLLM_TQ_FUSION_CAMPAIGN_CONFIG"] = config_path
            try:
                with open(config_path) as f:
                    campaign_config = json.load(f)
            except Exception:
                campaign_config = {}
            for key in (
                "VLLM_TQ_FUSION_V3_DECODE_HIP_FLASH_TQ",
                "VLLM_TQ_FUSION_V3_DECODE_HIP_V136_MFMA",
                "VLLM_TQ_FUSION_V3_DECODE_HIP_MFMA_QK",
                "VLLM_TQ_FUSION_V3_DECODE_HIP_SCALAR",
            ):
                env.pop(key, None)
            if campaign_config.get("decode_impl") == "hip_v3_flash_tq":
                env["VLLM_TQ_FUSION_V3_DECODE_HIP_FLASH_TQ"] = "1"
            if campaign_config.get("decode_impl") == "hip_v3_v136_mfma":
                env["VLLM_TQ_FUSION_V3_DECODE_HIP_V136_MFMA"] = "1"
            if campaign_config.get("decode_impl") == "hip_v3_mfma_qk":
                env["VLLM_TQ_FUSION_V3_DECODE_HIP_MFMA_QK"] = "1"
            if campaign_config.get("decode_impl") == "hip_v3_scalar":
                env["VLLM_TQ_FUSION_V3_DECODE_HIP_SCALAR"] = "1"
        external_source_root = candidate.artifacts.get("external_source_root", "")
        if external_source_root:
            env["VLLM_TQ_FUSION_V3_HIP_SOURCE_ROOT"] = external_source_root
        return env

    def _server_compilation_config_args(self, cfg: dict[str, Any]) -> list[str]:
        """Keep service gates runnable when local native ops are ABI-mismatched."""
        raw = os.environ.get("TQ_VLLM_COMPILATION_CONFIG")
        payload: dict[str, Any]
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"custom_ops": [part.strip() for part in raw.split(",") if part.strip()]}
        else:
            payload = dict(cfg.get("compilation_config", {}))
        payload.setdefault("custom_ops", ["all"])
        return ["--compilation-config", json.dumps(payload, sort_keys=True)]

    def _run_fusion_synthetic_roundtrip(
        self,
        state: PipelineState,
        candidate: CandidateResult,
        cfg: dict[str, Any],
    ) -> dict[str, float]:
        work_dir = self._candidate_dir(state, candidate)
        script_path = os.path.join(work_dir, "fusion_roundtrip_gate.py")
        script = textwrap.dedent(
            f"""\
            import json
            import math
            import os
            import torch
            import torch.nn.functional as F

            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )
            from vllm.model_executor.layers.quantization.turboquant.centroids import (
                solve_lloyd_max,
            )
            from vllm.v1.attention.ops.tq_fusion_v3_hip.external_ops import (
                _tq_full_dequant_kv,
                _use_fp8_e4b15,
                triton_turboquant_store,
            )

            torch.manual_seed(1234)
            device = torch.device("cuda:0")
            D = {int(cfg.get("head_dim", 128))}
            Hq = {int(cfg.get("heads_q", 64))}
            Hk = {int(cfg.get("heads_k", 8))}
            batch_size = {int(cfg.get("batch_size", 4))}
            block_size = 16
            seq_len = {int(cfg.get("seq_len", 256))}
            num_blocks = math.ceil(seq_len / block_size)
            num_tokens = num_blocks * block_size

            tq_cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
            slot_size = tq_cfg.slot_size_aligned
            mse_bytes = tq_cfg.key_packed_size - (0 if tq_cfg.key_fp8 else 4)
            val_data_bytes = tq_cfg.value_packed_size - 4
            key = torch.randn(num_tokens, Hk, D, device=device, dtype=torch.float16)
            value = torch.randn(num_tokens, Hk, D, device=device, dtype=torch.float16)
            kv_cache = torch.zeros(
                num_blocks,
                block_size,
                Hk,
                slot_size,
                device=device,
                dtype=torch.uint8,
            )
            slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int32)
            centroids, _ = solve_lloyd_max(D, tq_cfg.centroid_bits)
            centroids = centroids.float().to(device)

            H = torch.tensor([[1.0]], device=device)
            while H.shape[0] < D:
                H = torch.cat(
                    [
                        torch.cat([H, H], dim=1),
                        torch.cat([H, -H], dim=1),
                    ],
                    dim=0,
                )
            PiT = (H / math.sqrt(D)).contiguous()
            c_sorted, _ = centroids.sort()
            midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

            triton_turboquant_store(
                key=key,
                value=value,
                kv_cache=kv_cache,
                slot_mapping=slot_mapping,
                PiT=PiT,
                midpoints=midpoints,
                mse_bits=tq_cfg.key_mse_bits,
                key_packed_size=tq_cfg.key_packed_size,
                value_quant_bits=tq_cfg.effective_value_quant_bits,
                key_fp8=tq_cfg.key_fp8,
                centroids=centroids,
                norm_correction=tq_cfg.norm_correction,
            )

            alloc_len = num_tokens
            k_cached = torch.empty((1, Hk, alloc_len, D), dtype=torch.float16, device=device)
            v_cached = torch.empty((1, Hk, alloc_len, D), dtype=torch.float16, device=device)
            block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1)
            kv_cache_u16 = kv_cache.view(torch.uint16)

            key_data_bytes = D if tq_cfg.key_fp8 else mse_bytes
            data_bytes_per_slot = key_data_bytes + val_data_bytes
            meta_region_offset = block_size * Hk * data_bytes_per_slot
            num_soa_fields = 2 if tq_cfg.key_fp8 else 3
            soa_k_norm = 0
            soa_v_scale = 0 if tq_cfg.key_fp8 else 1
            soa_v_zero = 1 if tq_cfg.key_fp8 else 2

            _tq_full_dequant_kv[(alloc_len, Hk)](
                kv_cache,
                kv_cache_u16,
                block_table,
                centroids.float(),
                k_cached,
                v_cached,
                k_cached.stride(0),
                k_cached.stride(1),
                k_cached.stride(2),
                v_cached.stride(0),
                v_cached.stride(1),
                v_cached.stride(2),
                kv_cache.stride(0),
                block_table.stride(0),
                HEAD_DIM=D,
                BLOCK_SIZE=block_size,
                NUM_KV_HEADS=Hk,
                MSE_BYTES=mse_bytes,
                VQB=tq_cfg.effective_value_quant_bits,
                VAL_DATA_BYTES=val_data_bytes,
                MSE_BITS=tq_cfg.key_mse_bits,
                KEY_FP8=1 if tq_cfg.key_fp8 else 0,
                KEY_DATA_BYTES=key_data_bytes,
                META_REGION_OFFSET=meta_region_offset,
                NUM_SOA_FIELDS=num_soa_fields,
                SOA_K_NORM=soa_k_norm,
                SOA_V_SCALE=soa_v_scale,
                SOA_V_ZERO=soa_v_zero,
                BLOCK_D=128,
                NORM_CORRECTION=1 if tq_cfg.norm_correction else 0,
                FP8_E4B15=_use_fp8_e4b15(device.index or 0),
                num_warps=4,
            )

            k_ref = (key.reshape(-1, D).float() @ PiT.float()).reshape(num_tokens, Hk, D)
            k_out = k_cached[0, :, :num_tokens, :].transpose(0, 1).contiguous().float()
            v_ref = value.float()
            v_out = v_cached[0, :, :num_tokens, :].transpose(0, 1).contiguous().float()

            k_cos = F.cosine_similarity(
                k_ref.reshape(-1, D), k_out.reshape(-1, D), dim=-1
            ).mean().item()
            v_cos = F.cosine_similarity(
                v_ref.reshape(-1, D), v_out.reshape(-1, D), dim=-1
            ).mean().item()
            min_cos = min(k_cos, v_cos)
            passed = 1 if min_cos >= {float(cfg.get("min_cosine", 0.985))} else 0

            print("===CAMPAIGN_QUICK===")
            print(json.dumps({{
                "passed": passed,
                "key_cosine": k_cos,
                "value_cosine": v_cos,
                "min_cosine": min_cos,
            }}))
            """
        )
        with open(script_path, "w") as f:
            f.write(script)

        env = self._fusion_env(candidate, cfg)
        result = subprocess.run(
            [self.python, script_path],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        metrics = self._parse_marked_json(result.stdout, "===CAMPAIGN_QUICK===")
        if result.returncode != 0:
            candidate.failure_reason = (result.stderr or result.stdout)[-400:]
        return {k: float(v) for k, v in metrics.items()} if metrics else {}

    def _start_server(
        self,
        state: PipelineState,
        candidate: CandidateResult,
        cfg: dict[str, Any],
        stage_name: str,
    ) -> tuple[subprocess.Popen[Any] | None, str]:
        work_dir = self._candidate_dir(state, candidate)
        log_path = os.path.join(work_dir, f"{stage_name}_server.log")
        env = self._fusion_env(candidate, cfg)
        port = str(cfg.get("port", 8201))
        cmd = [
            self.python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(cfg["model"]),
            "--kv-cache-dtype",
            "turboquant_4bit_nc",
            "--port",
            port,
            "--max-model-len",
            str(cfg.get("max_model_len", 32768)),
            "--gpu-memory-utilization",
            str(cfg.get("gpu_memory_utilization", 0.84)),
            "--max-num-seqs",
            str(cfg.get("max_num_seqs", 256)),
            "--disable-log-stats",
            "--no-async-scheduling",
            "--enforce-eager",
        ]
        cmd.extend(self._server_compilation_config_args(cfg))
        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    cwd=self.repo_root,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except Exception:
            return None, log_path
        self._active_servers[proc.pid] = proc

        for _ in range(120):
            time.sleep(3)
            if proc.poll() is not None:
                break
            health = subprocess.run(
                ["curl", "-sf", f"http://localhost:{port}/health"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if health.returncode == 0:
                return proc, log_path
        self._stop_server(proc)
        return None, log_path

    def _stop_server(self, proc: subprocess.Popen[Any] | None) -> None:
        if proc is None:
            return
        self._active_servers.pop(proc.pid, None)
        if proc.poll() is not None:
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            return

        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            proc.terminate()

        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        time.sleep(2)

    def cleanup(self) -> None:
        for proc in list(self._active_servers.values()):
            self._stop_server(proc)

    def _run_service_benchmark(
        self,
        state: PipelineState,
        candidate: CandidateResult,
        cfg: dict[str, Any],
        stage_name: str,
    ) -> dict[str, float]:
        proc, log_path = self._start_server(state, candidate, cfg, stage_name)
        if proc is None:
            candidate.failure_reason = f"server failed to start: {log_path}"
            return {}

        work_dir = self._candidate_dir(state, candidate)
        bench_log = os.path.join(work_dir, f"{stage_name}_bench.log")
        result_json_name = f"{stage_name}_bench_results.json"
        result_json_path = os.path.join(work_dir, result_json_name)
        port = str(cfg.get("port", 8201))
        cmd = [
            self.python,
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--port",
            port,
            "--model",
            str(cfg["model"]),
            "--dataset-name",
            "random",
            "--random-input-len",
            str(cfg.get("input_len", 1024)),
            "--random-output-len",
            str(cfg.get("output_len", 64)),
            "--num-prompts",
            str(cfg.get("num_prompts", 4)),
            "--request-rate",
            "inf",
            "--max-concurrency",
            str(cfg.get("concurrency", 1)),
            "--percentile-metrics",
            "ttft,tpot,itl",
            "--metric-percentiles",
            "50,99",
            "--save-result",
            "--result-dir",
            work_dir,
            "--result-filename",
            result_json_name,
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=3600,
                env=self._fusion_env(candidate, cfg),
            )
        except Exception as exc:
            candidate.failure_reason = f"benchmark exception: {exc}"
            self._stop_server(proc)
            return {}
        else:
            self._stop_server(proc)

        with open(bench_log, "w") as f:
            f.write(result.stdout)
            f.write("\n")
            f.write(result.stderr)

        metrics = self._load_bench_metrics(result_json_path)
        if not metrics:
            metrics = self._parse_bench_metrics(result.stdout)
        if result.returncode != 0:
            candidate.failure_reason = (result.stderr or result.stdout)[-400:]
            return {}
        if not self._benchmark_metrics_complete(metrics):
            candidate.failure_reason = (
                f"incomplete benchmark metrics: {sorted(metrics.keys())}"
                if metrics else "benchmark result missing metrics"
            )
            return {}
        if metrics.get("failed", 0.0) > 0:
            candidate.failure_reason = (
                f"benchmark reported {int(metrics.get('failed', 0.0))} failed requests"
            )
            return {}
        return metrics

    def _run_quality_spotcheck(
        self,
        state: PipelineState,
        candidate: CandidateResult,
        cfg: dict[str, Any],
    ) -> dict[str, float]:
        proc, log_path = self._start_server(state, candidate, cfg, stage_name="quality")
        if proc is None:
            candidate.failure_reason = f"quality server failed to start: {log_path}"
            return {}

        import urllib.request

        fact_hits = 0
        errors = 0
        port = str(cfg.get("port", 8201))
        model = str(cfg.get("model"))
        try:
            for prompt, expected in self.QUALITY_PROMPTS:
                payload = json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 32,
                    "temperature": 0,
                    "top_p": 1.0,
                }).encode()
                req = urllib.request.Request(
                    f"http://localhost:{port}/v1/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        result = json.loads(resp.read())
                    text = result["choices"][0]["text"]
                    if expected.lower() in text.lower():
                        fact_hits += 1
                except Exception:
                    errors += 1
        finally:
            self._stop_server(proc)

        passed = 1 if errors == 0 and fact_hits >= 3 else 0
        return {
            "passed": float(passed),
            "fact_hits": float(fact_hits),
            "errors": float(errors),
        }

    def _run_decode_smoke(
        self,
        state: PipelineState,
        candidate: CandidateResult,
        cfg: dict[str, Any],
    ) -> dict[str, float]:
        proc, log_path = self._start_server(state, candidate, cfg, stage_name="quick_decode")
        if proc is None:
            candidate.failure_reason = f"decode smoke server failed to start: {log_path}"
            return {}

        import urllib.request

        port = str(cfg.get("port", 8201))
        model = str(cfg.get("model"))
        prompt = str(cfg.get("prompt", "Write one short sentence about TurboQuant."))
        text = ""
        try:
            payload = json.dumps({
                "model": model,
                "prompt": prompt,
                "max_tokens": int(cfg.get("max_tokens", 8)),
                "temperature": 0,
                "top_p": 1.0,
            }).encode()
            req = urllib.request.Request(
                f"http://localhost:{port}/v1/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read())
            text = str(result["choices"][0].get("text", ""))
        except Exception as exc:
            candidate.failure_reason = f"decode smoke failed: {exc}"
            return {}
        finally:
            self._stop_server(proc)

        return {
            "passed": 1.0 if text.strip() else 0.0,
            "response_chars": float(len(text)),
        }

    def _parse_bench_metrics(self, stdout: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        metric_map = {
            "Output token throughput": "output_token_throughput",
            "Request throughput": "request_throughput",
            "Mean TTFT": "mean_ttft_ms",
            "Median TPOT": "median_tpot_ms",
            "Median ITL": "median_itl_ms",
            "Mean ITL": "mean_itl_ms",
            "P99 ITL": "p99_itl_ms",
            "Successful requests": "completed",
            "Failed requests": "failed",
        }
        for line in stdout.splitlines():
            for needle, output_key in metric_map.items():
                if needle in line:
                    try:
                        metrics[output_key] = float(line.split()[-1])
                    except (IndexError, ValueError):
                        pass
        return metrics

    def _load_bench_metrics(self, path: str) -> dict[str, float]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return {
            "completed": float(payload.get("completed", 0.0)),
            "failed": float(payload.get("failed", 0.0)),
            "request_throughput": float(payload.get("request_throughput", 0.0) or 0.0),
            "output_token_throughput": float(payload.get("output_throughput", 0.0) or 0.0),
            "mean_ttft_ms": float(payload.get("mean_ttft_ms", 0.0) or 0.0),
            "median_tpot_ms": float(payload.get("median_tpot_ms", 0.0) or 0.0),
            "median_itl_ms": float(payload.get("median_itl_ms", 0.0) or 0.0),
            "mean_itl_ms": float(payload.get("mean_itl_ms", 0.0) or 0.0),
            "p99_itl_ms": float(payload.get("p99_itl_ms", 0.0) or 0.0),
        }

    def _benchmark_metrics_complete(self, metrics: dict[str, float]) -> bool:
        required = (
            "completed",
            "request_throughput",
            "output_token_throughput",
            "mean_ttft_ms",
            "median_tpot_ms",
            "median_itl_ms",
        )
        if any(metrics.get(key, 0.0) <= 0.0 for key in required):
            return False
        return metrics.get("completed", 0.0) > 0.0

    def _parse_marked_json(self, stdout: str, marker: str) -> dict[str, Any]:
        found = False
        for line in stdout.splitlines():
            if line.strip() == marker:
                found = True
                continue
            if found:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {}

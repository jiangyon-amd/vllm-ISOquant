"""Optional runtime stats for TurboQuant decode/store profiling.

Enabled with ``VLLM_TQ_PROFILE_STATS=1``. The collector is intentionally
lightweight when disabled so it can stay imported in production code paths.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from typing import Any

_ENABLED = os.environ.get("VLLM_TQ_PROFILE_STATS", "0") == "1"
_LOCK = threading.Lock()


def _new_state() -> dict[str, Any]:
    return {
        "step_mode_counts": Counter(),
        "max_query_len_hist": Counter(),
        "step_max_seq_len_hist": Counter(),
        "num_decode_tokens_hist": Counter(),
        "num_prefills_hist": Counter(),
        "decode_batch_hist": Counter(),
        "decode_max_seq_len_hist": Counter(),
        "decode_path_counts": Counter(),
        "decode_stage2_backend_counts": Counter(),
        "decode_custom_op_counts": Counter(),
        "decode_output_dtype_counts": Counter(),
        "num_kv_splits_hist": Counter(),
        "store_phase_counts": Counter(),
        "store_backend_counts": Counter(),
        "store_overlap_counts": Counter(),
        "store_wait_counts": Counter(),
        "host_time_us_total": {},
        "host_time_us_count": Counter(),
        "host_time_us_max": {},
    }


_STATE = _new_state()


def enabled() -> bool:
    return _ENABLED


def _bucket(value: int) -> str:
    if value <= 0:
        return "0"
    buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for upper in buckets:
        if value <= upper:
            return f"<={upper}"
    return ">8192"


def _observe_time(key: str, host_us: float | None) -> None:
    if host_us is None:
        return
    total = _STATE["host_time_us_total"]
    total[key] = total.get(key, 0.0) + host_us
    _STATE["host_time_us_count"][key] += 1
    prev_max = _STATE["host_time_us_max"].get(key, 0.0)
    _STATE["host_time_us_max"][key] = max(prev_max, host_us)


def reset_turboquant_runtime_stats() -> None:
    if not _ENABLED:
        return
    with _LOCK:
        global _STATE
        _STATE = _new_state()


def get_turboquant_runtime_stats() -> dict[str, Any]:
    if not _ENABLED:
        return {"enabled": False}

    with _LOCK:
        stats = {
            "enabled": True,
            "step_mode_counts": dict(_STATE["step_mode_counts"]),
            "max_query_len_hist": dict(_STATE["max_query_len_hist"]),
            "step_max_seq_len_hist": dict(_STATE["step_max_seq_len_hist"]),
            "num_decode_tokens_hist": dict(_STATE["num_decode_tokens_hist"]),
            "num_prefills_hist": dict(_STATE["num_prefills_hist"]),
            "decode_batch_hist": dict(_STATE["decode_batch_hist"]),
            "decode_max_seq_len_hist": dict(_STATE["decode_max_seq_len_hist"]),
            "decode_path_counts": dict(_STATE["decode_path_counts"]),
            "decode_stage2_backend_counts": dict(
                _STATE["decode_stage2_backend_counts"]
            ),
            "decode_custom_op_counts": dict(_STATE["decode_custom_op_counts"]),
            "decode_output_dtype_counts": dict(
                _STATE["decode_output_dtype_counts"]
            ),
            "num_kv_splits_hist": dict(_STATE["num_kv_splits_hist"]),
            "store_phase_counts": dict(_STATE["store_phase_counts"]),
            "store_backend_counts": dict(_STATE["store_backend_counts"]),
            "store_overlap_counts": dict(_STATE["store_overlap_counts"]),
            "store_wait_counts": dict(_STATE["store_wait_counts"]),
            "host_time_us_max": dict(_STATE["host_time_us_max"]),
        }
        host_time_us_avg: dict[str, float] = {}
        for key, total in _STATE["host_time_us_total"].items():
            count = _STATE["host_time_us_count"][key]
            if count:
                host_time_us_avg[key] = total / count
        stats["host_time_us_avg"] = host_time_us_avg
        stats["host_time_us_count"] = dict(_STATE["host_time_us_count"])
        return stats


def record_builder_step(
    *,
    max_query_len: int,
    max_seq_len: int,
    num_decodes: int,
    num_prefills: int,
    num_decode_tokens: int,
) -> None:
    if not _ENABLED:
        return
    mode = (
        "pure_decode"
        if max_query_len == 1 and num_prefills == 0
        else "pure_prefill"
        if num_decodes == 0
        else "mixed"
    )
    with _LOCK:
        _STATE["step_mode_counts"][mode] += 1
        _STATE["max_query_len_hist"][_bucket(max_query_len)] += 1
        _STATE["step_max_seq_len_hist"][_bucket(max_seq_len)] += 1
        _STATE["num_decode_tokens_hist"][_bucket(num_decode_tokens)] += 1
        _STATE["num_prefills_hist"][_bucket(num_prefills)] += 1


def record_decode_call(
    *,
    batch_size: int,
    max_seq_len: int,
    num_kv_splits: int,
    path: str,
    stage2_backend: str,
    custom_op: str,
    output_dtype: str,
    host_qrot_us: float | None = None,
    host_stage1_us: float | None = None,
    host_stage2_us: float | None = None,
) -> None:
    if not _ENABLED:
        return
    with _LOCK:
        _STATE["decode_batch_hist"][_bucket(batch_size)] += 1
        _STATE["decode_max_seq_len_hist"][_bucket(max_seq_len)] += 1
        _STATE["num_kv_splits_hist"][str(num_kv_splits)] += 1
        _STATE["decode_path_counts"][path] += 1
        _STATE["decode_stage2_backend_counts"][stage2_backend] += 1
        _STATE["decode_custom_op_counts"][custom_op] += 1
        _STATE["decode_output_dtype_counts"][output_dtype] += 1
        _observe_time("decode_qrot_host", host_qrot_us)
        _observe_time("decode_stage1_host", host_stage1_us)
        _observe_time("decode_stage2_host", host_stage2_us)


def record_store_call(
    *,
    phase: str | None = None,
    backend: str | None = None,
    overlap_mode: str | None = None,
    waited_for_store_read: bool = False,
    host_store_us: float | None = None,
) -> None:
    if not _ENABLED:
        return
    with _LOCK:
        if phase is not None:
            _STATE["store_phase_counts"][phase] += 1
        if backend is not None:
            _STATE["store_backend_counts"][backend] += 1
        if overlap_mode is not None:
            _STATE["store_overlap_counts"][overlap_mode] += 1
        if waited_for_store_read:
            _STATE["store_wait_counts"]["decode_wait_stream"] += 1
        _observe_time("store_host", host_store_us)


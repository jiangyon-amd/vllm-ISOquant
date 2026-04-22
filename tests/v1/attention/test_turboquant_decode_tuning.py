# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.ops.triton_turboquant_decode import (
    _resolve_num_kv_splits,
    _should_use_v56,
)


def test_should_use_v56_respects_batch_and_seq_thresholds() -> None:
    assert _should_use_v56(batch_size=4, max_seq_len_hint=1024, v56_max_seq_len=2048)
    assert not _should_use_v56(
        batch_size=16,
        max_seq_len_hint=1024,
        v56_max_seq_len=2048,
    )
    assert not _should_use_v56(
        batch_size=4,
        max_seq_len_hint=8192,
        v56_max_seq_len=2048,
    )


def test_resolve_num_kv_splits_keeps_fixed_value_when_adaptation_disabled() -> None:
    assert _resolve_num_kv_splits(
        max_num_kv_splits=8,
        eager_max_num_kv_splits=4,
        max_seq_len_hint=8192,
        batch_size=32,
        allow_adaptive_kv_splits=False,
    ) == 8


def test_resolve_num_kv_splits_prefers_lower_split_count_for_long_decode() -> None:
    assert _resolve_num_kv_splits(
        max_num_kv_splits=32,
        eager_max_num_kv_splits=8,
        max_seq_len_hint=8192,
        batch_size=16,
        allow_adaptive_kv_splits=True,
    ) == 4
    assert _resolve_num_kv_splits(
        max_num_kv_splits=32,
        eager_max_num_kv_splits=8,
        max_seq_len_hint=8192,
        batch_size=4,
        allow_adaptive_kv_splits=True,
    ) == 8

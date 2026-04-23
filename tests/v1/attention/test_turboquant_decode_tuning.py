# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TurboQuant decode dispatch heuristics.

Covers _should_use_fused (batch-adaptive fused-vs-split) and
_resolve_num_kv_splits (adaptive KV split count).
"""

from vllm.v1.attention.ops.triton_turboquant_decode import (
    _resolve_num_kv_splits,
    _should_use_fused,
)


# ---------------------------------------------------------------
# _should_use_fused: batch-adaptive fused threshold
# ---------------------------------------------------------------

def test_fused_small_batch_short_seq() -> None:
    """B<4 should use fused only for very short sequences."""
    assert _should_use_fused(batch_size=1, max_seq_len_hint=256)
    assert not _should_use_fused(batch_size=1, max_seq_len_hint=512)
    assert _should_use_fused(batch_size=2, max_seq_len_hint=128)


def test_fused_medium_batch() -> None:
    """B=4-15: fused up to seq=384."""
    assert _should_use_fused(batch_size=4, max_seq_len_hint=384)
    assert not _should_use_fused(batch_size=4, max_seq_len_hint=512)
    assert _should_use_fused(batch_size=8, max_seq_len_hint=256)
    assert not _should_use_fused(batch_size=8, max_seq_len_hint=1024)


def test_fused_large_batch() -> None:
    """B=16-31: fused up to seq=1024."""
    assert _should_use_fused(batch_size=16, max_seq_len_hint=1024)
    assert not _should_use_fused(batch_size=16, max_seq_len_hint=2048)


def test_fused_very_large_batch() -> None:
    """B>=32: fused up to seq=2048."""
    assert _should_use_fused(batch_size=32, max_seq_len_hint=2048)
    assert _should_use_fused(batch_size=64, max_seq_len_hint=1024)
    assert not _should_use_fused(batch_size=32, max_seq_len_hint=4096)


def test_fused_zero_seq_never_fused() -> None:
    """seq<=0 should never use fused."""
    assert not _should_use_fused(batch_size=100, max_seq_len_hint=0)
    assert not _should_use_fused(batch_size=100, max_seq_len_hint=-1)


# ---------------------------------------------------------------
# _resolve_num_kv_splits
# ---------------------------------------------------------------

def test_resolve_num_kv_splits_fixed_when_adaptation_disabled() -> None:
    """When allow_adaptive=False, return max_num_kv_splits verbatim."""
    assert _resolve_num_kv_splits(
        max_num_kv_splits=8,
        eager_max_num_kv_splits=4,
        max_seq_len_hint=8192,
        batch_size=32,
        allow_adaptive_kv_splits=False,
    ) == 8


def test_resolve_num_kv_splits_returns_one_when_max_is_one() -> None:
    assert _resolve_num_kv_splits(
        max_num_kv_splits=1,
        eager_max_num_kv_splits=32,
        max_seq_len_hint=4096,
        batch_size=16,
        allow_adaptive_kv_splits=True,
    ) == 1


def test_resolve_num_kv_splits_long_context_uses_eager_cap() -> None:
    """For seq>=4096, adaptive mode returns eager_cap."""
    assert _resolve_num_kv_splits(
        max_num_kv_splits=32,
        eager_max_num_kv_splits=8,
        max_seq_len_hint=8192,
        batch_size=16,
        allow_adaptive_kv_splits=True,
    ) == 8


def test_resolve_num_kv_splits_short_context_uses_tokens_per_split() -> None:
    """For short seq, use ceil(seq / 64) capped by eager_max."""
    # seq=128 → ceil(128/64) = 2
    assert _resolve_num_kv_splits(
        max_num_kv_splits=32,
        eager_max_num_kv_splits=32,
        max_seq_len_hint=128,
        batch_size=4,
        allow_adaptive_kv_splits=True,
    ) == 2

    # seq=512 → ceil(512/64) = 8
    assert _resolve_num_kv_splits(
        max_num_kv_splits=32,
        eager_max_num_kv_splits=32,
        max_seq_len_hint=512,
        batch_size=4,
        allow_adaptive_kv_splits=True,
    ) == 8


def test_resolve_num_kv_splits_zero_hint_returns_eager_cap() -> None:
    """When no seq len hint, return eager cap."""
    assert _resolve_num_kv_splits(
        max_num_kv_splits=32,
        eager_max_num_kv_splits=16,
        max_seq_len_hint=0,
        batch_size=4,
        allow_adaptive_kv_splits=True,
    ) == 16

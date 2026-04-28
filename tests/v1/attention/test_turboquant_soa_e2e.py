# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E correctness tests for TurboQuant SoA fusion path.

Tests the full self-contained pipeline:
  1. Store: raw KV -> TQ-quantized SoA KV cache (store kernel)
  2. Decode: query + TQ cache -> attention output (unified decode kernel)
  3. Correctness: compare against reference FP16 dense attention

Run:
  python -m pytest tests/v1/attention/test_turboquant_soa_e2e.py -v -s
"""

import math
import sys
import os

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="GPU not available")

from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.ops.turboquant_soa_fusion.external_ops import (
    triton_turboquant_store,
    triton_turboquant_decode_attention_v3,
)

DEVICE = "cuda:0"


def _build_hadamard(d, device):
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(device)


def _make_midpoints(centroids):
    c_sorted, _ = centroids.sort()
    return (c_sorted[:-1] + c_sorted[1:]) / 2


class SoATestCase:
    def __init__(self, preset_name, head_dim=128, num_kv_heads=8,
                 gqa_ratio=8, block_size=16, seq_len=64, batch_size=1,
                 device=DEVICE):
        self.preset_name = preset_name
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_kv_heads * gqa_ratio
        self.gqa_ratio = gqa_ratio
        self.block_size = block_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.cfg = TurboQuantConfig.from_cache_dtype(preset_name, head_dim=head_dim)
        self.scale = 1.0 / math.sqrt(head_dim)
        self.centroids = get_centroids(head_dim, self.cfg.centroid_bits).to(
            device=device, dtype=torch.float32)
        self.midpoints = _make_midpoints(self.centroids)
        self.Pi = _build_hadamard(head_dim, device).contiguous()
        self.PiT = self.Pi.T.contiguous()
        num_blocks_needed = math.ceil(seq_len / block_size) * batch_size
        self.num_blocks = num_blocks_needed + 4
        self.kv_cache = torch.zeros(
            (self.num_blocks, block_size, num_kv_heads, self.cfg.slot_size_aligned),
            dtype=torch.uint8, device=device)
        max_blocks_per_seq = math.ceil(seq_len / block_size)
        self.block_table = torch.zeros(
            (batch_size, max_blocks_per_seq), dtype=torch.int32, device=device)
        block_idx = 0
        for b in range(batch_size):
            for bl in range(max_blocks_per_seq):
                self.block_table[b, bl] = block_idx
                block_idx += 1
        self.seq_lens = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device)
        self.mse_bits = self.cfg.key_mse_bits if not self.cfg.key_fp8 else self.cfg.mse_bits

    def store(self, key, value, slot_mapping):
        triton_turboquant_store(
            key=key, value=value,
            kv_cache=self.kv_cache, slot_mapping=slot_mapping,
            PiT=self.PiT, midpoints=self.midpoints,
            mse_bits=self.mse_bits,
            key_packed_size=self.cfg.key_packed_size,
            value_quant_bits=self.cfg.effective_value_quant_bits,
            key_fp8=self.cfg.key_fp8,
            centroids=self.centroids,
            norm_correction=self.cfg.norm_correction)

    def decode(self, query, seq_lens=None, max_seq_len=None):
        if seq_lens is None:
            seq_lens = self.seq_lens
        if max_seq_len is None:
            max_seq_len = int(seq_lens.max().item())
        return triton_turboquant_decode_attention_v3(
            query=query, kv_cache=self.kv_cache,
            block_table=self.block_table, seq_lens=seq_lens,
            Pi=self.Pi, centroids=self.centroids, scale=self.scale,
            mse_bits=self.mse_bits,
            key_packed_size=self.cfg.key_packed_size,
            value_quant_bits=self.cfg.effective_value_quant_bits,
            value_packed_size=self.cfg.value_packed_size,
            key_fp8=self.cfg.key_fp8,
            norm_correction=self.cfg.norm_correction,
            PiT=self.PiT, max_seq_len=max_seq_len)


# ============================================================================
# 1. Store correctness
# ============================================================================

class TestSoAStoreBasic:
    @pytest.mark.parametrize("preset", list(TQ_PRESETS.keys()))
    def test_store_fills_cache(self, preset):
        tc = SoATestCase(preset, seq_len=32)
        N = tc.seq_len
        key = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                          device=tc.device, dtype=torch.bfloat16)
        value = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                            device=tc.device, dtype=torch.bfloat16)
        slot_mapping = torch.arange(N, device=tc.device, dtype=torch.int32)
        assert tc.kv_cache.sum() == 0
        tc.store(key, value, slot_mapping)
        nonzero = (tc.kv_cache != 0).sum().item()
        assert nonzero > 0, f"Cache still zero after store ({preset})"
        print(f"  {preset}: {nonzero} nonzero bytes in cache")


# ============================================================================
# 2. Decode correctness vs dense reference
# ============================================================================

class TestSoADecodeCorrectness:
    @pytest.mark.parametrize("preset", list(TQ_PRESETS.keys()))
    @pytest.mark.parametrize("seq_len", [16, 64, 256])
    def test_decode_vs_reference(self, preset, seq_len):
        torch.manual_seed(42)
        tc = SoATestCase(preset, seq_len=seq_len)
        D, Hk, Hq = tc.head_dim, tc.num_kv_heads, tc.num_query_heads
        N = seq_len

        key = torch.randn(N, Hk, D, device=tc.device, dtype=torch.bfloat16)
        value = torch.randn(N, Hk, D, device=tc.device, dtype=torch.bfloat16)
        slot_mapping = torch.arange(N, device=tc.device, dtype=torch.int32)
        tc.store(key, value, slot_mapping)

        query = torch.randn(1, Hq, D, device=tc.device, dtype=torch.bfloat16)
        tq_out = tc.decode(query)

        # Reference dense attention
        k_ref = key.float()
        v_ref = value.float()
        q_ref = query.float()
        k_exp = k_ref.unsqueeze(2).expand(N, Hk, tc.gqa_ratio, D).reshape(N, Hq, D)
        v_exp = v_ref.unsqueeze(2).expand(N, Hk, tc.gqa_ratio, D).reshape(N, Hq, D)
        scores = torch.einsum("bhd,nhd->bhn", q_ref, k_exp) * tc.scale
        attn_w = torch.softmax(scores, dim=-1)
        ref_out = torch.einsum("bhn,nhd->bhd", attn_w, v_exp)

        tq_f = tq_out.float()
        ref_f = ref_out.float()

        cos_sims = F.cosine_similarity(
            tq_f.reshape(-1, D), ref_f.reshape(-1, D), dim=-1)
        avg_cos = cos_sims.mean().item()
        min_cos = cos_sims.min().item()
        rel_err = (tq_f - ref_f).norm() / (ref_f.norm() + 1e-8)

        print(f"\n  {preset} seq={seq_len}: cos={avg_cos:.4f} "
              f"(min={min_cos:.4f}) rel_err={rel_err:.4f}")

        if tc.cfg.key_fp8:
            assert avg_cos > 0.85, f"FP8 cos too low: {avg_cos}"
        elif tc.cfg.key_quant_bits == 4:
            assert avg_cos > 0.75, f"4bit cos too low: {avg_cos}"
        else:
            assert avg_cos > 0.55, f"3bit cos too low: {avg_cos}"


# ============================================================================
# 3. Batch decode
# ============================================================================

class TestSoABatchDecode:
    @pytest.mark.parametrize("preset", ["turboquant_k8v4", "turboquant_4bit_nc"])
    def test_batch_decode(self, preset):
        torch.manual_seed(123)
        B = 4
        seq_lens_list = [16, 32, 48, 64]
        max_seq = max(seq_lens_list)
        D, Hk, gqa = 128, 8, 8
        Hq = Hk * gqa
        bs = 16

        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)
        scale = 1.0 / math.sqrt(D)
        centroids = get_centroids(D, cfg.centroid_bits).to(DEVICE, dtype=torch.float32)
        midpoints = _make_midpoints(centroids)
        Pi = _build_hadamard(D, DEVICE).contiguous()
        PiT = Pi.T.contiguous()
        mse_bits = cfg.key_mse_bits if not cfg.key_fp8 else cfg.mse_bits

        max_bps = math.ceil(max_seq / bs)
        total_blocks = B * max_bps + 4
        kv_cache = torch.zeros(
            (total_blocks, bs, Hk, cfg.slot_size_aligned),
            dtype=torch.uint8, device=DEVICE)
        block_table = torch.zeros((B, max_bps), dtype=torch.int32, device=DEVICE)
        bi = 0
        for b in range(B):
            for bl in range(max_bps):
                block_table[b, bl] = bi; bi += 1

        for b in range(B):
            sl = seq_lens_list[b]
            k = torch.randn(sl, Hk, D, device=DEVICE, dtype=torch.bfloat16)
            v = torch.randn(sl, Hk, D, device=DEVICE, dtype=torch.bfloat16)
            base = b * max_bps * bs
            sm = torch.arange(sl, device=DEVICE, dtype=torch.int32) + base
            triton_turboquant_store(
                key=k, value=v, kv_cache=kv_cache, slot_mapping=sm,
                PiT=PiT, midpoints=midpoints, mse_bits=mse_bits,
                key_packed_size=cfg.key_packed_size,
                value_quant_bits=cfg.effective_value_quant_bits,
                key_fp8=cfg.key_fp8, centroids=centroids,
                norm_correction=cfg.norm_correction)

        q = torch.randn(B, Hq, D, device=DEVICE, dtype=torch.bfloat16)
        sl_t = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
        out = triton_turboquant_decode_attention_v3(
            query=q, kv_cache=kv_cache, block_table=block_table,
            seq_lens=sl_t, Pi=Pi, centroids=centroids, scale=scale,
            mse_bits=mse_bits, key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            value_packed_size=cfg.value_packed_size,
            key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction,
            PiT=PiT, max_seq_len=max_seq)

        assert out.shape == (B, Hq, D)
        assert not torch.isnan(out).any(), "NaN"
        assert torch.isfinite(out).all(), "Inf"
        assert out.abs().max() > 0, "All zero"
        print(f"\n  {preset} B={B}: range=[{out.min():.4f},{out.max():.4f}]")


# ============================================================================
# 4. Self-containment
# ============================================================================

class TestSelfContained:
    def test_no_external_source_root(self):
        assert "VLLM_TQ_SOA_FUSION_SOURCE_ROOT" not in os.environ

    def test_import_chain(self):
        from vllm.v1.attention.ops.turboquant_soa_fusion.external_ops import (
            _decode_module, _store_module, _unified_module)
        assert "turboquant_soa_fusion" in _decode_module().__name__
        assert "turboquant_soa_fusion" in _store_module().__name__
        assert "turboquant_soa_fusion" in _unified_module().__name__

    def test_backend_impl_import(self):
        from vllm.v1.attention.ops.turboquant_soa_fusion import (
            FusionTurboQuantAttentionImpl)
        assert FusionTurboQuantAttentionImpl is not None

    def test_runtime_stats_import(self):
        from vllm.v1.attention.ops.turboquant_runtime_stats import (
            record_decode_call, record_store_call, record_builder_step)
        record_decode_call(batch_size=1, path="test")
        record_store_call(backend="test")
        record_builder_step(max_query_len=1)


# ============================================================================
# 5. HIP fallback
# ============================================================================

class TestHIPFallback:
    def test_hip_scalar_default_disabled(self):
        from vllm.v1.attention.ops.turboquant_soa_fusion.external_ops import (
            _load_hip_v3_scalar)
        assert _load_hip_v3_scalar() is None

    def test_triton_fallback_works(self):
        torch.manual_seed(99)
        tc = SoATestCase("turboquant_4bit_nc", seq_len=32)
        N = tc.seq_len
        k = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        v = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        sm = torch.arange(N, device=tc.device, dtype=torch.int32)
        tc.store(k, v, sm)
        q = torch.randn(1, tc.num_query_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        out = tc.decode(q)
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()
        print(f"\n  Triton fallback: norm={out.norm():.4f}")


# ============================================================================
# 6. Stress tests
# ============================================================================

class TestSoAStress:
    @pytest.mark.parametrize("preset", ["turboquant_k8v4", "turboquant_4bit_nc"])
    def test_long_seq_1024(self, preset):
        torch.manual_seed(7)
        tc = SoATestCase(preset, seq_len=1024)
        N = tc.seq_len
        k = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        v = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        sm = torch.arange(N, device=tc.device, dtype=torch.int32)
        tc.store(k, v, sm)
        q = torch.randn(1, tc.num_query_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        out = tc.decode(q)
        assert out.shape == (1, tc.num_query_heads, tc.head_dim)
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()
        print(f"\n  {preset} seq=1024: norm={out.norm():.4f}")

    def test_large_batch_32(self):
        torch.manual_seed(42)
        tc = SoATestCase("turboquant_4bit_nc", seq_len=128, batch_size=32)
        N = tc.seq_len
        for b in range(tc.batch_size):
            k = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                             device=tc.device, dtype=torch.bfloat16)
            v = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                             device=tc.device, dtype=torch.bfloat16)
            max_bps = math.ceil(N / tc.block_size)
            base = b * max_bps * tc.block_size
            sm = torch.arange(N, device=tc.device, dtype=torch.int32) + base
            tc.store(k, v, sm)
        q = torch.randn(tc.batch_size, tc.num_query_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        out = tc.decode(q)
        assert out.shape == (tc.batch_size, tc.num_query_heads, tc.head_dim)
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()
        print(f"\n  B=32 seq=128: norm={out.norm():.4f}")


# ============================================================================
# 7. Numerical stability
# ============================================================================

class TestNumericalStability:
    @pytest.mark.parametrize("preset", list(TQ_PRESETS.keys()))
    def test_output_not_constant(self, preset):
        torch.manual_seed(0)
        tc = SoATestCase(preset, seq_len=64)
        N = tc.seq_len
        k = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        v = torch.randn(N, tc.num_kv_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        sm = torch.arange(N, device=tc.device, dtype=torch.int32)
        tc.store(k, v, sm)
        q = torch.randn(1, tc.num_query_heads, tc.head_dim,
                         device=tc.device, dtype=torch.bfloat16)
        out = tc.decode(q)
        head_norms = out[0].float().norm(dim=-1)
        std = head_norms.std().item()
        assert std > 1e-4, f"Constant across heads (std={std})"
        out_std = out.float().std().item()
        assert out_std > 1e-4, f"No variation (std={out_std})"
        print(f"\n  {preset}: head_std={std:.6f} out_std={out_std:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])

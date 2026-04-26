# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import functools
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.triton_utils import triton
from vllm.utils.torch_utils import aux_stream
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.turboquant_runtime_stats import (
    record_builder_step,
    record_store_call,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

# CUDA stream overlap: disabled by default — degrades TTFT under concurrent
# load (489ms vs 338ms). Enable via TQ_STREAM_OVERLAP=1 for experimentation.
_USE_STREAM_OVERLAP = os.environ.get("TQ_STREAM_OVERLAP", "0") == "1"

# Experimental full-stack fusion path. This keeps the current production path
# untouched and only switches the implementation class when the env flag is set.
_USE_TQ_FUSION_V3_HIP = os.environ.get("VLLM_TQ_FUSION_V3_HIP", "0") == "1"

# Per-step batch mode flag: set by TurboQuantMetadataBuilder.build() so that
# do_kv_cache_update can detect pure-decode steps without receiving metadata.
# True = current step is pure decode (max_query_len == 1, no prefill tokens).
# Stream overlap is safe to enable only in pure-decode steps to avoid TTFT
# regression (prefill is latency-sensitive, decode is throughput-sensitive).
_STEP_IS_PURE_DECODE: bool = False

# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills


@functools.cache
def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), built on CPU.

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        if _USE_TQ_FUSION_V3_HIP:
            from vllm.v1.attention.ops.tq_fusion_v3_hip import (
                FusionTurboQuantAttentionImpl,
            )

            return FusionTurboQuantAttentionImpl
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Standard attention backends use (2, num_blocks, block_size, num_kv_heads,
        head_dim) with a leading 2 to separate K and V. TurboQuant packs K+V
        into a single interleaved slot per head per position, so the cache is:

            (num_blocks, block_size, num_kv_heads, slot_size_aligned)

        Each slot = [key_packed | value_packed | padding].
        This is safe because TQ has its own get_kv_cache_shape override and
        never shares cache tensors with other backends. Layers that fall back
        to native dtype via kv_cache_dtype_skip_layers get their own
        standard-shaped cache allocation.

        head_size is the model's real head_dim. slot_size_aligned is computed
        from the TQ config to ensure correct cache allocation for all head dims.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("turboquant_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor  # (num_reqs,) — total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests
    # Max seq_len of prefill-only requests (CPU int, set by builder to avoid
    # GPU sync in mixed-batch forward). Equals max_seq_len for pure-prefill
    # batches, and is max_seq_len (conservative upper bound) for mixed batches.
    prefill_max_seq_len: int = 0
    # Pre-computed sub-metadata for mixed batches (None for pure batches).
    # Avoids repeated TurboQuantMetadata construction in the forward() hot path
    # (80 layers × per-step construction = 80 Python object allocations saved).
    # decode_sub_meta: decode-portion metadata (is_prefill=False, first N decodes).
    # prefill_sub_meta: prefill-portion metadata (is_prefill=True, remaining reqs).
    decode_sub_meta: "TurboQuantMetadata | None" = None
    prefill_sub_meta: "TurboQuantMetadata | None" = None


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # Set seq_lens to 1 so CUDA graph capture is fast
        # (real seq_lens are filled at replay time).
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata."""
        global _STEP_IS_PURE_DECODE
        cam = common_attn_metadata

        # With reorder_batch_threshold=1, the model runner guarantees
        # decodes come first in the batch. split_decodes_and_prefills
        # finds the boundary (operates on CPU tensors — no GPU sync).
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        # Update per-step decode flag so do_kv_cache_update can use async
        # stream overlap only when it's safe (pure decode, no prefill tokens).
        _STEP_IS_PURE_DECODE = (cam.max_query_len == 1) and (num_prefills == 0)
        record_builder_step(
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
        )

        # Pre-build sub-metadata for mixed batches once in the builder,
        # so each of the 80 layers can reuse them without repeated Python object
        # construction. For pure decode/prefill batches, both are None.
        decode_sub_meta = None
        prefill_sub_meta = None
        if num_decodes > 0 and num_prefills > 0:
            # Mixed batch: decodes come first (reorder_batch_threshold=1).
            N = cam.num_actual_tokens
            # Precompute zero-shifted query_start_loc for prefill slice once
            # (avoids 80× tensor subtraction in forward() hot path).
            prefill_qsl_precomputed = (
                cam.query_start_loc[num_decodes:] - num_decode_tokens
            )

            # Compute the true max seq_len of prefill-only requests.
            # This is a single GPU→CPU sync but it runs once per scheduler
            # step (not per layer), and is required for the prefill fast path
            # (max_query_len == max_seq_len) to work correctly.  Without it,
            # decode sequences inflate max_seq_len and the fast path never
            # fires in mixed batches, silently falling back to the slow
            # per-request SDPA loop.
            prefill_seq_lens = cam.seq_lens[num_decodes:]
            prefill_max_sl = int(prefill_seq_lens.max().item())

            decode_sub_meta = TurboQuantMetadata(
                seq_lens=cam.seq_lens[:num_decodes],
                slot_mapping=cam.slot_mapping[:num_decode_tokens],
                block_table=cam.block_table_tensor[:num_decodes],
                query_start_loc=cam.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=cam.max_seq_len,
                is_prefill=False,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
            )
            prefill_sub_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=cam.slot_mapping[num_decode_tokens:N],
                block_table=cam.block_table_tensor[num_decodes:],
                query_start_loc=prefill_qsl_precomputed,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=cam.max_query_len,
                max_seq_len=prefill_max_sl,
                is_prefill=True,
                prefill_max_seq_len=prefill_max_sl,
            )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            # For mixed batches, use full max_seq_len as conservative upper
            # bound — avoids a GPU sync (seq_lens[num_decodes:].max().item()).
            # The prefill fast-path check (max_query_len == prefill_max_seq_len)
            # correctly fails for mixed batches anyway, since decode seq_lens
            # inflate the full max_seq_len above max_query_len.
            # For pure-prefill batches (num_decodes == 0), max_seq_len equals
            # the true prefill max context, so the fast path works correctly.
            prefill_max_seq_len=cam.max_seq_len,
            decode_sub_meta=decode_sub_meta,
            prefill_sub_meta=prefill_sub_meta,
        )


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        cfg = self.tq_config
        self._mse_bytes = (
            math.ceil(head_size * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_size
        )
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        self.eager_max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_eager
        )
        # Deprecated: V56 path removed; kept for config compat (always 0).
        self.v56_max_seq_len = vllm_config.attention_config.tq_v56_max_seq_len
        self.allow_adaptive_kv_splits = (
            vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        )

    def _ensure_on_device(self, layer, device):
        """One-time migration of TQ buffers to the correct device."""
        if layer._tq_signs.device != device:
            layer._tq_signs = layer._tq_signs.to(device)
            layer._tq_centroids = layer._tq_centroids.to(device)
        if not hasattr(layer, "_tq_cached"):
            D = layer._tq_signs.shape[0]
            signs = layer._tq_signs.float()

            # WHT rotation: orthonormal + self-inverse, enabling future
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = (signs.unsqueeze(1) * H).contiguous()
            layer._tq_Pi = layer._tq_PiT.T.contiguous()

            c = layer._tq_centroids.float()
            # Precompute midpoints for threshold-based quantization
            c_sorted, _ = c.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            # Pre-cache centroids as float32 to avoid per-call conversion
            layer._tq_centroids_f32 = c.contiguous()
            # Pre-cache PiT as float32 for continuation_prefill GEMM
            # (avoids repeated BF16→FP32 cast per call)
            layer._tq_PiT_f32 = layer._tq_PiT.float().contiguous()
            layer._tq_Pi_f32 = layer._tq_Pi.float().contiguous()
            # Decode buffers are lazily allocated on first decode call.
            # With fixed NUM_KV_SPLITS (cudagraph mode), the first warmup
            # allocates them and subsequent captures reuse via buf_holder.
            layer._tq_mid_o_buf = None
            layer._tq_output_buf = None
            layer._tq_lse_buf = None
            layer._tq_q_rot_buf = None  # pre-allocated q_rot buffer
            layer._tq_cached = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.

        With stream overlap enabled, the store runs on a secondary CUDA
        stream so it can overlap with the next layer's forward pass.
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        # Use stream overlap only during pure-decode steps (no prefill tokens).
        # Prefill is latency-sensitive: overlapping store with prefill compute
        # hurts TTFT (489ms vs 338ms in benchmarks). Decode is throughput-
        # sensitive: overlap hides store latency behind attention compute.
        stream = aux_stream() if (_USE_STREAM_OVERLAP and _STEP_IS_PURE_DECODE) else None
        use_overlap = (
            stream is not None and not torch.cuda.is_current_stream_capturing()
        )
        overlap_mode = "sync"
        if use_overlap:
            # The aux stream consumes key/value generated on the current stream,
            # so it must wait for the producing work before launching the store.
            stream.wait_stream(torch.cuda.current_stream(device))

            # Launch store on secondary stream
            with torch.cuda.stream(stream):
                self._store_kv(k, v, kv_cache, slot_mapping, layer._tq_centroids, layer)
            overlap_mode = "decode_overlap"
        else:
            self._store_kv(k, v, kv_cache, slot_mapping, layer._tq_centroids, layer)
            if _STEP_IS_PURE_DECODE:
                overlap_mode = (
                    "decode_overlap_disabled"
                    if _USE_STREAM_OVERLAP
                    else "decode_sync"
                )

        record_store_call(
            phase="decode" if _STEP_IS_PURE_DECODE else "prefill",
            overlap_mode=overlap_mode,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration)
        device = q.device
        self._ensure_on_device(layer, device)
        Pi = layer._tq_Pi
        PiT = layer._tq_PiT
        centroids = layer._tq_centroids

        # Ensure any async store has completed before decode reads cache.
        # Only needed when stream overlap is active (pure decode steps).
        if (
            _USE_STREAM_OVERLAP
            and _STEP_IS_PURE_DECODE
            and not attn_metadata.is_prefill
            and not torch.cuda.is_current_stream_capturing()
        ):
            stream = aux_stream()
            if stream is not None:
                torch.cuda.current_stream(device).wait_stream(stream)
                record_store_call(waited_for_store_read=True)

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q, k, v, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.zeros(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use pre-computed decode_sub_meta from builder (avoids repeated
            # TurboQuantMetadata construction across all 80 layers per step).
            decode_meta = attn_metadata.decode_sub_meta
            if decode_meta is None:
                # Fallback: builder did not pre-compute (should not happen
                # for mixed batches, but handle gracefully).
                decode_meta = TurboQuantMetadata(
                    seq_lens=attn_metadata.seq_lens[:num_decodes],
                    slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                    block_table=attn_metadata.block_table[:num_decodes],
                    query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                    num_actual_tokens=num_decode_tokens,
                    max_query_len=1,
                    max_seq_len=attn_metadata.max_seq_len,
                    is_prefill=False,
                )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, Pi, centroids, PiT, layer
            )

            # --- Prefill portion (remaining requests) ---
            # Use pre-computed prefill_sub_meta from builder (avoids repeated
            # TurboQuantMetadata construction + tensor subtraction across all
            # 80 layers per step; also eliminates prefill_seq_lens.max().item()
            # GPU sync that existed in the original code).
            prefill_meta = attn_metadata.prefill_sub_meta
            if prefill_meta is None:
                # Fallback: builder did not pre-compute.
                prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
                prefill_meta = TurboQuantMetadata(
                    seq_lens=prefill_seq_lens,
                    slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                    block_table=attn_metadata.block_table[num_decodes:],
                    query_start_loc=(
                        attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
                    ),
                    num_actual_tokens=N - num_decode_tokens,
                    max_query_len=attn_metadata.max_query_len,
                    max_seq_len=attn_metadata.prefill_max_seq_len,
                    is_prefill=True,
                )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                centroids,
                PiT,
                layer,
            )

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        centroids: torch.Tensor,
        layer: "AttentionLayer",
    ):
        """Quantize + store via fused Triton kernel."""
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            centroids,
            layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
        )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=True,
            )
            return output

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Convert to Python lists once (single CPU-GPU sync) instead of
        # per-request .item() calls that each force a sync.
        qsl = query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if _HAS_FLASH_ATTN:
                    cu = torch.tensor(
                        [0, q_len], device=query.device, dtype=torch.int32
                    )
                    out = flash_attn_varlen_func(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                        softmax_scale=self.scale,
                        causal=True,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly to
                # avoid O(cached_len) full-dequant per continuation.
                # For large continuations, fall back to _continuation_prefill.
                cached_len = seq_len - q_len
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    # Fast path: treat each query as a decode request
                    # with incremental seq_lens for causal masking.
                    synth_seq_lens = torch.arange(
                        cached_len + 1,
                        seq_len + 1,
                        device=query.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    out = triton_turboquant_decode_attention(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=synth_bt,
                        seq_lens=synth_seq_lens,
                        Pi=Pi,
                        centroids=centroids,
                        scale=self.scale,
                        mse_bits=self.tq_config.key_mse_bits,
                        key_packed_size=self.tq_config.key_packed_size,
                        value_quant_bits=(self.tq_config.effective_value_quant_bits),
                        key_fp8=self.tq_config.key_fp8,
                        norm_correction=self.tq_config.norm_correction,
                        PiT=PiT,
                    )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    # Use pre-cached f32 tensors from layer to avoid
                    # per-call BF16→FP32 conversions.
                    _centroids_f32 = (
                        getattr(layer, "_tq_centroids_f32", None)
                        if layer is not None else None
                    )
                    _PiT_f32 = (
                        getattr(layer, "_tq_PiT_f32", None)
                        if layer is not None else None
                    )
                    _Pi_f32 = (
                        getattr(layer, "_tq_Pi_f32", None)
                        if layer is not None else None
                    )
                    out = self._continuation_prefill(
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                        PiT=PiT,
                        centroids_f32=_centroids_f32,
                        PiT_f32=_PiT_f32,
                        Pi_f32=_Pi_f32,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    def _continuation_prefill(
        self,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        centroids_f32: torch.Tensor | None = None,  # pre-cached float32 centroids
        PiT_f32: torch.Tensor | None = None,        # pre-cached float32 PiT
        Pi_f32: torch.Tensor | None = None,         # pre-cached float32 Pi
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Rotated-space FA path (when PiT is available and key_fp8 is off):
          TQ cache stores K in rotated space (K_rot = K_raw @ PiT).
          Instead of inverse-rotating cached K back to raw space (O(cached_len*D^2)),
          we rotate Q and the current key_chunk into rotated space (O(q_len*D^2)),
          then run FA entirely in rotated space.
          Since Pi is orthogonal: (Q @ PiT) @ (K @ PiT)^T = Q @ K^T — attention
          scores are identical.

        Legacy path (key_fp8=True or PiT=None):
          Dequant cached K, inverse-rotate to raw space, cat with chunk, run FA.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes
        n_centroids = self._n_centroids
        qdtype = query.dtype

        # ── Dequant cached K/V from TQ cache ──────────────────────────────
        # _tq_full_dequant_kv outputs K in rotated space (K_rot) for MSE keys.
        # Use pre-cached float32 tensors when available to avoid per-call casts.
        _centroids_f32 = centroids_f32 if centroids_f32 is not None else centroids.float()
        _PiT_f32 = PiT_f32 if PiT_f32 is not None else (PiT.float() if PiT is not None else None)
        _Pi_f32 = Pi_f32 if Pi_f32 is not None else Pi.float()

        alloc_len = math.ceil(cached_len / block_size) * block_size
        k_cached = torch.zeros(1, Hk, alloc_len, D, dtype=qdtype, device=device)
        v_cached = torch.zeros(1, Hk, alloc_len, D, dtype=qdtype, device=device)

        grid = (alloc_len, 1 * Hk)
        _tq_full_dequant_kv[grid](
            kv_cache,
            block_table,
            _centroids_f32,
            k_cached,
            v_cached,
            k_cached.stride(0),
            k_cached.stride(1),
            k_cached.stride(2),
            v_cached.stride(0),
            v_cached.stride(1),
            v_cached.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=Hk,
            MSE_BYTES=mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=self.tq_config.key_mse_bits,
            N_CENTROIDS=n_centroids,
            KEY_FP8=1 if self.tq_config.key_fp8 else 0,
            BLOCK_D=BLOCK_D,
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(device.index or 0),
            OUT_BF16=1 if qdtype == torch.bfloat16 else 0,
            num_warps=4,
        )

        # ── Build K_full and V_full for attention ─────────────────────────
        use_rotated = (not self.tq_config.key_fp8) and (PiT is not None)

        if use_rotated:
            # Rotated-space path: avoid O(cached_len * D^2) inverse-rotate GEMM.
            # cached K is already in rotated space from _tq_full_dequant_kv.
            # Rotate Q and current key_chunk into rotated space instead.
            # Cost: O(q_len * (Hq + Hk) * D^2), which is << cached_len when
            # q_len <= _CONTINUATION_DECODE_THRESHOLD * 2 (a few hundred tokens).

            # k_cached_rot: (cached_len, Hk, D) — already rotated, trim padding
            k_cached_rot = (
                k_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
            )  # (cached_len, Hk, D)

            # Rotate current key_chunk: k_chunk_rot = key_chunk @ PiT
            # key_chunk shape: (q_len, Hk, D); reshape for batched GEMM
            k_chunk_rot = (
                key_chunk.reshape(-1, D).float() @ _PiT_f32
            ).to(qdtype).reshape(q_len, Hk, D)

            # Rotate query: q_rot = query @ PiT
            # query shape: (q_len, Hq, D)
            q_rot = (
                query.reshape(-1, D).float() @ _PiT_f32
            ).to(qdtype).reshape(q_len, Hq, D)

            k_full = torch.cat([k_cached_rot, k_chunk_rot], dim=0)
            v_full = torch.cat(
                [v_cached[0, :, :cached_len, :].transpose(0, 1).contiguous(),
                 val_chunk], dim=0
            )
            q_attn = q_rot
        else:
            # Legacy path: inverse-rotate cached K back to raw space.
            # Used when key_fp8=True (Pi rotation not applied during store)
            # or PiT is unavailable.
            if not self.tq_config.key_fp8:
                k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D).float()
                k_flat = k_flat @ _Pi_f32
                k_cached_trim = (
                    k_flat.to(qdtype).reshape(Hk, cached_len, D).transpose(0, 1)
                )  # (cached_len, Hk, D)
            else:
                k_cached_trim = (
                    k_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
                )  # (cached_len, Hk, D)

            v_cached_trim = (
                v_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
            )  # (cached_len, Hk, D)

            k_full = torch.cat([k_cached_trim, key_chunk], dim=0)
            v_full = torch.cat([v_cached_trim, val_chunk], dim=0)
            q_attn = query

        # Attention: q_len queries attending to seq_len K/V with causal mask
        if _HAS_FLASH_ATTN:
            cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
            output = flash_attn_varlen_func(
                q=q_attn,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
            )
            return output
        else:
            # SDPA fallback: expand KV for GQA, build causal mask
            q_t = q_attn.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)
            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            # Build causal mask: query position p can attend to K position j
            # where j <= cached_len + p (p is 0-indexed within chunk)
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos  # (q_len, seq_len)
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=(Hk < Hq),
            )  # (1, Hq, q_len, D)
            return out[0].transpose(0, 1)  # (q_len, Hq, D)

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Grab cached decode buffers from the layer (lazily allocated).
        mid_o_buf = output_buf = lse_buf = q_rot_buf = None
        centroids_f32 = None
        if layer is not None:
            mid_o_buf = getattr(layer, "_tq_mid_o_buf", None)
            output_buf = getattr(layer, "_tq_output_buf", None)
            lse_buf = getattr(layer, "_tq_lse_buf", None)
            q_rot_buf = getattr(layer, "_tq_q_rot_buf", None)
            centroids_f32 = getattr(layer, "_tq_centroids_f32", None)

        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            q_rot_buf=q_rot_buf,
            centroids_f32=centroids_f32,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
            eager_max_num_kv_splits=self.eager_max_num_kv_splits,
            max_seq_len_hint=attn_metadata.max_seq_len,
            allow_adaptive_kv_splits=self.allow_adaptive_kv_splits,
            v56_max_seq_len=self.v56_max_seq_len,
        )
        return result

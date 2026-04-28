"""Bridge to the v3 SoA/unified TurboQuant kernels.

The local experimental path lives in this repository, but for the first
bring-up we keep the Python/Triton source of truth in the sibling
`vllm_tq_rocm_v3_sinks` checkout. This lets the backend adopt the new SoA
layout + unified attention contract without perturbing the existing
production path while we iterate on the fused HIP/Triton integration.
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch

from vllm.platforms import current_platform

_DEFAULT_SOURCE_ROOT = Path(
    "/shareddata/amd/jiangyon/vllm_tq_rocm_v3_sinks/vllm/v1/attention/ops"
)
_WARNED_HIP_V3_SCALAR_KEYS: set[str] = set()


def _warn_hip_v3_scalar_once(key: str, message: str) -> None:
    if key in _WARNED_HIP_V3_SCALAR_KEYS:
        return
    _WARNED_HIP_V3_SCALAR_KEYS.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _source_root() -> Path:
    override = os.environ.get("VLLM_TQ_FUSION_V3_HIP_SOURCE_ROOT")
    return Path(override) if override else _DEFAULT_SOURCE_ROOT


def _load_module(module_name: str, file_name: str) -> ModuleType:
    source_path = _source_root() / file_name
    if not source_path.exists():
        raise ImportError(
            "TurboQuant fusion v3 source file is missing: "
            f"{source_path}. Set VLLM_TQ_FUSION_V3_HIP_SOURCE_ROOT to a checkout "
            "that contains the upstream v3 SoA/unified kernels."
        )

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TurboQuant fusion module from {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache
def _decode_module() -> ModuleType:
    return _load_module(
        "vllm.v1.attention.ops.tq_fusion_v3_hip._external_decode",
        "triton_turboquant_decode.py",
    )


@lru_cache
def _store_module() -> ModuleType:
    return _load_module(
        "vllm.v1.attention.ops.tq_fusion_v3_hip._external_store",
        "triton_turboquant_store.py",
    )


@lru_cache
def _unified_module() -> ModuleType:
    return _load_module(
        "vllm.v1.attention.ops.tq_fusion_v3_hip._external_unified_attention",
        "triton_turboquant_unified_attention.py",
    )


@lru_cache
def _load_hip_v3_scalar():
    if os.environ.get("VLLM_TQ_FUSION_V3_DECODE_HIP_SCALAR", "0") != "1":
        return None
    if os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1":
        _warn_hip_v3_scalar_once(
            "disabled",
            "TurboQuant fusion v3 HIP scalar decode is disabled by "
            "TQ_DISABLE_HIP_SO=1; falling back to Triton v3.",
        )
        return None
    if not current_platform.is_rocm():
        return None

    so_path = Path(
        os.environ.get(
            "VLLM_TQ_FUSION_V3_HIP_SCALAR_SO_PATH",
            str(Path(__file__).with_name("hip_v3_scalar.so")),
        )
    )
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "missing",
            f"TurboQuant fusion v3 HIP scalar decode library is missing: {so_path}; "
            "falling back to Triton v3.",
        )
        return None

    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.launch_tq_v3_scalar_decode
        fn.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 13
            + [ctypes.c_float]
            + [ctypes.c_int] * 4
            + [ctypes.c_void_p]
        )
        fn.restype = None
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "load-failed",
            f"Failed to load TurboQuant fusion v3 HIP scalar decode: {exc}; "
            "falling back to Triton v3.",
        )
        return None


@lru_cache
def _load_hip_v3_mfma_qk():
    if os.environ.get("VLLM_TQ_FUSION_V3_DECODE_HIP_MFMA_QK", "0") != "1":
        return None
    if os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1":
        _warn_hip_v3_scalar_once(
            "mfma-disabled",
            "TurboQuant fusion v3 HIP MFMA-QK decode is disabled by "
            "TQ_DISABLE_HIP_SO=1; falling back to scalar/Triton v3.",
        )
        return None
    if not current_platform.is_rocm():
        return None

    so_path = Path(
        os.environ.get(
            "VLLM_TQ_FUSION_V3_HIP_MFMA_QK_SO_PATH",
            str(Path(__file__).with_name("hip_v3_mfma_qk.so")),
        )
    )
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "mfma-missing",
            f"TurboQuant fusion v3 HIP MFMA-QK decode library is missing: {so_path}; "
            "falling back to scalar/Triton v3.",
        )
        return None

    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.launch_tq_v3_mfma_qk_decode
        fn.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 13
            + [ctypes.c_float]
            + [ctypes.c_int] * 3
            + [ctypes.c_void_p]
        )
        fn.restype = None
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "mfma-load-failed",
            f"Failed to load TurboQuant fusion v3 HIP MFMA-QK decode: {exc}; "
            "falling back to scalar/Triton v3.",
        )
        return None


@lru_cache
def _load_hip_v3_flash_tq():
    if os.environ.get("VLLM_TQ_FUSION_V3_DECODE_HIP_FLASH_TQ", "0") != "1":
        return None
    if os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1":
        _warn_hip_v3_scalar_once(
            "flash-tq-disabled",
            "TurboQuant fusion v3 HIP FlashTQ decode is disabled by "
            "TQ_DISABLE_HIP_SO=1; falling back to MFMA/scalar/Triton v3.",
        )
        return None
    if not current_platform.is_rocm():
        return None

    so_path = Path(
        os.environ.get(
            "VLLM_TQ_FUSION_V3_HIP_FLASH_TQ_SO_PATH",
            str(Path(__file__).with_name("hip_v3_flash_tq.so")),
        )
    )
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "flash-tq-missing",
            f"TurboQuant fusion v3 HIP FlashTQ decode library is missing: {so_path}; "
            "falling back to MFMA/scalar/Triton v3.",
        )
        return None

    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.launch_tq_v3_flash_tq_decode
        fn.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 13
            + [ctypes.c_float]
            + [ctypes.c_int] * 3
            + [ctypes.c_void_p]
        )
        fn.restype = None
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "flash-tq-load-failed",
            f"Failed to load TurboQuant fusion v3 HIP FlashTQ decode: {exc}; "
            "falling back to MFMA/scalar/Triton v3.",
        )
        return None


@lru_cache
def _load_hip_v3_v136_mfma_from_path(so_path: str):
    try:
        lib = ctypes.CDLL(so_path)
        fn = lib.launch_tq_v3_v136_mfma_decode
        fn.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 13
            + [ctypes.c_float]
            + [ctypes.c_int] * 3
            + [ctypes.c_void_p]
        )
        fn.restype = None
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "v136-mfma-load-failed",
            f"Failed to load TurboQuant fusion v3 HIP v136-MFMA decode: {exc}; "
            "falling back to FlashTQ/MFMA/scalar/Triton v3.",
        )
        return None


def _load_hip_v3_v136_mfma():
    if os.environ.get("VLLM_TQ_FUSION_V3_DECODE_HIP_V136_MFMA", "0") != "1":
        return None
    if os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1":
        _warn_hip_v3_scalar_once(
            "v136-mfma-disabled",
            "TurboQuant fusion v3 HIP v136-MFMA decode is disabled by "
            "TQ_DISABLE_HIP_SO=1; falling back to FlashTQ/MFMA/scalar/Triton v3.",
        )
        return None
    if not current_platform.is_rocm():
        return None

    so_path = Path(
        os.environ.get(
            "VLLM_TQ_FUSION_V3_HIP_V136_MFMA_SO_PATH",
            str(Path(__file__).with_name("hip_v3_v136_mfma.so")),
        )
    )
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "v136-mfma-missing",
            f"TurboQuant fusion v3 HIP v136-MFMA decode library is missing: "
            f"{so_path}; falling back to FlashTQ/MFMA/scalar/Triton v3.",
        )
        return None
    return _load_hip_v3_v136_mfma_from_path(str(so_path))


def _dtype_code(dtype: torch.dtype) -> int | None:
    if dtype is torch.bfloat16:
        return 0
    if dtype is torch.float16:
        return 1
    if dtype is torch.float32:
        return 2
    return None


def _hip_v3_scalar_safe(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    value_quant_bits: int,
    key_fp8: bool,
) -> bool:
    return (
        current_platform.is_rocm()
        and query.dim() == 3
        and query.shape[-1] == 128
        and kv_cache.dtype == torch.uint8
        and kv_cache.dim() == 4
        and block_table.dtype == torch.int32
        and seq_lens.dtype == torch.int32
        and centroids.dtype == torch.float32
        and (not key_fp8)
        and mse_bits == 4
        and value_quant_bits == 4
        and _dtype_code(query.dtype) is not None
    )


def _rotated_query_fp32(
    query: torch.Tensor,
    Pi: torch.Tensor,
    PiT: torch.Tensor | None,
) -> torch.Tensor:
    if PiT is None:
        PiT = Pi.T.contiguous()
    PiT_f32 = PiT if PiT.dtype == torch.float32 else PiT.to(torch.float32)
    if not PiT_f32.is_contiguous():
        PiT_f32 = PiT_f32.contiguous()
    return (query.float() @ PiT_f32).contiguous()


def _decode_num_splits(
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    max_num_kv_splits: int,
) -> int:
    max_seq_len_hint = max_seq_len if max_seq_len > 0 else int(seq_lens.max().item())
    use_3d = max_seq_len_hint >= 1024 and max_num_kv_splits > 1
    num_splits = max(1, max_num_kv_splits if use_3d else 1)
    max_possible_splits = max(1, math.ceil(max_seq_len_hint / block_size))
    return min(num_splits, max_possible_splits)


def _maybe_hip_v3_mfma_like_decode(
    fn,
    *args,
    q_rot_dtype: torch.dtype | None = None,
    require_query_dtype: torch.dtype | None = None,
    **kwargs,
):
    if fn is None:
        return None

    query = kwargs["query"]
    kv_cache = kwargs["kv_cache"]
    block_table = kwargs["block_table"]
    seq_lens = kwargs["seq_lens"]
    Pi = kwargs["Pi"]
    centroids = kwargs["centroids"]
    scale = float(kwargs["scale"])
    mse_bits = int(kwargs["mse_bits"])
    value_quant_bits = int(kwargs["value_quant_bits"])
    key_fp8 = bool(kwargs.get("key_fp8", False))
    PiT = kwargs.get("PiT", None)
    output_buf = kwargs.get("output_buf", None)
    max_seq_len = int(kwargs.get("max_seq_len", 0) or 0)
    max_num_kv_splits = int(kwargs.get("max_num_kv_splits", 32))

    if require_query_dtype is not None and query.dtype != require_query_dtype:
        return None
    if not _hip_v3_scalar_safe(
        query, kv_cache, block_table, seq_lens, centroids, mse_bits,
        value_quant_bits, key_fp8,
    ):
        return None

    B, Hq, _ = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    if kv_group_size != 8:
        return None

    q_rot = _rotated_query_fp32(query, Pi, PiT)
    if q_rot_dtype is not None:
        q_rot = q_rot.to(q_rot_dtype)
    q_rot = q_rot.contiguous()
    output = output_buf[:B] if output_buf is not None else torch.empty_like(query)
    if not output.is_contiguous():
        output = output.contiguous()
    out_dtype = _dtype_code(output.dtype)
    if out_dtype not in (0, 1):
        return None

    num_splits = _decode_num_splits(
        seq_lens, block_size, max_seq_len, max_num_kv_splits
    )
    mid_o = torch.empty(
        (B, Hq, num_splits, query.shape[-1] + 1),
        dtype=torch.float32,
        device=query.device,
    )

    stream_ptr = torch.cuda.current_stream(query.device).cuda_stream
    fn(
        ctypes.c_void_p(q_rot.data_ptr()),
        ctypes.c_void_p(kv_cache.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(seq_lens.data_ptr()),
        ctypes.c_void_p(centroids.data_ptr()),
        ctypes.c_void_p(mid_o.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        Hk,
        block_size,
        num_splits,
        kv_group_size,
        scale,
        out_dtype,
        B,
        Hq,
        ctypes.c_void_p(stream_ptr),
    )
    return output


def _maybe_hip_v3_v136_mfma_decode(*args, **kwargs):
    return _maybe_hip_v3_mfma_like_decode(
        _load_hip_v3_v136_mfma(),
        *args,
        q_rot_dtype=torch.bfloat16,
        require_query_dtype=torch.bfloat16,
        **kwargs,
    )


def _maybe_hip_v3_flash_tq_decode(*args, **kwargs):
    return _maybe_hip_v3_mfma_like_decode(
        _load_hip_v3_flash_tq(), *args, **kwargs)


def _maybe_hip_v3_mfma_qk_decode(*args, **kwargs):
    return _maybe_hip_v3_mfma_like_decode(
        _load_hip_v3_mfma_qk(), *args, **kwargs)


def _maybe_hip_v3_scalar_decode(*args, **kwargs):
    fn = _load_hip_v3_scalar()
    if fn is None:
        return None

    query = kwargs["query"]
    kv_cache = kwargs["kv_cache"]
    block_table = kwargs["block_table"]
    seq_lens = kwargs["seq_lens"]
    Pi = kwargs["Pi"]
    centroids = kwargs["centroids"]
    scale = float(kwargs["scale"])
    mse_bits = int(kwargs["mse_bits"])
    value_quant_bits = int(kwargs["value_quant_bits"])
    key_fp8 = bool(kwargs.get("key_fp8", False))
    PiT = kwargs.get("PiT", None)
    output_buf = kwargs.get("output_buf", None)
    max_seq_len = int(kwargs.get("max_seq_len", 0) or 0)
    max_num_kv_splits = int(kwargs.get("max_num_kv_splits", 32))

    if not _hip_v3_scalar_safe(
        query, kv_cache, block_table, seq_lens, centroids, mse_bits,
        value_quant_bits, key_fp8,
    ):
        return None

    # Scalar v3 validates the HIP SoA decode contract first. Fused Q rotation
    # moves into HIP in the following MFMA phase.
    q_rot = _rotated_query_fp32(query, Pi, PiT).to(query.dtype).contiguous()
    output = output_buf[: query.shape[0]] if output_buf is not None else torch.empty_like(query)
    if not output.is_contiguous():
        output = output.contiguous()

    B, Hq, _ = q_rot.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    num_splits = _decode_num_splits(
        seq_lens, block_size, max_seq_len, max_num_kv_splits
    )

    mid_o = torch.empty(
        (B, Hq, num_splits, query.shape[-1] + 1),
        dtype=torch.float32,
        device=query.device,
    )
    q_dtype = _dtype_code(q_rot.dtype)
    out_dtype = _dtype_code(output.dtype)
    if q_dtype is None or out_dtype is None:
        return None

    stream_ptr = torch.cuda.current_stream(query.device).cuda_stream
    fn(
        ctypes.c_void_p(q_rot.data_ptr()),
        ctypes.c_void_p(kv_cache.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(seq_lens.data_ptr()),
        ctypes.c_void_p(centroids.data_ptr()),
        ctypes.c_void_p(mid_o.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        Hk,
        block_size,
        num_splits,
        kv_group_size,
        scale,
        q_dtype,
        out_dtype,
        B,
        Hq,
        ctypes.c_void_p(stream_ptr),
    )
    return output


def _use_fp8_e4b15(device: int = 0) -> int:
    return _decode_module()._use_fp8_e4b15(device)


_tq_full_dequant_kv = _decode_module()._tq_full_dequant_kv


def triton_turboquant_store(*args, **kwargs):
    return _store_module().triton_turboquant_store(*args, **kwargs)


def triton_turboquant_unified_attention(*args, **kwargs):
    return _unified_module().triton_turboquant_unified_attention(*args, **kwargs)


def triton_turboquant_decode_attention_v3(*args, **kwargs):
    hip_out = _maybe_hip_v3_v136_mfma_decode(*args, **kwargs)
    if hip_out is not None:
        return hip_out
    hip_out = _maybe_hip_v3_flash_tq_decode(*args, **kwargs)
    if hip_out is not None:
        return hip_out
    hip_out = _maybe_hip_v3_mfma_qk_decode(*args, **kwargs)
    if hip_out is not None:
        return hip_out
    hip_out = _maybe_hip_v3_scalar_decode(*args, **kwargs)
    if hip_out is not None:
        return hip_out
    return _unified_module().triton_turboquant_decode_attention_v3(*args, **kwargs)

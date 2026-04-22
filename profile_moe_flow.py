#!/usr/bin/env python3
"""Profile MoE execution flow: capture kernel calls for quant+sort+GEMM."""
import os
import torch
import json

os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

from vllm import LLM, SamplingParams

LOG_PATH = "/data/jiangyon/.cursor/debug-88ac24.log"

def log(msg, data=None):
    entry = {"sessionId":"88ac24","location":"profile_moe_flow.py",
             "message":msg,"data":data or {},"timestamp":__import__('time').time_ns()//1000000}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

# Monkey-patch aiter to trace MoE kernel calls
import aiter
_orig_fused_moe = None
_orig_ck_stage1 = None
_orig_ck_stage2 = None
_orig_cktile_gemm1 = None
_orig_cktile_gemm2 = None

def _trace_fused_moe(*args, **kwargs):
    log("aiter.fused_moe_ called", {
        "args_types": [str(type(a).__name__) for a in args[:5]],
        "kwargs_keys": list(kwargs.keys()),
    })
    return _orig_fused_moe(*args, **kwargs)

# Patch the actual ops
from aiter.fused_moe import fused_moe_2stages as _orig_2stages
from aiter.fused_moe import moe_sorting as _orig_sorting
from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort as _orig_quant_sort

_call_count = {"sorting": 0, "quant_sort": 0, "stage1": 0, "stage2": 0, "fused_moe_2stages": 0}

_orig_sorting_fn = _orig_sorting
def _traced_sorting(*args, **kwargs):
    _call_count["sorting"] += 1
    if _call_count["sorting"] <= 3:
        log("moe_sorting called", {
            "count": _call_count["sorting"],
            "topk_ids_shape": list(args[0].shape) if len(args) > 0 else "?",
        })
    return _orig_sorting_fn(*args, **kwargs)

_orig_quant_sort_fn = _orig_quant_sort
def _traced_quant_sort(*args, **kwargs):
    _call_count["quant_sort"] += 1
    if _call_count["quant_sort"] <= 3:
        x = args[0]
        log("fused_dynamic_mxfp4_quant_moe_sort called", {
            "count": _call_count["quant_sort"],
            "x_shape": list(x.shape),
            "x_dtype": str(x.dtype),
            "token_num": kwargs.get("token_num"),
            "topk": kwargs.get("topk"),
        })
    result = _orig_quant_sort_fn(*args, **kwargs)
    if _call_count["quant_sort"] <= 3:
        log("fused_dynamic_mxfp4_quant_moe_sort result", {
            "count": _call_count["quant_sort"],
            "fp4_shape": list(result[0].shape),
            "fp4_dtype": str(result[0].dtype),
            "scale_shape": list(result[1].shape),
            "scale_dtype": str(result[1].dtype),
        })
    return result

_orig_2stages_fn = _orig_2stages
def _traced_2stages(*args, **kwargs):
    _call_count["fused_moe_2stages"] += 1
    if _call_count["fused_moe_2stages"] <= 3:
        hs = args[0]
        log("fused_moe_2stages called", {
            "count": _call_count["fused_moe_2stages"],
            "hidden_states_shape": list(hs.shape),
            "hidden_states_dtype": str(hs.dtype),
            "topk": args[3] if len(args) > 3 else kwargs.get("topk"),
            "quant_type": str(kwargs.get("quant_type", args[12] if len(args) > 12 else "?")),
            "activation": str(kwargs.get("activation", "?")),
        })
    return _orig_2stages_fn(*args, **kwargs)

# Patch CK stage1/stage2
_orig_ck_stage1_fn = aiter.ck_moe_stage1_fwd
def _traced_ck_stage1(*args, **kwargs):
    _call_count["stage1"] += 1
    if _call_count["stage1"] <= 3:
        hs = args[0]
        log("ck_moe_stage1_fwd called", {
            "count": _call_count["stage1"],
            "input_shape": list(hs.shape),
            "input_dtype": str(hs.dtype),
            "a1_scale": str(kwargs.get("a1_scale").dtype) if kwargs.get("a1_scale") is not None else "None",
            "quant_type": str(kwargs.get("quant_type", "?")),
        })
    return _orig_ck_stage1_fn(*args, **kwargs)

_orig_ck_stage2_fn = aiter.ck_moe_stage2_fwd
def _traced_ck_stage2(*args, **kwargs):
    _call_count["stage2"] += 1
    if _call_count["stage2"] <= 3:
        hs = args[0]
        log("ck_moe_stage2_fwd called", {
            "count": _call_count["stage2"],
            "input_shape": list(hs.shape),
            "input_dtype": str(hs.dtype),
            "a2_scale": str(kwargs.get("a2_scale").dtype) if kwargs.get("a2_scale") is not None else "None",
        })
    return _orig_ck_stage2_fn(*args, **kwargs)

_orig_cktile_gemm1_fn = aiter.moe_cktile2stages_gemm1
def _traced_cktile_gemm1(*args, **kwargs):
    _call_count["stage1"] += 1
    if _call_count["stage1"] <= 3:
        log("moe_cktile2stages_gemm1 called", {
            "count": _call_count["stage1"],
            "XQ_dtype": str(args[0].dtype),
            "XQ_shape": list(args[0].shape),
            "Y_dtype": str(args[2].dtype),
            "x_scale": str(args[10].dtype) if args[10] is not None else "None",
            "w_scale": str(args[11].dtype) if args[11] is not None else "None",
        })
    return _orig_cktile_gemm1_fn(*args, **kwargs)

_orig_cktile_gemm2_fn = aiter.moe_cktile2stages_gemm2
def _traced_cktile_gemm2(*args, **kwargs):
    _call_count["stage2"] += 1
    if _call_count["stage2"] <= 3:
        log("moe_cktile2stages_gemm2 called", {
            "count": _call_count["stage2"],
            "a2_dtype": str(args[0].dtype),
            "a2_shape": list(args[0].shape),
            "out_dtype": str(args[2].dtype),
        })
    return _orig_cktile_gemm2_fn(*args, **kwargs)

# Apply patches
import aiter.fused_moe as fm
fm.moe_sorting = _traced_sorting
fm.fused_dynamic_mxfp4_quant_moe_sort = _traced_quant_sort
fm.fused_moe_2stages = _traced_2stages
aiter.ck_moe_stage1_fwd = _traced_ck_stage1
aiter.ck_moe_stage2_fwd = _traced_ck_stage2
aiter.moe_cktile2stages_gemm1 = _traced_cktile_gemm1
aiter.moe_cktile2stages_gemm2 = _traced_cktile_gemm2

log("=== Starting separated mode profile ===")

if __name__ != '__main__':
    import sys
    sys.exit(0)

llm = LLM(
    model="/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2",
    trust_remote_code=True,
    max_model_len=256,
    gpu_memory_utilization=0.35,
    enforce_eager=True,
)

log("=== Model loaded, running inference ===")

out = llm.generate(
    ["1+1="],
    SamplingParams(max_tokens=5, temperature=0),
)
text = out[0].outputs[0].text
log("=== Inference result ===", {"text": text})

log("=== Call counts ===", _call_count)
print(f"Result: {text}")
print(f"Call counts: {_call_count}")

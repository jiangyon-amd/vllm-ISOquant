#!/usr/bin/env python3
"""
Profile MoE 30B model: separated vs fused attention path.
Uses manual HIP event timing to measure per-step GPU time.
"""
import os, sys, json, argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["separated", "fused"], required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="/data/jiangyon/vllm_rotation/profiling_results")
    args = parser.parse_args()

    model_path = "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2"

    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    os.environ["VLLM_ROCM_USE_AITER_FP4_ASM_GEMM"] = "1"
    os.environ["VLLM_MOE_UNIFIED_KERNEL"] = "0"
    os.environ["VLLM_MOE_FUSED_ROTATION"] = "0"

    if args.mode == "fused":
        os.environ["VLLM_USE_FUSED_ROTATION_QUANT"] = "1"
    else:
        os.environ["VLLM_USE_FUSED_ROTATION_QUANT"] = "0"

    from vllm import LLM, SamplingParams

    print(f"[profile] mode={args.mode}, batch_size={args.batch_size}, decode_steps={args.decode_steps}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=512,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.decode_steps)
    prompts = [f"Hello, tell me about topic {i}." for i in range(args.batch_size)]

    # Warmup
    print(f"[profile] Warming up...")
    for _ in range(args.warmup_steps):
        llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    print("[profile] Warmup done. Ready for rocprof.")

    # Signal that we're about to do the profiled run
    print("[profile] PROFILED_RUN_START", flush=True)
    torch.cuda.synchronize()
    
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    
    print("[profile] PROFILED_RUN_END", flush=True)
    print(f"[profile] Generated {sum(len(o.outputs[0].token_ids) for o in outputs)} tokens total")


if __name__ == "__main__":
    main()

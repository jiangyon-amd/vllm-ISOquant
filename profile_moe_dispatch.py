#!/usr/bin/env python3
"""
Profile MoE dispatch: separated vs moe-fused
Runs a few forward passes through MoE layer and captures PyTorch profiler trace.
"""
import os, sys, time, torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "4")
os.environ["TRITON_CACHE_DIR"] = "/tmp/jiangyon_triton_cache"

sys.path.insert(0, "/data/jiangyon/vllm_rotation")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["separated", "moe-fused"], required=True)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iters", type=int, default=20)
args = parser.parse_args()

from aiter.fused_moe import moe_sorting, fused_moe
from aiter import ActivationType, QuantType
from aiter.utility import dtypes

device = "cuda"
torch.manual_seed(42)

# Qwen3-30B-A3B shapes
M, K, E, topk = 1, 2048, 128, 8
RS = 128
N_experts_w1 = 2 * 1536  # intermediate_size * 2 for gate+up
N_experts_w2 = K  # hidden_size

print(f"Mode: {args.mode}, M={M}, K={K}, E={E}, topk={topk}")

# Create fake model weights (fp4 packed)
w1 = torch.randint(0, 255, (E, N_experts_w1, K // 2), dtype=torch.uint8, device=device)
w2 = torch.randint(0, 255, (E, N_experts_w2, 1536 // 2), dtype=torch.uint8, device=device)
w1_scale = torch.randint(100, 200, (E, N_experts_w1, K // 32), dtype=torch.uint8, device=device)
w2_scale = torch.randint(100, 200, (E, N_experts_w2, 1536 // 32), dtype=torch.uint8, device=device)

# View as proper types
w1_fp4 = w1.view(torch.float4_e2m1fn_x2)
w2_fp4 = w2.view(torch.float4_e2m1fn_x2)
w1_sc = w1_scale.view(torch.float8_e8m0fnu)
w2_sc = w2_scale.view(torch.float8_e8m0fnu)

# Rotation matrix
rot = torch.randn(RS, RS, dtype=torch.bfloat16, device=device) * 0.05

# Input
x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
topk_weights = torch.ones(M, topk, dtype=torch.float32, device=device) / topk
topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)

if args.mode == "moe-fused":
    rotation = rot
    rotation_size = RS
else:
    rotation = None
    rotation_size = 0

# Warmup
print(f"Warmup ({args.warmup} iters)...")
for _ in range(args.warmup):
    if args.mode == "separated" and rotation is None:
        # Simulate separated: rotate then call fused_moe without rotation
        x_rot = (x.reshape(-1, K // RS, RS) @ rot).reshape(M, K)
        fused_moe(x_rot, w1_fp4, w2_fp4, topk_weights, topk_ids,
                  expert_mask=None, activation=ActivationType.Silu,
                  quant_type=QuantType.per_1x32,
                  w1_scale=w1_sc, w2_scale=w2_sc)
    else:
        fused_moe(x, w1_fp4, w2_fp4, topk_weights, topk_ids,
                  expert_mask=None, activation=ActivationType.Silu,
                  quant_type=QuantType.per_1x32,
                  w1_scale=w1_sc, w2_scale=w2_sc,
                  rotation=rotation, rotation_size=rotation_size)
torch.cuda.synchronize()

# Profile
print(f"Profiling ({args.iters} iters)...")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(args.iters):
        if args.mode == "separated" and rotation is None:
            x_rot = (x.reshape(-1, K // RS, RS) @ rot).reshape(M, K)
            fused_moe(x_rot, w1_fp4, w2_fp4, topk_weights, topk_ids,
                      expert_mask=None, activation=ActivationType.Silu,
                      quant_type=QuantType.per_1x32,
                      w1_scale=w1_sc, w2_scale=w2_sc)
        else:
            fused_moe(x, w1_fp4, w2_fp4, topk_weights, topk_ids,
                      expert_mask=None, activation=ActivationType.Silu,
                      quant_type=QuantType.per_1x32,
                      w1_scale=w1_sc, w2_scale=w2_sc,
                      rotation=rotation, rotation_size=rotation_size)
    torch.cuda.synchronize()

# Print GPU kernel summary
print(f"\n{'='*80}")
print(f"  GPU Kernel Summary — {args.mode} (top 30 by total CUDA time)")
print(f"{'='*80}")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

# Save trace
trace_path = f"/data/jiangyon/vllm_rotation/bench_results/profile_{args.mode}.json"
prof.export_chrome_trace(trace_path)
print(f"\nTrace saved: {trace_path}")

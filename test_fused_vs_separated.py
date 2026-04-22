#!/usr/bin/env python3
"""Unit test: fused gluon v2 vs separated rotation+quant+GEMM."""
import os
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ROCM_USE_AITER"] = "1"
os.environ["VLLM_ROCM_USE_AITER_FP4_ASM_GEMM"] = "1"

import torch
import math
from safetensors import safe_open

MODEL_DIR = "/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-hadamard-r128"


def load_rotation(layer_name="model.layers.0.self_attn.q_proj"):
    f = safe_open(f"{MODEL_DIR}/model-00001-of-00002.safetensors", framework="pt")
    rot_raw = f.get_tensor(f"{layer_name}.input_rotation")
    if rot_raw.dtype == torch.bool:
        rot_int8 = rot_raw.to(torch.int8) * 2 - 1
    else:
        rot_int8 = rot_raw
    rot_bf16 = (rot_int8.to(torch.float) / math.sqrt(rot_int8.shape[0])).to(torch.bfloat16)
    return rot_bf16.cuda()


def load_weight_and_scale(layer_name="model.layers.0.self_attn.q_proj"):
    f = safe_open(f"{MODEL_DIR}/model-00001-of-00002.safetensors", framework="pt")
    w = f.get_tensor(f"{layer_name}.weight").cuda()
    ws = f.get_tensor(f"{layer_name}.weight_scale").cuda()
    return w, ws


def test_rotation_only():
    """Test 1: rotation matmul vs fused kernel rotation (fp4 output comparison)."""
    print("\n" + "="*60)
    print("TEST 1: Rotation + Quant (fp4 output comparison)")
    print("="*60)

    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import fused_rot_quant_gluon
    from aiter import per_1x32_f4_quant_hip

    rot = load_rotation()
    RS = rot.shape[0]

    for M in [1, 4, 32, 128]:
        x = torch.randn(M, 4096, dtype=torch.bfloat16, device="cuda")

        # Separated: rotation matmul + hip quant
        x_rot = x.reshape(M, -1, RS) @ rot
        x_rot = x_rot.reshape(M, 4096)
        x_q_sep, x_s_sep = per_1x32_f4_quant_hip(x_rot, shuffle=False)

        # Fused: gluon kernel (raw scales)
        fp4 = torch.empty((M, 2048), dtype=torch.uint8, device="cuda")
        sc = torch.empty((M, 128), dtype=torch.uint8, device="cuda")
        x_q_fus, x_s_fus = fused_rot_quant_gluon(x, rot, rotation_size=RS,
                                            fp4_out=fp4, scales_out=sc,
                                            shuffle_scales=False)

        fp4_match = (x_q_sep.view(torch.uint8) == x_q_fus).all().item()
        sc_match = (x_s_sep.view(torch.uint8)[:M, :128] == x_s_fus[:M, :128]).all().item()
        print(f"  M={M:>3}: fp4={'PASS' if fp4_match else 'FAIL'}  "
              f"scales={'PASS' if sc_match else 'FAIL'}")


def test_rotation_quant_shuffled():
    """Test 2: shuffled scales comparison."""
    print("\n" + "="*60)
    print("TEST 2: Rotation + Quant (shuffled scales comparison)")
    print("="*60)

    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import fused_rot_quant_gluon
    from aiter import per_1x32_f4_quant_hip

    rot = load_rotation()
    RS = rot.shape[0]

    for M in [1, 4, 32, 128]:
        x = torch.randn(M, 4096, dtype=torch.bfloat16, device="cuda")

        # Separated: rotation matmul + hip quant (shuffled)
        x_rot = x.reshape(M, -1, RS) @ rot
        x_rot = x_rot.reshape(M, 4096)
        x_q_sep, x_s_sep = per_1x32_f4_quant_hip(x_rot, shuffle=True)

        # Fused: gluon kernel (shuffled scales)
        sn_pad = (4096 // 32 + 7) // 8 * 8
        sm_pad = (M + 255) // 256 * 256
        fp4 = torch.empty((M, 2048), dtype=torch.uint8, device="cuda")
        sc = torch.empty((sm_pad, sn_pad), dtype=torch.uint8, device="cuda")
        x_q_fus, x_s_fus = fused_rot_quant_gluon(x, rot, rotation_size=RS,
                                            fp4_out=fp4, scales_out=sc,
                                            shuffle_scales=True)

        fp4_match = (x_q_sep.view(torch.uint8) == x_q_fus).all().item()
        # Compare scales in same shape
        sc_match = (x_s_sep.view(torch.uint8) == x_s_fus[:x_s_sep.shape[0], :x_s_sep.shape[1]]).all().item()
        print(f"  M={M:>3}: fp4={'PASS' if fp4_match else 'FAIL'}  "
              f"shuffled_scales={'PASS' if sc_match else 'FAIL'}  "
              f"sep_sc_shape={x_s_sep.shape} fus_sc_shape={x_s_fus.shape}")


def test_full_gemm():
    """Test 3: full rotation + quant + GEMM end-to-end."""
    print("\n" + "="*60)
    print("TEST 3: Full GEMM (rotation + quant + matmul)")
    print("="*60)

    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import fused_rot_quant_gluon
    from aiter import gemm_a4w4, per_1x32_f4_quant_hip
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    rot = load_rotation()
    w_raw, ws_raw = load_weight_and_scale()
    RS = rot.shape[0]
    N, K_packed = w_raw.shape
    K = K_packed * 2

    print(f"  Weight: N={N}, K_packed={K_packed}, K={K}")

    for M in [1, 4, 32]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        # === Separated path: rotation -> quant -> GEMM (using gemm_afp4wfp4, no shuffle needed) ===
        x_rot = x.reshape(M, -1, RS) @ rot
        x_rot = x_rot.reshape(M, K)
        x_q_sep, x_s_sep = dynamic_mxfp4_quant(x_rot)
        y_sep = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        gemm_afp4wfp4(x_q_sep, w_raw, x_s_sep, ws_raw.T, torch.bfloat16, y_sep)

        # === Fused path: fused rot+quant -> GEMM (using gemm_afp4wfp4, no shuffle) ===
        fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device="cuda")
        sc = torch.empty((M, K // 32), dtype=torch.uint8, device="cuda")
        x_q_fus, x_s_fus = fused_rot_quant_gluon(x, rot, rotation_size=RS,
                                            fp4_out=fp4, scales_out=sc,
                                            shuffle_scales=False)
        x_q_fus_v = x_q_fus.view(torch.float4_e2m1fn_x2)
        x_s_fus_v = x_s_fus.view(torch.float8_e8m0fnu)
        y_fus = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        gemm_afp4wfp4(x_q_fus_v, w_raw, x_s_fus_v, ws_raw.T, torch.bfloat16, y_fus)

        diff = (y_sep - y_fus).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_err = (diff / (y_sep.abs() + 1e-6)).mean().item()
        close = torch.allclose(y_sep, y_fus, atol=0.01, rtol=0.01)
        print(f"  M={M:>3}: max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
              f"rel_err={rel_err:.6f}  {'PASS' if close else 'FAIL'}")
        if not close:
            print(f"         sep sample: {y_sep[0,:5].tolist()}")
            print(f"         fus sample: {y_fus[0,:5].tolist()}")


def test_gemm_with_asm():
    """Test 4: full GEMM with ASM backend (shuffled weights, like vLLM uses)."""
    print("\n" + "="*60)
    print("TEST 4: Full GEMM with ASM backend (shuffled weights)")
    print("="*60)

    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import fused_rot_quant_gluon
    from aiter import gemm_a4w4, per_1x32_f4_quant_hip

    rot = load_rotation()
    w_raw, ws_raw = load_weight_and_scale()
    RS = rot.shape[0]
    N, K_packed = w_raw.shape
    K = K_packed * 2

    # Shuffle weight and weight_scale (like process_weights_after_loading)
    from vllm._aiter_ops import shuffle_weight
    sm, sn = ws_raw.shape
    ws_shuf = ws_raw.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    ws_shuf = ws_shuf.permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm, sn)
    w_shuf = shuffle_weight(w_raw, layout=(16, 16))

    print(f"  Weight: N={N}, K_packed={K_packed}, K={K}")
    print(f"  Weight shuffled: {w_shuf.shape}, Scale shuffled: {ws_shuf.shape}")

    for M in [1, 4, 32]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        # === Separated: rotation -> per_1x32_f4_quant_hip(shuffle=True) -> gemm_a4w4 ===
        x_rot = x.reshape(M, -1, RS) @ rot
        x_rot = x_rot.reshape(M, K)
        x_q_sep, x_s_sep = per_1x32_f4_quant_hip(x_rot, shuffle=True)
        y_sep = torch.empty((M + 31) // 32 * 32, N, device="cuda", dtype=torch.bfloat16)
        w_v = w_shuf.view(torch.float4_e2m1fn_x2)
        ws_v = ws_shuf.view(torch.float8_e8m0fnu)
        gemm_a4w4(x_q_sep, w_v, x_s_sep, ws_v.view(x_s_sep.dtype), y_sep, bpreshuffle=True)
        y_sep = y_sep[:M]

        # === Fused: fused rot+quant(shuffle=True) -> gemm_a4w4 ===
        sn_pad = (K // 32 + 7) // 8 * 8
        sm_pad = (M + 255) // 256 * 256
        fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device="cuda")
        sc = torch.empty((sm_pad, sn_pad), dtype=torch.uint8, device="cuda")
        x_q_fus, x_s_fus = fused_rot_quant_gluon(x, rot, rotation_size=RS,
                                            fp4_out=fp4, scales_out=sc,
                                            shuffle_scales=True)
        x_q_fus_v = x_q_fus.view(torch.float4_e2m1fn_x2)
        x_s_fus_v = x_s_fus.view(torch.float8_e8m0fnu)
        y_fus = torch.empty((M + 31) // 32 * 32, N, device="cuda", dtype=torch.bfloat16)
        gemm_a4w4(x_q_fus_v, w_v, x_s_fus_v, ws_v.view(x_s_fus_v.dtype), y_fus, bpreshuffle=True)
        y_fus = y_fus[:M]

        diff = (y_sep - y_fus).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        close = torch.allclose(y_sep, y_fus, atol=0.5)
        print(f"  M={M:>3}: max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
              f"{'PASS' if close else 'FAIL'}")
        if not close:
            print(f"         sep range: [{y_sep.min().item():.4f}, {y_sep.max().item():.4f}]")
            print(f"         fus range: [{y_fus.min().item():.4f}, {y_fus.max().item():.4f}]")
            print(f"         sep sample: {y_sep[0,:5].tolist()}")
            print(f"         fus sample: {y_fus[0,:5].tolist()}")


if __name__ == "__main__":
    print("Model:", MODEL_DIR)
    torch.manual_seed(42)

    test_rotation_only()
    test_rotation_quant_shuffled()
    test_full_gemm()
    test_gemm_with_asm()

    print("\n" + "="*60)
    print("All tests completed.")
    print("="*60)

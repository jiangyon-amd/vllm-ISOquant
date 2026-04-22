#!/usr/bin/env python3
"""
Debug v2 MoE kernel FP4 quantization bug.

Compares:
1. test_mfma_only acc values (verified correct)
2. v2_debug acc values + quantization intermediates
3. Python-side FP4 simulation

Diagnoses whether bug is in MFMA or quantization.
"""
import torch
import ctypes
import struct
import os
import subprocess
import sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"

def compile_kernel(hip_file, so_file):
    """Compile HIP kernel to .so if needed."""
    hip_path = os.path.join(DIR, hip_file)
    so_path = os.path.join(DIR, so_file)
    if os.path.exists(so_path) and os.path.getmtime(so_path) > os.path.getmtime(hip_path):
        print(f"  {so_file} is up-to-date")
        return so_path
    print(f"  Compiling {hip_file} -> {so_file} ...")
    cmd = [
        "hipcc", "-shared", "-fPIC", "-O2",
        "--offload-arch=gfx950",
        "-o", so_path, hip_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COMPILE ERROR:\n{result.stderr}")
        sys.exit(1)
    print(f"  Compiled OK")
    return so_path


def float_to_uint32(f):
    return struct.unpack('I', struct.pack('f', f))[0]


def uint32_to_float(u):
    return struct.unpack('f', struct.pack('I', u))[0]


def fp4_e2m1_encode(val):
    """Python simulation of FP4 E2M1 quantization (for reference)."""
    if val == 0:
        return 0
    sign = 1 if val < 0 else 0
    absval = abs(val)
    # E2M1: bias=1, max=6.0, min_subnormal=0.5
    if absval < 0.25:  # below half of min subnormal
        return sign << 3
    if absval >= 6.0:
        return (sign << 3) | 0b111  # max value = 6.0
    # Normal: exp bias=1, 1 mantissa bit
    # Values: 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    best_idx = 0
    best_err = float('inf')
    for idx, ref in enumerate(table):
        err = abs(absval - ref)
        if err < best_err:
            best_err = err
            best_idx = idx
    return (sign << 3) | best_idx


def main():
    print("=" * 70)
    print("MFMA v2 MoE Kernel FP4 Quantization Debug")
    print("=" * 70)

    # --- Setup ---
    RS = 128
    M = 1
    K = 128  # 1 rotation chunk for simplicity

    rot = torch.eye(RS, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    for i in range(32):
        x[0, i] = float(i + 1)

    print(f"\nSetup: M={M}, K={K}, RS={RS}")
    print(f"Input x[0, 0:32] = {x[0, :32].tolist()}")
    print(f"Rotation = identity (128x128)")

    # --- Step 1: Reference acc from test_mfma_only ---
    print("\n" + "=" * 70)
    print("Step 1: Reference MFMA output (test_mfma_only)")
    print("=" * 70)

    mfma_only_path = "/tmp/test_mfma_only.so"
    if not os.path.exists(mfma_only_path):
        compile_kernel("/tmp/test_mfma_only.hip", mfma_only_path)

    lib_ref = ctypes.CDLL(mfma_only_path)
    ref_result = torch.zeros(BLOCK_M := 32, RS, dtype=torch.float32, device="cuda")

    lib_ref.launch_test_mfma_only(
        ctypes.c_void_p(ref_result.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(rot.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(x.stride(0)),
        ctypes.c_void_p(0))
    torch.cuda.synchronize()

    print(f"ref_result[0, 0:32] = {ref_result[0, :32].tolist()}")
    print(f"ref_result[0, 32:64] = {ref_result[0, 32:64].tolist()}")
    ref_nonzero = (ref_result[0] != 0).sum().item()
    print(f"Non-zero elements in row 0: {ref_nonzero}/128")

    # --- Step 2: v2 debug kernel ---
    print("\n" + "=" * 70)
    print("Step 2: v2 debug kernel (acc + quant intermediates)")
    print("=" * 70)

    v2_debug_so = compile_kernel("mfma_rot_quant_moe_sort_v2_debug.hip",
                                  "mfma_rot_quant_moe_sort_v2_debug.so")
    lib_v2 = ctypes.CDLL(v2_debug_so)

    fp4_out = torch.zeros(M, K // 2, dtype=torch.uint8, device="cuda")
    debug_acc = torch.zeros(256, 16, dtype=torch.float32, device="cuda")
    debug_quant = torch.zeros(1024, 14, dtype=torch.float32, device="cuda")
    debug_count = torch.zeros(1, dtype=torch.int32, device="cuda")

    lib_v2.launch_mfma_rot_quant_moe_sort_v2_debug(
        ctypes.c_void_p(fp4_out.data_ptr()),
        ctypes.c_void_p(debug_acc.data_ptr()),
        ctypes.c_void_p(debug_quant.data_ptr()),
        ctypes.c_void_p(debug_count.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(rot.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(K),
        ctypes.c_int(x.stride(0)), ctypes.c_int(fp4_out.stride(0)),
        ctypes.c_void_p(0))
    torch.cuda.synchronize()

    n_quant_entries = debug_count[0].item()
    print(f"Quant debug entries written: {n_quant_entries}")

    # --- Step 3: Compare acc values ---
    print("\n" + "=" * 70)
    print("Step 3: Compare acc values (test_mfma_only vs v2_debug)")
    print("=" * 70)

    # Reconstruct v2's acc into [32, 128] grid for comparison
    v2_result = torch.zeros(32, 128, dtype=torch.float32, device="cuda")
    for tid in range(256):
        wave_id = tid // 64
        lane_id = tid % 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2
        c_m_group = lane_id // 16
        c_col = lane_id % 16
        n_base = wave_n * 64

        for nt in range(4):
            for i in range(4):
                row = wave_m * 16 + c_m_group * 4 + i
                col = n_base + nt * 16 + c_col
                if row < 32 and col < 128:
                    # bf16 truncation to match test_mfma_only
                    raw_val = debug_acc[tid, nt * 4 + i].item()
                    bf16_val = float(torch.tensor(raw_val, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).item())
                    v2_result[row, col] = bf16_val

    v2_result_cpu = v2_result.cpu()
    ref_result_cpu = ref_result.cpu()

    print(f"v2_result[0, 0:32]  = {v2_result_cpu[0, :32].tolist()}")
    print(f"ref_result[0, 0:32] = {ref_result_cpu[0, :32].tolist()}")

    acc_match = torch.allclose(v2_result_cpu[0], ref_result_cpu[0], atol=1e-6)
    acc_diff = (v2_result_cpu[0] - ref_result_cpu[0]).abs()
    max_diff = acc_diff.max().item()
    n_diff = (acc_diff > 1e-6).sum().item()
    print(f"\nAcc match: {acc_match} (max_diff={max_diff:.6f}, n_diff={n_diff}/128)")

    if not acc_match:
        print("\n*** BUG IS IN MFMA COMPUTATION ***")
        print("v2 kernel produces different acc values than test_mfma_only")
        diff_cols = torch.where(acc_diff > 1e-6)[0].tolist()
        print(f"Differing columns: {diff_cols[:20]}...")
        for col in diff_cols[:8]:
            print(f"  col {col}: ref={ref_result_cpu[0, col]:.6f}, v2={v2_result_cpu[0, col]:.6f}")
    else:
        print("\n*** ACC VALUES MATCH — BUG IS IN QUANTIZATION CODE ***")

    # --- Step 4: Analyze FP4 output ---
    print("\n" + "=" * 70)
    print("Step 4: FP4 output analysis")
    print("=" * 70)

    fp4_cpu = fp4_out[0].cpu()
    print(f"FP4 bytes[0:16]:  {fp4_cpu[:16].tolist()}")
    print(f"FP4 bytes[16:32]: {fp4_cpu[16:32].tolist()}")
    print(f"FP4 bytes[32:48]: {fp4_cpu[32:48].tolist()}")
    print(f"FP4 bytes[48:64]: {fp4_cpu[48:64].tolist()}")

    nonzero_bytes = (fp4_cpu != 0).sum().item()
    nonzero_positions = torch.where(fp4_cpu != 0)[0].tolist()
    print(f"\nNon-zero bytes: {nonzero_bytes}/{K // 2}")
    print(f"Non-zero positions: {nonzero_positions}")

    if nonzero_bytes > 0 and nonzero_bytes < K // 2:
        diffs = [nonzero_positions[i+1] - nonzero_positions[i]
                 for i in range(len(nonzero_positions)-1)]
        print(f"Stride between non-zero bytes: {diffs}")

    # --- Step 5: Analyze quant intermediates ---
    print("\n" + "=" * 70)
    print("Step 5: Quantization intermediates (per even-col store)")
    print("=" * 70)

    print(f"{'c_col':>5} {'qg_start':>8} {'acc_v0':>10} {'acc_v1':>10} "
          f"{'v0':>10} {'v1':>10} {'odd':>10} {'odd1':>10} "
          f"{'hw_sc':>8} {'e8m0':>5} {'pk0':>6} {'pk1':>6} "
          f"{'off0':>6} {'off1':>6}")
    print("-" * 130)

    for idx in range(min(n_quant_entries, 32)):
        entry = debug_quant[idx].cpu()
        acc_v0_raw = entry[0].item()
        acc_v1_raw = entry[1].item()
        v0 = entry[2].item()
        v1 = entry[3].item()
        odd = entry[4].item()
        odd1 = entry[5].item()
        hw_scale = entry[6].item()
        e8m0 = int(entry[7].item())
        pk0_bits = float_to_uint32(entry[8].item())
        pk1_bits = float_to_uint32(entry[9].item())
        store_off0 = int(entry[10].item())
        store_off1 = int(entry[11].item())
        c_col = int(entry[12].item())
        qg_start = int(entry[13].item())

        pk0_byte = pk0_bits & 0xFF
        pk1_byte = pk1_bits & 0xFF

        print(f"{c_col:5d} {qg_start:8d} {acc_v0_raw:10.4f} {acc_v1_raw:10.4f} "
              f"{v0:10.4f} {v1:10.4f} {odd:10.4f} {odd1:10.4f} "
              f"{hw_scale:8.4f} {e8m0:5d} {pk0_byte:6d} {pk1_byte:6d} "
              f"{store_off0:6d} {store_off1:6d}")

    # --- Step 6: Python FP4 simulation ---
    print("\n" + "=" * 70)
    print("Step 6: Python FP4 simulation vs v2 output")
    print("=" * 70)

    if acc_match:
        rotated = ref_result_cpu[0].numpy()  # [128]
        # Simulate quantization per quant group (32 elements)
        for qg in range(4):
            group = rotated[qg * 32: (qg + 1) * 32]
            amax = max(abs(group))
            if amax == 0:
                print(f"  QG {qg}: all zeros (expected for cols {qg*32}-{(qg+1)*32-1})")
                continue

            # Simulate v2 scale computation
            u32 = float_to_uint32(amax)
            u32_rounded = (u32 + 0x200000) & 0xFF800000
            raw_exp = (u32_rounded >> 23) & 0xFF
            e8m0 = max(raw_exp, 2) - 2
            hw_scale = uint32_to_float(e8m0 << 23) * 0.25

            print(f"\n  QG {qg} (cols {qg*32}-{(qg+1)*32-1}):")
            print(f"    amax={amax:.4f}, e8m0={e8m0}, hw_scale={hw_scale:.6f}")

            sim_fp4 = []
            for j in range(0, 32, 2):
                even_val = group[j]
                odd_val = group[j + 1]
                if hw_scale != 0:
                    even_scaled = even_val / hw_scale
                    odd_scaled = odd_val / hw_scale
                else:
                    even_scaled = 0.0
                    odd_scaled = 0.0
                even_fp4 = fp4_e2m1_encode(even_scaled)
                odd_fp4 = fp4_e2m1_encode(odd_scaled)
                byte_val = (odd_fp4 << 4) | (even_fp4 & 0xF)
                sim_fp4.append(byte_val)

            fp4_start = qg * 16  # 32 cols / 2 per byte = 16 bytes
            actual_fp4 = fp4_cpu[fp4_start:fp4_start + 16].tolist()

            print(f"    Simulated FP4: {sim_fp4}")
            print(f"    Actual FP4:    {actual_fp4}")
            print(f"    Match: {sim_fp4 == actual_fp4}")

            if sim_fp4 != actual_fp4:
                for j in range(16):
                    if sim_fp4[j] != actual_fp4[j]:
                        print(f"    DIFF at byte {j}: sim=0x{sim_fp4[j]:02x} "
                              f"actual=0x{actual_fp4[j]:02x} "
                              f"(cols {qg*32+j*2},{qg*32+j*2+1})")

    # --- Step 7: Check specific lanes ---
    print("\n" + "=" * 70)
    print("Step 7: Per-lane acc dump (wave_m=0, wave_n=0, first 16 lanes)")
    print("=" * 70)

    print(f"{'tid':>4} {'lane':>4} {'c_mg':>4} {'c_col':>5} "
          f"{'acc0[0]':>10} {'acc1[0]':>10} {'acc2[0]':>10} {'acc3[0]':>10}")
    print("-" * 70)
    for tid in range(16):
        lane_id = tid  # wave 0
        c_m_group = lane_id // 16
        c_col = lane_id % 16
        acc_vals = debug_acc[tid].cpu()
        print(f"{tid:4d} {lane_id:4d} {c_m_group:4d} {c_col:5d} "
              f"{acc_vals[0].item():10.4f} {acc_vals[4].item():10.4f} "
              f"{acc_vals[8].item():10.4f} {acc_vals[12].item():10.4f}")

    print("\nDone. Check above for diagnosis.")


if __name__ == "__main__":
    main()

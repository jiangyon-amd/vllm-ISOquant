#!/usr/bin/env python3
"""
Test ds_read_tr16_b64 with different address patterns to find the correct usage.

LDS contains: lds[0..63] = [1, 2, 3, ..., 64] as bf16
Viewed as a 4×16 bf16 matrix:
  Row 0: [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
  Row 1: [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
  Row 2: [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
  Row 3: [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]

Expected transpose: lane i gets [row0_col_i, row1_col_i, row2_col_i, row3_col_i]
  Lane 0: [1, 17, 33, 49]
  Lane 1: [2, 18, 34, 50]
  ...
"""
import torch, ctypes, subprocess, os, sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
HIP = os.path.join(DIR, "test_ds_read_tr_v2.hip")
SO = os.path.join(DIR, "test_ds_read_tr_v2.so")

if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(HIP):
    print("Compiling...")
    r = subprocess.run(["hipcc", "-shared", "-fPIC", "-O2", "--offload-arch=gfx950", "-o", SO, HIP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)

lib = ctypes.CDLL(SO)
out = torch.zeros(64 * 20, dtype=torch.float32, device="cuda")
lds_dump = torch.zeros(64, dtype=torch.int32, device="cuda")

lib.launch_test_ds_read_tr_asm(
    ctypes.c_void_p(out.data_ptr()),
    ctypes.c_void_p(lds_dump.data_ptr()),
    ctypes.c_void_p(0))
torch.cuda.synchronize()

out = out.cpu().view(64, 20)
lds = lds_dump.cpu()

print("LDS layout (4×16 matrix view):")
for r in range(4):
    print(f"  Row {r}: {lds[r*16:(r+1)*16].tolist()}")

# Expected: lane i gets column i from 4 rows = [i+1, i+17, i+33, i+49]
print(f"\nExpected transpose result for lane i: [i+1, i+17, i+33, i+49]")

tests = [
    ("A: builtin(same ptr)", 0),
    ("B: asm(same addr)",    4),
    ("C: asm(lane*8 addr)",  8),
    ("D: asm(lane%16*2)",   12),
]

for test_name, offset in tests:
    print(f"\n{'='*70}")
    print(f"  {test_name}")
    print(f"{'='*70}")
    print(f"  {'Lane':>4} {'lane%16':>6} {'result[0:3]':>30} {'expected':>30} {'Match':>6}")
    print(f"  {'-'*80}")

    match_count = 0
    for lane in range(16):  # first 16 lanes
        vals = out[lane, offset:offset+4].tolist()
        expected = [lane+1, lane+17, lane+33, lane+49]
        match = all(abs(v-e) < 0.5 for v, e in zip(vals, expected))
        if match: match_count += 1
        mark = "✓" if match else "✗"
        print(f"  {lane:4d} {lane%16:6d} {str([int(v) for v in vals]):>30s} {str(expected):>30s} {mark:>6}")

    print(f"\n  Matched: {match_count}/16")

    if match_count < 16:
        print(f"\n  Decode what was actually read:")
        for lane in range(min(8, 16)):
            vals = [int(v) for v in out[lane, offset:offset+4].tolist()]
            # val = position+1, so position = val-1, row = pos//16, col = pos%16
            decoded = [(max(0,v-1)//16, max(0,v-1)%16) for v in vals]
            print(f"    Lane {lane}: vals={vals} → (row,col)={decoded}")

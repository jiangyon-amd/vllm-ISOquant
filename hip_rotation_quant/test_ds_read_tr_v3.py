#!/usr/bin/env python3
"""Test DS_READ_B64_TR_B16 with contiguous, wide, and production LDS layouts."""
import torch, ctypes, subprocess, os, sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
HIP = os.path.join(DIR, "test_ds_read_tr_v3.hip")
SO = os.path.join(DIR, "test_ds_read_tr_v3.so")

if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(HIP):
    print("Compiling...")
    r = subprocess.run(["hipcc", "-shared", "-fPIC", "-O2", "--offload-arch=gfx950", "-o", SO, HIP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)
    print("OK")

lib = ctypes.CDLL(SO)

def run_test(name, launch_fn, decode_fn=None):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    out = torch.zeros(16 * 12, dtype=torch.float32, device="cuda")
    launch_fn(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(0))
    torch.cuda.synchronize()
    out = out.cpu().view(16, 12)

    print(f"  {'Lane':>4}  {'ds_read_tr':>30}  {'expected(scalar)':>30}  {'Match':>6}")
    print(f"  {'-'*75}")

    match_count = 0
    for lane in range(16):
        tr = [int(v) for v in out[lane, 0:4].tolist()]
        sc = [int(v) for v in out[lane, 4:8].tolist()]
        match = tr == sc
        if match: match_count += 1
        print(f"  {lane:4d}  {str(tr):>30}  {str(sc):>30}  {'✓' if match else '✗':>6}")

    print(f"\n  Matched: {match_count}/16")
    if match_count == 16:
        print("  ✅ ds_read_tr correctly transposes from this LDS layout!")
    else:
        print("  ❌ MISMATCH — ds_read_tr reads different data!")
        if decode_fn:
            decode_fn(out)

def decode_contiguous(out):
    """Decode: lds[row][col] = row*100 + col + 1"""
    print(f"\n  Decoding (value = row*100 + col + 1):")
    for lane in range(4):
        tr = [int(v) for v in out[lane, 0:4].tolist()]
        decoded = [((v-1)//100, (v-1)%100) if v > 0 else (-1,-1) for v in tr]
        print(f"    Lane {lane}: vals={tr} → (row,col)={decoded}")

def decode_production(out):
    """Decode: R_lds[row][col] = row*1000 + col + 1"""
    print(f"\n  Decoding (value = row*1000 + col + 1):")
    for lane in range(8):
        tr = [int(v) for v in out[lane, 0:4].tolist()]
        decoded = [((v-1)//1000, (v-1)%1000) if v > 0 else (-1,-1) for v in tr]
        print(f"    Lane {lane}: vals={tr} → (row,col)={decoded}")

    # Check if it matches contiguous 128-byte read pattern
    print(f"\n  Does it match contiguous 128-byte read (from same row)?:")
    for lane in range(4):
        tr = [int(v) for v in out[lane, 0:4].tolist()]
        contig = [int(v) for v in out[lane, 8:12].tolist()]
        match = tr == contig
        print(f"    Lane {lane}: ds_read_tr={tr}  contiguous_same_row={contig}  {'✓ MATCH' if match else '✗ no'}")

run_test("Test 1: CONTIGUOUS 4×16 (stride=32 bytes) — should work!",
         lib.launch_contiguous, decode_contiguous)

run_test("Test 2: WIDE 4×32 (stride=64 bytes) — stride mismatch",
         lib.launch_wide, decode_contiguous)

run_test("Test 3: PRODUCTION R_lds[8][128] (stride=256 bytes)",
         lib.launch_production, decode_production)

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("""
DS_READ_B64_TR_B16 transposes a 4×16 bf16 block (128 contiguous bytes).
It assumes stride = 32 bytes (16 bf16) between rows.

For R_lds[128][128] (stride=256 bytes), the FIX is:
  Option A: Pack 4×16 tiles into a contiguous buffer before ds_read_tr
  Option B: Use scalar LDS loads (slower but correct)
  Option C: Store rotation in a tiled layout with stride=32 bytes
""")

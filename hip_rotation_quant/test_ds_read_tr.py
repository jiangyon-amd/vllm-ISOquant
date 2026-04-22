#!/usr/bin/env python3
"""
Test ds_read_tr16_b64 LDS access pattern.

Fills LDS with known values (row*100 + col + 1) and compares:
- ds_read_tr output: what the hardware instruction actually returns
- scalar output: what we EXPECT (column access from 4 consecutive rows)

This reveals whether ds_read_tr reads contiguously (ignoring array stride)
or respects the logical 2D array layout.
"""
import torch, ctypes, subprocess, os, sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
HIP = os.path.join(DIR, "test_ds_read_tr.hip")
SO = os.path.join(DIR, "test_ds_read_tr.so")

# Compile
if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(HIP):
    print("Compiling...")
    r = subprocess.run(["hipcc", "-shared", "-fPIC", "-O2", "--offload-arch=gfx950", "-o", SO, HIP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)

lib = ctypes.CDLL(SO)


def run_test(name, launch_fn):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    out_tr = torch.zeros(64, dtype=torch.float32, device="cuda")  # 16 lanes × 4 values
    out_sc = torch.zeros(64, dtype=torch.float32, device="cuda")

    launch_fn(ctypes.c_void_p(out_tr.data_ptr()),
              ctypes.c_void_p(out_sc.data_ptr()),
              ctypes.c_void_p(0))
    torch.cuda.synchronize()

    tr = out_tr.cpu().view(16, 4)
    sc = out_sc.cpu().view(16, 4)

    print(f"\n  {'Lane':>4}  {'ds_read_tr[0:3]':>30}  {'scalar_expect[0:3]':>30}  {'Match':>6}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*30}  {'-'*6}")

    all_match = True
    for lane in range(16):
        tr_vals = tr[lane].tolist()
        sc_vals = sc[lane].tolist()
        match = all(abs(t - s) < 0.01 for t, s in zip(tr_vals, sc_vals))
        if not match:
            all_match = False
        mark = "✓" if match else "✗"
        print(f"  {lane:4d}  {str(tr_vals):>30s}  {str(sc_vals):>30s}  {mark:>6}")

    if all_match:
        print(f"\n  Result: ALL MATCH — ds_read_tr produces expected column access")
    else:
        print(f"\n  Result: MISMATCH — ds_read_tr reads DIFFERENT data than column access!")
        # Decode what ds_read_tr actually read
        print(f"\n  Decoding what ds_read_tr actually read:")
        print(f"  (value = row*100 + col + 1, so row = (val-1)//100, col = (val-1)%100)")
        for lane in range(min(4, 16)):
            vals = tr[lane].tolist()
            decoded = [(int(v-1)//100, int(v-1)%100) if v > 0 else (-1,-1) for v in vals]
            print(f"    Lane {lane}: values={vals}")
            print(f"           decoded (row,col) = {decoded}")

    return all_match


run_test("Test 1: WIDE LDS [8][128] (stride=256 bytes) — simulates R_lds[128][128]",
         lib.launch_test_wide)

run_test("Test 2: NARROW LDS [8][16] (stride=32 bytes) — contiguous 4×16 block",
         lib.launch_test_narrow)

run_test("Test 3: WIDE LDS offset at row=2, cols=32-47",
         lib.launch_test_offset)

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("""
If Test 1 (WIDE) MISMATCHES but Test 2 (NARROW) MATCHES:
  → ds_read_tr reads CONTIGUOUS 128 bytes, ignoring 2D array stride
  → R_lds[128][128] has stride=256 bytes → ds_read_tr reads wrong rows
  → FIX: Pack rotation into contiguous [4][16] tiles before ds_read_tr

If both MATCH:
  → ds_read_tr respects per-lane address, stride doesn't matter
  → Bug is elsewhere
""")

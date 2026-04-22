#!/usr/bin/env python3
"""Test DS_READ_B64_TR_B16 with per-lane addresses."""
import torch, ctypes, subprocess, os, sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
HIP = os.path.join(DIR, "test_ds_read_tr_v4.hip")
SO = os.path.join(DIR, "test_ds_read_tr_v4.so")

if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(HIP):
    print("Compiling...")
    r = subprocess.run(["hipcc", "-shared", "-fPIC", "-O2", "--offload-arch=gfx950", "-o", SO, HIP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)
    print("OK")

lib = ctypes.CDLL(SO)

# Test 1: Per-lane addresses
print(f"\n{'='*70}")
print("  Test 1: Per-lane addr → each lane points to its own 4-element group")
print(f"{'='*70}")
print("  LDS [4][16], lds[r][c] = r*100+c+1")
print("  Lane i → addr = &lds[i/4][(i%4)*4]")

out = torch.zeros(16 * 8, dtype=torch.float32, device="cuda")
lib.launch_per_lane(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(0))
torch.cuda.synchronize()
out = out.cpu().view(16, 8)

print(f"\n  {'Lane':>4} {'row':>3} {'col0':>4}  {'ds_read_tr':>30}  {'expected(col access)':>30}  {'Match':>6}")
print(f"  {'-'*82}")
match_count = 0
for lane in range(16):
    my_row = lane // 4
    my_col = (lane % 4) * 4
    tr = [int(v) for v in out[lane, 0:4].tolist()]
    sc = [int(v) for v in out[lane, 4:8].tolist()]
    match = tr == sc
    if match: match_count += 1
    print(f"  {lane:4d} {my_row:3d} {my_col:4d}  {str(tr):>30}  {str(sc):>30}  {'✓' if match else '✗':>6}")

print(f"\n  Matched: {match_count}/16")
if match_count == 16:
    print("  ✅ WITH PER-LANE ADDRESSES, ds_read_tr correctly transposes!")
else:
    print("  ❌ Still wrong. Decoding:")
    for lane in range(8):
        tr = [int(v) for v in out[lane, 0:4].tolist()]
        decoded = [((v-1)//100, (v-1)%100) if v > 0 else (-1,-1) for v in tr]
        print(f"    Lane {lane}: vals={tr} → (row,col)={decoded}")

# Test 2: Rotation tiled layout
print(f"\n{'='*70}")
print("  Test 2: Rotation matrix tiled (2 tiles of 4×16, K=8, N=16)")
print(f"{'='*70}")

out2 = torch.zeros(16 * 16, dtype=torch.float32, device="cuda")
lib.launch_rotation(ctypes.c_void_p(out2.data_ptr()), ctypes.c_void_p(0))
torch.cuda.synchronize()
out2 = out2.cpu().view(16, 16)

print(f"\n  Tile 0 (K[0:3]):")
print(f"  {'Lane':>4}  {'ds_read_tr':>30}  {'expected':>30}  {'Match':>6}")
match_t0 = 0
for lane in range(16):
    tr = [int(v) for v in out2[lane, 0:4].tolist()]
    sc = [int(v) for v in out2[lane, 8:12].tolist()]
    match = tr == sc
    if match: match_t0 += 1
    print(f"  {lane:4d}  {str(tr):>30}  {str(sc):>30}  {'✓' if match else '✗':>6}")

print(f"\n  Tile 1 (K[4:7]):")
match_t1 = 0
for lane in range(16):
    tr = [int(v) for v in out2[lane, 4:8].tolist()]
    sc = [int(v) for v in out2[lane, 12:16].tolist()]
    match = tr == sc
    if match: match_t1 += 1
    print(f"  {lane:4d}  {str(tr):>30}  {str(sc):>30}  {'✓' if match else '✗':>6}")

print(f"\n  Tile 0 match: {match_t0}/16, Tile 1 match: {match_t1}/16")
if match_t0 == 16 and match_t1 == 16:
    print("  ✅ ROTATION TILED LAYOUT WORKS WITH ds_read_tr!")

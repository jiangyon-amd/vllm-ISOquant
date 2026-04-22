#!/usr/bin/env python3
"""Debug script: dump per-lane MFMA output + quant values to understand layout mapping."""
import torch, ctypes, json, time, os, triton

LOG_PATH = "/data/jiangyon/.cursor/debug-17ff9e.log"

def log(msg, data, hypothesis="", location="debug_mfma_moe.py"):
    entry = {
        "sessionId": "17ff9e",
        "id": f"log_{int(time.time()*1000)}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": msg,
        "data": data,
        "runId": "run1",
        "hypothesisId": hypothesis,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Setup
RS = 128; K = 256; M = 1; topk = 8  # Small K=256 for easier debugging (2 rotation chunks)
n_i = K // 32; tile_n = triton.cdiv(n_i, 8)
rot = torch.eye(RS, dtype=torch.bfloat16, device="cuda")  # Identity → x_rot = x
torch.manual_seed(42)
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

# Make x simple: first 8 elements = [1,2,3,4,5,6,7,8], rest = 0
x.zero_()
for i in range(32):
    x[0, i] = float(i + 1)  # First quant group: 1,2,3,...,32

log("Input x[0, 0:32]", {"values": x[0, :32].tolist()}, "setup")
log("Rotation", {"type": "identity"}, "setup")

# === Reference: Triton tl.dot output ===
from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (
    _fused_decode_m1_topk8_general_kernel,
)
m_o = topk; m_pad = 32
si = torch.arange(m_o, dtype=torch.int32, device="cuda")
si = torch.cat([si, torch.full((m_pad - m_o,), 1, dtype=torch.int32, device="cuda")])
nv = torch.tensor([m_o], dtype=torch.int32, device="cuda")

fp4_tri = torch.empty((1, K//2), dtype=torch.uint8, device="cuda")
sc_tri = torch.zeros((m_pad, n_i), dtype=torch.uint8, device="cuda")
_fused_decode_m1_topk8_general_kernel[(K//RS,)](
    x, rot, fp4_tri, si, nv, sc_tri,
    x.stride(0), x.stride(1), rot.stride(0), rot.stride(1),
    fp4_tri.stride(0), sc_tri.stride(0), sc_tri.stride(1),
    token_num=1, m_o=m_o, N_I=n_i, TILE_N=tile_n, MAX_Q=64, num_warps=1)
torch.cuda.synchronize()

log("Triton FP4 output (first rotation chunk, 64 bytes)", 
    {"fp4": fp4_tri[0, :64].tolist(), "non_zero": (fp4_tri[0,:64] != 0).sum().item()}, "H1")

# Decode FP4 bytes to see paired values
fp4_bytes = fp4_tri[0, :16].tolist()
log("Triton FP4 bytes[0:16] (quant group 0, cols 0-31)", 
    {"bytes": fp4_bytes, 
     "hex": [f"0x{b:02x}" for b in fp4_bytes],
     "lo_nibble": [b & 0xF for b in fp4_bytes],
     "hi_nibble": [(b >> 4) & 0xF for b in fp4_bytes]}, "H2")

# === HIP MFMA kernel ===
lib = ctypes.CDLL("/data/jiangyon/vllm_rotation/hip_rotation_quant/mfma_rot_quant_moe_sort.so")
fp4_hip = torch.empty((M, K//2), dtype=torch.uint8, device="cuda")
sc_hip = torch.zeros((m_pad, n_i), dtype=torch.uint8, device="cuda")
lib.launch_mfma_rot_quant_moe_sort(
    ctypes.c_void_p(fp4_hip.data_ptr()), ctypes.c_void_p(sc_hip.data_ptr()),
    ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(rot.data_ptr()),
    ctypes.c_void_p(si.data_ptr()), ctypes.c_void_p(nv.data_ptr()),
    ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(1),
    ctypes.c_int(m_o), ctypes.c_int(m_pad),
    ctypes.c_int(x.stride(0)), ctypes.c_int(fp4_hip.stride(0)),
    ctypes.c_int(sc_hip.stride(0)), ctypes.c_int(sc_hip.stride(1)),
    ctypes.c_int(n_i), ctypes.c_void_p(0))
torch.cuda.synchronize()

log("HIP MFMA FP4 output (first rotation chunk, 64 bytes)", 
    {"fp4": fp4_hip[0, :64].tolist(), "non_zero": (fp4_hip[0,:64] != 0).sum().item()}, "H1")

fp4_hip_bytes = fp4_hip[0, :16].tolist()
log("HIP MFMA FP4 bytes[0:16]", 
    {"bytes": fp4_hip_bytes,
     "hex": [f"0x{b:02x}" for b in fp4_hip_bytes],
     "lo_nibble": [b & 0xF for b in fp4_hip_bytes],
     "hi_nibble": [(b >> 4) & 0xF for b in fp4_hip_bytes]}, "H2")

# === H3: Check scales ===
log("Triton scales (raw, first 8)", {"scales": sc_tri[0, :8].tolist()}, "H3")
log("HIP scales (from LDS scatter)", 
    {"scales": sc_hip[0, :n_i].tolist(), "non_zero": (sc_hip[0,:n_i] != 0).sum().item()}, "H3")

# === H5: Check if acc values are reasonable ===
# Dequantize FP4 to verify: with identity rotation, input [1..32],
# the rotated values should be [1..32], and FP4 quantization should preserve rough magnitudes
log("Expected: with identity rot and x=[1..32], FP4 should encode these values", 
    {"x_values": x[0, :32].tolist()}, "H5")

# Compare byte-by-byte
match_count = (fp4_tri[0,:K//2] == fp4_hip[0,:K//2]).sum().item()
total = K // 2
log("FP4 match summary", 
    {"match": match_count, "total": total, "pct": f"{match_count/total*100:.1f}%"}, "H1")

# Check for permutation: are the same BYTES present but in different positions?
tri_sorted = sorted(fp4_tri[0,:64].tolist())
hip_sorted = sorted(fp4_hip[0,:64].tolist())
log("Sorted byte values (detect permutation)",
    {"triton_sorted": tri_sorted, "hip_sorted": hip_sorted,
     "is_permutation": tri_sorted == hip_sorted}, "H4")

print(f"Debug logs written to {LOG_PATH}")

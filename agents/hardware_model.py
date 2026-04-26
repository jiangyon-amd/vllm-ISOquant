"""MI355X hardware performance model.

Provides theoretical peak calculations for kernel optimization targets.
All agents reference this model for convergence decisions.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HardwareSpec:
    """Hardware specification for target GPU."""
    name: str = "MI355X"
    arch: str = "gfx950"
    num_CUs: int = 304
    wavefront_size: int = 64
    max_waves_per_CU: int = 8          # typical, depends on VGPR usage
    hbm_bandwidth_GBs: float = 5300.0  # GB/s peak
    clock_MHz: float = 1500.0
    vgpr_per_CU: int = 512
    lds_per_CU_bytes: int = 65536
    l2_cache_MB: float = 256.0

    @property
    def hbm_bandwidth_bytes_per_us(self) -> float:
        """Bytes per microsecond at peak HBM bandwidth."""
        return self.hbm_bandwidth_GBs * 1e9 / 1e6  # = 5.3e6 B/µs

    @property
    def total_waves(self) -> int:
        return self.num_CUs * self.max_waves_per_CU

    def max_vgpr_for_occupancy(self, target_waves: int) -> int:
        """Max VGPRs per thread to achieve target waves/CU."""
        return self.vgpr_per_CU // target_waves


MI355X = HardwareSpec()


def compute_data_volume_bytes(
    target_kernel: str,
    cfg: dict[str, Any],
    *,
    head_dim: int = 128,
    slot_size: int = 136,
) -> int:
    """Estimate end-to-end memory traffic for one workload config."""
    D = head_dim
    if target_kernel == "tq_wht_rotate":
        M = int(cfg.get("M", 1))
        read_bytes = M * D * 2 + D * 4
        write_bytes = M * D * 2
    elif target_kernel == "tq_decode_stage2":
        B = int(cfg.get("B", 1))
        Hq = int(cfg.get("Hq", 64))
        splits = int(cfg.get("splits", 0))
        read_bytes = B * Hq * splits * (D + 1) * 4
        write_bytes = B * Hq * D * 2
    elif target_kernel == "tq_decode_stage1":
        B = int(cfg.get("B", 1))
        Hq = int(cfg.get("Hq", 64))
        Hk = int(cfg.get("Hk", 8))
        seq_len = int(cfg.get("seq", 128))
        splits = int(cfg.get("splits", 0))
        read_bytes = B * Hk * seq_len * slot_size
        write_bytes = B * Hq * splits * (D + 1) * 4
    elif target_kernel in ("tq_decode_fused", "tq_decode_fused_wht"):
        B = int(cfg.get("B", 1))
        Hq = int(cfg.get("Hq", 64))
        Hk = int(cfg.get("Hk", 8))
        seq_len = int(cfg.get("seq", 128))
        read_bytes = B * Hk * seq_len * slot_size
        write_bytes = B * Hq * D * 2
    else:
        B = int(cfg.get("B", 1))
        Hq = int(cfg.get("Hq", 64))
        splits = int(cfg.get("splits", 1))
        read_bytes = B * Hq * max(splits, 1) * (D + 1) * 4
        write_bytes = B * Hq * D * 2
    return read_bytes + write_bytes


@dataclass
class KernelProfile:
    """Measured + theoretical performance of a kernel variant."""
    name: str
    version: str
    # Measured
    gpu_time_us: dict[str, float] = field(default_factory=dict)  # config→µs
    vgpr_count: int = 0
    sgpr_count: int = 0
    lds_bytes: int = 0
    spill_bytes: int = 0
    # Theoretical
    theoretical_us: dict[str, float] = field(default_factory=dict)

    def efficiency(self, config: str) -> float:
        """HBM bandwidth efficiency = theoretical / actual."""
        theo = self.theoretical_us.get(config, 0)
        actual = self.gpu_time_us.get(config, 0)
        if actual <= 0 or theo <= 0:
            return 0.0
        return theo / actual

    def avg_efficiency(self) -> float:
        configs = set(self.theoretical_us) & set(self.gpu_time_us)
        if not configs:
            return 0.0
        return sum(self.efficiency(c) for c in configs) / len(configs)


def stage2_theoretical_us(
    B: int, Hq: int = 64, num_kv_splits: int = 32, D: int = 128,
    hw: HardwareSpec = MI355X,
) -> float:
    """Theoretical minimum time for Stage2 reduce (memory-bound)."""
    total_bytes = compute_data_volume_bytes(
        "tq_decode_stage2",
        {"B": B, "Hq": Hq, "splits": num_kv_splits},
        head_dim=D,
    )
    return total_bytes / hw.hbm_bandwidth_bytes_per_us


def stage1_theoretical_us(
    B: int, seq_len: int, Hq: int = 64, Hk: int = 8,
    num_kv_splits: int = 32, slot_size: int = 136,
    hw: HardwareSpec = MI355X,
) -> float:
    """Theoretical minimum time for Stage1.

    Stage1 is COMPUTE-BOUND: for each KV token it does:
      - Load KPS=68 bytes (quantized KV slot)
      - Decode 4-bit quantized key → 128 FP32 values (via centroid lookup)
      - Dot product Q·K (128 FMA ops)
      - Softmax (exp, sum)
      - Decode 4-bit value → 128 FP32 values
      - Accumulate V weighted by attention (128 FMA ops)
    Total: ~256 FMA + 128 exp + centroid lookups per token

    We model as max(memory_time, compute_time).
    """
    D = 128

    # Memory-bound estimate
    tokens_per_split = math.ceil(seq_len / num_kv_splits)
    total_bytes = compute_data_volume_bytes(
        "tq_decode_stage1",
        {
            "B": B,
            "seq": tokens_per_split * num_kv_splits,
            "Hq": Hq,
            "Hk": Hk,
            "splits": num_kv_splits,
        },
        head_dim=D,
        slot_size=slot_size,
    )
    mem_time = total_bytes / hw.hbm_bandwidth_bytes_per_us

    # Compute-bound estimate
    # Grid = (B, Hq, splits), Block = 128 threads
    # Each block processes tokens_per_split KV tokens
    # Per token: ~256 FMA (Q·K + V accumulate) + centroid lookup overhead
    # At ~1500 MHz, 304 CUs, 128 FLOPs/CU/cycle (FMA on 2 SIMD units):
    # Peak = 304 * 128 * 1500e6 = 58.4 TFLOPS
    total_flops = B * Hq * tokens_per_split * (D * 2)  # Q·K + V·attn
    peak_tflops = hw.num_CUs * 128 * hw.clock_MHz * 1e-6  # TFLOPS
    compute_time = total_flops / (peak_tflops * 1e6)  # µs

    # Occupancy-adjusted: with 58 VGPRs only 8 waves/CU (vs 10+ theoretical)
    occupancy_factor = 0.8  # approximate
    compute_time /= occupancy_factor

    return max(mem_time, compute_time)


def fused_theoretical_us(
    B: int, seq_len: int, Hq: int = 64, Hk: int = 8,
    slot_size: int = 136, D: int = 128,
    hw: HardwareSpec = MI355X,
) -> float:
    """Theoretical minimum for fused kernel (no mid_o)."""
    total_bytes = compute_data_volume_bytes(
        "tq_decode_fused",
        {"B": B, "seq": seq_len, "Hq": Hq, "Hk": Hk},
        head_dim=D,
        slot_size=slot_size,
    )
    return total_bytes / hw.hbm_bandwidth_bytes_per_us


def wht_rotate_theoretical_us(
    M: int, D: int = 128,
    hw: HardwareSpec = MI355X,
) -> float:
    """Theoretical minimum for WHT rotation (memory-bound).

    WHT rotation reads M×D bf16 + D f32 signs, writes M×D bf16.
    The kernel is launch-latency bound for small M.
    """
    total_bytes = compute_data_volume_bytes(
        "tq_wht_rotate",
        {"M": M},
        head_dim=D,
    )
    mem_time = total_bytes / hw.hbm_bandwidth_bytes_per_us
    # Launch overhead floor (empirical ~5µs on MI300X/MI355X)
    launch_floor = 5.0
    return max(mem_time, launch_floor)


# ---- Quick self-test ----
if __name__ == "__main__":
    print(f"MI355X HBM BW: {MI355X.hbm_bandwidth_bytes_per_us/1e6:.1f} MB/µs")
    for B in [1, 4, 16, 64, 128, 200]:
        t = stage2_theoretical_us(B)
        print(f"  Stage2 B={B:3d}: {t:.1f} µs theoretical")
    print()
    for B, seq in [(4, 128), (4, 8192), (64, 128), (200, 128)]:
        t = fused_theoretical_us(B, seq)
        print(f"  Fused  B={B:3d} seq={seq:5d}: {t:.1f} µs theoretical")
    print()
    for B, seq in [(4, 8192), (20, 8192)]:
        t = stage1_theoretical_us(B, seq)
        print(f"  Stage1 B={B:3d} seq={seq:5d}: {t:.1f} µs theoretical")

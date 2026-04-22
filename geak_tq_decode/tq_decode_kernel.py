# TQ Decode Stage1 Kernel - extracted for GEAK optimization
# Source: vllm/v1/attention/ops/triton_turboquant_decode.py
#
# This is the #1 bottleneck (56.8% GPU time) in TurboQuant 4bit_nc on MI300X.
# Target: reduce avg latency from 266.6us → <180us (match CUDA H100 level)

import math
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _tq_decode_stage1(
    # Precomputed query projection
    Q_rot_ptr,              # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,           # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,        # [B, max_num_blocks] int32
    Seq_lens_ptr,           # [B] int32
    # TQ parameters
    Centroids_ptr,          # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,              # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb, stride_qh,
    stride_cache_block, stride_cache_pos, stride_cache_head,
    stride_bt_b,
    stride_mid_b, stride_mid_h, stride_mid_s,
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    # TQ layout constants
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Score constants
    ATTN_SCALE: tl.constexpr,
    # Block tile sizes
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    KEY_FP8: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)
    kv_head = hid // KV_GROUP_SIZE

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0)
        slot_bases = (block_nums * stride_cache_block
                      + page_off * stride_cache_pos
                      + kv_head * stride_cache_head)

        # ---- KEY SCORE COMPUTATION ----
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(KV_cache_ptr + k_addrs, mask=kv_mask[:, None] & d_mask[None, :], other=0)
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = tl.sum(tl.where(d_mask[None, :], q_rot[None, :] * k_float, 0.0), axis=1) * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))
        else:
            # MSE unpack: 2 byte loads per dimension → 16-bit → shift+mask
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(KV_cache_ptr + mse_addrs0, mask=kv_mask[:, None] & d_mask[None, :], other=0).to(tl.int32)
            mse_raw1 = tl.load(KV_cache_ptr + mse_addrs0 + 1, mask=kv_mask[:, None] & d_mask[None, :], other=0).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            c_vals = tl.load(Centroids_ptr + mse_idx, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)

            if NORM_CORRECTION:
                c_norm_sq = tl.sum(tl.where(d_mask[None, :], c_vals * c_vals, 0.0), axis=1)
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            term1 = tl.sum(tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0), axis=1)

            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(tl.uint16)
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms * term1 * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))

        # ---- ONLINE SOFTMAX ----
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # ---- VALUE DEQUANT ----
        val_bases = slot_bases + KPS
        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(KV_cache_ptr + val_addrs0, mask=kv_mask[:, None] & d_mask[None, :], other=0).to(tl.int32)
            val_raw1 = tl.load(KV_cache_ptr + val_addrs0 + 1, mask=kv_mask[:, None] & d_mask[None, :], other=0).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)
            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(KV_cache_ptr + val_addrs, mask=kv_mask[:, None] & d_mask[None, :], other=0).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)
            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ---- ACCUMULATE ----
        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)

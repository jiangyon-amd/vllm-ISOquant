/**
 * TQ Decode Stage1 - V62: BLOCK_KV=8 with V60's branch elimination
 *
 * GEAK Iteration 3: Double the tokens per warp-iteration.
 *
 * Rationale: V60 showed 11.6% improvement by eliminating bounds-check branches.
 * V61 showed that branchless page-dedup pushes VGPRs to 85, hurting occupancy.
 * 
 * V62 approach: Keep V60's structure (page-dedup branches ok) but increase
 * BLOCK_KV from 4 to 8. This:
 *   - Doubles tokens per iteration: 8 warps × 8 tokens = 64 tokens/iter
 *   - For B=4 seq=8192 splits=32: 256 tokens/split, only 4 iterations 
 *     (vs 8 with BLOCK_KV=4)
 *   - Amortizes loop overhead and page_table lookups (more tokens per page)
 *   - Risk: more VGPRs for 8 score/p/value arrays
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HEAD_DIM 128
#define MSE_BYTES 64
#define N_CENTROIDS 16
#define KPS 68
#define VAL_DATA_BYTES 64
#define THREADS_PER_WARP 32
#define NUM_WARPS 8
#define THREADS (THREADS_PER_WARP * NUM_WARPS)
#define DIMS_PER_THREAD 4
#define BLOCK_KV 8

__device__ __forceinline__ void wave_sum2_32(float& a, float& b) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_xor(a, offset, 32);
        b += __shfl_xor(b, offset, 32);
    }
}

__device__ __forceinline__ void process_full_block_8(
    const unsigned char* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    int bt_base, int warp_base,
    int block_shift, int block_mask,
    int stride_cb, int stride_cp,
    long long kv_head_offset,
    int byte_idx,
    const float* __restrict__ s_c,
    float q0, float q1, float q2, float q3,
    float attn_scale, int norm_correction,
    float& m_prev, float& l_prev,
    float& acc0, float& acc1, float& acc2, float& acc3)
{
    unsigned short mse_data[BLOCK_KV];
    unsigned short norm_data[BLOCK_KV];
    unsigned short val_data[BLOCK_KV];
    unsigned int sz_data[BLOCK_KV];
    
    int prev_page_idx = -1;
    int prev_block_num = 0;

    #pragma unroll
    for (int kv = 0; kv < BLOCK_KV; kv++) {
        const int kv_pos = warp_base + kv;
        const int page_idx = kv_pos >> block_shift;
        const int page_off = kv_pos & block_mask;
        int block_num;
        if (page_idx != prev_page_idx) {
            block_num = block_table[bt_base + page_idx];
            prev_page_idx = page_idx;
            prev_block_num = block_num;
        } else {
            block_num = prev_block_num;
        }
        const long long slot_base =
            (long long)block_num * stride_cb + page_off * stride_cp + kv_head_offset;
        mse_data[kv] = *reinterpret_cast<const unsigned short*>(kv_cache + slot_base + byte_idx);
        norm_data[kv] = *reinterpret_cast<const unsigned short*>(kv_cache + slot_base + MSE_BYTES);
        val_data[kv] = *reinterpret_cast<const unsigned short*>(kv_cache + slot_base + KPS + byte_idx);
        sz_data[kv] = *reinterpret_cast<const unsigned int*>(kv_cache + slot_base + KPS + VAL_DATA_BYTES);
    }

    float scores[BLOCK_KV];
    #pragma unroll
    for (int kv = 0; kv < BLOCK_KV; kv++) {
        const unsigned short mse_u16 = mse_data[kv];
        const float c0 = s_c[mse_u16 & 0xF];
        const float c1 = s_c[(mse_u16 >> 4) & 0xF];
        const float c2 = s_c[(mse_u16 >> 8) & 0xF];
        const float c3 = s_c[(mse_u16 >> 12)];
        float score = __fmaf_rn(q0, c0, __fmaf_rn(q1, c1, __fmaf_rn(q2, c2, q3 * c3)));
        float norm_sq = __fmaf_rn(c0, c0, __fmaf_rn(c1, c1, __fmaf_rn(c2, c2, c3 * c3)));
        wave_sum2_32(score, norm_sq);
        if (norm_correction) score *= rsqrtf(norm_sq + 1e-16f);
        const float vec_norm = __half2float(*reinterpret_cast<const __half*>(&norm_data[kv]));
        scores[kv] = vec_norm * score * attn_scale;
    }

    float block_max = scores[0];
    #pragma unroll
    for (int kv = 1; kv < BLOCK_KV; kv++) block_max = fmaxf(block_max, scores[kv]);
    
    const float n_e_max = fmaxf(block_max, m_prev);
    const float re_scale = expf(m_prev - n_e_max);
    float new_l = 0.0f;
    float p[BLOCK_KV];
    #pragma unroll
    for (int kv = 0; kv < BLOCK_KV; kv++) {
        p[kv] = expf(scores[kv] - n_e_max);
        new_l += p[kv];
    }
    acc0 *= re_scale; acc1 *= re_scale; acc2 *= re_scale; acc3 *= re_scale;
    #pragma unroll
    for (int kv = 0; kv < BLOCK_KV; kv++) {
        const unsigned int sz = sz_data[kv];
        const float v_scale = __half2float(*reinterpret_cast<const __half*>(&sz));
        const unsigned short zr_u16 = (sz >> 16);
        const float v_zero = __half2float(*reinterpret_cast<const __half*>(&zr_u16));
        const unsigned short val_u16 = val_data[kv];
        const float v0 = (float)(val_u16 & 0xF) * v_scale + v_zero;
        const float v1 = (float)((val_u16 >> 4) & 0xF) * v_scale + v_zero;
        const float v2 = (float)((val_u16 >> 8) & 0xF) * v_scale + v_zero;
        const float v3 = (float)((val_u16 >> 12) & 0xF) * v_scale + v_zero;
        acc0 = __fmaf_rn(p[kv], v0, acc0);
        acc1 = __fmaf_rn(p[kv], v1, acc1);
        acc2 = __fmaf_rn(p[kv], v2, acc2);
        acc3 = __fmaf_rn(p[kv], v3, acc3);
    }
    l_prev = __fmaf_rn(l_prev, re_scale, new_l);
    m_prev = n_e_max;
}

extern "C" __global__ __launch_bounds__(256, 2)
void tq_decode_stage1_hip(
    const float* __restrict__ q_rot,
    const unsigned char* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const float* __restrict__ centroids,
    float* __restrict__ mid_o,
    int stride_qb, int stride_qh,
    int stride_cb, int stride_cp, int stride_ch,
    int stride_bt,
    int stride_mb, int stride_mh, int stride_ms,
    int num_kv_heads, int block_size,
    int num_kv_splits, int kv_group_size,
    float attn_scale, int norm_correction
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int sid = blockIdx.z;
    const int kv_head = hid / kv_group_size;
    const int tid = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;

    const int seq_len = seq_lens[bid];
    const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = split_len * sid;
    const int split_end = min(split_start + split_len, seq_len);
    if (split_start >= split_end) return;

    __shared__ float s_c[N_CENTROIDS];
    __shared__ float s_m[NUM_WARPS];
    __shared__ float s_l[NUM_WARPS];
    __shared__ float s_acc[NUM_WARPS][HEAD_DIM];
    if (tid < N_CENTROIDS) s_c[tid] = centroids[tid];
    __syncthreads();

    const int q_base = bid * stride_qb + hid * stride_qh + lane_id * DIMS_PER_THREAD;
    const float4 q_vec = *reinterpret_cast<const float4*>(q_rot + q_base);
    float m_prev = -1e30f, l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int bt_base = bid * stride_bt;
    const long long kv_head_offset = (long long)kv_head * stride_ch;
    const int byte_idx = lane_id * 2;
    const int block_shift = __ffs(block_size) - 1;
    const int block_mask = block_size - 1;
    const int stride_iter = BLOCK_KV * NUM_WARPS;  // 64 tokens/iter
    const int total_tokens = split_end - split_start;
    const int full_iters = total_tokens / stride_iter;
    const int tail_start = split_start + full_iters * stride_iter;

    for (int iter = 0; iter < full_iters; iter++) {
        const int warp_base = split_start + iter * stride_iter + warp_id * BLOCK_KV;
        process_full_block_8(kv_cache, block_table, bt_base, warp_base,
            block_shift, block_mask, stride_cb, stride_cp, kv_head_offset,
            byte_idx, s_c, q_vec.x, q_vec.y, q_vec.z, q_vec.w, attn_scale, norm_correction,
            m_prev, l_prev, acc0, acc1, acc2, acc3);
    }

    // Tail with bounds check
    if (tail_start < split_end) {
        const int warp_base = tail_start + warp_id * BLOCK_KV;
        const int actual_kv = min(BLOCK_KV, split_end - warp_base);
        if (actual_kv > 0) {
            unsigned short mse_data[BLOCK_KV], norm_data[BLOCK_KV], val_data[BLOCK_KV];
            unsigned int sz_data[BLOCK_KV];
            int prev_page_idx = -1, prev_block_num = 0;
            #pragma unroll
            for (int kv = 0; kv < BLOCK_KV; kv++) {
                if (kv < actual_kv) {
                    const int kv_pos = warp_base + kv;
                    const int page_idx = kv_pos >> block_shift;
                    const int page_off = kv_pos & block_mask;
                    int block_num;
                    if (page_idx != prev_page_idx) { block_num = block_table[bt_base + page_idx]; prev_page_idx = page_idx; prev_block_num = block_num; } else { block_num = prev_block_num; }
                    const long long slot_base = (long long)block_num * stride_cb + page_off * stride_cp + kv_head_offset;
                    mse_data[kv] = *reinterpret_cast<const unsigned short*>(kv_cache + slot_base + byte_idx);
                    norm_data[kv] = *reinterpret_cast<const unsigned short*>(kv_cache + slot_base + MSE_BYTES);
                    val_data[kv] = *reinterpret_cast<const unsigned short*>(kv_cache + slot_base + KPS + byte_idx);
                    sz_data[kv] = *reinterpret_cast<const unsigned int*>(kv_cache + slot_base + KPS + VAL_DATA_BYTES);
                }
            }
            float scores[BLOCK_KV];
            #pragma unroll
            for (int kv = 0; kv < BLOCK_KV; kv++) {
                if (kv < actual_kv) {
                    const unsigned short mse_u16 = mse_data[kv];
                    const float c0 = s_c[mse_u16 & 0xF], c1 = s_c[(mse_u16>>4)&0xF], c2 = s_c[(mse_u16>>8)&0xF], c3 = s_c[(mse_u16>>12)];
                    float score = __fmaf_rn(q_vec.x, c0, __fmaf_rn(q_vec.y, c1, __fmaf_rn(q_vec.z, c2, q_vec.w * c3)));
                    float norm_sq = __fmaf_rn(c0, c0, __fmaf_rn(c1, c1, __fmaf_rn(c2, c2, c3 * c3)));
                    wave_sum2_32(score, norm_sq);
                    if (norm_correction) score *= rsqrtf(norm_sq + 1e-16f);
                    scores[kv] = __half2float(*reinterpret_cast<const __half*>(&norm_data[kv])) * score * attn_scale;
                } else scores[kv] = -1e30f;
            }
            float block_max = scores[0];
            #pragma unroll
            for (int kv = 1; kv < BLOCK_KV; kv++) block_max = fmaxf(block_max, scores[kv]);
            const float n_e_max = fmaxf(block_max, m_prev);
            const float re_scale = expf(m_prev - n_e_max);
            float new_l = 0.0f; float p[BLOCK_KV];
            #pragma unroll
            for (int kv = 0; kv < BLOCK_KV; kv++) { p[kv] = expf(scores[kv] - n_e_max); new_l += p[kv]; }
            acc0 *= re_scale; acc1 *= re_scale; acc2 *= re_scale; acc3 *= re_scale;
            #pragma unroll
            for (int kv = 0; kv < BLOCK_KV; kv++) {
                if (kv < actual_kv) {
                    const unsigned int sz = sz_data[kv];
                    const float v_scale = __half2float(*reinterpret_cast<const __half*>(&sz));
                    const unsigned short zr = (sz >> 16);
                    const float v_zero = __half2float(*reinterpret_cast<const __half*>(&zr));
                    const unsigned short val_u16 = val_data[kv];
                    acc0 = __fmaf_rn(p[kv], (float)(val_u16 & 0xF) * v_scale + v_zero, acc0);
                    acc1 = __fmaf_rn(p[kv], (float)((val_u16>>4)&0xF) * v_scale + v_zero, acc1);
                    acc2 = __fmaf_rn(p[kv], (float)((val_u16>>8)&0xF) * v_scale + v_zero, acc2);
                    acc3 = __fmaf_rn(p[kv], (float)((val_u16>>12)&0xF) * v_scale + v_zero, acc3);
                }
            }
            l_prev = __fmaf_rn(l_prev, re_scale, new_l);
            m_prev = n_e_max;
        }
    }

    if (lane_id == 0) { s_m[warp_id] = m_prev; s_l[warp_id] = l_prev; }
    s_acc[warp_id][lane_id*DIMS_PER_THREAD] = acc0;
    s_acc[warp_id][lane_id*DIMS_PER_THREAD+1] = acc1;
    s_acc[warp_id][lane_id*DIMS_PER_THREAD+2] = acc2;
    s_acc[warp_id][lane_id*DIMS_PER_THREAD+3] = acc3;
    __syncthreads();

    if (warp_id == 0) {
        float m_final = s_m[0];
        #pragma unroll
        for (int w = 1; w < NUM_WARPS; w++) m_final = fmaxf(m_final, s_m[w]);
        float l_final = 0.0f; float scales[NUM_WARPS];
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) { scales[w] = expf(s_m[w] - m_final); l_final += s_l[w] * scales[w]; }
        const float inv_l = 1.0f / fmaxf(l_final, 1e-16f);
        const int db = lane_id * DIMS_PER_THREAD;
        float o0=0,o1=0,o2=0,o3=0;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) { o0+=s_acc[w][db]*scales[w]; o1+=s_acc[w][db+1]*scales[w]; o2+=s_acc[w][db+2]*scales[w]; o3+=s_acc[w][db+3]*scales[w]; }
        o0*=inv_l; o1*=inv_l; o2*=inv_l; o3*=inv_l;
        const int out_base = bid * stride_mb + hid * stride_mh + sid * stride_ms;
        *reinterpret_cast<float4*>(mid_o + out_base + db) = make_float4(o0, o1, o2, o3);
        if (lane_id == 0) mid_o[out_base + HEAD_DIM] = m_final + logf(fmaxf(l_final, 1e-16f));
    }
}

extern "C" void launch_tq_decode_stage1(
    const float* q_rot, const unsigned char* kv_cache,
    const int* block_table, const int* seq_lens,
    const float* centroids, float* mid_o,
    int stride_qb, int stride_qh, int stride_cb, int stride_cp, int stride_ch,
    int stride_bt, int stride_mb, int stride_mh, int stride_ms,
    int num_kv_heads, int block_size, int num_kv_splits, int kv_group_size,
    float attn_scale, int norm_correction, int B, int Hq, hipStream_t stream
) {
    dim3 grid(B, Hq, num_kv_splits);
    dim3 block(256, 1, 1);
    hipLaunchKernelGGL(tq_decode_stage1_hip, grid, block, 0, stream,
        q_rot, kv_cache, block_table, seq_lens, centroids, mid_o,
        stride_qb, stride_qh, stride_cb, stride_cp, stride_ch,
        stride_bt, stride_mb, stride_mh, stride_ms,
        num_kv_heads, block_size, num_kv_splits, kv_group_size,
        attn_scale, norm_correction);
}

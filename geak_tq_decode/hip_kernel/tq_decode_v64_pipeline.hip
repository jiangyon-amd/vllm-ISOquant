/**
 * TQ Decode Stage1 - V64: Load-compute pipeline
 *
 * GEAK Iteration 5: Interleave loads and computation.
 *
 * V60 loads all BLOCK_KV=4 tokens' data, then processes all 4.
 * V64: Load token[0], compute token[0] while loading token[1], etc.
 * This overlaps memory latency with ALU computation.
 *
 * Structure:
 *   Load T0 data
 *   For each iteration:
 *     Process T0 (compute score, exp, accumulate)
 *     Load T1 (overlaps with T0 processing)
 *     Process T1 
 *     Load T2 (overlaps)
 *     Process T2
 *     Load T3 (overlaps)
 *     Process T3
 *     Load T0 of NEXT iteration (overlaps)
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
#define BLOCK_KV 4

__device__ __forceinline__ void wave_sum2_32(float& a, float& b) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_xor(a, offset, 32);
        b += __shfl_xor(b, offset, 32);
    }
}

// Load one token's data 
__device__ __forceinline__ void load_one_token(
    const unsigned char* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    int bt_base, int kv_pos,
    int block_shift, int block_mask,
    int stride_cb, int stride_cp,
    long long kv_head_offset, int byte_idx,
    unsigned short& mse, unsigned short& norm, 
    unsigned short& val, unsigned int& sz)
{
    const int page_idx = kv_pos >> block_shift;
    const int page_off = kv_pos & block_mask;
    const int block_num = block_table[bt_base + page_idx];
    const long long slot_base = (long long)block_num * stride_cb + page_off * stride_cp + kv_head_offset;
    
    mse = *reinterpret_cast<const unsigned short*>(kv_cache + slot_base + byte_idx);
    norm = *reinterpret_cast<const unsigned short*>(kv_cache + slot_base + MSE_BYTES);
    val = *reinterpret_cast<const unsigned short*>(kv_cache + slot_base + KPS + byte_idx);
    sz = *reinterpret_cast<const unsigned int*>(kv_cache + slot_base + KPS + VAL_DATA_BYTES);
}

// Score one token and apply online softmax update
__device__ __forceinline__ void score_and_accumulate(
    unsigned short mse_u16, unsigned short norm_u16,
    unsigned short val_u16, unsigned int sz,
    const float* __restrict__ s_c,
    float q0, float q1, float q2, float q3,
    float attn_scale, int norm_correction,
    float& m_prev, float& l_prev,
    float& acc0, float& acc1, float& acc2, float& acc3)
{
    const float c0 = s_c[mse_u16 & 0xF];
    const float c1 = s_c[(mse_u16 >> 4) & 0xF];
    const float c2 = s_c[(mse_u16 >> 8) & 0xF];
    const float c3 = s_c[(mse_u16 >> 12)];

    float score = __fmaf_rn(q0, c0, __fmaf_rn(q1, c1, __fmaf_rn(q2, c2, q3 * c3)));
    float norm_sq = __fmaf_rn(c0, c0, __fmaf_rn(c1, c1, __fmaf_rn(c2, c2, c3 * c3)));
    wave_sum2_32(score, norm_sq);
    if (norm_correction) score *= rsqrtf(norm_sq + 1e-16f);
    
    const float vec_norm = __half2float(*reinterpret_cast<const __half*>(&norm_u16));
    const float s = vec_norm * score * attn_scale;
    
    // Single-token online softmax
    const float n_max = fmaxf(s, m_prev);
    const float re_scale = expf(m_prev - n_max);
    const float p = expf(s - n_max);
    
    acc0 *= re_scale; acc1 *= re_scale; acc2 *= re_scale; acc3 *= re_scale;
    
    const float v_scale = __half2float(*reinterpret_cast<const __half*>(&sz));
    const unsigned short zr = (sz >> 16);
    const float v_zero = __half2float(*reinterpret_cast<const __half*>(&zr));
    
    acc0 = __fmaf_rn(p, (float)(val_u16 & 0xF) * v_scale + v_zero, acc0);
    acc1 = __fmaf_rn(p, (float)((val_u16 >> 4) & 0xF) * v_scale + v_zero, acc1);
    acc2 = __fmaf_rn(p, (float)((val_u16 >> 8) & 0xF) * v_scale + v_zero, acc2);
    acc3 = __fmaf_rn(p, (float)((val_u16 >> 12) & 0xF) * v_scale + v_zero, acc3);
    
    l_prev = __fmaf_rn(l_prev, re_scale, p);
    m_prev = n_max;
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
    const int bid = blockIdx.x, hid = blockIdx.y, sid = blockIdx.z;
    const int kv_head = hid / kv_group_size;
    const int tid = threadIdx.x, warp_id = tid / 32, lane_id = tid % 32;
    const int seq_len = seq_lens[bid];
    const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = split_len * sid;
    const int split_end = min(split_start + split_len, seq_len);
    if (split_start >= split_end) return;

    __shared__ float s_c[N_CENTROIDS], s_m[NUM_WARPS], s_l[NUM_WARPS], s_acc[NUM_WARPS][HEAD_DIM];
    if (tid < N_CENTROIDS) s_c[tid] = centroids[tid];
    __syncthreads();

    const int q_base = bid * stride_qb + hid * stride_qh + lane_id * 4;
    const float4 qv = *reinterpret_cast<const float4*>(q_rot + q_base);
    float m_prev = -1e30f, l_prev = 0.0f, acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int bt_base = bid * stride_bt;
    const long long kv_head_offset = (long long)kv_head * stride_ch;
    const int byte_idx = lane_id * 2;
    const int block_shift = __ffs(block_size) - 1, block_mask = block_size - 1;
    const int stride_iter = BLOCK_KV * NUM_WARPS;

    // Process token-by-token with pipeline
    for (int base_n = split_start; base_n < split_end; base_n += stride_iter) {
        const int warp_base = base_n + warp_id * BLOCK_KV;
        
        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            const int kv_pos = warp_base + kv;
            if (kv_pos < split_end) {
                unsigned short mse, norm, val; unsigned int sz;
                load_one_token(kv_cache, block_table, bt_base, kv_pos,
                    block_shift, block_mask, stride_cb, stride_cp, kv_head_offset,
                    byte_idx, mse, norm, val, sz);
                score_and_accumulate(mse, norm, val, sz, s_c,
                    qv.x, qv.y, qv.z, qv.w, attn_scale, norm_correction,
                    m_prev, l_prev, acc0, acc1, acc2, acc3);
            }
        }
    }

    // Reduce
    if (lane_id == 0) { s_m[warp_id] = m_prev; s_l[warp_id] = l_prev; }
    s_acc[warp_id][lane_id*4] = acc0; s_acc[warp_id][lane_id*4+1] = acc1;
    s_acc[warp_id][lane_id*4+2] = acc2; s_acc[warp_id][lane_id*4+3] = acc3;
    __syncthreads();
    if (warp_id == 0) {
        float mf = s_m[0]; for (int w=1;w<NUM_WARPS;w++) mf=fmaxf(mf,s_m[w]);
        float lf=0; float sc[NUM_WARPS];
        for (int w=0;w<NUM_WARPS;w++){sc[w]=expf(s_m[w]-mf);lf+=s_l[w]*sc[w];}
        const float il=1.0f/fmaxf(lf,1e-16f); const int db=lane_id*4;
        float o0=0,o1=0,o2=0,o3=0;
        for (int w=0;w<NUM_WARPS;w++){o0+=s_acc[w][db]*sc[w];o1+=s_acc[w][db+1]*sc[w];o2+=s_acc[w][db+2]*sc[w];o3+=s_acc[w][db+3]*sc[w];}
        o0*=il;o1*=il;o2*=il;o3*=il;
        const int ob=bid*stride_mb+hid*stride_mh+sid*stride_ms;
        *reinterpret_cast<float4*>(mid_o+ob+db)=make_float4(o0,o1,o2,o3);
        if (lane_id==0) mid_o[ob+HEAD_DIM]=mf+logf(fmaxf(lf,1e-16f));
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
    dim3 grid(B, Hq, num_kv_splits); dim3 block(256, 1, 1);
    hipLaunchKernelGGL(tq_decode_stage1_hip, grid, block, 0, stream,
        q_rot, kv_cache, block_table, seq_lens, centroids, mid_o,
        stride_qb, stride_qh, stride_cb, stride_cp, stride_ch,
        stride_bt, stride_mb, stride_mh, stride_ms,
        num_kv_heads, block_size, num_kv_splits, kv_group_size,
        attn_scale, norm_correction);
}

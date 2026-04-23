/**
 * TQ Decode Stage2 V4 — Vectorized float4 loads + online softmax
 *
 * GEAK V4 over V3:
 *   1. float4 vector loads: 128B per load instead of 32B
 *      Each thread handles 4 dimensions → 32 threads cover D=128
 *   2. Block = 64 threads (1 warp on CDNA4)
 *      32 threads for data (4 dims each) + 32 threads for data
 *      Actually: 64 threads, each reads 2×float4 = 8 dims → 64*2=128 dims
 *      Wait, better: 32 threads, each handles 4 dims → need 128/4=32 threads
 *   3. Single warp = no LDS needed, pure register + shuffle
 *
 * Approach: 32 active threads (lanes 0-31), each handles 4 consecutive dims.
 * Lane 0 loads LSE and shuffles to all.
 * Remaining 32 lanes in warp are unused (but warp executes in lockstep anyway).
 *
 * Grid: (B, Hq)
 * Block: 32 threads
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define HEAD_DIM 128
#define WARP_SIZE 64
#define DIMS_PER_THREAD 4
#define NUM_DATA_THREADS 32  // 128 / 4 = 32

__device__ __forceinline__ hip_bfloat16 f32_to_bf16(float f) {
    return hip_bfloat16(f);
}

extern "C" __global__ __launch_bounds__(32, 16)
void tq_decode_stage2_v4_bf16(
    const float* __restrict__ mid_o,     // [B, Hq, NUM_KV_SPLITS, D+1]
    hip_bfloat16* __restrict__ output,   // [B, Hq, D] bf16
    const int* __restrict__ seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;  // 0..31

    const int seq_len = seq_lens[bid];
    if (seq_len <= 0) return;

    const int mid_base = bid * stride_mb + hid * stride_mh;
    const int dim_start = tid * DIMS_PER_THREAD;  // 0, 4, 8, ..., 124

    // Online softmax accumulators (4 dims per thread)
    float running_max = -1e30f;
    float l_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
        const int split_start = split_len * s;
        if (split_start >= seq_len) break;

        const int s_offset = mid_base + s * stride_ms;

        // Lane 0 loads LSE, broadcast via __shfl
        float lse;
        if (tid == 0) {
            lse = mid_o[s_offset + HEAD_DIM];
        }
        // On CDNA4 with 64-wide warps, all 32 threads are in same warp
        lse = __shfl(lse, 0, WARP_SIZE);

        float new_max = fmaxf(running_max, lse);
        float rescale = expf(running_max - new_max);
        float weight = expf(lse - new_max);

        // Vector load 4 floats (16 bytes) — coalesced across 32 threads = 512B
        const float4* src = reinterpret_cast<const float4*>(&mid_o[s_offset + dim_start]);
        float4 vals = *src;

        acc0 = acc0 * rescale + weight * vals.x;
        acc1 = acc1 * rescale + weight * vals.y;
        acc2 = acc2 * rescale + weight * vals.z;
        acc3 = acc3 * rescale + weight * vals.w;
        l_sum = l_sum * rescale + weight;
        running_max = new_max;
    }

    // Write output as bf16
    const int out_base = bid * stride_ob + hid * stride_oh;
    float safe_l = l_sum > 0.0f ? l_sum : 1.0f;
    float inv_l = 1.0f / safe_l;
    
    output[out_base + dim_start + 0] = f32_to_bf16(acc0 * inv_l);
    output[out_base + dim_start + 1] = f32_to_bf16(acc1 * inv_l);
    output[out_base + dim_start + 2] = f32_to_bf16(acc2 * inv_l);
    output[out_base + dim_start + 3] = f32_to_bf16(acc3 * inv_l);
}

extern "C" __global__ __launch_bounds__(32, 16)
void tq_decode_stage2_v4_f32(
    const float* __restrict__ mid_o,
    float* __restrict__ output,
    const int* __restrict__ seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[bid];
    if (seq_len <= 0) return;

    const int mid_base = bid * stride_mb + hid * stride_mh;
    const int dim_start = tid * DIMS_PER_THREAD;

    float running_max = -1e30f;
    float l_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
        const int split_start = split_len * s;
        if (split_start >= seq_len) break;

        const int s_offset = mid_base + s * stride_ms;

        float lse;
        if (tid == 0) {
            lse = mid_o[s_offset + HEAD_DIM];
        }
        lse = __shfl(lse, 0, WARP_SIZE);

        float new_max = fmaxf(running_max, lse);
        float rescale = expf(running_max - new_max);
        float weight = expf(lse - new_max);

        const float4* src = reinterpret_cast<const float4*>(&mid_o[s_offset + dim_start]);
        float4 vals = *src;

        acc0 = acc0 * rescale + weight * vals.x;
        acc1 = acc1 * rescale + weight * vals.y;
        acc2 = acc2 * rescale + weight * vals.z;
        acc3 = acc3 * rescale + weight * vals.w;
        l_sum = l_sum * rescale + weight;
        running_max = new_max;
    }

    const int out_base = bid * stride_ob + hid * stride_oh;
    float safe_l = l_sum > 0.0f ? l_sum : 1.0f;
    float inv_l = 1.0f / safe_l;

    float4* dst = reinterpret_cast<float4*>(&output[out_base + dim_start]);
    *dst = make_float4(acc0 * inv_l, acc1 * inv_l, acc2 * inv_l, acc3 * inv_l);
}

// Launch wrappers — same ABI as V2/V3
extern "C" void launch_tq_decode_stage2_bf16(
    const float* mid_o, hip_bfloat16* output, const int* seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits, int B, int Hq, hipStream_t stream
) {
    dim3 grid(B, Hq);
    dim3 block(32, 1, 1);
    hipLaunchKernelGGL(tq_decode_stage2_v4_bf16, grid, block, 0, stream,
        mid_o, output, seq_lens, stride_mb, stride_mh, stride_ms,
        stride_ob, stride_oh, num_kv_splits);
}

extern "C" void launch_tq_decode_stage2_f32(
    const float* mid_o, float* output, const int* seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits, int B, int Hq, hipStream_t stream
) {
    dim3 grid(B, Hq);
    dim3 block(32, 1, 1);
    hipLaunchKernelGGL(tq_decode_stage2_v4_f32, grid, block, 0, stream,
        mid_o, output, seq_lens, stride_mb, stride_mh, stride_ms,
        stride_ob, stride_oh, num_kv_splits);
}

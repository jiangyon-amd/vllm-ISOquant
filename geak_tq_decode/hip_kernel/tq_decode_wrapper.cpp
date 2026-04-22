/**
 * PyTorch/pybind11 wrapper for TQ Decode Stage1 HIP kernel (v52).
 *
 * NOTE: This wrapper is a LEGACY build path.  The production path uses
 *       the extern "C" launcher compiled directly into tq_decode_hip.so
 *       and called via ctypes from triton_turboquant_decode.py.
 *       If you need to rebuild, use hipcc directly:
 *         hipcc --offload-arch=gfx950 -shared -fPIC -O3 \
 *               -o tq_decode_v52_no_nt.so tq_decode_v52_no_nt.hip
 *
 * ABI: matches tq_decode_v52_no_nt.hip (no block_kv parameter).
 */

#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Forward declaration — must match tq_decode_v52_no_nt.hip
extern "C" __global__ void tq_decode_stage1_hip(
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
    int num_kv_heads,
    int block_size,
    int num_kv_splits,
    int kv_group_size,
    float attn_scale,
    int norm_correction
);

void launch_tq_decode_stage1(
    torch::Tensor q_rot,
    torch::Tensor kv_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor centroids,
    torch::Tensor mid_o,
    int num_kv_heads,
    int block_size,
    int num_kv_splits,
    int kv_group_size,
    float attn_scale,
    int norm_correction
) {
    int B = q_rot.size(0);
    int Hq = q_rot.size(1);

    dim3 grid(B, Hq, num_kv_splits);
    dim3 block(64, 1, 1);

    hipLaunchKernelGGL(
        tq_decode_stage1_hip,
        grid, block, 0, 0,
        q_rot.data_ptr<float>(),
        kv_cache.data_ptr<unsigned char>(),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        centroids.data_ptr<float>(),
        mid_o.data_ptr<float>(),
        q_rot.stride(0), q_rot.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        num_kv_heads, block_size, num_kv_splits, kv_group_size,
        attn_scale, norm_correction
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_tq_decode_stage1", &launch_tq_decode_stage1, "TQ Decode Stage1 HIP kernel");
}

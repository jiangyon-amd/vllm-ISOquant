"""
LEGACY build script for TQ decode HIP kernels via setuptools/pybind11.

The PRODUCTION build path uses hipcc directly:

    # v52 Stage1
    hipcc --offload-arch=gfx950 -shared -fPIC -O3 \
          -o tq_decode_v52_no_nt.so tq_decode_v52_no_nt.hip

    # v56 GEMV-fused Stage1
    hipcc --offload-arch=gfx950 -shared -fPIC -O3 \
          -o tq_decode_v56_gemv_fused.so tq_decode_v56_gemv_fused.hip

    # Stage2 reduce
    hipcc --offload-arch=gfx950 -shared -fPIC -O3 \
          -o tq_decode_stage2_v2.so tq_decode_stage2_v2.hip

Then copy the .so files to vllm/v1/attention/ops/:
    cp tq_decode_v52_no_nt.so    <repo>/vllm/v1/attention/ops/tq_decode_hip.so
    cp tq_decode_v56_gemv_fused.so <repo>/vllm/v1/attention/ops/tq_decode_v56_hip.so
    cp tq_decode_stage2_v2.so    <repo>/vllm/v1/attention/ops/tq_decode_stage2_hip.so

This setup.py is kept for reference only.  If you must use it, update
the --offload-arch to match your GPU (gfx942 for MI300X, gfx950 for MI325X).
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tq_decode_hip',
    ext_modules=[
        CUDAExtension(
            name='tq_decode_hip',
            sources=['tq_decode_wrapper.cpp', 'tq_decode_v52_no_nt.hip'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--offload-arch=gfx950']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

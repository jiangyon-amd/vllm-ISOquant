# TurboQuant SoA Fusion Path

This package hosts the opt-in full-stack TurboQuant SoA fusion path enabled by
`VLLM_TQ_SOA_FUSION=1`.

Current structure:

- `backend_impl.py`: local backend implementation that reuses the existing
  metadata builder / scheduler glue while switching the store + decode
  contract to the SoA / unified attention path.
- `external_ops.py`: loader for the sibling `vllm_tq_rocm_v3_sinks` checkout,
  which remains the source of truth for the Python/Triton kernels during
  bring-up.

The default production backend is intentionally unchanged.

import torch
import triton
import triton.language as tl


VARIANT_NAME = "round2"
MAIN_CHANGE = "Round 2 switches from linear ordering to grouped program ordering for better locality."


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
                  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
                  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
                  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=5, num_warps=8),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def round2_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    group_offset = pid % num_pid_in_group
    pid_m = first_pid_m + (group_offset % group_size_m)
    pid_n = group_offset // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k_start * BLOCK_K + offs_k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def _validate_inputs(a: torch.Tensor, b: torch.Tensor):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected two rank-2 tensors.")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 inputs.")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes.")


def _make_output(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    return M, N, K, c


def _make_grid(M: int, N: int):
    return lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )


def _serialize_config(config: triton.Config) -> dict:
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def get_best_config(a: torch.Tensor, b: torch.Tensor) -> dict:
    _validate_inputs(a, b)
    matmul(a, b)
    return _serialize_config(round2_matmul_kernel.best_config)


def _matmul_fixed_config(a: torch.Tensor, b: torch.Tensor, fixed_config: dict) -> torch.Tensor:
    M, N, K, c = _make_output(a, b)
    grid = _make_grid(M, N)
    round2_matmul_kernel.fn[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        **fixed_config["kwargs"],
        num_warps=fixed_config["num_warps"],
        num_stages=fixed_config["num_stages"],
    )
    return c


def matmul(a: torch.Tensor, b: torch.Tensor, fixed_config: dict | None = None) -> torch.Tensor:
    _validate_inputs(a, b)
    if fixed_config is not None:
        return _matmul_fixed_config(a, b, fixed_config)

    M, N, K, c = _make_output(a, b)

    grid = _make_grid(M, N)
    round2_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c

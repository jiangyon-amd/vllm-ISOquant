"""Test tl.range with mask load/store in Triton 3.5.1."""
import torch
import triton
import triton.language as tl


@triton.jit
def _test_dynamic_range(in_ptr, out_ptr, N):
    """Dynamic loop - NOT unrolled at compile time."""
    for i in tl.range(0, N, 32):
        offs = i + tl.arange(0, 32)
        mask = offs < N
        x = tl.load(in_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, x * 2, mask=mask)


@triton.jit
def _test_scatter_range(src_ptr, dst_ptr, ids_ptr, N_ids, N_src):
    """Dynamic loop scatter - similar to MoE sort."""
    pid = tl.program_id(0)
    for qb in tl.range(0, N_ids, 32):
        q = qb + tl.arange(0, 32)
        mask = q < N_ids
        tok = tl.load(ids_ptr + q, mask=mask, other=N_src)
        val = tl.load(src_ptr + tok, mask=mask & (tok < N_src), other=0)
        tl.store(dst_ptr + q, val, mask=mask)


if __name__ == "__main__":
    device = "cuda"

    # Test 1: dynamic range load/store
    x = torch.arange(100, dtype=torch.float32, device=device)
    y = torch.zeros(100, dtype=torch.float32, device=device)
    _test_dynamic_range[(1,)](x, y, 100)
    assert torch.equal(y, x * 2), f"FAIL: {y[:5]}"
    print("Test 1 (dynamic range): PASS")

    # Test 2: scatter with dynamic range
    N_src = 8
    N_ids = 4096  # like MoE sorted_ids
    src = torch.arange(N_src, dtype=torch.float32, device=device)
    ids = torch.randint(0, N_src, (N_ids,), dtype=torch.int32, device=device)
    dst = torch.zeros(N_ids, dtype=torch.float32, device=device)
    _test_scatter_range[(1,)](src, dst, ids, N_ids, N_src)

    expected = src[ids.long()]
    match = (dst == expected).sum().item()
    print(f"Test 2 (scatter 4096 ids): {match}/{N_ids} match = {'PASS' if match == N_ids else 'FAIL'}")

    # Test 3: compile time - dynamic N
    import time
    for N in [64, 4096, 8192]:
        t0 = time.perf_counter()
        _test_dynamic_range[(1,)](x[:N] if N <= 100 else torch.zeros(N, device=device),
                                   torch.zeros(N, device=device), N)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"Test 3 (N={N}): compile+run in {dt:.3f}s")

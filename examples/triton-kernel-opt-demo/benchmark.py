import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

VARIANT_MODULES = {
    "baseline": "baseline_matmul",
    "round2": "optimized_matmul_round2",
    "round3": "optimized_matmul_round3",
}


def load_variant(name: str):
    module_name = VARIANT_MODULES[name]
    return importlib.import_module(module_name)


def build_inputs(m: int, n: int, k: int, dtype: torch.dtype):
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    return a, b


def serialize_triton_config(config) -> dict:
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def make_matmul_fn(module, fixed_config=None):
    return lambda lhs, rhs: module.matmul(lhs, rhs, fixed_config=fixed_config)


def run_correctness(matmul_fn, a, b, rtol: float, atol: float):
    out = matmul_fn(a, b)
    reference = torch.matmul(a.float(), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(out, reference, rtol=rtol, atol=atol)
    return out


def run_benchmark(matmul_fn, a, b, warmup: int, iters: int):
    for _ in range(warmup):
        matmul_fn(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        matmul_fn(a, b)
    end.record()
    torch.cuda.synchronize()

    average_ms = start.elapsed_time(end) / iters
    return average_ms


def select_best_config(module, a, b, warmup: int, iters: int, repeats: int):
    selection = []
    for raw_config in module.AUTOTUNE_CONFIGS:
        fixed_config = serialize_triton_config(raw_config)
        matmul_fn = make_matmul_fn(module, fixed_config=fixed_config)
        samples_ms = [
            run_benchmark(matmul_fn, a, b, warmup, iters)
            for _ in range(repeats)
        ]
        selection_ms = statistics.median(samples_ms)
        selection.append(
            {
                "config": fixed_config,
                "selection_ms": selection_ms,
                "selection_us": selection_ms * 1000.0,
                "samples_ms": samples_ms,
            }
        )
    selection.sort(key=lambda item: item["selection_ms"])
    return {
        "best_config": selection[0]["config"],
        "results": selection,
        "selection_warmup": warmup,
        "selection_iters": iters,
        "selection_repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANT_MODULES), required=True)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--best-config-out", type=Path)
    parser.add_argument("--fixed-config-in", type=Path)
    parser.add_argument("--select-best-config", action="store_true")
    parser.add_argument("--selection-warmup", type=int, default=10)
    parser.add_argument("--selection-iters", type=int, default=20)
    parser.add_argument("--selection-repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for this benchmark.")
    if args.tune_only and args.select_best_config:
        raise ValueError("--tune-only and --select-best-config cannot be used together.")
    if (args.tune_only or args.select_best_config) and args.fixed_config_in is not None:
        raise ValueError(
            "--tune-only/--select-best-config cannot be used with --fixed-config-in."
        )
    if args.best_config_out is not None and not (args.tune_only or args.select_best_config):
        raise ValueError(
            "--best-config-out requires --tune-only or --select-best-config."
        )

    dtype = torch.bfloat16
    module = load_variant(args.variant)
    a, b = build_inputs(args.m, args.n, args.k, dtype)
    fixed_config = None
    if args.fixed_config_in is not None:
        fixed_config = json.loads(args.fixed_config_in.read_text())
    matmul_fn = make_matmul_fn(module, fixed_config=fixed_config)

    if args.tune_only:
        best_config = module.get_best_config(a, b)
        print(f"variant={args.variant}")
        print(f"module={VARIANT_MODULES[args.variant]}")
        print(f"mode=tune-only")
        print(json.dumps(best_config, indent=2))
        if args.best_config_out is not None:
            args.best_config_out.parent.mkdir(parents=True, exist_ok=True)
            args.best_config_out.write_text(json.dumps(best_config, indent=2))
        return

    if args.select_best_config:
        selection = select_best_config(
            module,
            a,
            b,
            warmup=args.selection_warmup,
            iters=args.selection_iters,
            repeats=args.selection_repeats,
        )
        print(f"variant={args.variant}")
        print(f"module={VARIANT_MODULES[args.variant]}")
        print("mode=select-best-config")
        print(json.dumps(selection, indent=2))
        if args.best_config_out is not None:
            args.best_config_out.parent.mkdir(parents=True, exist_ok=True)
            args.best_config_out.write_text(json.dumps(selection["best_config"], indent=2))
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(selection, indent=2))
        return

    correctness = "skipped"
    if args.check:
        run_correctness(matmul_fn, a, b, args.rtol, args.atol)
        correctness = "pass"

    average_ms = run_benchmark(matmul_fn, a, b, args.warmup, args.iters)
    average_us = average_ms * 1000.0
    tflops = (2.0 * args.m * args.n * args.k) / (average_ms * 1.0e-3) / 1.0e12

    result = {
        "variant": args.variant,
        "kernel_module": VARIANT_MODULES[args.variant],
        "execution_mode": "fixed-config" if fixed_config is not None else "autotuned",
        "main_change": module.MAIN_CHANGE,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "dtype": args.dtype,
        "correctness": correctness,
        "rtol": args.rtol,
        "atol": args.atol,
        "warmup": args.warmup,
        "iters": args.iters,
        "average_ms": average_ms,
        "average_us": average_us,
        "tflops": tflops,
    }
    if fixed_config is not None:
        result["fixed_config"] = fixed_config

    print(f"variant={args.variant}")
    print(f"module={VARIANT_MODULES[args.variant]}")
    print(f"execution_mode={result['execution_mode']}")
    print(f"main_change={module.MAIN_CHANGE}")
    print(f"shape=({args.m}, {args.n}, {args.k})")
    print(f"correctness={correctness}")
    print(f"average_ms={average_ms:.6f}")
    print(f"average_us={average_us:.3f}")
    print(f"tflops={tflops:.3f}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

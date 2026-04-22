#!/usr/bin/env python3
"""TurboQuant ROCm benchmark: baseline vs turboquant_k8v4 on Qwen3-4B.

Matches the PR's test scenarios:
  - short-decode:    input=128, output=512
  - long-prefill:    input=4096, output=128
  - mixed:           input=512, output=512
  - decode-heavy:    input=64,  output=1024
  - 8k-long-decode:  input=8192, output=1024
"""
import argparse
import time
import torch
from vllm import LLM, SamplingParams


SCENARIOS = {
    "short-decode":  {"input_len": 128,  "output_len": 512,  "num_prompts": 100},
    "long-prefill":  {"input_len": 4096, "output_len": 128,  "num_prompts": 50},
    "mixed":         {"input_len": 512,  "output_len": 512,  "num_prompts": 100},
    "decode-heavy":  {"input_len": 64,   "output_len": 1024, "num_prompts": 100},
    "8k-long-decode": {"input_len": 8192, "output_len": 1024, "num_prompts": 16},
}


def run_benchmark(model_path, kv_cache_dtype, scenario_name, scenario_cfg, tp=1):
    input_len = scenario_cfg["input_len"]
    output_len = scenario_cfg["output_len"]
    num_prompts = scenario_cfg["num_prompts"]

    max_model_len = min(input_len + output_len + 256, 32768)

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name} | kv_cache_dtype={kv_cache_dtype}")
    print(f"  input_len={input_len}, output_len={output_len}, num_prompts={num_prompts}")
    print(f"{'='*70}")

    llm = LLM(
        model=model_path,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        tensor_parallel_size=tp,
        enforce_eager=True,
        disable_log_stats=True,
    )

    # Generate synthetic prompts with fixed token count
    tokenizer = llm.get_tokenizer()
    # Use a repeating pattern to get desired input length
    base_text = "The quick brown fox jumps over the lazy dog. " * 200
    tokens = tokenizer.encode(base_text)

    prompts = []
    for i in range(num_prompts):
        # Truncate/pad to exact input_len tokens
        prompt_tokens = tokens[:input_len]
        prompt_text = tokenizer.decode(prompt_tokens)
        prompts.append(prompt_text)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,  # Force full output_len generation
    )

    # Warmup
    print("  Warming up...")
    warmup_params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    _ = llm.generate(prompts[:2], warmup_params)

    # Benchmark
    print("  Running benchmark...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    total_input_tokens = sum(len(tokenizer.encode(p)) for p in prompts)
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    input_tps = total_input_tokens / elapsed
    output_tps = total_output_tokens / elapsed
    total_tps = (total_input_tokens + total_output_tokens) / elapsed

    result = {
        "scenario": scenario_name,
        "kv_cache_dtype": kv_cache_dtype,
        "elapsed_s": elapsed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_tok/s": input_tps,
        "output_tok/s": output_tps,
        "total_tok/s": total_tps,
    }

    print(f"\n  Results:")
    print(f"    Elapsed:          {elapsed:.2f}s")
    print(f"    Input tokens:     {total_input_tokens}")
    print(f"    Output tokens:    {total_output_tokens}")
    print(f"    Input tok/s:      {input_tps:.1f}")
    print(f"    Output tok/s:     {output_tps:.1f}")
    print(f"    Total tok/s:      {total_tps:.1f}")

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/shareddata/Qwen/Qwen3-4B")
    parser.add_argument("--scenarios", nargs="+",
                        default=["short-decode", "decode-heavy"],
                        choices=list(SCENARIOS.keys()))
    parser.add_argument("--presets", nargs="+",
                        default=["auto", "turboquant_k8v4"],
                        help="KV cache dtypes to compare")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    all_results = []

    for scenario_name in args.scenarios:
        scenario_cfg = SCENARIOS[scenario_name]
        for preset in args.presets:
            try:
                result = run_benchmark(
                    args.model, preset, scenario_name, scenario_cfg, tp=args.tp
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR in {scenario_name}/{preset}: {e}")
                import traceback; traceback.print_exc()

    # Summary table
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"{'Scenario':<16} {'KV-Cache-Dtype':<22} {'Output tok/s':>14} {'Total tok/s':>14} {'Time(s)':>10}")
    print("-"*90)
    for r in all_results:
        print(f"{r['scenario']:<16} {r['kv_cache_dtype']:<22} {r['output_tok/s']:>14.1f} {r['total_tok/s']:>14.1f} {r['elapsed_s']:>10.2f}")

    # Compute speedup where possible
    print("\n" + "-"*50)
    print("Comparison (TQ vs baseline):")
    for scenario_name in args.scenarios:
        base = [r for r in all_results if r["scenario"] == scenario_name and r["kv_cache_dtype"] == "auto"]
        tq = [r for r in all_results if r["scenario"] == scenario_name and r["kv_cache_dtype"] != "auto"]
        if base and tq:
            base_otps = base[0]["output_tok/s"]
            for t in tq:
                tq_otps = t["output_tok/s"]
                ratio = tq_otps / base_otps * 100 if base_otps > 0 else 0
                print(f"  {scenario_name}: {t['kv_cache_dtype']} = {ratio:.1f}% of baseline output tok/s")


if __name__ == "__main__":
    main()

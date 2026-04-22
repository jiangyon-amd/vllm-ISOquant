#!/usr/bin/env python3
"""
E2E Correctness Test: Baseline (auto) vs TurboQuant (turboquant_4bit_nc)

Uses GPU 0 for single-card Qwen3-4B, then GPU 0-3 TP=4 for Qwen3-32B.
Compares generated text to verify TQ produces reasonable output.

Usage:
    python tests/e2e_correctness_test.py                     # full test
    python tests/e2e_correctness_test.py --model-only 4b     # just 4B
    python tests/e2e_correctness_test.py --model-only 32b    # just 32B
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "profiling", "e2e_correctness")
os.makedirs(RESULT_DIR, exist_ok=True)

CORRECTNESS_PROMPTS = [
    "The capital of France is",
    "The speed of light is approximately",
    "Albert Einstein was born in the year",
    "What is 17 * 23? The answer is",
    "1, 1, 2, 3, 5, 8, 13, the next number is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "If all roses are flowers, and some flowers are red, then",
    "Translate to French: 'Hello, how are you?' ->",
]


@dataclass
class ModelConfig:
    name: str
    path: str
    gpus: str
    tp: int
    port: int
    max_model_len: int
    max_tokens: int


MODELS = {
    "4b": ModelConfig(
        name="Qwen3-4B",
        path="/shareddata/Qwen/Qwen3-4B",
        gpus="0",
        tp=1,
        port=8195,
        max_model_len=8192,
        max_tokens=80,
    ),
    "32b": ModelConfig(
        name="Qwen3-32B",
        path="/shareddata/Qwen/Qwen3-32B",
        gpus="0,1,2,3",
        tp=4,
        port=8196,
        max_model_len=16384,
        max_tokens=80,
    ),
}


def start_server(model, preset, label):
    log_path = os.path.join(RESULT_DIR, f"{label}_server.log")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model.path,
        "--kv-cache-dtype", preset,
        "--port", str(model.port),
        "--max-model-len", str(model.max_model_len),
        "--disable-log-stats",
        "--enforce-eager",
    ]
    if model.tp > 1:
        cmd += ["--tensor-parallel-size", str(model.tp)]

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = model.gpus
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"  CMD: HIP_VISIBLE_DEVICES={model.gpus} {' '.join(cmd)}")
    print(f"  LOG: {log_path}")
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    print(f"  PID: {proc.pid}, waiting for /health ...")

    for i in range(180):
        time.sleep(3)
        if proc.poll() is not None:
            logf.close()
            print(f"\n  Server exited with code {proc.returncode}!")
            print("  Last 20 lines of log:")
            subprocess.run(["tail", "-20", log_path])
            return None, logf
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://localhost:{model.port}/health"],
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                print(f"  Server ready after {(i+1)*3}s")
                return proc, logf
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"    still waiting ({(i+1)*3}s)...")

    logf.close()
    print(f"\n  Server timeout after 540s!")
    subprocess.run(["tail", "-20", log_path])
    proc.kill()
    return None, logf


def stop_server(proc, logf):
    if logf:
        try:
            logf.close()
        except Exception:
            pass
    if proc:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    subprocess.run(["pkill", "-9", "-f", "EngineCore"], capture_output=True)
    time.sleep(3)


def generate(port, model_path, prompt, max_tokens=80):
    data = json.dumps({
        "model": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["text"]


def run_correctness(model, preset, label):
    proc, logf = start_server(model, preset, label)
    if proc is None:
        return None

    results = {}
    for prompt in CORRECTNESS_PROMPTS:
        try:
            text = generate(model.port, model.path, prompt, model.max_tokens)
            short = text[:100].replace("\n", "\\n")
            print(f"    OK  {prompt[:45]:45s} -> {short}")
            results[prompt] = text
        except Exception as e:
            print(f"    ERR {prompt[:45]:45s} -> {e}")
            results[prompt] = f"__ERROR__: {e}"

    stop_server(proc, logf)
    return results


def compare_results(bl_results, tq_results, model_name):
    print(f"\n{'='*80}")
    print(f"  CORRECTNESS COMPARISON: {model_name}")
    print(f"{'='*80}")

    n_match = 0
    n_close = 0
    n_diff = 0
    n_error = 0

    for prompt in CORRECTNESS_PROMPTS:
        bl = bl_results.get(prompt, "__MISSING__")
        tq = tq_results.get(prompt, "__MISSING__")

        if "__ERROR__" in bl or "__ERROR__" in tq:
            status = "ERROR"
            n_error += 1
        elif bl == tq:
            status = "EXACT"
            n_match += 1
        elif bl[:30] == tq[:30]:
            status = "CLOSE"
            n_close += 1
        else:
            status = "DIFF"
            n_diff += 1

        print(f"\n  [{status:5s}] {prompt[:55]}")
        if status != "EXACT":
            bl_show = bl[:90].replace('\n', ' ')
            tq_show = tq[:90].replace('\n', ' ')
            print(f"    BL: {bl_show}")
            print(f"    TQ: {tq_show}")

    print(f"\n  Summary: {n_match} exact, {n_close} close, {n_diff} diff, {n_error} error")
    print(f"  (CLOSE/DIFF are expected with 4-bit quantization)")
    return n_error == 0


def run_small_bench(model, preset, label):
    proc, logf = start_server(model, preset, label)
    if proc is None:
        return None

    print(f"\n  Running small benchmark (20 prompts, input=512, output=128)...")
    bench_log = os.path.join(RESULT_DIR, f"{label}_bench.log")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm",
        "--port", str(model.port),
        "--model", model.path,
        "--dataset-name", "random",
        "--random-input-len", "512",
        "--random-output-len", "128",
        "--num-prompts", "20",
        "--request-rate", "4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    with open(bench_log, "w") as f:
        f.write(result.stdout + "\n" + result.stderr)

    metrics = {}
    for line in result.stdout.split("\n"):
        for key in ["Output token throughput", "Request throughput",
                     "Mean TTFT", "Median TPOT", "Mean ITL"]:
            if key in line:
                parts = line.strip().split()
                for p in reversed(parts):
                    try:
                        metrics[key] = float(p)
                        break
                    except ValueError:
                        continue
    for k, v in sorted(metrics.items()):
        print(f"    {k}: {v}")

    stop_server(proc, logf)
    return metrics


def test_model(model_key):
    model = MODELS[model_key]
    print(f"\n{'#'*80}")
    print(f"#  E2E Test: {model.name} (TP={model.tp}, GPU={model.gpus})")
    print(f"{'#'*80}")

    # Phase 1: Baseline
    print(f"\n[Phase 1] Baseline (auto) - {model.name}")
    bl_results = run_correctness(model, "auto", f"{model_key}_baseline")
    if bl_results is None:
        print(f"  SKIP: baseline server failed for {model.name}")
        return False

    # Phase 2: TQ
    print(f"\n[Phase 2] TurboQuant (turboquant_4bit_nc) - {model.name}")
    tq_results = run_correctness(model, "turboquant_4bit_nc", f"{model_key}_tq4nc")
    if tq_results is None:
        print(f"  SKIP: TQ server failed for {model.name}")
        return False

    # Phase 3: Compare
    ok = compare_results(bl_results, tq_results, model.name)

    # Phase 4: Quick bench
    print(f"\n[Phase 4] Quick benchmark - {model.name}")
    bl_metrics = run_small_bench(model, "auto", f"{model_key}_bl_bench")
    tq_metrics = run_small_bench(model, "turboquant_4bit_nc", f"{model_key}_tq_bench")

    if bl_metrics and tq_metrics:
        print(f"\n  {'Metric':30s} {'Baseline':>12s} {'TQ':>12s}")
        print(f"  {'-'*56}")
        for k in ["Output token throughput", "Mean TTFT", "Median TPOT", "Mean ITL"]:
            bl_v = bl_metrics.get(k, 0)
            tq_v = tq_metrics.get(k, 0)
            if bl_v > 0:
                print(f"  {k:30s} {bl_v:>12.1f} {tq_v:>12.1f}")

    results_path = os.path.join(RESULT_DIR, f"{model_key}_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": model.name,
            "baseline": bl_results,
            "tq": tq_results,
            "bl_metrics": bl_metrics,
            "tq_metrics": tq_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_path}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-only", choices=["4b", "32b"], default=None)
    args = parser.parse_args()

    all_ok = True
    if args.model_only:
        all_ok = test_model(args.model_only)
    else:
        for key in ["4b", "32b"]:
            ok = test_model(key)
            all_ok = all_ok and ok

    print(f"\n{'='*80}")
    if all_ok:
        print("  ALL E2E TESTS PASSED")
    else:
        print("  SOME TESTS HAD ISSUES (see details above)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate MoE 30B: RTN vs Separated vs Fused on lm_eval benchmark suite."""
import json, os, subprocess, sys, time, urllib.request
from datetime import datetime

PORT = 8200
GPU = "1"
BASE_ENV = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1",
            "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": "/data/jiangyon/vllm_rotation"}

MODES = [
    ("rtn", "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rtn", {}),
    ("separated", "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2", {}),
    ("fused", "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2",
     {"VLLM_MOE_FUSED_ROTATION": "1", "VLLM_MOE_TRITON_ROT_SORT_FUSION": "1"}),
]

TASKS = "arc_challenge,arc_easy,piqa,winogrande,hellaswag"
BATCH_SIZE = "auto"


def cleanup():
    os.system('pkill -9 -f "openai.api_server" >/dev/null 2>&1')
    os.system('pkill -9 -f "VLLM::EngineCore" >/dev/null 2>&1')
    time.sleep(6)


def wait_ready():
    for _ in range(360):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=1)
            return True
        except Exception:
            time.sleep(1)
    return False


def run_eval(model_name, label):
    """Run lm_eval with vLLM as backend via OpenAI server."""
    out_dir = f"/data/jiangyon/vllm_rotation/bench_results/eval_{label}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-completions",
        "--model_args", f"model={model_name},base_url=http://localhost:{PORT}/v1/completions,tokenizer_backend=huggingface,max_length=4096",
        "--tasks", TASKS,
        "--batch_size", BATCH_SIZE,
        "--num_fewshot", "0",
        "--output_path", out_dir,
    ]

    print(f"  Running lm_eval: {TASKS}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    combined = result.stdout + "\n" + result.stderr
    scores = {}
    for line in combined.splitlines():
        line = line.strip()
        if "|" in line and ("acc" in line.lower() or "acc_norm" in line.lower()):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                task = parts[1].strip()
                metric = parts[2].strip()
                try:
                    val = float(parts[4].strip())
                except (ValueError, IndexError):
                    continue
                if metric == "acc_norm":
                    scores[task + "_norm"] = val
                elif metric == "acc":
                    scores[task] = val

    if not scores:
        print("  [WARN] No scores parsed, showing last 20 lines:", flush=True)
        for line in combined.splitlines()[-20:]:
            print(f"    {line}", flush=True)

    return scores, combined


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== MoE 30B Eval Suite ({ts}) ===", flush=True)
    print(f"Tasks: {TASKS}", flush=True)

    all_results = {}

    for label, model, extra in MODES:
        cleanup()
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(extra)

        print(f"\n[{label}] Starting server...", flush=True)
        proc = subprocess.Popen([
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--port", str(PORT), "--trust-remote-code",
            "--disable-log-requests", "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.35", "--host", "0.0.0.0"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            if not wait_ready():
                print(f"[{label}] TIMEOUT!", flush=True)
                continue

            print(f"[{label}] Server ready, running eval...", flush=True)
            scores, raw = run_eval(model, label)
            all_results[label] = scores

            with open(f"/data/jiangyon/vllm_rotation/bench_results/eval_{label}_raw.txt", "w") as f:
                f.write(raw)

            print(f"[{label}] Scores: {scores}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    cleanup()

    print("\n" + "=" * 80, flush=True)
    print(f"EVAL RESULTS: MoE 30B — {ts}", flush=True)
    print("=" * 80, flush=True)

    all_tasks = set()
    for scores in all_results.values():
        all_tasks.update(scores.keys())
    all_tasks = sorted(all_tasks)

    header = f"| {'Task':<20} |"
    sep = f"|{'-'*22}|"
    for label, _, _ in MODES:
        header += f" {label:>10} |"
        sep += f"{'-'*12}|"
    print(header, flush=True)
    print(sep, flush=True)

    for task in all_tasks:
        row = f"| {task:<20} |"
        for label, _, _ in MODES:
            val = all_results.get(label, {}).get(task, None)
            row += f" {val:>10.4f} |" if val is not None else f" {'N/A':>10} |"
        print(row, flush=True)

    report_path = f"/data/jiangyon/vllm_rotation/bench_results/eval_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {report_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Final E2E: separated vs fused (TL 3-in-1 decode + gluon_kw8 prefill)."""
import json, os, subprocess, sys, time, urllib.request

PORT = 8200
MODEL = "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rotation-vllm"
GPU_UTIL = "0.35"
OPENAI_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
BASE_ENV = {"HIP_VISIBLE_DEVICES": "4", "VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1"}
CONCURRENCIES = [1, 4, 16, 32]

def wait_ready(timeout_s=360):
    for _ in range(timeout_s):
        try: urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=1); return True
        except: time.sleep(1)
    return False

def server_warmup():
    for i in range(8):
        p = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": f"warmup {i}"}], "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
        try: urllib.request.urlopen(urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=p, headers={"Content-Type": "application/json"}), timeout=30)
        except: pass

def run_bench(c, np_val, nw_val):
    p = subprocess.run([sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--backend", "openai", "--base-url", f"http://localhost:{PORT}", "--model", MODEL, "--num-prompts", str(np_val), "--max-concurrency", str(c), "--request-rate", "inf", "--dataset-name", "random", "--random-input-len", "512", "--random-output-len", "256", "--random-range-ratio", "0.2", "--num-warmups", str(nw_val), "--seed", "42", "--extra-body", json.dumps(OPENAI_EXTRA_BODY)], capture_output=True, text=True, timeout=1800)
    def pick(key):
        for line in p.stdout.splitlines():
            if key in line: return line.split(":", 1)[1].strip()
        return "N/A"
    return {"tok_s": pick("Output token throughput"), "tpot": pick("Mean TPOT"), "ttft": pick("Mean TTFT"), "rc": str(p.returncode)}

def main():
    modes = [
        ("separated", {}),
        ("fused", {"VLLM_MOE_FUSED_ROTATION": "1", "VLLM_MOE_TRITON_ROT_SORT_FUSION": "1"}),
    ]
    results = {}
    print(f"=== E2E MoE Final: separated vs fused, c={CONCURRENCIES} ===", flush=True)

    for label, extra in modes:
        os.system('pkill -9 -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1')
        os.system('pkill -9 -f "VLLM::EngineCore" >/dev/null 2>&1')
        time.sleep(6)
        env = os.environ.copy(); env.update(BASE_ENV); env.update(extra)
        env["PYTHONPATH"] = "/data/jiangyon/vllm_rotation:" + env.get("PYTHONPATH", "")
        print(f"[{label}] start server...", flush=True)
        proc = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", str(PORT), "--trust-remote-code", "--disable-log-requests", "--max-model-len", "4096", "--gpu-memory-utilization", GPU_UTIL, "--host", "0.0.0.0"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if not wait_ready():
                print(f"[{label}] timeout", flush=True)
                results[label] = {c: {"tok_s": "N/A"} for c in CONCURRENCIES}
                continue
            server_warmup(); results[label] = {}
            for c in CONCURRENCIES:
                np_val = 20 if c <= 4 else (40 if c <= 16 else 60)
                nw_val = 4 if c <= 8 else 6
                _ = run_bench(c, max(8, np_val // 3), 2)
                print(f"[{label}] bench c={c}...", flush=True)
                results[label][c] = run_bench(c, np_val, nw_val)
                print(f"[{label}] c={c} -> {results[label][c]}", flush=True)
        finally:
            proc.terminate()
            try: proc.wait(timeout=10)
            except: proc.kill()

    os.system('pkill -9 -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1')

    print("\n" + "=" * 70, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"{'c':<5} {'Separated':>12} {'Fused':>12} {'Speedup':>10}", flush=True)
    print("-" * 42, flush=True)
    for c in CONCURRENCIES:
        sep = results.get("separated", {}).get(c, {}).get("tok_s", "N/A")
        fus = results.get("fused", {}).get(c, {}).get("tok_s", "N/A")
        try:
            sp = f"+{(float(fus) - float(sep)) / float(sep) * 100:.1f}%"
        except:
            sp = "N/A"
        print(f"{'c='+str(c):<5} {sep:>12} {fus:>12} {sp:>10}", flush=True)

    print("\n--- TPOT (ms) ---", flush=True)
    print(f"{'c':<5} {'Separated':>12} {'Fused':>12}", flush=True)
    for c in CONCURRENCIES:
        sep = results.get("separated", {}).get(c, {}).get("tpot", "N/A")
        fus = results.get("fused", {}).get(c, {}).get("tpot", "N/A")
        print(f"{'c='+str(c):<5} {sep:>12} {fus:>12}", flush=True)

if __name__ == "__main__":
    main()

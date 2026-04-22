#!/usr/bin/env python3
"""Compare outputs of RTN vs Separated vs Fused on same prompts."""
import json, os, subprocess, sys, time, urllib.request

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

PROMPTS = [
    "只回答数字：1+1等于几？",
    "只回答数字：100-37等于几？",
    "What is 15 * 17? Only answer the number.",
    "法国的首都是哪里？只回答城市名。",
    "Write a haiku about the moon.",
    "Explain quantum computing in one sentence.",
]


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


def query(model, prompt, max_tokens=32):
    p = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                f"http://localhost:{PORT}/v1/chat/completions",
                data=p, headers={"Content-Type": "application/json"}), timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERR: {e}"


def main():
    results = {}
    for label, model, extra in MODES:
        cleanup()
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(extra)
        print(f"[{label}] Starting...", flush=True)
        proc = subprocess.Popen([
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--port", str(PORT), "--trust-remote-code",
            "--disable-log-requests", "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.35", "--host", "0.0.0.0"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if not wait_ready():
                print(f"[{label}] TIMEOUT", flush=True)
                continue
            for i in range(4):
                query(model, f"warmup {i}", 16)
            results[label] = []
            for prompt in PROMPTS:
                ans = query(model, prompt)
                results[label].append(ans)
                print(f"  [{label}] {prompt[:25]}... → {ans[:40]}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    cleanup()

    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON: RTN vs Separated vs Fused")
    print("=" * 80)
    for i, prompt in enumerate(PROMPTS):
        print(f"\nQ{i+1}: {prompt}")
        for label, _, _ in MODES:
            ans = results.get(label, ["N/A"] * len(PROMPTS))[i]
            tag = ""
            if label == "separated" and "rtn" in results:
                tag = " [= RTN]" if ans == results["rtn"][i] else " [≠ RTN]"
            elif label == "fused" and "separated" in results:
                tag = " [= Sep]" if ans == results["separated"][i] else " [≠ Sep]"
            print(f"  {label:>10}: {ans[:70]}{tag}")

    if "separated" in results and "fused" in results:
        identical = sum(1 for s, f in zip(results["separated"], results["fused"]) if s == f)
        print(f"\nSeparated vs Fused: {identical}/{len(PROMPTS)} identical outputs")


if __name__ == "__main__":
    main()

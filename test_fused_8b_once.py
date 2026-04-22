#!/usr/bin/env python3
"""单次测试：仅 Fused 路径启动 vLLM + 一次推理，结果写文件便于查看。"""
import sys
import os
import time
import signal
import subprocess
import requests

# 与 bench_3way_8b 一致
ROT_MODEL = "/data/jiangyon/vllm_rotation/qwen3-8b-mxfp4-hadamard-r128"
PORT = 8200
GPU = os.environ.get("BENCH_GPU", "1")
LOG = "/data/jiangyon/vllm_rotation/test_fused_8b_result.txt"

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

def main():
    if os.path.exists(LOG):
        os.remove(LOG)
    log("=== Fused 8B 单次测试开始 ===")

    # 先清理可能占端口的进程
    subprocess.run(["pkill", "-f", f"api_server.*{PORT}"], capture_output=True, timeout=5)
    time.sleep(5)

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = GPU
    env["VLLM_ROCM_USE_AITER"] = "1"
    env["VLLM_ROCM_USE_AITER_FP4_ASM_GEMM"] = "1"
    # 不设 VLLM_DISABLE_FUSED_ROT_QUANT，即 Fused 路径
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", ROT_MODEL, "--port", str(PORT),
        "--trust-remote-code", "--disable-log-requests",
        "--max-model-len", "4096", "--gpu-memory-utilization", "0.35",
        "--host", "0.0.0.0",
    ]
    log("启动服务 (Fused)...")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                r = requests.get(f"http://localhost:{PORT}/health", timeout=5)
                if r.status_code == 200:
                    log("服务已就绪.")
                    break
            except Exception as e:
                log(f"  等待健康检查: {e}")
            time.sleep(10)
        else:
            log("ERROR: 服务 600s 内未就绪")
            return 1

        # 一次推理
        log("发送一次 chat completion...")
        t0 = time.time()
        r = requests.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            json={
                "model": ROT_MODEL,
                "messages": [{"role": "user", "content": "What is 2+2? Reply in one short sentence."}],
                "max_tokens": 64,
                "temperature": 0,
            },
            timeout=120,
        )
        elapsed = time.time() - t0
        d = r.json()
        if "choices" not in d or not d["choices"]:
            log(f"ERROR 响应: {d}")
            return 1
        usage = d.get("usage", {})
        content = d["choices"][0]["message"]["content"]
        ct = usage.get("completion_tokens", 0)
        tps = ct / elapsed if elapsed > 0 else 0
        log(f"完成: {ct} tokens, {elapsed:.2f}s, {tps:.1f} tok/s")
        log(f"回复预览: {content[:200]}")
        log("=== Fused 8B 单次测试通过 ===")
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log("服务已停止.")

if __name__ == "__main__":
    sys.exit(main())

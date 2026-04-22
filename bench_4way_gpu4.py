#!/usr/bin/env python3
"""
4-Way Benchmark: Separated / Attn-Fused / MoE-Fused / Both-Fused
Qwen3-30B-A3B on GPU 4

Tests: correctness (QA), performance (TPOT), quality (wikitext PPL, gsm8k, arc)
"""
import json, os, subprocess, sys, time, urllib.request, csv
from datetime import datetime

GPU = "4"
PORT = 8400
RTN_MODEL = "qwen3-30b-mxfp4-rtn"
ROT_MODEL = "qwen3-30b-mxfp4-trained-r128-vllm-v2"
DATASET = "/data/jiangyon/hf_cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
OUT = f"bench_results/4way_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

MODES = [
    ("separated", ROT_MODEL, {
        "VLLM_USE_FUSED_ROTATION_QUANT": "0",
        "VLLM_MOE_FUSED_ROTATION": "0",
    }),
    ("attn-fused", ROT_MODEL, {
        "VLLM_USE_FUSED_ROTATION_QUANT": "1",
        "VLLM_MOE_FUSED_ROTATION": "0",
    }),
    ("moe-fused", ROT_MODEL, {
        "VLLM_USE_FUSED_ROTATION_QUANT": "0",
        "VLLM_MOE_FUSED_ROTATION": "1",
    }),
    ("both-fused", ROT_MODEL, {
        "VLLM_USE_FUSED_ROTATION_QUANT": "1",
        "VLLM_MOE_FUSED_ROTATION": "1",
    }),
]

BASE_ENV = {
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1",
    "TRITON_CACHE_DIR": "/tmp/jiangyon_triton_cache",
    "VLLM_CACHE_ROOT": "/tmp/jiangyon_vllm_cache",
    "VLLM_NO_USAGE_STATS": "1",
    "XDG_CONFIG_HOME": "/tmp/jiangyon_config",
    "HIP_VISIBLE_DEVICES": GPU,
}

PERF_ROUNDS = 2
LM_EVAL_LIMIT = 500

def log(m):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {m}", flush=True)

def cleanup():
    os.system("kill -9 $(pgrep -f 'api_server.*8400') $(pgreg -f EngineCore) 2>/dev/null")
    os.system("kill -9 $(pgrep -f EngineCore) 2>/dev/null")
    time.sleep(10)

def wait_gpu_free(timeout=120):
    for _ in range(timeout // 5):
        try:
            out = subprocess.check_output(
                f"rocm-smi --showmeminfo vram 2>/dev/null | grep 'GPU\\[{GPU}\\]' | grep Used",
                shell=True, text=True)
            used = int(out.strip().split(":")[-1].strip())
            if used < 2_000_000_000:
                return True
        except: pass
        time.sleep(5)
    return False

def start_server(model, extra_env, label):
    env = os.environ.copy()
    env.update(BASE_ENV)
    for k in ["VLLM_USE_FUSED_ROTATION_QUANT", "VLLM_MOE_FUSED_ROTATION"]:
        env.pop(k, None)
    env.update(extra_env)

    server_log = open(f"{OUT}/server_{label}.log", "w")
    proc = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--port", str(PORT), "--trust-remote-code",
        "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
        "--host", "0.0.0.0", "--disable-log-requests",
    ], env=env, stdout=server_log, stderr=subprocess.STDOUT)

    for w in range(600):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            log(f"  Server ready ({w}s)")
            return proc, server_log
        except:
            if proc.poll() is not None:
                log(f"  Server DIED! Check {OUT}/server_{label}.log")
                server_log.close()
                return None, None
            time.sleep(1)
    log("  Server TIMEOUT!")
    proc.terminate(); server_log.close()
    return None, None

def stop_server(proc, server_log):
    if proc:
        proc.terminate()
        try: proc.wait(15)
        except: proc.kill()
    if server_log:
        server_log.close()
    cleanup()
    wait_gpu_free(90)

def check_correctness(model):
    results = {}
    for q, expect in [
        ("只回答数字：1+1等于几？", "2"),
        ("只回答数字：3*3等于几？", "9"),
        ("What is the capital of France? One word.", "paris"),
    ]:
        try:
            data = json.dumps({
                "model": model, "messages": [{"role": "user", "content": q}],
                "max_tokens": 32, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}
            }).encode()
            req = urllib.request.Request(
                f"http://localhost:{PORT}/v1/chat/completions",
                data=data, headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            answer = resp["choices"][0]["message"]["content"].strip()
            ok = expect.lower() in answer.lower()
            results[q] = {"answer": answer, "pass": ok}
            log(f"    Q: {q} → A: {answer} {'✓' if ok else '✗'}")
        except Exception as e:
            results[q] = {"answer": str(e), "pass": False}
            log(f"    Q: {q} → ERROR: {e}")
    return results

def check_server_log(label):
    logf = f"{OUT}/server_{label}.log"
    info = {}
    try:
        with open(logf) as f:
            content = f.read()
        # CUDAGraph
        if "CUDAGraphMode.FULL_AND_PIECEWISE" in content:
            info["cudagraph"] = "FULL_AND_PIECEWISE ✓"
        else:
            info["cudagraph"] = "NOT FOUND ✗"
        # Fused dispatch
        attn_fused = content.count("Using fused Triton rotation")
        attn_sep = content.count("Using separated rotation")
        info["attn_dispatch"] = f"fused={attn_fused} separated={attn_sep}"
        # MoE dispatch
        moe_fused = content.count("mode=fused")
        moe_sep = content.count("mode=separated")
        moe_mfma = content.count("HIP MFMA rotation branch")
        info["moe_dispatch"] = f"fused={moe_fused} sep={moe_sep} hip_mfma={moe_mfma}"
        # enforce_eager
        if "enforce_eager=False" in content or "enforce_eager" not in content:
            info["enforce_eager"] = "False ✓"
    except:
        info["error"] = "log read failed"
    return info

def run_perf(model, label, csv_writer):
    # Warmup
    log("  Perf: warmup (64 reqs)...")
    subprocess.run([
        "vllm", "bench", "serve", "--backend", "vllm",
        "--base-url", f"http://localhost:{PORT}",
        "--model", model, "--dataset-name", "sharegpt",
        "--dataset-path", DATASET,
        "--num-prompts", "64", "--request-rate", "16",
    ], capture_output=True, timeout=300)
    time.sleep(2)

    for r in range(1, PERF_ROUNDS + 1):
        log(f"  Perf: round {r}/{PERF_ROUNDS}")
        for c in [1, 4, 16, 32]:
            try:
                p = subprocess.run([
                    "vllm", "bench", "serve", "--backend", "vllm",
                    "--base-url", f"http://localhost:{PORT}",
                    "--model", model, "--dataset-name", "sharegpt",
                    "--dataset-path", DATASET,
                    "--num-prompts", "128", "--request-rate", str(c),
                ], capture_output=True, text=True, timeout=600)
                tpot = ttft = thr = ""
                for line in p.stdout.splitlines():
                    if "Mean TPOT" in line: tpot = line.split()[-1]
                    if "Mean TTFT" in line: ttft = line.split()[-1]
                    if "Output token throughput" in line and "Peak" not in line:
                        thr = line.split()[-1]
                log(f"    c={c}: TPOT={tpot}ms TTFT={ttft}ms thr={thr}")
                csv_writer.writerow([label, r, c, tpot, ttft, thr])
            except Exception as e:
                log(f"    c={c}: ERROR {e}")
                csv_writer.writerow([label, r, c, "ERR", "ERR", "ERR"])

def run_quality(model, label):
    results = {}
    tasks = [
        ("wikitext", 0),
        ("gsm8k", 5),
        ("arc_challenge", 25),
    ]
    lm_env = os.environ.copy()
    lm_env["PYTHONPATH"] = "/data/jiangyon/vllm_rotation:" + lm_env.get("PYTHONPATH", "")

    for task, nshot in tasks:
        log(f"  Quality: {task} ({nshot}-shot)...")
        try:
            p = subprocess.run([
                sys.executable, "-m", "lm_eval",
                "--model", "local-completions",
                "--model_args", f"model={model},base_url=http://localhost:{PORT}/v1/completions,num_concurrent=4,tokenized_requests=False",
                "--tasks", task,
                "--num_fewshot", str(nshot),
                "--limit", str(LM_EVAL_LIMIT),
                "--output_path", f"{OUT}/lm_eval_{label}_{task}",
                "--log_samples",
            ], capture_output=True, text=True, timeout=7200, env=lm_env)

            # Save raw output
            with open(f"{OUT}/lm_eval_{label}_{task}_raw.txt", "w") as f:
                f.write(p.stdout)

            # Parse table
            for line in p.stdout.splitlines():
                if "|" not in line: continue
                parts = [x.strip() for x in line.split("|")]
                parts = [x for x in parts if x]
                if len(parts) >= 6 and parts[0] not in ("-", "Tasks", ""):
                    try:
                        metric, val = parts[3], float(parts[5])
                        stderr = float(parts[7]) if len(parts) > 7 else 0
                        key = f"{parts[0]}/{metric}"
                        results[key] = {"value": val, "stderr": stderr}
                        log(f"    {key}: {val:.4f} ± {stderr:.4f}")
                    except: pass
                elif len(parts) >= 6 and parts[0] == "":
                    try:
                        metric, val = parts[3], float(parts[5])
                        stderr = float(parts[7]) if len(parts) > 7 else 0
                        key = f"{task}/{metric}"
                        results[key] = {"value": val, "stderr": stderr}
                        log(f"    {key}: {val:.4f} ± {stderr:.4f}")
                    except: pass

            if not results:
                # Fallback: show last lines
                for line in p.stdout.splitlines()[-8:]:
                    if line.strip() and "|" in line:
                        log(f"    {line.strip()}")
        except Exception as e:
            log(f"    ERROR: {e}")
    return results

# ============ MAIN ============
os.makedirs(OUT, exist_ok=True)
for d in ["/tmp/jiangyon_triton_cache", "/tmp/jiangyon_vllm_cache", "/tmp/jiangyon_config"]:
    os.makedirs(d, exist_ok=True)

log("=" * 75)
log(f"  4-Way Benchmark: Separated / Attn-Fused / MoE-Fused / Both-Fused")
log(f"  Model: Qwen3-30B-A3B | GPU {GPU} | Output: {OUT}")
log("=" * 75)

all_results = {}
perf_csv = open(f"{OUT}/perf.csv", "w", newline="")
perf_writer = csv.writer(perf_csv)
perf_writer.writerow(["mode", "round", "concurrency", "tpot_ms", "ttft_ms", "throughput_tps"])

for label, model, extra in MODES:
    cleanup()
    wait_gpu_free()

    log("")
    log(f"{'='*60}")
    log(f"  [{label}]")
    log(f"  VLLM_USE_FUSED_ROTATION_QUANT={extra.get('VLLM_USE_FUSED_ROTATION_QUANT','?')}")
    log(f"  VLLM_MOE_FUSED_ROTATION={extra.get('VLLM_MOE_FUSED_ROTATION','?')}")
    log(f"{'='*60}")

    proc, slog = start_server(model, extra, label)
    if proc is None:
        all_results[label] = {"status": "SERVER_FAILED"}
        continue

    # 1. Check server log (CUDAGraph, dispatch)
    log("  --- Server Log Check ---")
    loginfo = check_server_log(label)
    for k, v in loginfo.items():
        log(f"    {k}: {v}")

    # 2. Correctness
    log("  --- Correctness ---")
    correctness = check_correctness(model)

    # 3. Performance
    log("  --- Performance ---")
    run_perf(model, label, perf_writer)
    perf_csv.flush()

    # 4. Quality
    log("  --- Quality ---")
    quality = run_quality(model, label)

    all_results[label] = {
        "loginfo": loginfo,
        "correctness": correctness,
        "quality": quality,
    }

    stop_server(proc, slog)

perf_csv.close()

# ============ SUMMARY ============
log("")
log("=" * 75)
log("  FINAL SUMMARY")
log("=" * 75)

# Performance summary
log("")
log("  === TPOT (ms) ===")
try:
    perf_data = {}
    with open(f"{OUT}/perf.csv") as f:
        for row in csv.DictReader(f):
            m, c = row["mode"], int(row["concurrency"])
            try:
                t = float(row["tpot_ms"])
                perf_data.setdefault(m, {}).setdefault(c, []).append(t)
            except: pass

    labels = [m[0] for m in MODES]
    header = f"  {'c':>3}" + "".join(f" | {l:>12}" for l in labels)
    log(header)
    log("  " + "-" * len(header))
    for c in [1, 4, 16, 32]:
        vals = []
        for l in labels:
            d = perf_data.get(l, {}).get(c, [])
            avg = sum(d)/len(d) if d else 0
            vals.append(f"{avg:.2f}ms" if avg > 0 else "N/A")
        log(f"  {c:>3}" + "".join(f" | {v:>12}" for v in vals))
except: pass

# Quality summary
log("")
log("  === Quality ===")
labels = [m[0] for m in MODES]
all_metrics = set()
for l in labels:
    q = all_results.get(l, {}).get("quality", {})
    all_metrics.update(q.keys())
if all_metrics:
    header = f"  {'metric':<35}" + "".join(f"{l:>14}" for l in labels)
    log(header)
    log("  " + "-" * (35 + 14 * len(labels)))
    for metric in sorted(all_metrics):
        vals = []
        for l in labels:
            v = all_results.get(l, {}).get("quality", {}).get(metric, {})
            if isinstance(v, dict) and "value" in v:
                vals.append(f"{v['value']:.4f}")
            else:
                vals.append("N/A")
        log(f"  {metric:<35}" + "".join(f"{v:>14}" for v in vals))

with open(f"{OUT}/all_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
log(f"")
log(f"  Results: {OUT}/")
log("=" * 75)

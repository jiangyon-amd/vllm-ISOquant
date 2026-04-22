#!/usr/bin/env python3
"""
3-Way Quality Evaluation: RTN / Separated / MoE-Fused
Tasks: wikitext (PPL), gsm8k (accuracy), arc_challenge (accuracy)
"""
import json, os, subprocess, sys, time, urllib.request
from datetime import datetime

PORT = 8400
GPU = "4"
MODEL = "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2"
RTN_MODEL = "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rtn"
OUT = "/data/jiangyon/vllm_rotation/bench_results/eval_3way_" + datetime.now().strftime("%Y%m%d_%H%M%S")

MODES = [
    ("rtn", RTN_MODEL, {}),
    ("separated", MODEL, {
        "VLLM_USE_FUSED_ROTATION_QUANT": "0",
        "VLLM_MOE_FUSED_ROTATION": "0",
    }),
    ("moe-fused", MODEL, {
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
}

LM_EVAL_LIMIT = 500

EVAL_RUNS = [
    # (task, num_fewshot, model_type, base_url_template)
    ("wikitext", 0, "local-completions",
     "base_url=http://localhost:{port}/v1/completions"),
    ("gsm8k", 5, "local-completions",
     "base_url=http://localhost:{port}/v1/completions"),
    ("arc_challenge", 25, "local-completions",
     "base_url=http://localhost:{port}/v1/completions"),
]

def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def cleanup():
    os.system("kill -9 $(pgrep -f 'api_server.*8400') $(pgrep -f EngineCore) 2>/dev/null")
    time.sleep(30)

def wait_gpu_free(timeout=120):
    """Wait until GPU memory drops below 2GB."""
    for i in range(timeout // 5):
        try:
            out = subprocess.check_output(
                "rocm-smi --showmeminfo vram 2>/dev/null | grep 'GPU\\[4\\]' | grep Used",
                shell=True, text=True)
            used = int(out.strip().split(":")[-1].strip())
            if used < 2_000_000_000:
                return True
        except:
            pass
        time.sleep(5)
    return False

def run_eval(model, task, num_fewshot, model_type, base_url_tpl, label, out_dir):
    log(f"  {task} ({num_fewshot}-shot)...")
    base_url = base_url_tpl.format(port=PORT)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", model_type,
        "--model_args", f"model={model},{base_url},num_concurrent=4,tokenized_requests=False",
        "--tasks", task,
        "--num_fewshot", str(num_fewshot),
        "--limit", str(LM_EVAL_LIMIT),
        "--output_path", f"{out_dir}/lm_eval_{label}_{task}",
        "--log_samples",
    ]
    lm_env = os.environ.copy()
    lm_env["PYTHONPATH"] = "/data/jiangyon/vllm_rotation:" + lm_env.get("PYTHONPATH", "")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=lm_env)
        raw_file = f"{out_dir}/lm_eval_{label}_{task}_raw.txt"
        with open(raw_file, "w") as f:
            f.write(p.stdout)
            if p.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(p.stderr[-3000:])

        results = {}
        for line in p.stdout.splitlines():
            if "|" not in line:
                continue
            parts = [x.strip() for x in line.split("|")]
            parts = [pp for pp in parts if pp]
            if len(parts) >= 7 and parts[0] not in ("-", "Tasks", ""):
                try:
                    metric = parts[3]
                    value = float(parts[5])
                    stderr = float(parts[7]) if len(parts) > 7 else 0
                    key = f"{parts[0]}/{metric}"
                    results[key] = {"value": value, "stderr": stderr}
                    log(f"    {key}: {value:.4f} +/- {stderr:.4f}")
                except (ValueError, IndexError):
                    pass
            elif len(parts) >= 7 and parts[0] == "":
                try:
                    metric = parts[3]
                    value = float(parts[5])
                    stderr = float(parts[7]) if len(parts) > 7 else 0
                    key = f"{task}/{metric}"
                    results[key] = {"value": value, "stderr": stderr}
                    log(f"    {key}: {value:.4f} +/- {stderr:.4f}")
                except (ValueError, IndexError):
                    pass

        if not results:
            log(f"    (no parsed results, last 15 lines:)")
            for line in p.stdout.splitlines()[-15:]:
                if line.strip():
                    log(f"      {line.strip()}")
        return results
    except Exception as e:
        log(f"    ERROR: {e}")
        return {}

# ============ MAIN ============
os.makedirs(OUT, exist_ok=True)
for d in ["/tmp/jiangyon_triton_cache", "/tmp/jiangyon_vllm_cache", "/tmp/jiangyon_config"]:
    os.makedirs(d, exist_ok=True)

all_results = {}
labels = [m[0] for m in MODES]

log(f"{'='*70}")
log(f"  3-Way Quality Eval (GPU {GPU}, limit={LM_EVAL_LIMIT})")
log(f"  Tasks: wikitext(PPL), gsm8k(5-shot), arc_challenge(25-shot)")
log(f"  Output: {OUT}")
log(f"{'='*70}")

for label, model, extra in MODES:
    cleanup()
    wait_gpu_free()
    log(f"")
    log(f"{'='*50}")
    log(f"  [{label}] model={os.path.basename(model)}")
    log(f"{'='*50}")

    env = os.environ.copy()
    env.update(BASE_ENV)
    for k in ["VLLM_USE_FUSED_ROTATION_QUANT", "VLLM_MOE_FUSED_ROTATION"]:
        env.pop(k, None)
    env.update(extra)
    env["HIP_VISIBLE_DEVICES"] = GPU

    server_log = open(f"{OUT}/server_{label}.log", "w")
    proc = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--port", str(PORT), "--trust-remote-code",
        "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
        "--host", "0.0.0.0", "--disable-log-requests",
    ], env=env, stdout=server_log, stderr=subprocess.STDOUT)

    ready = False
    for w in range(600):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            ready = True
            break
        except:
            if proc.poll() is not None:
                log(f"  Server died! Check {OUT}/server_{label}.log")
                break
            time.sleep(1)
    if not ready:
        log(f"  TIMEOUT!")
        all_results[label] = {"status": "TIMEOUT"}
        proc.terminate()
        server_log.close()
        continue
    log(f"  Server ready ({w}s)")

    mode_results = {}
    for task, nshot, model_type, base_url_tpl in EVAL_RUNS:
        r = run_eval(model, task, nshot, model_type, base_url_tpl, label, OUT)
        mode_results.update(r)

    all_results[label] = mode_results
    proc.terminate()
    try:
        proc.wait(15)
    except:
        proc.kill()
    server_log.close()

cleanup()

# ============ SUMMARY ============
log(f"")
log(f"{'='*70}")
log(f"  QUALITY RESULTS SUMMARY")
log(f"{'='*70}")

all_metrics = set()
for l in labels:
    r = all_results.get(l, {})
    if isinstance(r, dict):
        all_metrics.update(k for k in r.keys() if k != "status")

if all_metrics:
    header = f"  {'metric':<35}" + "".join(f"{l:>14}" for l in labels)
    log(header)
    log(f"  {'-'*35}" + "-" * 14 * len(labels))
    for metric in sorted(all_metrics):
        vals = []
        for l in labels:
            r = all_results.get(l, {})
            v = r.get(metric, {})
            if isinstance(v, dict) and "value" in v:
                vals.append(f"{v['value']:.4f}")
            else:
                vals.append("N/A")
        log(f"  {metric:<35}" + "".join(f"{v:>14}" for v in vals))

with open(f"{OUT}/results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
log(f"")
log(f"  JSON: {OUT}/results.json")
log(f"{'='*70}")

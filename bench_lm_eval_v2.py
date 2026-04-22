#!/usr/bin/env python3
"""
lm_eval quality benchmark v2 — Fixed:
  1. Proper few-shot: arc_challenge=25, hellaswag=10, gsm8k=5(default)
  2. Use local-completions for all tasks (loglikelihood works properly)
  3. limit=500 for tighter stderr
"""
import json, os, subprocess, sys, time, urllib.request
from datetime import datetime

PORT = 8300; GPU = "4"
MODEL = "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2"
RTN_MODEL = "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rtn"
OUT = "/data/jiangyon/vllm_rotation/bench_results/lm_eval_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S")

MODES = [
    ("rtn", RTN_MODEL, {}),
    ("separated", MODEL, {
        "VLLM_USE_FUSED_ROTATION_QUANT": "0",
        "VLLM_MOE_FUSED_ROTATION": "0",
    }),
    ("moe-fused", MODEL, {
        "VLLM_USE_FUSED_ROTATION_QUANT": "0",
        "VLLM_MOE_FUSED_ROTATION": "1",
    }),
    ("both-fused", MODEL, {
        "VLLM_USE_FUSED_ROTATION_QUANT": "1",
        "VLLM_MOE_FUSED_ROTATION": "1",
    }),
]

BASE = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1"}

LM_EVAL_LIMIT = 500

# Each task with proper num_fewshot
EVAL_RUNS = [
    # (tasks, num_fewshot, model_type, extra_args)
    ("arc_challenge", 25, "local-completions", 
     "base_url=http://localhost:{port}/v1/completions"),
    ("hellaswag", 10, "local-completions",
     "base_url=http://localhost:{port}/v1/completions"),
    ("gsm8k", 5, "local-completions",
     "base_url=http://localhost:{port}/v1/completions"),
]

def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def cleanup():
    os.system("kill -9 $(pgrep -f 'openai.api_server.*8300') $(pgrep -f EngineCore) 2>/dev/null")
    time.sleep(8)

def run_eval(model, task, num_fewshot, model_type, base_url_tpl, label, out_dir, env):
    log(f"  {task} ({num_fewshot}-shot, {model_type})...")
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
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
        raw_file = f"{out_dir}/lm_eval_{label}_{task}_raw.txt"
        with open(raw_file, "w") as f:
            f.write(p.stdout)
            if p.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(p.stderr[-2000:])
        
        # Parse results
        results = {}
        for line in p.stdout.splitlines():
            if "|" not in line:
                continue
            parts = [x.strip() for x in line.split("|")]
            parts = [p for p in parts if p]
            if len(parts) >= 7 and not parts[0].startswith("-") and parts[0] != "Tasks":
                task_name = parts[0] if parts[0] else task
                try:
                    metric = parts[3]  # Metric column
                    value = float(parts[5])  # Value column (after ↑)
                    stderr = float(parts[7]) if len(parts) > 7 else 0
                    key = f"{task_name}/{metric}"
                    results[key] = {"value": value, "stderr": stderr}
                    log(f"    {key}: {value:.4f} ± {stderr:.4f}")
                except (ValueError, IndexError):
                    pass
            # Also handle continuation rows (task name empty)
            elif len(parts) >= 7 and parts[0] == "":
                try:
                    metric = parts[3]
                    value = float(parts[5])
                    stderr = float(parts[7]) if len(parts) > 7 else 0
                    key = f"{task}/{metric}"
                    results[key] = {"value": value, "stderr": stderr}
                    log(f"    {key}: {value:.4f} ± {stderr:.4f}")
                except (ValueError, IndexError):
                    pass
        
        if not results:
            log(f"    (no parsed results, last 10 lines:)")
            for line in p.stdout.splitlines()[-10:]:
                if line.strip():
                    log(f"    {line.strip()}")
        return results
    except Exception as e:
        log(f"    ERROR: {e}")
        return {}

os.makedirs(OUT, exist_ok=True)
all_results = {}
labels_list = [m[0] for m in MODES]

log(f"=== lm_eval v2 (GPU {GPU}, limit={LM_EVAL_LIMIT}, proper few-shot) ===")
log(f"Output: {OUT}")

for label, model, extra in MODES:
    cleanup()
    log(f"======== [{label}] ========")
    log(f"  Model: {os.path.basename(model)}")

    env = os.environ.copy(); env.update(BASE)
    for k in ["VLLM_USE_FUSED_ROTATION_QUANT", "VLLM_MOE_FUSED_ROTATION"]:
        env.pop(k, None)
    env.update(extra)
    env["HIP_VISIBLE_DEVICES"] = GPU
    env["HF_HOME"] = "/data/jiangyon/hf_cache"
    env["PYTHONPATH"] = "/data/jiangyon/vllm_rotation:" + env.get("PYTHONPATH", "")

    lf = open(f"{OUT}/server_{label}.log", "w")
    proc = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--port", str(PORT), "--trust-remote-code",
        "--max-model-len", "4096", "--gpu-memory-utilization", "0.35",
        "--host", "0.0.0.0", "--disable-log-requests",
    ], env=env, stdout=lf, stderr=subprocess.STDOUT)

    ready = False
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            ready = True; break
        except: time.sleep(1)
    if not ready:
        log(f"  TIMEOUT!"); all_results[label] = {"status": "TIMEOUT"}
        proc.terminate(); lf.close(); continue
    log(f"  Server ready")

    lm_env = os.environ.copy()
    lm_env["PYTHONPATH"] = "/data/jiangyon/vllm_rotation:" + lm_env.get("PYTHONPATH", "")

    mode_results = {}
    for task, nshot, model_type, base_url_tpl in EVAL_RUNS:
        r = run_eval(model, task, nshot, model_type, base_url_tpl, label, OUT, lm_env)
        mode_results.update(r)
    
    all_results[label] = mode_results

    proc.terminate()
    try: proc.wait(10)
    except: proc.kill()
    lf.close()

cleanup()

# ============ SUMMARY ============
log("")
log(f"======== Quality Results (GPU {GPU}, limit={LM_EVAL_LIMIT}) ========")

all_metrics = set()
for label in labels_list:
    r = all_results.get(label, {})
    if isinstance(r, dict):
        all_metrics.update(k for k in r.keys() if k != "status")

if all_metrics:
    header = f"  {'metric':<35}" + "".join(f"{l:>14}" for l in labels_list)
    log(header)
    log(f"  {'-'*35}" + "-"*14*len(labels_list))
    for metric in sorted(all_metrics):
        vals = []
        for l in labels_list:
            r = all_results.get(l, {})
            v = r.get(metric, {})
            if isinstance(v, dict) and "value" in v:
                vals.append(f"{v['value']:.4f}±{v['stderr']:.3f}")
            else:
                vals.append("N/A")
        log(f"  {metric:<35}" + "".join(f"{v:>14}" for v in vals))

with open(f"{OUT}/results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
log(f"\nJSON: {OUT}/results.json")

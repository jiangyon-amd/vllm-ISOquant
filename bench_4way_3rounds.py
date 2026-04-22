#!/usr/bin/env python3
"""
4-Way MoE test, 3 rounds per mode, with warmup
  1. RTN           — no rotation
  2. Separated     — attn=sep, moe=sep
  3. MoE-Fused     — attn=sep, moe=fused(aiter Triton)
  4. Both-Fused    — attn=fused(Gluon), moe=fused(aiter Triton)
"""
import json, os, subprocess, sys, time, urllib.request, statistics
from datetime import datetime

PORT = 8300; GPU = "5"
MODEL = "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-trained-r128-vllm-v2"
RTN_MODEL = "/data/jiangyon/vllm_rotation/qwen3-30b-mxfp4-rtn"
OUT = "/data/jiangyon/vllm_rotation/bench_results/4way_3r_" + datetime.now().strftime("%Y%m%d_%H%M%S")
EXTRA = json.dumps({"chat_template_kwargs": {"enable_thinking": False}})
ROUNDS = 3

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

EXPECTED = {
    "separated":  (0, 48, 0, 48),
    "moe-fused":  (0, 48, 48, 0),
    "both-fused": (48, 0, 48, 0),
}

BASE = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1"}

TESTS = [
    ("只回答数字：1+1等于几？", lambda r: "2" in r),
    ("只回答数字：3*3等于几？", lambda r: "9" in r),
    ("What is the capital of France?", lambda r: "paris" in r.lower()),
    ("只回答数字：5+7等于几？", lambda r: "12" in r),
    ("What is the capital of Japan? One word only.", lambda r: "tokyo" in r.lower()),
]

def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def cleanup():
    os.system("kill -9 $(pgrep -f openai.api_server) $(pgrep -f EngineCore) 2>/dev/null")
    time.sleep(8)

def do_bench(model, np_v, conc, benv):
    p = subprocess.run([
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "openai", "--base-url", f"http://localhost:{PORT}",
        "--model", model, "--num-prompts", str(np_v),
        "--max-concurrency", str(conc), "--request-rate", "inf",
        "--dataset-name", "random", "--random-input-len", "512",
        "--random-output-len", "256", "--random-range-ratio", "0.2",
        "--num-warmups", "4", "--seed", "42",
        "--extra-body", EXTRA,
    ], capture_output=True, text=True, timeout=600, env=benv)
    def pick(k):
        for l in p.stdout.splitlines():
            if k in l: return float(l.split(":", 1)[1].strip().split()[0])
        return None
    return {"tpot": pick("Mean TPOT"), "tok_s": pick("Output token throughput")}

os.makedirs(OUT, exist_ok=True)
results = {}
labels_list = [m[0] for m in MODES]

log(f"=== 4-Way MoE, 3 Rounds (GPU {GPU}, CUDAGraph) ===")
log(f"Output: {OUT}")

for label, model, extra in MODES:
    cleanup()
    log(f"======== [{label}] ========")
    log(f"  Model: {os.path.basename(model)}")
    log(f"  Env: {extra}")

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
        log(f"  TIMEOUT!"); results[label] = {"status": "TIMEOUT"}; lf.close(); continue
    log(f"  Server ready")

    # Config check
    lf.flush(); os.fsync(lf.fileno())
    with open(f"{OUT}/server_{label}.log") as rf: slog = rf.read()
    cg = "FULL_AND_PIECEWISE" if "FULL_AND_PIECEWISE" in slog else "CHECK"
    log(f"  CUDAGraph: {cg}")

    if label in EXPECTED:
        af = slog.count("Using fused Triton")
        a_s = slog.count("Using separated")
        mf = slog.count("mode=fused")
        ms = slog.count("mode=separated")
        exp = EXPECTED[label]
        ok_cfg = (af == exp[0] and a_s == exp[1] and mf == exp[2] and ms == exp[3])
        log(f"  attn: fused={af} sep={a_s} | moe: fused={mf} sep={ms}")
        log(f"  Config: {'✓ CORRECT' if ok_cfg else '✗ MISMATCH!'}")
        if not ok_cfg:
            results[label] = {"status": "CONFIG_ERROR"}; proc.terminate(); lf.close(); continue

    # Correctness
    log(f"  Correctness:")
    corr_ok = True
    for prompt, chk in TESTS:
        try:
            d = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 20, "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False}}).encode()
            r = json.loads(urllib.request.urlopen(
                urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",
                                      data=d, headers={"Content-Type": "application/json"}),
                timeout=60).read())
            reply = r["choices"][0]["message"]["content"]
            ok = chk(reply)
            if not ok: corr_ok = False
            log(f"    {'PASS' if ok else 'FAIL'} {prompt[:30]} -> {reply[:30]}")
        except Exception as e:
            corr_ok = False; log(f"    FAIL {prompt[:30]} -> {e}")
    if not corr_ok:
        results[label] = {"status": "INCORRECT"}; proc.terminate(); lf.close(); continue

    # Fused kernel check (via aiter Triton fused_rot_quant_moe_sort)
    if "fused" in label and label != "separated":
        lf.flush(); os.fsync(lf.fileno())
        with open(f"{OUT}/server_{label}.log") as rf:
            slog2 = rf.read()
            triton_ok = "mode=fused" in slog2
            if "moe" in label or "both" in label:
                log(f"  Fused MoE dispatch: {'confirmed ✓' if triton_ok else 'NOT seen'}")

    # Warmup: 30 requests
    log(f"  Warming up (30 requests)...")
    benv = os.environ.copy()
    benv["PYTHONPATH"] = "/data/jiangyon/vllm_rotation:" + benv.get("PYTHONPATH", "")
    do_bench(model, 30, 4, benv)

    # 3 rounds
    results[label] = {"status": "OK", "rounds": [], "avg": {}}
    for rd in range(1, ROUNDS + 1):
        log(f"  --- Round {rd}/{ROUNDS} ---")
        round_data = {}
        for c in [1, 4, 16, 32]:
            np_val = {1: 40, 4: 40, 16: 60, 32: 80}[c]
            do_bench(model, 8, c, benv)  # mini warmup per concurrency
            r = do_bench(model, np_val, c, benv)
            round_data[c] = r
            log(f"    c={c}: TPOT={r['tpot']:.2f}ms  tok/s={r['tok_s']:.2f}")
        results[label]["rounds"].append(round_data)

    # Average
    for c in [1, 4, 16, 32]:
        tpots = [results[label]["rounds"][rd][c]["tpot"] for rd in range(ROUNDS) if results[label]["rounds"][rd][c]["tpot"]]
        toks = [results[label]["rounds"][rd][c]["tok_s"] for rd in range(ROUNDS) if results[label]["rounds"][rd][c]["tok_s"]]
        avg_tpot = statistics.mean(tpots) if tpots else None
        avg_tok = statistics.mean(toks) if toks else None
        std_tpot = statistics.stdev(tpots) if len(tpots) > 1 else 0
        results[label]["avg"][c] = {"tpot": avg_tpot, "tok_s": avg_tok, "tpot_std": std_tpot}
        log(f"  Avg c={c}: TPOT={avg_tpot:.2f}±{std_tpot:.2f}ms  tok/s={avg_tok:.2f}")

    proc.terminate()
    try: proc.wait(10)
    except: proc.kill()
    lf.close()

cleanup()

# ============ SUMMARY ============
log("")
log(f"======== RESULTS (GPU {GPU}, 3-Round Avg, CUDAGraph) ========")

# Per-round detail
for rd in range(ROUNDS):
    log(f"\n--- Round {rd+1} TPOT (ms) ---")
    log(f"  {'c':>3}  {'RTN':>8}  {'Sep':>8}  {'MoE-F':>8}  {'Both-F':>8}")
    for c in [1, 4, 16, 32]:
        vals = []
        for l in labels_list:
            v = results.get(l, {}).get("rounds", [{}]*(rd+1))[rd].get(c, {}).get("tpot")
            vals.append(f"{v:.2f}" if v else "N/A")
        log(f"  {c:>3}  {'  '.join(f'{v:>8}' for v in vals)}")

# Average
log(f"\n--- 3-Round Average TPOT (ms) ---")
log(f"  {'c':>3}  {'RTN':>10}  {'Sep':>10}  {'MoE-F':>10}  {'Both-F':>10}  {'Sep/RTN':>8}  {'MF/Sep':>8}  {'BF/Sep':>8}  {'BF/MF':>8}")
for c in [1, 4, 16, 32]:
    v = {}
    for l in labels_list:
        avg = results.get(l, {}).get("avg", {}).get(c, {})
        t = avg.get("tpot")
        s = avg.get("tpot_std", 0)
        v[l] = t
    def pct(a, b):
        try: return f"{(v[a]-v[b])/v[b]*100:+.1f}%"
        except: return "N/A"
    def fmt(l):
        t = v[l]
        s = results.get(l, {}).get("avg", {}).get(c, {}).get("tpot_std", 0)
        return f"{t:.2f}±{s:.2f}" if t else "N/A"
    log(f"  {c:>3}  {fmt('rtn'):>10}  {fmt('separated'):>10}  {fmt('moe-fused'):>10}  {fmt('both-fused'):>10}  {pct('separated','rtn'):>8}  {pct('moe-fused','separated'):>8}  {pct('both-fused','separated'):>8}  {pct('both-fused','moe-fused'):>8}")

log(f"\n--- 3-Round Average Throughput (tok/s) ---")
log(f"  {'c':>3}  {'RTN':>8}  {'Sep':>8}  {'MoE-F':>8}  {'Both-F':>8}  {'Sep/RTN':>8}  {'MF/Sep':>8}  {'BF/Sep':>8}  {'BF/MF':>8}")
for c in [1, 4, 16, 32]:
    v = {}
    for l in labels_list:
        v[l] = results.get(l, {}).get("avg", {}).get(c, {}).get("tok_s")
    def pct(a, b):
        try: return f"{(v[a]-v[b])/v[b]*100:+.1f}%"
        except: return "N/A"
    def fmt(l):
        return f"{v[l]:.1f}" if v[l] else "N/A"
    log(f"  {c:>3}  {fmt('rtn'):>8}  {fmt('separated'):>8}  {fmt('moe-fused'):>8}  {fmt('both-fused'):>8}  {pct('separated','rtn'):>8}  {pct('moe-fused','separated'):>8}  {pct('both-fused','separated'):>8}  {pct('both-fused','moe-fused'):>8}")

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nLogs: {OUT}/server_*.log")
log(f"JSON: {OUT}/results.json")

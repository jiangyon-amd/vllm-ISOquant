# Triton Kernel Optimization Skill Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a project skill that teaches an agent to generate a Triton kernel, profile it with `rocprofv3`/`rocprof`, and then perform two additional optimization rounds with profiling after every code generation or modification.

**Architecture:** The implementation has three parts: a project skill with reusable templates, a minimal Triton matmul demo scaffold, and a verification document that records RED/GREEN/REFACTOR behavior. The demo should preserve a three-round workflow: round 1 generates the baseline kernel, round 2 applies one focused optimization, and round 3 applies a second focused optimization, with correctness, benchmark, and `rocprof` analysis after every round.

**Tech Stack:** Markdown project skills, Python, Triton 3.5.1, PyTorch ROCm, `rocprofv3`, `rocprof`, shell scripts, lightweight CSV/JSON parsing.

---

## Chunk 1: Planning and Verification Contract

### Task 1: Lock the verification rubric and fixed prompts

**Files:**
- Modify if needed: `docs/superpowers/specs/2026-04-03-triton-kernel-optimization-skill-design.md`
- Create: `.cursor/skills/triton-kernel-optimization/verify-skill.md`

- [ ] **Step 1: Re-read the spec and extract the required verification criteria**

Run: no command required; use the spec document as the source of truth.
Expected: A short checklist covering baseline, correctness, profiling evidence, one-change-per-round discipline, and keep/revert/revise tracking.

- [ ] **Step 2: Write the fixed verification prompts**

Add three prompts to `verify-skill.md`:
- a prompt asking for three-round Triton GEMM optimization
- a prompt asking for `rocprof` analysis of a Triton kernel
- a prompt that adds time pressure like "just make it faster quickly"

Expected: The prompts are stable, copyable, and written in English only.

- [ ] **Step 3: Define the pass/fail rubric**

Write a scoring section in `verify-skill.md` with explicit 0/1 checks for:
- baseline established
- correctness checked
- profiling required
- one focused change per round
- keep/revert/revise recorded

Expected: A concrete threshold such as "pass if the skill-guided run satisfies all required checks and the no-skill baseline misses at least one."

- [ ] **Step 4: Add the profiler selection contract**

Document that verification should prefer `rocprofv3`, fall back to `rocprof`, and fail loudly if neither is available.

Expected: The verification document makes tool choice explicit and reproducible.

- [ ] **Step 5: Update the design spec only if execution details force a scope clarification**

If verification requirements or artifact boundaries materially drift from the approved design spec, sync the relevant section of the spec before proceeding. If no material drift appears, leave the spec unchanged.

Expected: The spec and verification document stay aligned without making unnecessary spec edits.

## Chunk 2: RED Phase for Skill Creation

### Task 2: Run baseline pressure scenarios before the new skill exists

**Files:**
- Modify: `.cursor/skills/triton-kernel-optimization/verify-skill.md`

- [ ] **Step 1: Create the skill directory without creating `SKILL.md` yet**

Run: `mkdir -p .cursor/skills/triton-kernel-optimization`
Expected: The directory exists, but the new skill is still absent.

- [ ] **Step 2: Run the fixed prompts without the new skill**

Use subagents in readonly mode and record the behavior in `verify-skill.md`.

Expected: At least one baseline run skips profiling, mixes multiple optimization ideas, or fails to require round-by-round evidence.

- [ ] **Step 3: Summarize the baseline failure patterns**

Write a short RED summary covering the rationalizations and missing workflow steps.

Expected: The RED summary directly informs what the skill must enforce.

## Chunk 3: Skill Content

### Task 3: Create the project skill and supporting templates

**Files:**
- Create: `.cursor/skills/triton-kernel-optimization/SKILL.md`
- Create: `.cursor/skills/triton-kernel-optimization/baseline-template.md`
- Create: `.cursor/skills/triton-kernel-optimization/round-prompt-template.md`
- Create: `.cursor/skills/triton-kernel-optimization/rocprof-template.md`
- Create: `.cursor/skills/triton-kernel-optimization/round-log-template.md`
- Create: `.cursor/skills/triton-kernel-optimization/stop-continue-template.md`

- [ ] **Step 1: Write the frontmatter and overview**

`SKILL.md` must use:
- a lowercase hyphenated name
- a `description` that starts with "Use when..."
- English-only wording

Expected: The description talks only about trigger conditions, not the internal workflow.

- [ ] **Step 2: Write the optimization workflow**

The skill must explicitly require:
- round 1 baseline generation
- round 2 first focused optimization
- round 3 second focused optimization
- correctness, benchmark, and `rocprof` analysis after every round
- exactly one main optimization hypothesis per round

Expected: The workflow is actionable and matches the user requirement exactly.

- [ ] **Step 3: Cover the optimization principles**

The skill or templates must cover:
- memory coalescing
- vectorized access
- tiling
- LDS / shared memory
- bank conflicts
- occupancy
- loop unrolling
- MFMA / `tl.dot`
- fusion / launch overhead
- persistent-kernel thinking
- double buffering

Expected: There is a visible coverage checklist or section tying these ideas to the workflow.

- [ ] **Step 4: Write the templates**

Required templates:
- `baseline-template.md` for round-1 recording
- `round-prompt-template.md` for round-2 and round-3 implementation prompts
- `rocprof-template.md` for the profiler command flow and fallback behavior
- `round-log-template.md` for recording correctness, benchmark, profiler evidence, and keep/revert/revise decisions
- `stop-continue-template.md` for deciding whether to continue, revert, or stop after a round

Expected: Every template is English-only and copyable.

## Chunk 4: Demo Scaffold

### Task 4: Build a runnable Triton matmul demo with three round variants

**Files:**
- Create: `examples/triton-kernel-opt-demo/README.md`
- Create: `examples/triton-kernel-opt-demo/baseline_matmul.py`
- Create: `examples/triton-kernel-opt-demo/optimized_matmul_round2.py`
- Create: `examples/triton-kernel-opt-demo/optimized_matmul_round3.py`
- Create: `examples/triton-kernel-opt-demo/benchmark.py`
- Create: `examples/triton-kernel-opt-demo/profile_rocprof.sh`
- Create: `examples/triton-kernel-opt-demo/analyze_rocprof.py`
- Create: `examples/triton-kernel-opt-demo/logs/.gitkeep`

- [ ] **Step 1: Write the baseline kernel module**

Implement a BF16 Triton matmul that:
- uses `tl.dot`
- has 6+ autotune configs
- uses `num_stages >= 3`
- handles arbitrary sizes with masks
- exposes a direct CLI or a benchmark-driven CLI path for `M`, `N`, `K`, dtype, and autotune control

Expected: `python3 examples/triton-kernel-opt-demo/benchmark.py --variant baseline --check` passes correctness.

- [ ] **Step 2: Write the round-2 optimized module**

Apply one focused optimization only, such as improved program mapping or better tiling/locality.

Expected: The round-2 file is clearly derived from round 1 and explains the one main change in comments or README text.

- [ ] **Step 3: Write the round-3 optimized module**

Apply one more focused optimization only, distinct from round 2.

Expected: The round-3 file preserves the round-2 gain path and adds exactly one main new optimization direction.

- [ ] **Step 4: Write the benchmark harness**

`benchmark.py` should:
- accept `--variant baseline|round2|round3`
- accept `--m`, `--n`, `--k`
- run correctness against `torch.matmul`
- use explicit `torch.testing.assert_close` tolerances
- report latency and TFLOPS
- optionally write round results to JSON

Expected: The benchmark output is machine-readable enough for verification.

- [ ] **Step 5: Write the profiling and analysis scripts**

`profile_rocprof.sh` should:
- auto-detect `rocprofv3` first, then `rocprof`
- take a variant name and output directory
- collect kernel trace data

`analyze_rocprof.py` should:
- find the relevant output files
- parse kernel durations
- print the top kernels
- highlight the likely Triton target kernel

Expected: A round can be profiled and analyzed without manual directory spelunking.

- [ ] **Step 6: Write the README**

The README must show the exact round sequence:
- round 1: generate baseline
- round 2: first optimization
- round 3: second optimization

For each round, document:
- correctness command
- benchmark command
- profiling command
- analysis command

Expected: A reader can reproduce the three rounds end-to-end.

## Chunk 5: GREEN Phase Verification

### Task 5: Verify that the new skill changes agent behavior

**Files:**
- Modify: `.cursor/skills/triton-kernel-optimization/verify-skill.md`

- [ ] **Step 1: Re-run the fixed prompts with the new skill present**

Use subagents again after `SKILL.md` exists.

Expected: The skill-guided runs now require a baseline, correctness, profiling, and one focused optimization per round.

- [ ] **Step 2: Record the GREEN results**

Add a comparison section in `verify-skill.md` that shows before/after behavior against the rubric.

Expected: The difference is specific and not hand-wavy.

## Chunk 6: Execute the Three-Round Demo Workflow

### Task 6: Run round 1, round 2, and round 3 with profiling after every code generation or modification

**Files:**
- Modify: `.cursor/skills/triton-kernel-optimization/verify-skill.md`
- Modify: `examples/triton-kernel-opt-demo/README.md` if commands or tolerances need correction

- [ ] **Step 1: Run round 1 correctness and benchmark**

Working directory: repository root
Run: `python3 examples/triton-kernel-opt-demo/benchmark.py --variant baseline --m 1024 --n 1024 --k 1024 --check --json-out examples/triton-kernel-opt-demo/logs/round1_benchmark.json`
Expected: Correctness passes and benchmark JSON is written.

- [ ] **Step 2: Run round 1 profiling and analysis**

Run: `bash examples/triton-kernel-opt-demo/profile_rocprof.sh baseline examples/triton-kernel-opt-demo/logs/round1_profile`
Run: `python3 examples/triton-kernel-opt-demo/analyze_rocprof.py --input examples/triton-kernel-opt-demo/logs/round1_profile --variant baseline`
Expected: Profiler output exists and analysis prints the top kernels and likely target kernel.

- [ ] **Step 3: Run round 2 correctness, benchmark, profiling, and analysis**

Run: `python3 examples/triton-kernel-opt-demo/benchmark.py --variant round2 --m 1024 --n 1024 --k 1024 --check --json-out examples/triton-kernel-opt-demo/logs/round2_benchmark.json`
Run: `bash examples/triton-kernel-opt-demo/profile_rocprof.sh round2 examples/triton-kernel-opt-demo/logs/round2_profile`
Run: `python3 examples/triton-kernel-opt-demo/analyze_rocprof.py --input examples/triton-kernel-opt-demo/logs/round2_profile --variant round2`
Expected: Round 2 has a complete evidence trail before moving to round 3.

- [ ] **Step 4: Run round 3 correctness, benchmark, profiling, and analysis**

Run: `python3 examples/triton-kernel-opt-demo/benchmark.py --variant round3 --m 1024 --n 1024 --k 1024 --check --json-out examples/triton-kernel-opt-demo/logs/round3_benchmark.json`
Run: `bash examples/triton-kernel-opt-demo/profile_rocprof.sh round3 examples/triton-kernel-opt-demo/logs/round3_profile`
Run: `python3 examples/triton-kernel-opt-demo/analyze_rocprof.py --input examples/triton-kernel-opt-demo/logs/round3_profile --variant round3`
Expected: Round 3 also has a complete evidence trail.

- [ ] **Step 5: Record the three-round comparison**

Use the round log template to summarize:
- the optimization principle used in round 2
- the optimization principle used in round 3
- how the profiler evidence changed
- whether each change should be kept, reverted, or revised

Expected: The verification document demonstrates the exact workflow the user requested.

## Chunk 7: REFACTOR and Final Checks

### Task 7: Tighten the skill based on verification results

**Files:**
- Modify: `.cursor/skills/triton-kernel-optimization/SKILL.md`
- Modify: `.cursor/skills/triton-kernel-optimization/verify-skill.md`
- Modify: any template file that needs clarification

- [ ] **Step 1: Identify new loopholes**

Review the GREEN results and the three-round demo for any failure modes, such as:
- ambiguous profiler instructions
- too many optimization ideas in one round
- missing keep/revert/revise logging

Expected: New loopholes are listed explicitly.

- [ ] **Step 2: Patch the skill and templates**

Update the language so the loopholes are closed without bloating the skill.

Expected: The skill becomes more robust and still stays concise.

- [ ] **Step 3: Re-run at least one verification prompt**

Run one final skill-guided prompt to ensure the revised language still works.

Expected: The final verification shows no regression in behavior.

### Task 8: Validate the deliverables

**Files:**
- Review: `.cursor/skills/triton-kernel-optimization/`
- Review: `examples/triton-kernel-opt-demo/`

- [ ] **Step 1: Run targeted lint/diagnostic checks**

Run: `python3 -m py_compile examples/triton-kernel-opt-demo/baseline_matmul.py examples/triton-kernel-opt-demo/optimized_matmul_round2.py examples/triton-kernel-opt-demo/optimized_matmul_round3.py examples/triton-kernel-opt-demo/benchmark.py examples/triton-kernel-opt-demo/analyze_rocprof.py`
Expected: No syntax errors.

- [ ] **Step 2: Check edited files for IDE diagnostics**

Use the linter/diagnostic tooling on the newly created files only.
Expected: No new blocking diagnostics remain.

- [ ] **Step 3: Final review**

Request a review of the new skill and demo artifacts before declaring completion.

Expected: Any issues found are fixed before wrapping up.

## Chunk 8: Follow-Up Hardening

### Task 9: Make profiling steady-state only

**Files:**
- Modify: `examples/triton-kernel-opt-demo/profile_rocprof.sh`
- Modify: `examples/triton-kernel-opt-demo/benchmark.py`
- Modify: `examples/triton-kernel-opt-demo/README.md`
- Modify: `.cursor/skills/triton-kernel-optimization/verify-skill.md`

- [ ] **Step 1: Reproduce the profiling pollution**

Demonstrate that the current profiling flow still mixes in autotune or other non-steady-state work.

- [ ] **Step 2: Implement a steady-state-only profiling path**

Change the benchmark/profile flow so the profiled region excludes autotune and one-time compilation work.

- [ ] **Step 3: Verify the cleaner profile**

Re-run profiling and confirm the target kernel trace now reflects steady-state execution more directly.

### Task 10: Replace round-3 with a stronger single optimization

**Files:**
- Modify: `examples/triton-kernel-opt-demo/optimized_matmul_round3.py`
- Modify: `examples/triton-kernel-opt-demo/README.md`
- Modify: `.cursor/skills/triton-kernel-optimization/verify-skill.md`

- [ ] **Step 1: Use round-1 and round-2 evidence to pick a better round-3 hypothesis**

The new round-3 change must be one focused optimization and should be more plausible than the previous mask-removal path.

- [ ] **Step 2: Implement the new round-3 change**

Keep round 2 intact and change only the round-3 optimization idea.

- [ ] **Step 3: Re-run the three-round comparison**

Run correctness, benchmark, and profiling again for rounds 1, 2, and 3 and update the decision log.

### Task 11: Port the validated skill to `gpu_kernel_skill`

**Files:**
- Target repository path to be determined from the GitEnterprise checkout

- [ ] **Step 1: Access the target repository**

Clone or open `https://gitenterprise.xilinx.com/AMDNeuralOpt/gpu_kernel_skill` using the available credentials.

- [ ] **Step 2: Copy the validated skill pack**

Port the English-only skill, templates, verification document, and any needed scaffold references into the target repository structure.

- [ ] **Step 3: Check the result in the target repository**

Verify the expected files exist and that the ported skill remains internally consistent.

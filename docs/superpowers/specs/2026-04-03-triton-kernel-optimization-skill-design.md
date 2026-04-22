# Triton Kernel Optimization Skill Design

**Date:** 2026-04-03

**Status:** Approved for implementation after review updates

**Goal:** Create a reusable project skill that teaches an agent how to optimize Triton kernels through evidence-driven, multi-round iteration with benchmarking and `rocprof` analysis. The skill should be generic enough for arbitrary Triton kernels, while using AMD/ROCm profiling and GEMM-style optimization as the default path.

## Problem Statement

The desired workflow is not "generate one faster kernel once." It is an optimization loop:

1. Start from a runnable baseline kernel.
2. Measure correctness and performance.
3. Profile with `rocprof` or `rocprofv3`.
4. Make one focused optimization change at a time.
5. Re-measure and decide whether to keep, revert, or revise.
6. Repeat until the target is reached or gains flatten out.

The PPT material emphasizes the same loop and the same optimization themes:

- global memory coalescing
- kernel fusion
- LDS / shared memory usage
- tiling hierarchy
- vectorized memory access
- warp / wave primitives
- occupancy tuning
- bank-conflict avoidance
- loop unrolling
- MFMA / tensor core usage
- persistent kernels
- double buffering
- graph-style replay / launch overhead reduction

The repository currently contains Triton kernels, benchmarks, and ROCm profiling helpers, but there is no single reusable skill that packages the optimization workflow into a repeatable process.

## Source Material Coverage

This skill is derived from the user-provided PPT themes discussed in this session. The implementation must explicitly cover the following ideas somewhere in the skill, templates, scaffold README, or verification notes:

- global memory coalescing
- kernel fusion
- LDS usage and conflict awareness
- tiling hierarchy
- vectorized loads and stores
- warp / wave communication
- occupancy tuning
- bank-conflict avoidance
- loop unrolling
- MFMA / matrix-core usage
- persistent kernel thinking
- double buffering
- profiling-after-each-round discipline

The implementation should include a short coverage section or checklist so the mapping from PPT themes to the skill content is auditable.

## Scope

### In Scope

- A new project skill under `.cursor/skills/`
- A skill focused on optimizing Triton kernels using a repeatable loop
- AMD/ROCm-first profiling guidance using `rocprof` / `rocprofv3`
- Templates for optimization prompts, profiling commands, and round logging
- A minimal scaffold that demonstrates how the skill should be used on a small Triton kernel example
- Skill verification that checks both discovery and actual behavior

### Out of Scope

- Delivering a production-best GEMM kernel for every shape
- Building a full benchmark framework for all kernels in the repository
- Auto-tuning infrastructure beyond what is needed for the demo scaffold
- Rewriting existing repository profiling scripts

## Primary Users

- A developer asking an agent to optimize a Triton kernel
- A developer who wants structured profiling evidence instead of ad hoc tuning
- A developer working on AMD/ROCm who wants `rocprof` to be a required part of the loop

## High-Level Design

The solution has three layers:

1. **Skill layer**
   A new skill explains when to use the workflow, how to run the optimization loop, which questions to answer before each round, and which evidence must be collected before claiming improvement.

2. **Template layer**
   Supporting markdown templates capture the reusable artifacts around the skill:
   - baseline checklist
   - round prompt template
   - `rocprof` command template
   - per-round results log
   - stop / continue decision template

3. **Scaffold layer**
   A minimal demo directory shows the workflow on a small Triton kernel so the skill is not just abstract documentation. The scaffold exists to validate the skill and to serve as a copyable example.

## Proposed Directory Layout

```text
vllm_rotation/
  .cursor/
    skills/
      triton-kernel-optimization/
        SKILL.md
        baseline-template.md
        round-prompt-template.md
        rocprof-template.md
        round-log-template.md
        verify-skill.md
  docs/
    superpowers/
      specs/
        2026-04-03-triton-kernel-optimization-skill-design.md
      plans/
        2026-04-03-triton-kernel-optimization-skill.md
  examples/
    triton-kernel-opt-demo/
      README.md
      baseline_matmul.py
      benchmark.py
      profile_rocprof.sh
      analyze_rocprof.py
      logs/
        .gitkeep
```

## Skill Behavior

The skill should trigger when the user asks for any of the following kinds of work:

- optimize a Triton kernel
- tune a GEMM / matmul kernel
- improve MFMA or matrix-core utilization
- profile a kernel with `rocprof` / `rocprofv3`
- perform multi-round GPU optimization
- use profiling evidence to guide Triton changes

The skill should instruct the agent to:

1. Confirm the baseline kernel and success metric.
2. Verify correctness before optimizing.
3. Benchmark the baseline and record timing / TFLOPS.
4. Run `rocprof` profiling and inspect the collected evidence.
5. Form one optimization hypothesis for the next round.
6. Make one focused change only.
7. Re-run correctness, benchmark, and profiling.
8. Record whether the change helped and why.
9. Repeat until gains flatten out or the target is achieved.

The skill must also define a fallback path:

- on AMD/ROCm, use `rocprofv3` by default
- if `rocprofv3` is unavailable but `rocprof` exists, fall back to `rocprof`
- on non-ROCm environments, keep the same optimization loop but replace the profiling tool with the platform-equivalent profiler and state clearly that the detailed profiling instructions are ROCm-specific

## Optimization Framework

The skill should organize optimization choices into concrete categories instead of vague "make it faster" instructions:

### 1. Memory

- improve global memory coalescing
- use wider vectorized loads / stores when alignment allows
- reduce redundant reads
- reduce write amplification

### 2. Tiling and Mapping

- tune `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
- tune `num_warps` and `num_stages`
- match tile shape to hardware execution width
- handle arbitrary sizes without polluting the hot path

### 3. On-Chip Data Reuse

- use LDS / shared-memory staging when beneficial
- reduce bank conflicts with layout or padding changes
- use double-buffering or software pipelining where supported

### 4. Compute Utilization

- prefer `tl.dot` / matrix-core paths when applicable
- raise MFMA usage and reduce scalar fallback behavior
- consider persistent-kernel style mappings if launch or reload overhead dominates

### 5. Fusion and Overhead

- fuse adjacent operations when it reduces traffic or launch overhead
- avoid unnecessary host synchronizations in the benchmark harness
- consider graph replay only if the workload and harness justify it

## Data Flow

The expected optimization loop is:

```text
baseline kernel
  -> correctness check
  -> benchmark
  -> rocprof collection
  -> metric analysis
  -> one optimization hypothesis
  -> one kernel change
  -> correctness re-check
  -> benchmark re-run
  -> rocprof re-run
  -> keep / revert / revise
  -> next round or stop
```

The scaffold should mirror this flow directly so that the skill and the example reinforce each other.

## Profiling Tool Contract

The implementation should standardize on the following behavior:

- default profiler: `rocprofv3`
- fallback profiler: `rocprof`
- the scaffold profiling script should auto-detect which one is available and print the selected tool
- the scaffold analysis script should parse the output format produced by the selected tool, or fail loudly with a clear message if the format is unsupported

The default AMD path should assume:

- kernel trace collection is available
- the analysis step can at least extract per-kernel duration
- the analysis step can identify the target Triton kernel name
- optional counters may be added later, but the minimum viable result is a ranked kernel-duration report plus the target kernel summary

The scaffold does not need to solve every ROCm version difference, but it must make the tool choice explicit and observable.

## Verification Strategy

The skill itself must be verified, not just written.

### RED

Before creating the new skill, run pressure scenarios without it and observe likely failure modes, such as:

- skipping `rocprof`
- proposing multiple unrelated optimizations in one round
- changing the kernel without recording the baseline
- claiming improvement based on intuition rather than evidence
- stopping after one benchmark without a profiling pass

The verification package should use fixed prompts so results are reproducible. At minimum, it should include:

1. a prompt asking the agent to optimize a Triton GEMM with multi-round tuning
2. a prompt asking the agent to analyze a Triton kernel with `rocprof`
3. a prompt asking the agent to improve a Triton kernel but with conflicting pressure, such as "just make it faster quickly"

For each prompt, verification should record:

- whether the agent establishes a baseline
- whether the agent asks for or runs correctness checks
- whether the agent requires profiling evidence
- whether the agent proposes exactly one focused optimization per round
- whether the agent records keep / revert / revise decisions

Passing means the behavior with the skill is materially better than the behavior without it on these criteria.

### GREEN

Write the minimal skill, templates, and scaffold needed to prevent those failures.

### REFACTOR

Re-run the same scenarios with the skill present, collect new loopholes, and tighten the language until the workflow becomes robust.

## Error Handling

The skill should explicitly cover common failure modes:

- no correctness baseline available
- profiling command fails or tool is unavailable
- performance changes but correctness regresses
- results vary too much between runs
- the round mixes several optimization ideas and becomes non-attributable
- arbitrary-size boundary handling accidentally dominates the hot path
- the profiler selected by the environment differs from the default tool
- the analysis script cannot map output files to the current round

The default behavior should be:

- stop and diagnose correctness failures before more tuning
- log tool failures instead of pretending profiling succeeded
- revert or isolate mixed optimization rounds
- require evidence before claiming a win

## Success Criteria

Implementation is successful when all of the following are true:

1. The new skill can be discovered from a prompt about optimizing Triton kernels with `rocprof`.
2. The skill directs the agent into a multi-round optimization loop rather than a one-shot rewrite.
3. The skill requires baseline benchmarking and profiling evidence.
4. The templates make each round easy to record and compare.
5. The demo scaffold is runnable and clearly shows the intended workflow.
6. Verification demonstrates a behavioral difference before and after the skill exists.
7. The profiler selection and output format are explicit enough that the demo can either run or fail loudly for a known reason.
8. Correctness checks in the demo are tied to a defined reference and tolerance.

## Recommended Defaults

- Store the skill as a project skill in this repository.
- Make AMD/ROCm profiling the default detailed path.
- Keep the skill generic to Triton kernels, not only GEMM.
- Use GEMM / matmul as the default example because it maps well to the PPT guidance.
- Prefer concise instructions in `SKILL.md`, with templates moved to supporting files.
- Write all skill-facing artifacts in English only. This includes `SKILL.md`, template files, scaffold `README.md`, verification docs, and any generated prompt text. Do not include Chinese in those artifacts.

## Demo Scaffold Contract

The demo scaffold should be intentionally small and should not attempt to be a full benchmark suite.

### `baseline_matmul.py`

- contains a simple Triton kernel with arbitrary-size masking
- exposes a small CLI for `M`, `N`, `K`, dtype, and autotune toggle
- compares against a PyTorch reference

### `benchmark.py`

- runs warmup and timed repeats
- reports average latency and estimated TFLOPS
- writes a compact baseline or round result record

### `profile_rocprof.sh`

- detects `rocprofv3` first, then `rocprof`
- runs the target command with kernel tracing
- writes outputs to a round-specific subdirectory

### `analyze_rocprof.py`

- accepts the profiling output directory as input
- extracts at least the top kernels by duration
- highlights the target Triton kernel
- prints enough evidence to support the next optimization hypothesis

### `README.md`

- shows the exact round sequence: baseline -> benchmark -> profile -> analyze -> modify -> re-run
- explains that one round should contain one main optimization hypothesis

## Correctness Standard

The scaffold should define correctness relative to a PyTorch reference implementation.

- primary reference: `torch.matmul`
- default dtype path: BF16 inputs with FP32 accumulation where applicable
- comparison method: `torch.testing.assert_close`
- default tolerance for the demo should be documented in the README and benchmark script

The exact tolerance can be tuned during implementation, but it must be explicit rather than implied.

## Spec vs Plan Responsibilities

This spec defines the target behavior, structure, and validation requirements.

The implementation plan document will translate this spec into step-by-step tasks, exact file paths, execution order, and verification commands. If the plan materially changes scope, the spec must be updated as well.

## Implementation Notes

- Avoid touching existing user-modified files outside the new skill, docs, and example directories.
- Keep the demo self-contained.
- Favor ASCII-only content unless an existing file pattern requires otherwise.
- Keep all newly created skill artifacts English-only.
- Do not claim the skill works until the RED/GREEN/REFACTOR verification loop has been executed.

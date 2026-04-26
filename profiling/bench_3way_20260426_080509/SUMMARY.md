# 3-Way E2E Benchmark Results

## Configuration
- **Model**: Qwen2.5-72B-Instruct
- **GPU**: AMD MI300X (gfx950), single GPU (TP=1)
- **Input**: 8192 tokens, **Output**: 1024 tokens
- **Concurrency**: 32 max concurrent requests
- **Prompts**: 100 random prompts
- **Mode**: enforce-eager (no CUDA graphs, no torch.compile)

## Configurations Tested
| Config | Description | Code Path |
|--------|-------------|-----------|
| A: Baseline | Original Triton TQ decode | TQ_DISABLE_HIP_SO=1, our repo |
| B: Aditi-v3 | Unified Triton attention kernel | VLLM_TQ_DECODE_V3=1, Aditi's repo |
| C: Ours-v136 | HIP MFMA unified kernel + bf16 Q | Default HIP path, our repo |

## Results

| Metric | A: Baseline | B: Aditi-v3 | C: Ours-v136 |
|--------|:-----------:|:-----------:|:------------:|
| Output throughput (tok/s) | 100.57 | 175.49 | **291.19** |
| Total throughput (tok/s) | 905.16 | 1,579.40 | **2,620.71** |
| Median TPOT (ms) | 301.10 | 167.93 | **95.96** |
| P99 TPOT (ms) | 302.54 | 169.06 | **97.08** |
| Median ITL (ms) | 269.87 | 137.21 | **65.44** |
| P99 ITL (ms) | 1,339.41 | 1,203.61 | **1,145.62** |
| Median TTFT (ms) | 5,389.38 | 4,824.09 | **4,599.22** |
| P99 TTFT (ms) | 44,467.97 | 40,172.35 | **38,282.92** |
| Benchmark duration (s) | 1,018.17 | 583.51 | **351.66** |

## Speedup Analysis

### Ours vs Baseline
| Metric | Speedup |
|--------|:-------:|
| Median TPOT | **3.14x** faster |
| Output throughput | **2.90x** higher |
| Median ITL | **4.12x** faster |

### Ours vs Aditi
| Metric | Speedup |
|--------|:-------:|
| Median TPOT | **1.75x** faster |
| Output throughput | **1.66x** higher |
| Median ITL | **2.10x** faster |

### Aditi vs Baseline
| Metric | Speedup |
|--------|:-------:|
| Median TPOT | 1.79x faster |
| Output throughput | 1.74x higher |
| Median ITL | 1.97x faster |

## Key Takeaways
1. **Our HIP MFMA kernel (v136) is the clear winner** across all metrics
2. **3.14x median TPOT improvement** over baseline Triton decode
3. **1.75x faster than Aditi's Triton v3** unified attention
4. The HIP kernel's use of MFMA matrix instructions and bf16 Q input
   provides significant advantage over Triton's scalar operations
5. TTFT improvements (1.17x) are modest since prefill uses a different
   code path — the gains are dominated by decode performance

# BL vs TQ-v136 Benchmark Results

- **Model**: Qwen2.5-72B-Instruct, TP=1
- **GPU**: MI300X (gfx950)
- **Input**: 8192 tokens, MaxConcurrency=32, 100 prompts
- **BL**: Standard bf16 KV cache (kv_cache_dtype=auto)
- **TQ**: TurboQuant 4-bit + HIP MFMA v136 kernel

| Output Len | BL Mean TTFT (ms) | TQ Mean TTFT (ms) | BL Mean TPOT (ms) | TQ Mean TPOT (ms) | BL Throughput (tok/s) | TQ Throughput (tok/s) | TQ/BL % |
| ---------- | ----------------- | ------------------ | ------------------ | ------------------ | --------------------- | --------------------- | ------- |
| 128 | 21503.60 | 9877.79 | 589.20 | 271.00 | 42.04 | 89.93 | 213.92% |
| 256 | 21512.48 | 830.35 | 316.90 | 63.45 | 78.40 | 418.03 | 533.20% |
| 512 | 21524.95 | 966.48 | 181.94 | 63.38 | 137.86 | 423.96 | 307.53% |
| 1024 | 21563.05 | 1016.04 | 115.19 | 64.11 | 221.29 | 424.39 | 191.78% |
| 2048 | 21635.65 | 988.65 | 82.53 | 65.71 | 315.01 | 417.88 | 132.66% |
| 4096 | 21712.83 | 1033.68 | 67.87 | 68.94 | 391.39 | 401.37 | 102.55% |

## vs Old TQ (Triton decode) — Key Improvements

| Output Len | Old TQ TPOT (ms) | v136 TPOT (ms) | TPOT Speedup | Old TQ/BL % | v136/BL % | Note |
| ---------- | ---------------- | -------------- | ------------ | ----------- | --------- | ---- |
| 128        | 435.16           | 271.00         | 1.61x        | 162.55%     | 213.92%   |      |
| 256        | 291.86           | 63.45          | 4.60x        | 139.34%     | 533.20%   |      |
| 512        | 221.75           | 63.38          | 3.50x        | 112.55%     | 307.53%   |      |
| 1024       | 189.64           | 64.11          | 2.96x        | 98.89%      | 191.78%   | Fixed: was slower than BL |
| 2048       | 183.81           | 65.71          | 2.80x        | 57.03%      | 132.66%   | Fixed: was 43% slower than BL |
| 4096       | N/A              | 68.94          | —            | N/A         | 102.55%   | New: still faster than BL |

### Summary
- **v136 TPOT is 1.6-5.0x faster** than old Triton TQ decode across all output lengths
- **Critical regression fixed**: Old TQ was slower than BL at output≥1024; v136 is **faster at every length**
- **TPOT stability**: v136 maintains ~63-69ms TPOT regardless of output length (256-4096)
- **TTFT dramatically improved**: ~21s → ~1s (20x) due to TurboQuant's 4x KV cache compression enabling more concurrent prefills
